#!/usr/bin/env python3
"""
Collect, from the DiscoForcing runs on this machine, the numbers the TMR supplementary still needs for DiscoForcing
(seeds/variance and the user studies are left out), and write one file to fill the paper from:
<out>/SUPP_FILL.md, with everything raw in <out>/supp_collect.json. CPU only; reads, never modifies, the runs and
raw_data/cl_sessions/<benchmark>/eval.

Stages (paper: sec/X_suppl.tex):
  sessions   app:disco "Sessions": train / val / test sequences per session, candidate clips and buffer budget, the
             genre distance from every earlier session to each later one, whether every such curve is monotone, and a
             re-check that the manifest's pairing and order are what the stated rules give for its distances
  protocol   app:disco "Generator": the reported runs' saved configs (outputs/.../sanity_check/*.yaml) and, unless
             --no_ckpt_meta, their last.ckpt (global step, LR at the end, EMA decay): cosine decay over each session,
             EMA 0.999, steps, seed, replay settings; which runs deviate
  buffers    per method and session: clips, budget, candidates, selection seconds, weights (effective size), and TMR's
             log line (ranks under each N_eff, OMP stop, modal error)
  facts      tab:instantiation: T x D of the trajectory; whether the music encoder changes from session to session
  tables     the screen's paper_disco_tables.{md,tex} and matrix tables, and which runs are complete
  sr_snr     tab:sr_snr on the final pairing: FID_k / FID_g with per-call (metrics/m2d.py) and frozen normalisation,
             the handcrafted genre classifier and StyleRetention (same split halves as metrics.cl_metrics build)
  sr_valid   tab:sr_valid and the numbers in the text of app:sr, from metrics.style_encoder validate's validation.json
  sanity     app:sr "Validation": SR of real dance, autoencoder reconstructions, an untrained generator and the partly
             trained S0 model, from the scripts/eval_cl_session.py outputs that run_supp_collect.sh writes

  python scripts/supp/collect_disco_supp.py --tag v2_40000_full_s0v2 --out results/aistpp_genre5/supp
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())

PAPER_VARIANTS = ["vanilla", "random", "random_kd", "random_star", "ocs", "gcr", "gcr_kd", "mir", "mir_kd", "csrel",
                  "csrel_kd", "tmr_joint_neff", "tmr_joint_neff_kd"]
NAME = {"vanilla": "Fine-tuning", "random": "ER", "random_kd": "ER+KD", "random_star": "STAR", "ocs": "OCS",
        "gcr": "GCR", "gcr_kd": "GCR+KD", "mir": "MIR", "mir_kd": "MIR+KD", "csrel": "CSReL", "csrel_kd": "CSReL+KD",
        "tmr_joint_neff": "TMR", "tmr_joint_neff_kd": "TMR+KD"}
PROTOCOL_KEYS = ["seed", "trainer.max_steps", "trainer.precision", "lr_scheduler.target", "lr_scheduler.params.T_max",
                 "lr_scheduler.params.eta_min", "model.ema_decay", "optimizer.target", "optimizer.params.lr",
                 "optimizer.params.betas", "optimizer.params.weight_decay", "data.train_bs", "data.train_max_tokens",
                 "init_from_ema", "replay.weight", "replay.batch_size", "replay.use_sample_weights", "replay.balance",
                 "replay.distill.enable", "replay.distill.gt_weight", "replay.distill.kd_weight", "replay.star.enable",
                 "model.params.noise_steps", "model.params.cfg_scale", "model.params.history_guidance_scale",
                 "model.params.prediction_type", "model.params.music_dim", "data.music_path", "data.token_path"]


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def read_list(p: Path) -> list[str]:
    return [l.strip() for l in Path(p).open() if l.strip()]


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, Path):
        return str(o)
    return o


def f3(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


class Ctx:
    def __init__(self, a):
        self.a = a
        self.B = a.benchmark
        self.tag = a.tag
        self.res = Path("results") / self.B / "screen" / self.tag
        self.outp = Path("outputs") / "screen" / self.tag
        self.bufr = Path("buffers") / self.B / "screen" / self.tag
        self.logs = Path("logs") / self.B / "screen" / self.tag
        self.root = Path("raw_data/cl_sessions") / self.B
        self.out = Path(a.out)
        self.man = json.loads((self.root / "manifest.json").read_text())
        self.K = len(self.man["sessions"])
        self.variants = self._variants()
        self.s0_run = self._s0_run()

    def _variants(self) -> list[str]:
        found = sorted(p.name for p in self.res.iterdir() if p.is_dir() and not p.name.startswith(".")
                       and p.name != "s0_model") if self.res.is_dir() else []
        if self.a.variants:
            return [v for v in found if v in self.a.variants or re.sub(r"_seed\d+$", "", v) in self.a.variants]
        return found

    def _s0_run(self) -> Path | None:
        if os.environ.get("S0_RUN"):
            return Path(os.environ["S0_RUN"])
        p = self.res / "s0_model" / "after_s0_sess0.json"
        if p.exists():
            ck = json.loads(p.read_text()).get("ckpt")
            if ck:
                return Path(ck).parent
        return None

    def run_dir(self, variant: str, k: int) -> Path | None:
        if k == 0:
            return self.s0_run
        ds = sorted(glob.glob(str(self.outp / f"*_cl_ldf_{self.B}_s{k}_{variant}")))
        return Path(ds[-1]) if ds else None


def base_method(v: str) -> str:
    return re.sub(r"_seed\d+$", "", v)


# ----------------------------------------------------------------------------------------------
# sessions
# ----------------------------------------------------------------------------------------------
def stage_sessions(c: Ctx) -> dict:
    man, K = c.man, c.K
    sess = []
    for k, s in enumerate(man["sessions"]):
        row = {"session": k, "groups": s["groups"], "names": s.get("group_names"), "train_minutes": s.get("train_minutes"),
               "manifest_counts": {sp: s.get(sp) for sp in ("train", "val", "test")}}
        for sp in ("train", "val", "test"):
            p = c.root / f"s{k}" / f"{sp}.txt"
            names = read_list(p) if p.exists() else []
            row[sp] = len(names)
            row[f"{sp}_per_genre"] = {g: sum(n[1:3] == g for n in names) for g in s["groups"]}
            lp = Path(man["feature_dir"]).parent / f"cl_{c.B}_s{k}_{sp}.txt"  # the loader's copy of the list
            row[f"{sp}_loader_list_matches"] = (sorted(read_list(lp)) == sorted(names)) if lp.exists() else None
        # candidate clips / budget of the buffer of this session (any method: all tile the same training list)
        for p in sorted(c.bufr.glob(f"*/s{k}.json")):
            b = json.loads(p.read_text())
            row["n_candidates"], row["budget"] = b.get("n_candidates"), b.get("budget")
            row["candidate_tokens"] = b.get("candidate_tokens")
            break
        sess.append(row)

    gd = man.get("genre_distance", {})
    d = {}
    for key, v in gd.items():
        x, y = key.split("-")
        d[(x, y)] = d[(y, x)] = float(v)

    def dist(a, b):
        return 0.0 if a == b else d[(a, b)]

    groups = [tuple(s["groups"]) for s in man["sessions"]]
    G = lambda g1, g2: float(np.mean([dist(a, b) for a in g1 for b in g2]))
    D = [[G(groups[i], groups[j]) if i != j else float(np.mean([dist(a, b) for a in groups[i] for b in groups[i]
                                                                   if a != b] or [0.0]))
          for j in range(K)] for i in range(K)]
    curves = []
    for i in range(K - 1):
        cv = [G(groups[i], groups[t]) for t in range(i + 1, K)]
        curves.append({"session": i, "distance_to_later": cv,
                       "monotone_nondecreasing": bool(all(b >= a - 1e-12 for a, b in zip(cv, cv[1:]))),
                       "strictly_increasing": bool(all(b > a for a, b in zip(cv, cv[1:])))})
    within = {f"S{k}": dist(*groups[k]) if len(groups[k]) == 2 else None for k in range(K)}
    out = {"sessions": sess, "group_distance_matrix": D, "curves": curves, "within_session_distance": within,
           "manifest_order_stats": man.get("order_stats"), "pairing_rule": man.get("pairing"), "order_rule": man.get("order"),
           "test_protocol": man.get("test_protocol"), "held_out_music": man.get("held_out_music")}
    # pairwise curve check as the session builder counts it (fraction of (i, t < t2) with d(i,t) <= d(i,t2))
    pairs = [G(groups[i], groups[t]) <= G(groups[i], groups[t2])
             for i in range(K - 2) for t in range(i + 1, K) for t2 in range(t + 1, K)]
    out["monotone_fraction"] = float(np.mean(pairs)) if pairs else None
    # re-derive pairing and order from the stored distances with the stated rules (scripts/make_cl_sessions.py)
    try:
        sys.path.insert(0, "scripts")
        import make_cl_sessions as M

        genres = sorted({g for gr in groups for g in gr})
        dd = {(x, y): v for (x, y), v in d.items() if x < y}
        pairing = [tuple(p) for p in M.best_pairing(genres, dd)]
        n_train = {g: 0 for g in genres}
        for k in range(K):
            for n in read_list(c.root / f"s{k}" / "train.txt"):
                if n[1:3] in n_train:
                    n_train[n[1:3]] += 1
        order = M.escalating_order(pairing, dd, lambda grp: sum(n_train[g] for g in grp))
        same_pairing = sorted(map(lambda g: tuple(sorted(g)), pairing)) == sorted(map(lambda g: tuple(sorted(g)), groups))
        same_order = [tuple(sorted(g)) for g in order] == [tuple(sorted(g)) for g in groups]
        out["rederived"] = {"pairing": [list(p) for p in pairing], "order": [list(g) for g in order],
                            "pairing_matches_manifest": bool(same_pairing), "order_matches_manifest": bool(same_order),
                            "order_stats_rederived": M.order_stats(order, dd),
                            "order_stats_manifest_sessions": M.order_stats(groups, dd)}
    except Exception as e:  # noqa: BLE001
        out["rederived"] = {"error": repr(e)}
    return out


# ----------------------------------------------------------------------------------------------
# protocol
# ----------------------------------------------------------------------------------------------
def _sel(cfg, key):
    from omegaconf import OmegaConf

    v = OmegaConf.select(cfg, key, default="<absent>", throw_on_resolution_failure=False)
    if OmegaConf.is_config(v):
        v = OmegaConf.to_container(v, resolve=False)
    return v


def ckpt_meta(path: Path) -> dict:
    import torch

    try:
        ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception:  # noqa: BLE001 (older torch: no mmap)
        ck = torch.load(path, map_location="cpu", weights_only=False)
    out = {"global_step": ck.get("global_step"), "epoch": ck.get("epoch")}
    sch = (ck.get("lr_schedulers") or [{}])[0]
    out.update({"sched_T_max": sch.get("T_max"), "sched_eta_min": sch.get("eta_min"), "sched_last_epoch": sch.get("last_epoch"),
                "sched_base_lrs": sch.get("base_lrs"), "sched_last_lr": sch.get("_last_lr")})
    try:
        out["optimizer_lr_now"] = ck["optimizer_states"][0]["param_groups"][0]["lr"]
    except Exception:  # noqa: BLE001
        pass
    ema = ck.get("ema_state") or {}
    out["ema_decay"] = ema.get("decay")
    out["ema_num_updates"] = ema.get("num_updates")
    del ck
    return out


def stage_protocol(c: Ctx) -> dict:
    from omegaconf import OmegaConf

    runs = []
    todo = [("s0_model", 0, c.s0_run)] + [(v, k, c.run_dir(v, k)) for v in c.variants for k in range(1, c.K)]
    for v, k, d in todo:
        if d is None or not Path(d).is_dir():
            continue
        row = {"variant": v, "session": k, "run_dir": str(d)}
        ys = sorted(Path(d, "sanity_check").glob("*.yaml"))
        if ys:
            cfg = OmegaConf.load(ys[0])
            row["config_file"] = str(ys[0])
            row["config"] = {kk: _sel(cfg, kk) for kk in PROTOCOL_KEYS}
            row["init_ckpt"] = _sel(cfg, "init_ckpt")
            row["replay_buffer_dir"] = _sel(cfg, "replay.buffer_dir")
            row["train_list"] = _sel(cfg, "data.train_meta_paths")
        last = Path(d) / "last.ckpt"
        row["has_last_ckpt"] = last.exists()
        if last.exists() and not c.a.no_ckpt_meta:
            try:
                row["ckpt"] = ckpt_meta(last)
            except Exception as e:  # noqa: BLE001
                row["ckpt"] = {"error": repr(e)}
        found = (re.search(r"step=(\d+)", p.name) for p in Path(d).glob("*step=*.ckpt"))
        row["step_ckpts"] = sorted(int(mm.group(1)) for mm in found if mm)
        runs.append(row)
        log(f"[protocol] {v} S{k}: {d}")
    # summary per field: values over the runs, and the runs that differ from the most common one
    summ = {}
    for kk in PROTOCOL_KEYS:
        vals = [(r["variant"], r["session"], json.dumps(r["config"][kk])) for r in runs if "config" in r
                and r["config"][kk] != "<absent>"]  # e.g. S0 has no replay block
        if not vals:
            continue
        counts = {}
        for _, _, v in vals:
            counts[v] = counts.get(v, 0) + 1
        common = max(counts, key=counts.get)
        summ[kk] = {"values": counts, "most_common": json.loads(common),
                    "deviating": [f"{a} S{b}={v}" for a, b, v in vals if v != common]}
    checks = []
    for r in runs:
        cf = r.get("config", {})
        ck = r.get("ckpt", {}) or {}
        cos = str(cf.get("lr_scheduler.target", "")).endswith("CosineAnnealingLR")
        tm, ms = cf.get("lr_scheduler.params.T_max"), cf.get("trainer.max_steps")
        checks.append({"run": f"{r['variant']} S{r['session']}", "cosine": cos, "T_max": tm, "max_steps": ms,
                       "cosine_over_session": bool(cos and tm is not None and ms is not None and int(tm) == int(ms)),
                       "ema_decay_config": cf.get("model.ema_decay"), "ema_decay_ckpt": ck.get("ema_decay"),
                       "global_step": ck.get("global_step"), "lr_end": ck.get("optimizer_lr_now"),
                       "eta_min": cf.get("lr_scheduler.params.eta_min"), "seed": cf.get("seed"),
                       "init_from_ema": cf.get("init_from_ema")})
    return {"runs": runs, "summary": summ, "checks": checks}


# ----------------------------------------------------------------------------------------------
# buffers
# ----------------------------------------------------------------------------------------------
LOG_PATTERNS = {
    "M": r"M=(\d+) modes by (\w+)", "energy90_M": r"\(90% energy: (\d+)", "M_bic_N": r"BIC N_eff=N: (\d+)",
    "M_bic_ND": r"BIC N_eff=N\*D: (\d+)", "D_eff": r"D_eff=([\d.]+)\): (\d+)", "n_eff": r"BIC n_eff=(\d+) \((\d+) same-music pairs\): (\d+)",
    "variance_kept": r"([\d.]+)% of temporal variance", "omp": r"OMP picked (\d+)/(\d+)", "fill": r"(k-center|pursuit) fill (\d+)",
    "objective": r"objective ([\d.eE+-]+) -> ([\d.eE+-]+)", "modal_error": r"modal error ([\d.eE+-]+)",
    "effective_size": r"effective size ([\d.]+) of (\d+)", "music_windows": r"music windows: (\d+) clips -> (\d+) distinct",
}


def parse_buffer_log(path: Path) -> dict:
    text = path.read_text(errors="replace")
    lines = [l for l in text.splitlines() if l.startswith("[") or l.startswith("saved") or l.startswith("session ")]
    out = {"log": str(path), "lines": lines[-12:]}
    for k, pat in LOG_PATTERNS.items():
        m = None
        for m in re.finditer(pat, text):
            pass
        if m:
            out[k] = list(m.groups())
    return out


def buffer_logs(c: Ctx, variant: str, j: int) -> list[Path]:
    cands = [c.logs / f"buffer_s{j}_{variant}.log"]
    if j == 0:
        cands += [c.logs / f"buffer_{variant}.log", c.logs / f"buffer_s0_{variant}.log",
                  c.logs / f"buffer_{base_method(variant)}.log", c.logs / f"buffer_s0_{base_method(variant)}.log",
                  c.logs / f"buffer_{re.sub(r'_kd$', '', base_method(variant))}.log"]
    return [p for p in cands if p.exists()]


def buffer_hash(clips) -> str:
    key = sorted((cl["name"], int(cl["start"]), int(cl["length"]), round(float(cl.get("weight", 1.0)), 6)) for cl in clips)
    return hashlib.md5(json.dumps(key).encode()).hexdigest()[:12]


def stage_buffers(c: Ctx) -> dict:
    rows = []
    for vdir in sorted(p for p in c.bufr.iterdir() if p.is_dir()) if c.bufr.is_dir() else []:
        v = vdir.name
        for p in sorted(vdir.glob("s*.json")):
            m = re.fullmatch(r"s(\d+)\.json", p.name)
            if not m:
                continue
            j = int(m.group(1))
            b = json.loads(p.read_text())
            w = np.array([float(cl.get("weight", 1.0)) for cl in b["clips"]])
            row = {"variant": v, "session": j, "method": b.get("method"), "clips": len(b["clips"]),
                   "budget": b.get("budget"), "n_candidates": b.get("n_candidates"), "ratio": b.get("ratio"),
                   "sequences": b.get("n_sequences"), "buffer_tokens": b.get("buffer_tokens"),
                   "candidate_tokens": b.get("candidate_tokens"), "seconds": b.get("seconds"), "ckpt": b.get("ckpt"),
                   "params": b.get("params"), "hash": buffer_hash(b["clips"]),
                   "weights": {"mean": float(w.mean()), "min": float(w.min()), "max": float(w.max()),
                               "effective_size": float(w.sum() ** 2 / (w ** 2).sum()), "n": int(len(w))}}
            row["clips_below_budget"] = (row["budget"] is not None and row["clips"] < row["budget"])
            logs = buffer_logs(c, v, j)
            if logs:
                row["log"] = parse_buffer_log(logs[0])
            rows.append(row)
    # identical buffers (e.g. <x>_kd and seeds share the S0 buffer of <x>)
    same = {}
    for r in rows:
        same.setdefault((r["session"], r["hash"]), []).append(r["variant"])
    # selection time per method (mean over sessions)
    times = {}
    for r in rows:
        if r["seconds"] is not None and r["method"]:
            times.setdefault(r["method"], []).append(float(r["seconds"]))
    return {"buffers": rows, "identical": {f"S{s} {h}": v for (s, h), v in same.items() if len(v) > 1},
            "selection_seconds": {m: {"mean": float(np.mean(t)), "min": float(np.min(t)), "max": float(np.max(t)),
                                      "n": len(t)} for m, t in times.items()}}


# ----------------------------------------------------------------------------------------------
# facts for tab:instantiation
# ----------------------------------------------------------------------------------------------
def _state(path: Path) -> dict:
    import torch

    try:
        ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception:  # noqa: BLE001
        ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck["state_dict"]


def stage_facts(c: Ctx) -> dict:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(f"configs/cl/ldf_{c.B}_s1.yaml")
    music_path = str(cfg.data.music_path)
    W = re.search(r"window_(\d+)", music_path)
    out = {"clip_tokens_T_max": int(cfg.data.train_max_tokens), "latent_channels": int(cfg.model.params.input_dim),
           "music_features_per_frame": int(cfg.model.params.music_dim), "music_window_frames": int(W.group(1)) if W else None,
           "music_path": music_path, "noise_steps": int(cfg.model.params.noise_steps),
           "cfg_scale": float(cfg.model.params.cfg_scale),
           "history_guidance_scale": float(cfg.model.params.get("history_guidance_scale", float("nan"))),
           "inputs_besides_music": "none: the denoiser reads the motion latents (x, with its clean history frames) and "
                                   "the per-frame music windows only (models/diffusion_forcing_wan.py forward: feature, "
                                   "feature_length, music)"}
    if c.s0_run and (c.s0_run / "last.ckpt").exists():
        try:
            sd0 = _state(c.s0_run / "last.ckpt")
            me = {k: tuple(v.shape) for k, v in sd0.items() if "music_embedding" in k}
            out["music_embedding_shapes"] = me
            emb_dim = [s[0] for k, s in me.items() if k.endswith("weight") and len(s) == 2]
            out["music_embedding_dim"] = emb_dim[-1] if emb_dim else None
            if out["music_embedding_dim"]:
                out["trajectory_D"] = out["music_embedding_dim"] + out["latent_channels"]
            out["n_params"] = int(sum(v.numel() for k, v in sd0.items() if v.dtype.is_floating_point))
            # does the music encoder change from session to session? (TMR+KD chain, raw weights)
            drift = []
            for v in ("tmr_joint_neff_kd", "tmr_joint_neff", "vanilla"):
                prev, prev_k = sd0, 0
                for k in range(1, c.K):
                    d = c.run_dir(v, k)
                    if d is None or not (d / "last.ckpt").exists():
                        break
                    sdk = _state(d / "last.ckpt")
                    num = sum(float((sdk[n].float() - prev[n].float()).norm() ** 2) for n in me)
                    den = sum(float(prev[n].float().norm() ** 2) for n in me)
                    allk = [n for n in sdk if n in prev and sdk[n].dtype.is_floating_point]
                    num_a = sum(float((sdk[n].float() - prev[n].float()).norm() ** 2) for n in allk)
                    den_a = sum(float(prev[n].float().norm() ** 2) for n in allk)
                    drift.append({"variant": v, "from": f"S{prev_k}", "to": f"S{k}",
                                  "music_embedding_rel_change": math.sqrt(num / max(den, 1e-30)),
                                  "all_params_rel_change": math.sqrt(num_a / max(den_a, 1e-30))})
                    prev, prev_k = sdk, k
                if drift:
                    break
            out["music_embedding_drift"] = drift
        except Exception as e:  # noqa: BLE001
            out["error"] = repr(e)
    return out


# ----------------------------------------------------------------------------------------------
# tables and completeness
# ----------------------------------------------------------------------------------------------
def stage_tables(c: Ctx) -> dict:
    dst = c.out / "tables"
    dst.mkdir(parents=True, exist_ok=True)
    copied = []
    for n in ("paper_disco_tables.md", "paper_disco_tables.tex", "matrix_by_genre.md", "matrix_by_session.md", "summary.md"):
        p = c.res / n
        if p.exists():
            shutil.copy2(p, dst / n)
            copied.append(n)
    comp = {}
    for v in c.variants:
        have = {}
        for t in range(1, c.K):
            sc = []
            for i in range(t + 1):
                p = c.res / v / f"after_s{t}_sess{i}.json"
                ok = p.exists()
                dances = ok and p.with_suffix(".joints.npz").exists() and '"names"' in p.read_text()
                sc.append("D" if dances else ("j" if ok else "-"))
            have[f"S{t}"] = "".join(sc)
        comp[v] = have
    sr = {}  # session-level SR of every run (for the coverage / theory correlations and a quick look)
    for v in c.variants + ["s0_model"]:
        for p in sorted((c.res / v).glob("after_s*_sess*.json")):
            m = re.fullmatch(r"after_s(\d+)_sess(\d+)\.json", p.name)
            if m:
                try:
                    sr.setdefault(v, {})[f"S{m[1]}_sess{m[2]}"] = json.loads(p.read_text()).get("StyleRetention")
                except Exception:  # noqa: BLE001
                    pass
    return {"copied": copied, "completeness": comp, "legend": "per session t: one char per scored session i<=t, "
            "D = score + dances + names, j = score only, - = missing", "session_sr": sr}


# ----------------------------------------------------------------------------------------------
# tab:sr_snr on the final pairing (mirrors metrics.cl_metrics.build, nothing is written to eval/)
# ----------------------------------------------------------------------------------------------
def snr_table(benchmark: str, probe_max_windows: int = 4000, seed: int = 0, with_style: bool = True) -> dict:
    from metrics.cl_metrics import (JOINTS, GenreProbe, cached_window_features, frozen_fid, joints_from_272, style_fd,
                                    window_features)
    from metrics.tools.utils import calculate_activation_statistics_np, calculate_frechet_distance_np
    from utils.motion_process import recover_joint_positions_272

    root = Path("raw_data/cl_sessions") / benchmark
    man = json.loads((root / "manifest.json").read_text())
    gt_dir = Path(man["feature_dir"])
    K = len(man["sessions"])
    rng = np.random.default_rng(seed)
    fine_label = (lambda n, s: n[1:3]) if man["dataset"] == "aistpp" else (lambda n, s: str(s))

    test, owners, probe_X, probe_lab, names = {}, {}, [], [], {}
    seqfeat = {}
    for s in range(K):
        nm = read_list(root / f"s{s}" / "test.txt")
        names[s] = nm
        per_seq = [cached_window_features(gt_dir, n) for n in nm]
        test[s] = (np.concatenate([p[0] for p in per_seq]), np.concatenate([p[1] for p in per_seq]))
        owners[s] = np.concatenate([[i] * len(p[0]) for i, p in enumerate(per_seq)])
        for n in read_list(root / f"s{s}" / "train.txt"):
            k, g = cached_window_features(gt_dir, n)
            probe_X.append(np.concatenate([k, g], 1))
            probe_lab += [fine_label(n, s)] * len(k)
        # one feature vector per test sequence, as metrics/m2d.py (whole-sequence window = the repo extractors)
        ks, gs = [], []
        for n in nm:
            j = joints_from_272(np.load(gt_dir / f"{n}.npy"))
            kk, gg = window_features(j, win=len(j), stride=len(j))
            ks.append(kk[0])
            gs.append(gg[0])
        seqfeat[s] = (np.stack(ks), np.stack(gs))
        log(f"[sr_snr] {benchmark} S{s}: {len(nm)} test sequences, {len(test[s][0])} windows")

    classes = sorted(set(probe_lab))
    session_of_class = [next(s for s in range(K) for n in read_list(root / f"s{s}" / "train.txt")
                             if fine_label(n, s) == cc) for cc in classes]
    X, y = np.concatenate(probe_X), np.array([classes.index(cc) for cc in probe_lab])
    keep_idx = np.concatenate([rng.permutation(np.where(y == cc)[0])[:probe_max_windows] for cc in range(len(classes))])
    probe = GenreProbe.fit(X[keep_idx], y[keep_idx], session_of_class)

    pool_k = np.concatenate([test[s][0] for s in range(K)])
    pool_g = np.concatenate([test[s][1] for s in range(K)])
    geo_keep = pool_g.std(0) > 1e-6
    refs = {"kin_mean": pool_k.mean(0), "kin_std": pool_k.std(0) + 1e-10, "geo_mean": pool_g[:, geo_keep].mean(0),
            "geo_std": pool_g[:, geo_keep].std(0)}
    fidk = lambda a, b: frozen_fid(a, b, refs["kin_mean"], refs["kin_std"])
    fidg = lambda a, b: frozen_fid(a[:, geo_keep], b[:, geo_keep], refs["geo_mean"], refs["geo_std"])
    acc = lambda kin, geo, s: float((probe.predict_session(np.concatenate([kin, geo], 1)) == s).mean())

    def percall(gt, pred):  # metrics/m2d.py normalize + Frechet distance, normalised by this call's reference
        mean, std = gt.mean(0), gt.std(0)
        g, p = (gt - mean) / (std + 1e-10), (pred - mean) / (std + 1e-10)
        mu1, s1 = calculate_activation_statistics_np(g)
        mu2, s2 = calculate_activation_statistics_np(p)
        try:
            return float(calculate_frechet_distance_np(mu1, s1, mu2, s2))
        except Exception:  # noqa: BLE001
            return float("inf")

    rows = {}
    for s in range(K):
        kin, geo = test[s]
        sk, sg = seqfeat[s]
        fk, fg, ga, pk, pg = [], [], [], [], []
        nseq = owners[s].max() + 1
        for _ in range(20):  # the same halves (and random draws) as cl_metrics build
            half = rng.permutation(nseq)[: nseq // 2]
            a = np.isin(owners[s], half)
            fk.append(fidk(kin[a], kin[~a]))
            fg.append(fidg(geo[a], geo[~a]))
            ga.append(acc(kin[~a], geo[~a], s))
            A = np.isin(np.arange(nseq), half)
            pk.append(percall(sk[A], sk[~A]))
            pg.append(percall(sg[A], sg[~A]))
        others = [o for o in range(K) if o != s]
        ck_ = {o: fidk(kin, test[o][0]) for o in others}
        cg_ = {o: fidg(geo, test[o][1]) for o in others}
        ca_ = {o: acc(*test[o], s) for o in others}
        pk_c = {o: percall(sk, seqfeat[o][0]) for o in others}
        pg_c = {o: percall(sg, seqfeat[o][1]) for o in others}
        closest = min(others, key=lambda o: ck_[o])

        def snr(comp, noise):  # None when the noise has no spread (or is not finite)
            sd = float(np.std(noise))
            return float((comp - np.mean(noise)) / sd) if sd > 1e-9 and np.isfinite(sd) and np.isfinite(comp) else None

        def snr_acc(halves, comp):
            sd = float(np.std(halves))
            return float((np.mean(halves) - comp) / sd) if sd > 1e-9 else None
        r = {"n_test_seqs": int(nseq), "n_windows": int(len(kin)),
             "FID_k_frozen": {"noise_mean": float(np.mean(fk)), "noise_std": float(np.std(fk)),
                              "complete": float(min(ck_.values())), "snr": snr(min(ck_.values()), fk)},
             "FID_g_frozen": {"noise_mean": float(np.mean(fg)), "noise_std": float(np.std(fg)),
                              "complete": float(min(cg_.values())), "snr": snr(min(cg_.values()), fg)},
             "FID_k_percall": {"noise_mean": float(np.mean(pk)), "noise_std": float(np.std(pk)),
                               "complete": float(min(pk_c.values())), "snr": snr(min(pk_c.values()), pk)},
             "FID_g_percall": {"noise_mean": float(np.mean(pg)), "noise_std": float(np.std(pg)),
                               "noise_max": float(np.max(pg)), "complete": float(min(pg_c.values())),
                               "complete_max": float(max(pg_c.values())), "snr": snr(min(pg_c.values()), pg)},
             "GenreAcc": {"gt": acc(kin, geo, s), "halfset_mean": float(np.mean(ga)), "halfset_std": float(np.std(ga)),
                          "complete": float(max(ca_.values())), "snr": snr_acc(ga, max(ca_.values()))},
             "closest_other_session_by_FID_k": int(closest)}
        rows[s] = r
    if with_style:
        from metrics.style_encoder import StyleEncoder, benchmark_session_of_class, ckpt_path

        style = StyleEncoder(ckpt_path(man["dataset"], -1))
        soc = benchmark_session_of_class(style.classes, man)
        scores = lambda prob, s: style.session_scores(prob, s, soc)
        enc = {s: style.encode([recover_joint_positions_272(np.load(gt_dir / f"{n}.npy"), JOINTS) for n in names[s]])
               for s in range(K)}
        for s in range(K):
            e, others = enc[s], [o for o in range(K) if o != s]
            seqs, fd_n, ret_n = np.unique(e["owner"]), [], []
            for _ in range(20):
                a = np.isin(e["owner"], rng.permutation(seqs)[: len(seqs) // 2])
                fd_n.append(style_fd(e["emb"][a], e["emb"][~a]))
                ret_n.append(scores(e["prob"][~a], s)["StyleRetention"])
            gt = scores(e["prob"], s)
            comp = max(scores(enc[o]["prob"], s)["StyleRetention"] for o in others)
            sd_r = float(np.std(ret_n))
            rows[s]["StyleRetention"] = {"gt": gt["StyleRetention"], "StyleAcc_gt": gt["StyleAcc"],
                                         "DanceRate_gt": gt.get("DanceRate"), "halfset_mean": float(np.mean(ret_n)),
                                         "halfset_std": sd_r, "complete": float(comp),
                                         "snr": float((np.mean(ret_n) - comp) / sd_r) if sd_r > 1e-9 else None}
            comp_fd = min(style_fd(e["emb"], enc[o]["emb"]) for o in others)
            sd_f = float(np.std(fd_n))
            rows[s]["StyleFD"] = {"noise_mean": float(np.mean(fd_n)), "noise_std": sd_f, "complete": float(comp_fd),
                                  "snr": float((comp_fd - np.mean(fd_n)) / sd_f) if sd_f > 1e-9 else None}
    summary = {}
    for key in ("FID_k_percall", "FID_g_percall", "FID_k_frozen", "FID_g_frozen", "GenreAcc", "StyleRetention", "StyleFD"):
        v = [rows[s][key]["snr"] for s in range(K) if key in rows[s] and rows[s][key]["snr"] is not None]
        if v:
            summary[key] = {"min": float(np.min(v)), "median": float(np.median(v)), "per_session": v}
    summary["FID_g_percall_max_value"] = float(max(max(rows[s]["FID_g_percall"]["noise_max"],
                                                       rows[s]["FID_g_percall"]["complete_max"]) for s in range(K)))
    # is eval/ (used by every eval_cl_session.py score for FID / noise) built for the current test lists?
    stale = {}
    try:
        noise = json.loads((root / "eval" / "noise.json").read_text())
        for s in range(K):
            stale[f"s{s}"] = {"noise_json_test_seqs": noise.get(f"s{s}", {}).get("n_test_seqs"),
                              "current_test_seqs": rows[s]["n_test_seqs"],
                              "noise_json_windows": noise.get(f"s{s}", {}).get("n_windows"),
                              "current_windows": rows[s]["n_windows"]}
        refs_npz = np.load(root / "eval" / "refs.npz")
        for s in range(K):
            stale[f"s{s}"]["refs_kin_windows"] = int(refs_npz[f"s{s}_kin"].shape[0]) if f"s{s}_kin" in refs_npz else None
        stale["probe_classes_session"] = noise.get("session_of_class")
        stale["current_session_of_class"] = session_of_class
        stale["eval_dir_matches_current_lists"] = all(
            stale[f"s{s}"]["noise_json_windows"] == rows[s]["n_windows"] for s in range(K)) and \
            noise.get("session_of_class") == session_of_class
    except Exception as e:  # noqa: BLE001
        stale["error"] = repr(e)
    return {"benchmark": benchmark, "per_session": rows, "summary": summary, "eval_dir_check": stale,
            "probe_classes": classes, "session_of_class": session_of_class}


# ----------------------------------------------------------------------------------------------
# tab:sr_valid from validation.json
# ----------------------------------------------------------------------------------------------
def stage_sr_valid(path: Path) -> dict:
    rep = json.loads(path.read_text())
    v = rep.get("verdict", {})
    inv = rep.get("invariance", [])
    clean = inv[0] if inv else {}
    out = {"file": str(path), "verdict": v, "criteria": rep.get("criteria")}
    enc, prb = rep["clean"]["encoder"], rep["clean"]["probe"]
    out["clean"] = {"encoder_session_acc": enc["session_acc"], "encoder_genre_acc": enc["genre_acc"],
                    "probe_session_acc": prb["session_acc"], "probe_genre_acc": prb.get("genre_acc"),
                    "ece": enc.get("ece"), "rejected_windows": enc.get("rejected_windows"), "dances": enc.get("dances"),
                    "dances_session_majority_correct": enc.get("dances_session_majority_correct"),
                    "dances_genre_majority_correct": enc.get("dances_genre_majority_correct"),
                    "per_genre_acc": enc.get("per_genre_acc"), "per_session": enc.get("per_session")}
    seen = [r for r in inv[1:] if r.get("seen_in_training")]
    unseen = [r for r in inv[1:] if not r.get("seen_in_training")]
    c_enc, c_prb = clean.get("encoder_session_acc"), clean.get("probe_session_acc")
    if seen:
        out["seen_worst"] = {"encoder": max(c_enc - r["encoder_session_acc"] for r in seen),
                             "probe": max(c_prb - r["probe_session_acc"] for r in seen),
                             "encoder_case": max(seen, key=lambda r: c_enc - r["encoder_session_acc"])["kind"]}
    if unseen:
        w = max(unseen, key=lambda r: c_enc - r["encoder_session_acc"])
        out["unseen_worst"] = {"encoder": c_enc - w["encoder_session_acc"], "encoder_case": f"{w['kind']} {w['level']}",
                               "probe": max(c_prb - r["probe_session_acc"] for r in unseen)}
    out["probe_text"] = {f"{r['kind']} {r['level']}": r["probe_session_acc"] for r in inv
                         if (r["kind"], r["level"]) in (("clean", 0), ("jitter", 0.5), ("smooth", 2))}
    out["encoder_text"] = {f"{r['kind']} {r['level']}": r["encoder_session_acc"] for r in inv
                           if (r["kind"], r["level"]) in (("clean", 0), ("jitter", 0.5), ("smooth", 2), ("wobble", 3))}
    sw = rep.get("genre_swap", {})
    ret = np.asarray(sw.get("retention", [[np.nan]]))
    K = len(ret)
    out["swap"] = {"own_min": float(min(ret[i][i] for i in range(K))), "own_max": float(max(ret[i][i] for i in range(K))),
                   "other_max": float(max(ret[i][j] for i in range(K) for j in range(K) if i != j)),
                   "matrix": ret.tolist(), "snr_retention": [n.get("snr_retention") for n in sw.get("noise", [])]}
    rej = rep.get("rejection", [])
    must = [r for r in rej if r.get("must_reject")]
    unseen_rej = [r for r in must if r["probe"].startswith("unseen")]
    seen_rej = [r for r in must if not r["probe"].startswith("unseen")]
    out["reject"] = {"max_session_mass_nondance": max((r["max_session_mass"] for r in seen_rej), default=None),
                     "max_session_mass_unseen": max((r["max_session_mass"] for r in unseen_rej), default=None),
                     "includes_untrained_generator": any(r["probe"] == "untrained generator" for r in must),
                     "rows": rej}
    if rep.get("held_out_dancers"):
        out["held_out"] = {"encoder_genre_acc": v.get("held_out_encoder_genre_acc"),
                           "probe_genre_acc": v.get("held_out_probe_genre_acc"),
                           "folds": [{k: r["all_sequences"][k] for k in ("encoder_genre_acc", "probe_genre_acc")}
                                     for r in rep["held_out_dancers"]]}
    return out


# ----------------------------------------------------------------------------------------------
# sanity rows (eval_cl_session.py --gt / --vae_recon / --untrained / partly trained S0 model)
# ----------------------------------------------------------------------------------------------
def stage_sanity(c: Ctx, sdir: Path) -> dict:
    out = {}
    enc = soc = None
    for p in sorted(sdir.glob("*.json")):
        try:
            m = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        row = {k: m.get(k) for k in ("StyleRetention", "StyleAcc", "DanceRate", "StyleFD", "n_windows", "mode", "ckpt",
                                      "session", "max_tokens", "samples", "n_sequences")}
        npz = p.with_suffix(".joints.npz")
        if npz.exists() and ("s0step" in p.stem or "untrained" in p.stem):  # where the dance-like output goes
            try:
                from metrics.style_encoder import OTHER, StyleEncoder, benchmark_session_of_class, ckpt_path

                if enc is None:
                    enc = StyleEncoder(ckpt_path(c.man["dataset"], -1))
                    soc = np.asarray(benchmark_session_of_class(enc.classes, c.man))
                z = np.load(npz)
                e = enc.encode([z[k] for k in sorted(z.files, key=int)])
                prob = e["prob"]
                other = float(prob[:, enc.classes.index(OTHER)].mean()) if OTHER in enc.classes else 0.0
                mass = [float(prob[:, soc == s].sum(1).mean()) for s in range(c.K)]
                row["session_mass"] = mass
                row["p_other"] = other
                row["dance_like_share_per_session"] = [x / max(1e-12, 1 - other) for x in mass]
                win_other = prob.argmax(1) == enc.classes.index(OTHER) if OTHER in enc.classes else np.zeros(len(prob), bool)
                row["windows_argmax_other"] = float(win_other.mean())
            except Exception as ex:  # noqa: BLE001
                row["rescore_error"] = repr(ex)
        out[p.stem] = row
    return out


# ----------------------------------------------------------------------------------------------
# SUPP_FILL.md
# ----------------------------------------------------------------------------------------------
def write_md(c: Ctx, R: dict) -> Path:
    L = ["# DiscoForcing numbers for the TMR supplementary", "",
         f"Collected {time.strftime('%Y-%m-%d %H:%M')} from screen `{c.tag}` (S0 model `{c.s0_run}`), benchmark `{c.B}`.",
         "Paper file: `sec/X_suppl.tex`. Every section names the red item it fills.", ""]
    S = R.get("sessions")
    if S and "sessions" in S:
        L += ["## 1. app:disco, Sessions: `[Training, validation and test sequences per session; monotone curves]`", "",
              "| Session | Genres | Train | Val | Test | Train min | Candidate clips | Budget (10%) |", "|---|---|---|---|---|---|---|---|"]
        for s in S["sessions"]:
            L.append(f"| S{s['session']} | {' + '.join(s['names'] or s['groups'])} | {s['train']} | {s['val']} | {s['test']} | "
                     f"{s.get('train_minutes')} | {s.get('n_candidates', 'n/a')} | {s.get('budget', 'n/a')} |")
        tot = {sp: sum(s[sp] for s in S["sessions"]) for sp in ("train", "val", "test")}
        L += [f"| all | | {tot['train']} | {tot['val']} | {tot['test']} | | | |", ""]
        mism = [f"S{s['session']} {sp}" for s in S["sessions"] for sp in ("train", "val", "test")
                if s["manifest_counts"].get(sp) not in (None, s[sp]) or s.get(f"{sp}_loader_list_matches") is False]
        L.append(f"List files agree with manifest and loader lists: {'yes' if not mism else 'NO: ' + ', '.join(mism)}")
        L += ["", "Distance from each earlier session to every later one (mean genre distance, as the session builder):", ""]
        for cv in S["curves"]:
            L.append(f"- S{cv['session']}: " + " -> ".join(f3(x) for x in cv["distance_to_later"])
                     + f"  monotone: {'yes' if cv['monotone_nondecreasing'] else 'NO'}")
        L.append(f"- fraction of (i, t < t') with d(S_i,S_t) <= d(S_i,S_t'): {f3(S.get('monotone_fraction'))}")
        rd = S.get("rederived", {})
        if "error" in rd:
            L.append(f"- re-derivation failed: {rd['error']}")
        else:
            L.append(f"- the stated rules applied to the manifest's distances give pairing {rd['pairing']} "
                     f"(matches manifest: {rd['pairing_matches_manifest']}) and order {rd['order']} "
                     f"(matches: {rd['order_matches_manifest']})")
        L += ["", "Suggested text: `Sessions hold X/Y/Z training/validation/test sequences (Tab.~...)`; if every curve "
              "is monotone: `for this pairing every earlier session's distance grows monotonically`, otherwise name the "
              "exception from the list above.", ""]
    P = R.get("protocol")
    if P and P.get("checks"):
        L += ["## 2. app:disco, Generator: `[Confirm for the reported runs: cosine decay over each session and EMA decay 0.999]`", "",
              "| Run | cosine over session (T_max = max_steps) | T_max | steps (ckpt) | LR at end | EMA cfg / ckpt | seed | init EMA |",
              "|---|---|---|---|---|---|---|---|"]
        for ch in P["checks"]:
            L.append(f"| {ch['run']} | {ch['cosine_over_session']} | {ch['T_max']} | {ch['global_step']} | "
                     f"{f3(ch['lr_end'], 8) if ch['lr_end'] is not None else 'n/a'} | {ch['ema_decay_config']} / "
                     f"{ch['ema_decay_ckpt']} | {ch['seed']} | {ch['init_from_ema']} |")
        L += ["", "Fields that are not identical across the reported runs (most common value, then the deviating runs):", ""]
        for k, s in P["summary"].items():
            if s["deviating"]:
                L.append(f"- `{k}`: most common {s['most_common']}; differs in {', '.join(s['deviating'][:12])}"
                         + (" ..." if len(s["deviating"]) > 12 else ""))
        L += ["", "Every other field has one value in all runs (see supp_collect.json protocol.summary).", ""]
    F = R.get("facts")
    if F:
        L += ["## 3. tab:instantiation, DiscoForcing row", "",
              f"- T x D: T <= {F.get('clip_tokens_T_max')} latent frames, D = {F.get('music_embedding_dim')} (music embedding) + "
              f"{F.get('latent_channels')} (motion latents) = {F.get('trajectory_D')}",
              f"- music input: {F.get('music_features_per_frame')} librosa features per frame, causal window of "
              f"{F.get('music_window_frames')} frames; generator parameters {F.get('n_params')}",
              f"- `Inputs with x` cell: {F.get('inputs_besides_music')}  -> suggested cell: `none`", ""]
        for d in F.get("music_embedding_drift", []):
            L.append(f"- {d['variant']} {d['from']}->{d['to']}: relative change of the music embedding "
                     f"{f3(d['music_embedding_rel_change'], 4)}, of all weights {f3(d['all_params_rel_change'], 4)}")
        L += ["", "If the music embedding changes in every session, the feature-replay row of Tab. paradigms "
              "(`the audio encoder is updated in every session`) holds for DiscoForcing.", ""]
    T = R.get("sr_snr")
    if T and "summary" in T:
        sm = T["summary"]
        g = lambda k, q: f3(sm[k][q], 1) if k in sm else "n/a"
        L += ["## 4. app:sr, Tab. sr_snr (final pairing)", "",
              "| Metric | Minimum | Median |", "|---|---|---|",
              f"| FID_k, per-call normalization | {g('FID_k_percall', 'min')} | {g('FID_k_percall', 'median')} |",
              f"| FID_g, per-call normalization | max value {f3(sm.get('FID_g_percall_max_value'), 1)} (SNR {g('FID_g_percall', 'min')} / {g('FID_g_percall', 'median')}) | |",
              f"| FID_k, frozen normalization | {g('FID_k_frozen', 'min')} | {g('FID_k_frozen', 'median')} |",
              f"| FID_g, frozen normalization | {g('FID_g_frozen', 'min')} | {g('FID_g_frozen', 'median')} |",
              f"| Handcrafted genre classifier | {g('GenreAcc', 'min')} | {g('GenreAcc', 'median')} |",
              f"| StyleRetention (ours) | {g('StyleRetention', 'min')} | {g('StyleRetention', 'median')} |",
              f"| (StyleFD, not in the table) | {g('StyleFD', 'min')} | {g('StyleFD', 'median')} |", "",
              "LaTeX rows:", "```latex",
              f"FID$_k$, per-call normalization & ${g('FID_k_percall', 'min')}$ & ${g('FID_k_percall', 'median')}$\\\\",
              f"FID$_k$, frozen normalization & ${g('FID_k_frozen', 'min')}$ & ${g('FID_k_frozen', 'median')}$\\\\",
              f"FID$_g$, frozen normalization & ${g('FID_g_frozen', 'min')}$ & ${g('FID_g_frozen', 'median')}$\\\\",
              f"Handcrafted genre classifier & {g('GenreAcc', 'min')} & {g('GenreAcc', 'median')}\\\\",
              f"StyleRetention (ours) & {g('StyleRetention', 'min')} & {g('StyleRetention', 'median')}\\\\", "```", "",
              "SNR = (complete forgetting - mean split-half noise) / sd of the noise for the distances, and "
              "(mean half-set score - complete forgetting) / sd for the classifiers (20 split halves by sequence; "
              "complete forgetting = the closest other session's real test dance).", ""]
        ev = T.get("eval_dir_check", {})
        if "eval_dir_matches_current_lists" in ev:
            L.append(f"eval/ (refs.npz, noise.json, probe.npz) built for the current test lists and pairing: "
                     f"{ev['eval_dir_matches_current_lists']} (if False, FID values in eval jsons use old references; "
                     f"SR is unaffected)")
        for s, r in T["per_session"].items():
            sr = r.get("StyleRetention", {})
            L.append(f"- S{s}: {r['n_test_seqs']} test dances, {r['n_windows']} windows; SR real {f3(sr.get('gt'))} "
                     f"(half-set sd {f3(sr.get('halfset_std'))}), closest other session {f3(sr.get('complete'))}; "
                     f"classifier {f3(r['GenreAcc']['gt'])} -> {f3(r['GenreAcc']['complete'])}")
        L.append("")
    FD = R.get("sr_snr_finedance")
    if FD and "summary" in FD:
        sm = FD["summary"]
        L += [f"FineDance (text: `on FineDance the result is the same (-1.3 and -0.9)`): per-call FID_k SNR min "
              f"{f3(sm['FID_k_percall']['min'], 1)}, median {f3(sm['FID_k_percall']['median'], 1)}", ""]
    V = R.get("sr_valid")
    if V and "clean" in V:
        cl, sw, rj = V["clean"], V["swap"], V["reject"]
        ho = V.get("held_out", {})
        L += ["## 5. app:sr, Tab. sr_valid (final pairing)", "",
              "| Test | Ours | Handcrafted |", "|---|---|---|",
              f"| Genre accuracy, dancers unseen in training (10 genres) | {f3(ho.get('encoder_genre_acc'), 2)} | {f3(ho.get('probe_genre_acc'), 2)} |",
              f"| Session accuracy, clean test windows | {f3(cl['encoder_session_acc'])} | {f3(cl['probe_session_acc'], 2)} |",
              f"| Test dances assigned to their session (majority vote) | {cl['dances_session_majority_correct']}/{cl['dances']} | -- |",
              f"| Largest change, artifacts within the training ranges | {f3(V.get('seen_worst', {}).get('encoder'))} | {f3(V.get('seen_worst', {}).get('probe'), 2)} |",
              f"| Largest change, unseen distortions | {f3(V.get('unseen_worst', {}).get('encoder'))} ({V.get('unseen_worst', {}).get('encoder_case')}) | {f3(V.get('unseen_worst', {}).get('probe'), 2)} |",
              f"| SR of real dance, own session / other sessions | {f3(sw['own_min'], 2)}--{f3(sw['own_max'], 2)} / <= {f3(sw['other_max'], 2)} | -- |",
              f"| Largest SR of non-dance inputs, any session | {f3(rj['max_session_mass_nondance'])} (untrained generator included: {rj['includes_untrained_generator']}) | -- |",
              f"| Largest SR of unseen non-dance inputs, any session | {f3(rj['max_session_mass_unseen'], 2)} | -- |", "",
              f"Text of app:sr: the handcrafted classifier's session accuracy {V.get('probe_text')} "
              f"(clean -> jitter 0.5 cm, clean -> smoothing sigma 2); the encoder at the same settings {V.get('encoder_text')}.",
              f"Limitations paragraph: windows rejected {f3(cl.get('rejected_windows'))}; dances not assigned to their session "
              f"{(cl['dances'] or 0) - (cl['dances_session_majority_correct'] or 0)} of {cl['dances']}; per-genre accuracy "
              + ", ".join(f"{k} {f3(v, 2)}" for k, v in (cl.get("per_genre_acc") or {}).items()), ""]
    Q = R.get("sanity")
    if Q:
        L += ["## 6. app:sr, Validation paragraph: SR on generated motion (first 10 s unless named full)", "",
              "| Probe | session | SR | DanceRate | mass per session | windows |", "|---|---|---|---|---|---|"]
        for k, r in sorted(Q.items()):
            mass = r.get("session_mass")
            L.append(f"| {k} | S{r.get('session')} | {f3(r.get('StyleRetention'))} | {f3(r.get('DanceRate'))} | "
                     f"{'/'.join(f3(x, 2) for x in mass) if mass else ''} | {r.get('n_windows')} |")
        L += ["", "Mapping to the text: untrained generator = `untrained_s0_t76`; Session-0 model after 2,500 steps = "
              "`s0step*_s0_t76`; autoencoder reconstructions = `vae_s*_t76`; real dance = `gt_s*_t76`; the same partly "
              "trained model on Session-4 music = `s0step*_s4_t76` (its dance-like share in S0 style = "
              "session_mass[0] / (1 - P(other))). The `[SR of real test dance and of autoencoder reconstructions for "
              "every session]` item: the gt_* and vae_* rows (t76 and full).", ""]
    Bf = R.get("buffers")
    if Bf and Bf.get("buffers"):
        L += ["## 7. Buffers of the reported runs (selection time, sizes, TMR's rank and OMP stop)", "",
              "| Variant | S | clips/budget | candidates | sequences | seconds | eff. size | TMR log (M by rule; N, ND, n_eff ranks; OMP picked; fill) |",
              "|---|---|---|---|---|---|---|---|"]
        for b in Bf["buffers"]:
            lg = b.get("log", {})
            t = ""
            if "M" in lg:
                t = (f"M={lg['M'][0]} ({lg['M'][1]}); N:{lg.get('M_bic_N', ['?'])[0]} ND:{lg.get('M_bic_ND', ['?'])[0]} "
                     f"n_eff={lg.get('n_eff', ['?', '?', '?'])[0]}:{lg.get('n_eff', ['?', '?', '?'])[2]}; "
                     f"OMP {'/'.join(lg.get('omp', ['?', '?']))}; fill {lg.get('fill', ['-', '0'])[1]}")
            L.append(f"| {b['variant']} | {b['session']} | {b['clips']}/{b['budget']} | {b['n_candidates']} | {b['sequences']} | "
                     f"{b['seconds']} | {f3(b['weights']['effective_size'], 1)} | {t} |")
        L += ["", "Selection seconds per method (whole build: model load, features, selection): "
              + "; ".join(f"{m} {f3(s['mean'], 0)} s" for m, s in sorted(Bf["selection_seconds"].items())), ""]
        if Bf.get("identical"):
            L.append("Identical buffers: " + "; ".join(f"{k}: {', '.join(v)}" for k, v in Bf["identical"].items()))
            L.append("")
    Tb = R.get("tables")
    if Tb:
        L += ["## 8. Tables and completeness", "", f"Copied to tables/: {', '.join(Tb['copied']) or 'none'}", "",
              f"Completeness ({Tb['legend']}):", ""]
        for v, h in Tb["completeness"].items():
            L.append(f"- {v}: " + " ".join(f"{k}:{x}" for k, x in h.items()))
        L.append("")
    TH = Path(c.out) / "theory" / "THEORY.md"
    if TH.exists():
        L += ["## 9. Fast theory checks", "", f"See `theory/THEORY.md` ({TH})", ""]
    if R.get("errors"):
        L += ["## Errors", ""] + [f"- {k}: {v.splitlines()[-1] if v else v}" for k, v in R["errors"].items()] + [""]
    p = Path(c.out) / "SUPP_FILL.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="aistpp_genre5")
    ap.add_argument("--tag", default="v2_40000_full_s0v2", help="screen folder of the reported runs")
    ap.add_argument("--out", default=None, help="default results/<benchmark>/supp")
    ap.add_argument("--variants", nargs="*", default=None,
                    help="variants to read (default: every run of the screen); their _seed<k> runs are included")
    ap.add_argument("--stages", nargs="*", default=["sessions", "protocol", "buffers", "facts", "tables", "sr_snr",
                                                    "sr_valid", "sanity"])
    ap.add_argument("--validation", default=None, help="validation.json (default <out>/validation.json, else the "
                                                       "style encoder's own)")
    ap.add_argument("--finedance", default="finedance_style4", help="also the per-call FID_k SNR on this benchmark if "
                                                                    "it exists ('' to skip)")
    ap.add_argument("--no_ckpt_meta", action="store_true", help="do not open last.ckpt files (global step, final LR, EMA)")
    a = ap.parse_args()
    a.out = a.out or f"results/{a.benchmark}/supp"
    c = Ctx(a)
    c.out.mkdir(parents=True, exist_ok=True)
    log(f"screen {c.res} | S0 run {c.s0_run} | variants {c.variants}")
    jp = c.out / "supp_collect.json"
    R = json.loads(jp.read_text()) if jp.exists() else {}
    R["errors"] = {}
    R["meta"] = {"tag": c.tag, "benchmark": c.B, "s0_run": str(c.s0_run), "variants": c.variants,
                 "collected": time.strftime("%Y-%m-%d %H:%M:%S")}
    stages = {
        "sessions": lambda: stage_sessions(c),
        "protocol": lambda: stage_protocol(c),
        "buffers": lambda: stage_buffers(c),
        "facts": lambda: stage_facts(c),
        "tables": lambda: stage_tables(c),
        "sr_snr": lambda: snr_table(c.B),
        "sr_valid": lambda: stage_sr_valid(Path(a.validation) if a.validation else (
            c.out / "validation.json" if (c.out / "validation.json").exists() else
            Path("raw_data/cl_sessions/_style_encoder") / c.man["dataset"] / "validation.json")),
        "sanity": lambda: stage_sanity(c, c.out / "sanity"),
    }
    for name in a.stages:
        t0 = time.time()
        try:
            log(f"stage {name}")
            R[name] = stages[name]()
            if name == "sr_snr" and a.finedance and (Path("raw_data/cl_sessions") / a.finedance / "manifest.json").exists():
                try:
                    R["sr_snr_finedance"] = snr_table(a.finedance, with_style=False)
                except Exception:  # noqa: BLE001
                    R["errors"]["sr_snr_finedance"] = traceback.format_exc()
        except Exception:  # noqa: BLE001
            R["errors"][name] = traceback.format_exc()
            log(f"stage {name} FAILED:\n{R['errors'][name]}")
        log(f"stage {name}: {time.time() - t0:.0f} s")
        jp.write_text(json.dumps(jsonable(R), indent=1))
    md = write_md(c, R)
    log(f"wrote {jp} and {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
