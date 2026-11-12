#!/usr/bin/env python3
"""
Pack the stimuli of the StyleRetention human study. Runs on the server (from the repo root); the laptop renders and
hosts the study (scripts/user_study/render_study.py) and scores the answers (scripts/user_study/analyze_study.py).

Uses the dances scripts/eval_cl_session.py saved next to its scores (<results>/<variant>/after_s{t}_sess{i}.joints.npz)
and the style probabilities scripts/cl/screen_matrix_table.py cached next to them (.style.npz; computed if missing).

Trials (5 s windows, the style encoder's unit, so every clip has exactly one score per metric):
  * style (main): two windows generated for the SAME test music, sample and time window by two different models or
    checkpoints, shown under two real reference clips of the music's genre; "which one dances more like the
    references?". Stratified by how far apart the style encoder puts them, |P_A(genre) - P_B(genre)| in
    [0, .1) / [.1, .4) / [.4, 1], by type: "forget" = a model right after learning the session vs the same lineage
    after a later session, "cross" = any two models, and by composition: in --dance_pairs (half) of each cell both
    clips are dance to the encoder (P(other) < .5), so the choice is about style rather than dance vs no dance.
  * catch: a real window of the genre vs a real window of a distant genre (attention check, answer known; both
    clips are recognised as their genre by the encoder, so the check is unambiguous).
Real clips (references, catch, practice) come from the more active half of their genre's windows (by joint speed).
  * practice: like catch, shown first with feedback.
  * dance: one generated window, "is the figure dancing?", spread over P(other) (checks DanceRate / the reject class).
Every clip carries its scores under each compared metric, computed on that exact window:
  ours_genre    P(music's genre), style encoder               (genre-level StyleRetention)
  ours_session  P(the session's genres)                       (StyleRetention as reported)
  p_other       P(other)                                      (1 - DanceRate)
  probe_genre / probe_session  handcrafted kinetic+geometric logistic probe (eval/probe.npz; Robust Dancer-style)
  kin / geo     minus the standardised distance of the window's kinetic / geometric features to the genre's real
                windows (per-clip analogue of FID_k / FID_g; FID itself needs sets)
  emb_cos       cosine of the style embedding to the genre's real centroid (per-clip analogue of StyleFD)
Real clips come from the metric's reference features (noise.json reference_dir: VAE reconstructions of the test
dance, the same decoder domain as generated dance). Clip ids are random, so file names reveal nothing.

  python scripts/user_study/pack_study.py --results results/aistpp_genre5/screen/v2_40000_full_s0v2
  -> study_pack.zip (study_pack/study.json + study_pack/clips.npz, ~20-40 MB); env STYLE_ENCODER_DEVICE (cpu)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
FPS, WIN, STRIDE = 30, 150, 30
BINS = (0.0, 0.1, 0.4, 1.0 + 1e-9)  # |delta P(genre)| strata of the style trials
DANCE_BINS = (0.0, 0.2, 0.8, 1.0 + 1e-9)  # P(other) strata of the dance trials
TYPES = ("forget", "cross")
GENRE = {"BR": "Break", "PO": "Pop", "LO": "Lock", "WA": "Waack", "MH": "Middle Hip-hop", "LH": "LA Hip-hop",
         "HO": "House", "KR": "Krump", "JS": "Street Jazz", "JB": "Ballet Jazz"}


def problems(J: np.ndarray) -> str:
    """Why a window is unfit to show (empty when fine); same checks as scripts/cl/demo_render.py."""
    if not np.isfinite(J).all():
        return "NaN"
    if np.linalg.norm(J - J[:, :1], axis=2).max() > 1.5:
        return "a joint > 1.5 m from the pelvis"
    if np.linalg.norm(np.diff(J[:, 0, [0, 2]], axis=0), axis=1).max() * FPS > 6:
        return "root > 6 m/s"
    return ""


def speed(J: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(J, axis=0), axis=-1).mean() * FPS)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def load_module(path: str):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def score_files(res: Path) -> list[tuple[str, int, int, Path, list[str]]]:
    """(variant, trained up to t, session tested i, score json, music names) of every score with saved dances."""
    out = []
    for d in sorted(p for p in res.iterdir() if p.is_dir()):
        for p in sorted(d.glob("after_s*_sess*.json")):
            m = re.fullmatch(r"after_s(\d+)_sess(\d+)\.json", p.name)
            if not m or int(m[2]) > int(m[1]) or not p.with_suffix(".joints.npz").exists():
                continue
            names = json.loads(p.read_text()).get("names")
            if names:
                out.append((d.name, int(m[1]), int(m[2]), p, names))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True, help="screen results folder (one subfolder per variant)")
    ap.add_argument("--benchmark", default="aistpp_genre5")
    ap.add_argument("--ref_dir", type=Path, default=None, help="272-dim real features (default: noise.json reference_dir)")
    ap.add_argument("--pairs", type=int, default=180, help="style trials")
    ap.add_argument("--catch", type=int, default=12)
    ap.add_argument("--practice", type=int, default=2)
    ap.add_argument("--dance", type=int, default=36, help="dance trials (generated; plus 1/6 as many real ones)")
    ap.add_argument("--refs_per_genre", type=int, default=2)
    ap.add_argument("--dance_pairs", type=float, default=0.5,
                    help="share of each (type, delta) cell whose two clips are both dance to the encoder")
    ap.add_argument("--dance_thresh", type=float, default=0.5, help="P(other) below which a clip counts as dance")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("study_pack.zip"))
    a = ap.parse_args()
    t0 = time.time()

    from metrics.cl_metrics import GenreProbe, cached_window_features, root_relative, window_features
    from metrics.style_encoder import StyleEncoder, benchmark_session_of_class, ckpt_path
    from utils.motion_process import recover_joint_positions_272

    tbl = load_module("scripts/cl/screen_matrix_table.py")  # its .style.npz cache of the window probabilities
    root = Path("raw_data/cl_sessions") / a.benchmark
    man = json.loads((root / "manifest.json").read_text())
    if man["dataset"] != "aistpp":
        raise SystemExit("genres are read from AIST++ sequence names; other datasets need their own labelling")
    enc = StyleEncoder(ckpt_path(man["dataset"], -1))
    classes = list(enc.classes)
    ci = {c: k for k, c in enumerate(classes)}
    soc = np.asarray(benchmark_session_of_class(classes, man))
    sess_of = {g: s for s, sess in enumerate(man["sessions"]) for g in sess["groups"]}
    genres = [g for s in man["sessions"] for g in s["groups"]]
    noise = json.loads((root / "eval" / "noise.json").read_text())
    ref_dir = a.ref_dir or Path(noise["reference_dir"])
    refs = np.load(root / "eval" / "refs.npz")
    probe = GenreProbe.load(root / "eval" / "probe.npz")
    probe_classes = list(noise["probe_classes"])
    keep = refs["geo_keep"]
    rng = np.random.default_rng(a.seed)
    print(f"[study] benchmark {a.benchmark}: sessions {[s['groups'] for s in man['sessions']]}; references {ref_dir}")

    # ---------------- generated windows: every window of every saved dance, with its encoder probabilities -------
    files = score_files(a.results)
    if not files:
        raise SystemExit(f"no scores with saved dances under {a.results}")
    print(f"[study] {len(files)} score files with saved dances, {len({f[0] for f in files})} variants")
    music_id, cols = {}, {k: [] for k in ("fi", "q", "k", "g", "i", "pg", "ps", "po", "key")}
    K_MAX, S_MAX = 4096, 64
    for fi, (v, t, i, p, names) in enumerate(files):
        prob, owner = tbl.encoded(p, enc)
        n = len(names)
        first = np.searchsorted(owner, owner, side="left")  # owner is sorted: windows of dance q are contiguous
        k = np.arange(len(owner)) - first
        g_of_q = np.array([genres.index(names[q % n][1:3]) for q in range(int(owner.max()) + 1)])
        gw = g_of_q[owner]
        mid = np.array([music_id.setdefault(names[q % n], len(music_id)) for q in range(int(owner.max()) + 1)])[owner]
        cols["fi"].append(np.full(len(owner), fi))
        cols["q"].append(owner)
        cols["k"].append(k)
        cols["g"].append(gw)
        cols["i"].append(np.full(len(owner), i))
        cols["pg"].append(prob[np.arange(len(owner)), [ci[genres[x]] for x in gw]])
        cols["ps"].append(prob[:, soc == i].sum(1))
        cols["po"].append(prob[:, ci["other"]] if "other" in ci else np.zeros(len(owner)))
        # same music, same sample (generation seed), same time window -> one key
        if (owner // n).max() >= S_MAX or k.max() >= K_MAX:
            raise SystemExit(f"{p}: more than {S_MAX} samples or {K_MAX} windows per dance")
        cols["key"].append((mid * S_MAX + owner // n) * K_MAX + k)
    Wn = {k: np.concatenate(v) for k, v in cols.items()}
    print(f"[study] {len(Wn['fi'])} generated windows, {len(music_id)} test music")
    variants = sorted({f[0] for f in files})
    Wn["v"] = np.array([variants.index(f[0]) for f in files])[Wn["fi"]]  # variant of each window
    Wn["t"] = np.array([f[1] for f in files])[Wn["fi"]]  # the session its model was trained up to
    s0_id = variants.index("s0_model") if "s0_model" in variants else -1

    dance_cache: dict[tuple[int, int], np.ndarray] = {}

    def dance(fi: int, q: int) -> np.ndarray:
        if (fi, q) not in dance_cache:
            dance_cache[fi, q] = np.load(files[fi][3].with_suffix(".joints.npz"))[str(q)].astype(np.float32)
        return dance_cache[fi, q]

    def window_ok(w: int) -> bool:
        J = dance(int(Wn["fi"][w]), int(Wn["q"][w]))
        s = STRIDE * int(Wn["k"][w])
        return len(J) >= s + WIN and not problems(J[s:s + WIN])

    def forget_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """x[:, None] vs y[None, :] (window indices of one music): x right after learning the session, y the same
        lineage after a later session (for session 0 every model descends from the s0 model)."""
        i = Wn["i"][x][:, None]
        vx, vy = Wn["v"][x][:, None], Wn["v"][y][None, :]
        return (Wn["t"][x][:, None] == i) & (Wn["t"][y][None, :] > i) & ((vx == vy) | ((i == 0) & (vx == s0_id)))

    def forget_pair(wa: int, wb: int) -> bool:
        return bool(forget_matrix(np.array([wa]), np.array([wb]))[0, 0])

    # ---------------- style pairs: greedy fill of the (type, |delta| bin) quotas -----------------------------------
    order = np.argsort(Wn["key"], kind="stable")
    bounds = np.flatnonzero(np.diff(Wn["key"][order])) + 1
    groups = [g for g in np.split(order, bounds) if len(g) >= 2]
    rng.shuffle(groups)
    # cells: (type, |delta| bin, composition); "dance" = both clips are dance to the encoder (P(other) < threshold),
    # so humans must judge style, not dance vs no dance; "any" = no constraint
    per_cell = a.pairs // (len(TYPES) * (len(BINS) - 1))
    quota = {}
    for ty in TYPES:
        for b in range(len(BINS) - 1):
            quota[ty, b, "dance"] = int(round(per_cell * a.dance_pairs))
            quota[ty, b, "any"] = per_cell - quota[ty, b, "dance"]
    for x in [k for k in quota if k[2] == "any"][: a.pairs - sum(quota.values())]:
        quota[x] += 1
    dancing = Wn["po"] < a.dance_thresh
    n_genres = len({int(x) for x in Wn["g"]})
    genre_cap = math.ceil(1.5 * a.pairs / n_genres)
    music_cap = max(2, math.ceil(3 * a.pairs / len(music_id)))
    used_g, used_m, pairs, used_grp = {}, {}, [], set()
    sample = lambda xs, n: [xs[j] for j in rng.permutation(len(xs))[:n]]
    for strict in (True, False):  # second pass: fill whatever quota is left with any cell
        for gi, grp in enumerate(groups):
            if len(pairs) >= a.pairs:
                break
            if gi in used_grp:
                continue
            g, m = int(Wn["g"][grp[0]]), int(Wn["key"][grp[0]] // (S_MAX * K_MAX))
            if used_g.get(g, 0) >= genre_cap or used_m.get(m, 0) >= music_cap:
                continue
            # forget pairs are rare among random draws, so they are enumerated; cross pairs are drawn at random;
            # both from all windows and from the dancing ones only (dancing pairs with a large delta are rarer)
            forget = [(grp[x], grp[y]) for x, y in np.argwhere(forget_matrix(grp, grp))]
            dg = grp[dancing[grp]]
            drawn = sample(forget, 12) + sample([p for p in forget if dancing[p[0]] and dancing[p[1]]], 12)
            drawn += [tuple(rng.choice(grp, 2, replace=False)) for _ in range(12)]
            drawn += [tuple(rng.choice(dg, 2, replace=False)) for _ in range(12)] if len(dg) >= 2 else []
            cand = []
            for wa, wb in drawn:
                ty = "forget" if forget_pair(wa, wb) else "cross"
                d = abs(float(Wn["pg"][wa] - Wn["pg"][wb]))
                b = int(np.searchsorted(BINS, d, side="right") - 1)
                both = bool(dancing[wa] and dancing[wb])
                cell = "dance" if both and quota[ty, b, "dance"] > 0 else "any"
                need = quota[ty, b, cell] if strict else 1
                if need > 0:
                    cand.append((need + rng.random(), ty, b, cell, both, int(wa), int(wb), d))
            for _, ty, b, cell, both, wa, wb, d in sorted(cand, reverse=True):
                if window_ok(wa) and window_ok(wb):
                    pairs.append({"type": ty, "bin": b, "a": wa, "b": wb, "delta": d, "both_dance": both})
                    quota[ty, b, cell] -= 1
                    used_g[g] = used_g.get(g, 0) + 1
                    used_m[m] = used_m.get(m, 0) + 1
                    used_grp.add(gi)  # one pair per (music, sample, window)
                    break
    left = {f"{k[0]}/bin{k[1]}/{k[2]}": v for k, v in quota.items() if v > 0}
    print(f"[study] {len(pairs)} style pairs ({sum(p['both_dance'] for p in pairs)} with both clips dancing); "
          f"unfilled quota {left or 'none'}")

    # ---------------- real dance: references, catch and practice clips, genre centroids --------------------------
    gt = []
    for s in range(len(man["sessions"])):
        for n in [l.strip() for l in (root / f"s{s}" / "test.txt").open() if l.strip()]:
            J = recover_joint_positions_272(np.load(ref_dir / f"{n}.npy"), 22).astype(np.float32)
            if len(J) >= WIN:
                kin, geo = cached_window_features(ref_dir, n)
                gt.append({"name": n, "genre": n[1:3], "J": J, "kin": kin, "geo": geo})
    e = enc.encode([d["J"] for d in gt])
    for x, d in enumerate(gt):
        sel = e["owner"] == x
        d["emb"], d["prob"] = e["emb"][sel], e["prob"][sel]
        nw = min(len(d["emb"]), len(d["kin"]))
        d["emb"], d["prob"], d["kin"], d["geo"] = d["emb"][:nw], d["prob"][:nw], d["kin"][:nw], d["geo"][:nw]
    by_genre = {g: [x for x, d in enumerate(gt) if d["genre"] == g] for g in genres}
    for d in gt:
        d["speed"] = np.array([speed(d["J"][STRIDE * k:STRIDE * k + WIN]) for k in range(len(d["emb"]))])
    # real clips are taken from the genre's more active half (sequences open and close standing still); chosen by
    # motion alone, so the references do not lean towards what the encoder thinks the genre is
    med_speed = {g: float(np.median(np.concatenate([gt[x]["speed"] for x in xs]))) for g, xs in by_genre.items() if xs}

    def gt_window(x: int, check: bool) -> int | None:
        d, g = gt[x], gt[x]["genre"]
        ok = [k for k in range(len(d["emb"])) if not problems(d["J"][STRIDE * k:STRIDE * k + WIN])
              and d["speed"][k] >= max(0.3, med_speed[g])
              # attention checks must be unambiguous: the encoder also sees the clip's own genre
              and (not check or classes[int(d["prob"][k].argmax())] == g)]
        return int(rng.choice(ok)) if ok else None

    used_seq: set[int] = set()

    def take_gt(g: str, check: bool = False) -> tuple[int, int] | None:
        for x in rng.permutation(by_genre[g]):
            if int(x) in used_seq:
                continue
            k = gt_window(int(x), check)
            if k is not None:
                used_seq.add(int(x))
                return int(x), k
        return None

    pair_genres = sorted({genres[int(Wn["g"][p["a"]])] for p in pairs}, key=genres.index)
    references = {g: [r for r in (take_gt(g) for _ in range(a.refs_per_genre)) if r] for g in pair_genres}
    ref_seqs = {x for rs in references.values() for x, _ in rs}

    cent, kin_mu, geo_mu = {}, {}, {}
    for g in genres:
        xs = [x for x in by_genre[g] if x not in ref_seqs]
        if not xs:
            continue
        E = np.concatenate([gt[x]["emb"] for x in xs])
        cent[g] = E.mean(0) / (np.linalg.norm(E.mean(0)) + 1e-9)
        kin_mu[g] = np.concatenate([gt[x]["kin"] for x in xs]).mean(0)
        geo_mu[g] = np.concatenate([gt[x]["geo"] for x in xs])[:, keep].mean(0)

    def far_genre(g: str) -> str:  # a genre of another session, among the 3 farthest in style space
        cands = sorted((h for h in cent if sess_of[h] != sess_of[g]), key=lambda h: float(cent[g] @ cent[h]))
        return str(rng.choice(cands[:3]))

    def scores(prob, emb, kin, geo, g: str, i: int) -> dict:
        x = np.concatenate([kin, geo])
        pp = softmax(((x - probe.mu) / probe.sd) @ probe.W.T + probe.b)
        return {"ours_genre": float(prob[ci[g]]), "ours_session": float(prob[soc == i].sum()),
                "p_other": float(prob[ci["other"]]) if "other" in ci else 0.0,
                "argmax": classes[int(prob.argmax())],
                "probe_genre": float(pp[probe_classes.index(g)]),
                "probe_session": float(sum(pp[c] for c, h in enumerate(probe_classes) if sess_of.get(h) == i)),
                "kin": -float(np.sqrt(np.mean(((kin - kin_mu[g]) / refs["kin_std"]) ** 2))),
                "geo": -float(np.sqrt(np.mean(((geo[keep] - geo_mu[g]) / refs["geo_std"]) ** 2))),
                "emb_cos": float(emb @ cent[g])}

    clips, joints = {}, {}

    def new_id() -> str:
        while True:
            c = f"{int(rng.integers(16 ** 8)):08x}"
            if c not in clips:
                return c

    def add_gt(x: int, k: int) -> str:
        d, cid = gt[x], new_id()
        g = d["genre"]
        joints[cid] = d["J"][STRIDE * k:STRIDE * k + WIN]
        clips[cid] = {"kind": "gt", "genre": g, "session": sess_of[g], "variant": "gt", "method": "gt", "after": None,
                      "music": d["name"], "sample": None, "window": k, "start_frame": STRIDE * k,
                      "scores": scores(d["prob"][k], d["emb"][k], d["kin"][k], d["geo"][k], g, sess_of[g])}
        return cid

    gen_ids: dict[int, str] = {}

    def add_gen(w: int) -> str:
        if w not in gen_ids:
            fi, q, k = int(Wn["fi"][w]), int(Wn["q"][w]), int(Wn["k"][w])
            v, t, i, _, names = files[fi]
            cid = new_id()
            joints[cid] = dance(fi, q)[STRIDE * k:STRIDE * k + WIN]
            clips[cid] = {"kind": "gen", "genre": genres[int(Wn["g"][w])], "session": i, "variant": v,
                          "method": re.sub(r"_seed\d+$", "", v), "after": t, "music": names[q % len(names)],
                          "sample": q // len(names), "window": k, "start_frame": STRIDE * k, "_w": w}
            gen_ids[w] = cid
        return gen_ids[w]

    trials = []
    for p in pairs:
        trials.append({"task": "style", "type": p["type"], "genre": genres[int(Wn["g"][p["a"]])],
                       "a": add_gen(p["a"]), "b": add_gen(p["b"]), "delta": p["delta"], "bin": p["bin"],
                       "both_dance": p["both_dance"]})
    rng.shuffle(trials)
    for n, tr in enumerate(trials, 1):
        tr["id"] = f"m{n:03d}"
    for kind, count in (("catch", a.catch), ("practice", a.practice)):
        for n in range(count):
            g = pair_genres[n % len(pair_genres)] if kind == "catch" else str(rng.choice(pair_genres))
            h = far_genre(g)
            ra, rb = take_gt(g, check=True), take_gt(h, check=True)
            if not (ra and rb):  # no window the encoder recognises: fall back to any active one
                print(f"[study] WARNING: {kind} trial {n + 1}: no real {g if not ra else h} window recognised as its "
                      f"genre, using an unchecked one")
                ra, rb = ra or take_gt(g), rb or take_gt(h)
            if ra and rb:
                trials.append({"id": f"{kind[0]}{n + 1:02d}", "task": "style", "type": kind, "genre": g,
                               "a": add_gt(*ra), "b": add_gt(*rb), "expected": "a", "delta": None, "bin": None})
            else:
                print(f"[study] WARNING: {kind} trial {n + 1}: no unused real sequence of {g if not ra else 'a far genre'}")

    # ---------------- dance trials: generated windows over P(other), plus some real ones --------------------------
    in_pairs = {p["a"] for p in pairs} | {p["b"] for p in pairs}
    per_bin = a.dance // (len(DANCE_BINS) - 1)
    dance_trials = []
    for b in range(len(DANCE_BINS) - 1):
        pool = np.flatnonzero((Wn["po"] >= DANCE_BINS[b]) & (Wn["po"] < DANCE_BINS[b + 1]))
        got = 0
        for w in rng.permutation(pool)[: 50 * per_bin]:
            if got == per_bin:
                break
            if int(w) not in in_pairs and window_ok(int(w)):
                dance_trials.append({"task": "dance", "clip": add_gen(int(w)), "bin": b})
                got += 1
        if got < per_bin:
            print(f"[study] dance trials: only {got}/{per_bin} windows with P(other) in "
                  f"[{DANCE_BINS[b]:.1f}, {DANCE_BINS[b + 1]:.1f})")
    for _ in range(max(1, a.dance // 6)):
        r = take_gt(str(rng.choice(pair_genres)))
        if r:
            dance_trials.append({"task": "dance", "clip": add_gt(*r), "bin": None})
    rng.shuffle(dance_trials)
    for n, tr in enumerate(dance_trials, 1):
        tr["id"] = f"d{n:02d}"

    # ---------------- scores of the generated clips on their exact window --------------------------------------
    gen = [c for c in clips if clips[c]["kind"] == "gen"]
    emb, logits = enc.encode_windows(np.stack([joints[c] for c in gen]))
    prob = softmax(logits / enc.temperature)
    worst = 0.0
    feats: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for c, pr, em in zip(gen, prob, emb):
        w = clips[c].pop("_w")
        fi, q, k = int(Wn["fi"][w]), int(Wn["q"][w]), int(Wn["k"][w])
        worst = max(worst, abs(float(pr[ci[clips[c]["genre"]]] - Wn["pg"][w])))
        if (fi, q) not in feats:
            feats[fi, q] = window_features(root_relative(dance(fi, q)))
        kin, geo = feats[fi, q]
        clips[c]["scores"] = scores(pr, em, kin[k], geo[k], clips[c]["genre"], clips[c]["session"])
    print(f"[study] window alignment check: max |P(genre) recomputed - cached| = {worst:.2e}"
          + ("  WARNING: windows misaligned" if worst > 1e-3 else ""))

    # ---------------- write -------------------------------------------------------------------------------
    ref_ids = {g: [add_gt(x, k) for x, k in rs] for g, rs in references.items()}
    stage = a.out.with_suffix("")
    stage.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(stage / "clips.npz", **{c: j.astype(np.float32) for c, j in joints.items()})
    study = {"benchmark": a.benchmark, "results": str(a.results), "created": time.strftime("%Y-%m-%d %H:%M"),
             "fps": FPS, "window": WIN, "seed": a.seed, "args": {k: str(v) for k, v in vars(a).items()},
             "reference_dir": str(ref_dir), "classes": classes, "genres": {g: GENRE.get(g, g) for g in genres},
             "sessions": [s["groups"] for s in man["sessions"]], "bins": list(BINS[:-1]), "dance_bins": list(DANCE_BINS[:-1]),
             "references": ref_ids,
             "trials": trials, "dance_trials": dance_trials, "clips": clips,
             "variants": sorted({f[0] for f in files})}
    (stage / "study.json").write_text(json.dumps(study, indent=1))
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("study.json", "clips.npz"):
            z.write(stage / f, f"study_pack/{f}")
    main_n = sum(t["type"] in TYPES for t in trials)
    cells = {f"{ty}/bin{b}": sum(t["type"] == ty and t["bin"] == b for t in trials) for ty in TYPES for b in range(3)}
    print(f"[study] {main_n} style trials {cells}, {sum(t['type'] == 'catch' for t in trials)} catch, "
          f"{sum(t['type'] == 'practice' for t in trials)} practice, {len(dance_trials)} dance; {len(clips)} clips")
    per_genre = {g: sum(t["genre"] == g for t in trials if t["type"] in TYPES) for g in pair_genres}
    print(f"[study] style trials per genre: {per_genre}")
    print(f"[study] wrote {a.out} ({a.out.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
