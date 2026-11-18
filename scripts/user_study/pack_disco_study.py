#!/usr/bin/env python3
"""
Pack the stimuli of the DiscoForcing preference study (TMR+KD against fine-tuning, judged on Google Forms). Runs on the
server from the repo root; the laptop renders the videos (scripts/user_study/render_form_studies.py disco).

Reads the dances scripts/eval_cl_session.py saved next to its scores (<results>/<variant>/after_s{t}_sess{i}.joints.npz),
the real test dance (noise.json reference_dir: the reference StyleRetention uses, VAE reconstructions of the test dance)
and the test music (raw_data/aistpp/wavs/<sequence>.wav, aligned with the dance at frame 0).

Design, fixed before any output is inspected; every random choice comes from --seed:
  retention   after the final session (--final, default the last), one trial per earlier genre (8 on aistpp_genre5),
              on that genre's test music: does the final model still dance the genres it learned first?
  plasticity  one trial per later session k = 1..final, with the model right after session k, on the test music of
              one of its two genres (drawn at random; the other genre in the next version): does replay cost the new
              genre?
  check       attention checks (--checks, default 2): one option is the reference dance itself, the other a real dance
              of a genre two or more sessions away, both played with the reference's music
A trial shows the real dance of one test sequence (the reference) next to the two models' dances for its music, with
the same generation sample (same noise seed in eval_cl_session.py) and the same --seconds window, whose start is drawn
uniformly at least --offset s into the dance. Music is drawn among the genre's test music not yet used in the version
(preferring music no other version uses), then a sequence of that music, a sample and a start. A draw is redrawn only
when a panel fails the automatic checks (NaN, a joint > 1.5 m from the pelvis, root > 6 m/s); nothing else is
filtered. Which side shows TMR is balanced within each condition and randomised; the order is random with the checks
spread over the form (never among the first three). --versions draws that many forms with the same rule. --extra packs
more models (default GCR+KD) on the same windows; they do not affect the draw and are shown only if chosen later.

  python scripts/user_study/pack_disco_study.py --results results/aistpp_genre5/screen/v2_40000_full_s0v2
  -> disco_study_pack.zip (disco_study_pack/{study.json, clips.npz, wavs/*.wav, pack_log.txt}); tens of MB (the music
     is cut to the trial windows). CPU only: the per-window style scores reported next to each panel run on
     STYLE_ENCODER_DEVICE (default cpu); a score file without a .style.npz cache is encoded once and cached, as
     scripts/user_study/pack_study.py does
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import sys
import time
import wave
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
FPS = 30
GENRE = {"BR": "Break", "PO": "Pop", "LO": "Lock", "WA": "Waack", "MH": "Middle Hip-hop", "LH": "LA Hip-hop",
         "HO": "House", "KR": "Krump", "JS": "Street Jazz", "JB": "Ballet Jazz"}


def problems(J: np.ndarray) -> str:
    """Why a window is unfit to show (empty when fine); the checks of scripts/user_study/pack_study.py."""
    if not np.isfinite(J).all():
        return "NaN"
    if np.linalg.norm(J - J[:, :1], axis=2).max() > 1.5:
        return "a joint > 1.5 m from the pelvis"
    if len(J) > 1 and np.linalg.norm(np.diff(J[:, 0, [0, 2]], axis=0), axis=1).max() * FPS > 6:
        return "root > 6 m/s"
    return ""


def music_id(name: str) -> str:
    """gLO_sFM_cAll_d13_mLO4_ch02 -> mLO4 (sequences of one song share their music)."""
    m = re.search(r"_(m[A-Za-z]{2}\d+)_", name)
    return m.group(1) if m else name


def load_module(path: str):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def wav_seconds(p: Path) -> float | None:
    try:
        with wave.open(str(p), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except (wave.Error, EOFError, OSError):
        return None


def cut_wav(src: Path, dst: Path, t0: float, dur: float) -> dict:
    """The [t0, t0 + dur) seconds of a PCM wav; any other wav is copied whole and cut by the renderer."""
    try:
        with wave.open(str(src), "rb") as w:
            sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
            a = min(n, int(round(t0 * sr)))
            b = min(n, a + int(round(dur * sr)))
            w.setpos(a)
            data = w.readframes(b - a)
        with wave.open(str(dst), "wb") as o:
            o.setnchannels(ch)
            o.setsampwidth(sw)
            o.setframerate(sr)
            o.writeframes(data)
        return {"offset": 0.0, "seconds": round((b - a) / sr, 3), "rate": sr, "channels": ch}
    except (wave.Error, EOFError):
        shutil.copy(src, dst)
        return {"offset": round(t0, 3), "seconds": round(dur, 3), "whole_file": True}


class Dances:
    """The dances one score file saved: entry q belongs to names[q % len(names)], sample q // len(names)."""

    def __init__(self, score: Path):
        self.score = score
        self.meta = json.loads(score.read_text())
        self.names = list(self.meta.get("names") or [])
        self.index = {n: k for k, n in enumerate(self.names)}
        self.samples = int(self.meta.get("samples") or 1)
        self.z = np.load(score.with_suffix(".joints.npz"))
        if len(self.z.files) < self.samples * len(self.names):
            self.samples = len(self.z.files) // max(1, len(self.names))

    def entry(self, name: str, sample: int) -> int:
        return sample * len(self.names) + self.index[name]

    def get(self, name: str, sample: int) -> np.ndarray:
        return self.z[str(self.entry(name, sample))].astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True, help="screen results folder (one subfolder per variant)")
    ap.add_argument("--benchmark", default="aistpp_genre5")
    ap.add_argument("--methods", nargs=2, default=["tmr_joint_neff_kd", "vanilla"], metavar=("OURS", "REFERENCE"),
                    help="the two compared variants (default: TMR+KD against fine-tuning)")
    ap.add_argument("--extra", nargs="*", default=["gcr_kd"], help="more variants packed on the same windows")
    ap.add_argument("--final", type=int, default=None, help="final session (default: the benchmark's last)")
    ap.add_argument("--seconds", type=float, default=15.0, help="length of every trial's window")
    ap.add_argument("--offset", type=float, default=3.0, help="windows start at least this far into the dance")
    ap.add_argument("--checks", type=int, default=2, help="attention checks per version")
    ap.add_argument("--versions", type=int, default=2, help="forms with disjoint draws (1, 2, ...)")
    ap.add_argument("--ref_dir", type=Path, default=None, help="272-dim real dance (default: noise.json reference_dir)")
    ap.add_argument("--wav_dir", type=Path, default=Path("raw_data/aistpp/wavs"))
    ap.add_argument("--no_scores", action="store_true", help="skip the style encoder (the draw does not use it)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("disco_study_pack.zip"))
    a = ap.parse_args()
    t0 = time.time()
    lines: list[str] = []

    def say(msg: str) -> None:
        print(f"[disco-study] {msg}", flush=True)
        lines.append(msg)

    from utils.motion_process import recover_joint_positions_272

    root = Path("raw_data/cl_sessions") / a.benchmark
    man = json.loads((root / "manifest.json").read_text())
    sessions = [list(s["groups"]) for s in man["sessions"]]
    final = len(sessions) - 1 if a.final is None else a.final
    if not 1 <= final < len(sessions):
        raise SystemExit(f"--final {final}: the benchmark has sessions 0..{len(sessions) - 1}")
    sess_of = {g: s for s, gs in enumerate(sessions) for g in gs}
    noise = json.loads((root / "eval" / "noise.json").read_text())
    ref_dir = a.ref_dir or Path(noise["reference_dir"])
    ours, base = a.methods
    extra = [m for m in a.extra if m not in a.methods]
    say(f"benchmark {a.benchmark}: sessions {sessions}, final S{final}; {ours} against {base}"
        + (f", extra {extra}" if extra else "") + f"; real dance from {ref_dir}")

    # ---------------- saved dances of every checkpoint the design needs ---------------------------------------
    need = sorted({(final, i) for i in range(final)} | {(k, k) for k in range(1, final + 1)})
    dances: dict[tuple[str, int, int], Dances] = {}
    missing = []
    for m in [ours, base] + extra:
        for t, i in need:
            p = a.results / m / f"after_s{t}_sess{i}.json"
            ok = p.exists() and p.with_suffix(".joints.npz").exists()
            if ok:
                d = Dances(p)
                ok = bool(d.names)
            if ok:
                dances[m, t, i] = d
            else:
                missing.append((m, t, i, p))
    for m, t, i, p in missing:
        say(f"MISSING dances of {m} after S{t} on session {i} ({p}{'' if p.exists() else ': no score file'})")
        if p.exists():
            meta = json.loads(p.read_text())
            say(f"  regenerate: CUDA_VISIBLE_DEVICES=<gpu> python scripts/eval_cl_session.py --config "
                f"configs/cl/ldf_{a.benchmark}_s{t}.yaml --vae_ckpt {meta.get('vae_ckpt')} --ckpt {meta.get('ckpt')} "
                f"--session {i} --samples {meta.get('samples', 3)}"
                + (f" --max_tokens {meta['max_tokens']}" if meta.get("max_tokens") else "")
                + f" --out {p}   (writes the dances next to the score; rerun this script afterwards)")

    # ---------------- real dance and music -------------------------------------------------------------------
    gt_cache: dict[str, np.ndarray | None] = {}

    def gt(name: str) -> np.ndarray | None:
        if name not in gt_cache:
            p = ref_dir / f"{name}.npy"
            gt_cache[name] = recover_joint_positions_272(np.load(p), 22).astype(np.float32) if p.exists() else None
        return gt_cache[name]

    wav_len: dict[str, float | None] = {}

    def music_frames(name: str) -> int | None:
        if name not in wav_len:
            p = a.wav_dir / f"{name}.wav"
            wav_len[name] = (wav_seconds(p) or -1.0) if p.exists() else None
            if wav_len[name] == -1.0:  # not plain PCM: length unknown, trust the dance
                wav_len[name] = 1e9
        return None if wav_len[name] is None else int(wav_len[name] * FPS)

    test_names = {s: [l.strip() for l in (root / f"s{s}" / "test.txt").open() if l.strip()]
                  for s in range(len(sessions)) if (root / f"s{s}" / "test.txt").exists()}
    L, lo = int(round(a.seconds * FPS)), int(round(a.offset * FPS))
    music_versions: dict[str, set[int]] = {}  # music id -> versions that use it

    def music_order(rng, musics: list[str], v: int, used: set[str]) -> list[str]:
        """Unused in this version; music no other version uses first, random order within each group."""
        free = [mu for mu in musics if mu not in used]
        perm = [free[k] for k in rng.permutation(len(free))]
        return sorted(perm, key=lambda mu: len(music_versions.get(mu, set()) - {v}) > 0)

    def draw_gen(rng, v: int, used: set[str], genre: str, da: Dances, db: Dances) -> dict | None:
        """Music, sequence, sample and window for the two compared dances and the real one."""
        names = [n for n in da.names if n[1:3] == genre and n in db.index]
        by_music: dict[str, list[str]] = {}
        for n in names:
            by_music.setdefault(music_id(n), []).append(n)
        for mu in music_order(rng, sorted(by_music), v, used):
            for k in rng.permutation(len(by_music[mu])):
                n = by_music[mu][int(k)]
                G, mf = gt(n), music_frames(n)
                if G is None or mf is None:
                    say(f"  v{v} {genre}: {n} skipped, no {'real dance' if G is None else 'music'}")
                    continue
                samples = min(da.samples, db.samples)
                for _ in range(20):
                    s = int(rng.integers(samples))
                    Ja, Jb = da.get(n, s), db.get(n, s)
                    T = min(len(Ja), len(Jb), len(G), mf)
                    if T < L:
                        say(f"  v{v} {genre}: {n} skipped, {T / FPS:.1f} s < {a.seconds:g} s")
                        break
                    st = int(rng.integers(min(lo, T - L), T - L + 1))
                    seg = {"ours": Ja[st:st + L], "base": Jb[st:st + L], "ref": G[st:st + L]}
                    bad = {k2: problems(x) for k2, x in seg.items() if problems(x)}
                    if bad:
                        say(f"  v{v} {genre}: {n} sample {s} @{st / FPS:.1f} s redrawn ({bad})")
                        continue
                    used.add(mu)
                    music_versions.setdefault(mu, set()).add(v)
                    return {"music": n, "music_id": mu, "sample": s, "start_frame": st, "T": T, "seg": seg}
        return None

    def draw_real(rng, v: int, used: set[str] | None, genre: str, avoid: set[str]) -> dict | None:
        """A window of a real test dance of the genre (attention checks)."""
        names = [n for n in test_names.get(sess_of[genre], []) if n[1:3] == genre and n not in avoid]
        for k in rng.permutation(len(names)):
            n = names[int(k)]
            if used is not None and music_id(n) in used:
                continue
            G, mf = gt(n), music_frames(n) if used is not None else 10 ** 9
            if G is None or mf is None:
                continue
            T = min(len(G), mf)
            if T < L:
                continue
            for _ in range(20):
                st = int(rng.integers(min(lo, T - L), T - L + 1))
                if not problems(G[st:st + L]):
                    if used is not None:
                        used.add(music_id(n))
                        music_versions.setdefault(music_id(n), set()).add(v)
                    return {"music": n, "music_id": music_id(n), "start_frame": st, "T": T, "seg": G[st:st + L]}
        return None

    # ---------------- style scores of every packed window (reported, never used by the draw) -------------------
    enc = None
    if not a.no_scores:
        os.environ.setdefault("STYLE_ENCODER_DEVICE", "cpu")  # leave the GPUs to the queued runs
        try:
            from metrics.style_encoder import StyleEncoder, benchmark_session_of_class, ckpt_path

            enc = StyleEncoder(ckpt_path(man["dataset"], -1))
            classes = list(enc.classes)
            ci = {c: k for k, c in enumerate(classes)}
            soc = np.asarray(benchmark_session_of_class(classes, man))
            tbl = load_module("scripts/cl/screen_matrix_table.py")
        except Exception as e:  # noqa: BLE001 - the scores are a report, the pack is still complete without them
            say(f"WARNING: no style scores ({type(e).__name__}: {e})")
            enc = None

    def seg_scores(J: np.ndarray, genre: str) -> dict | None:
        if enc is None:
            return None
        p = enc.encode([J])["prob"]
        top = Counter(classes[int(k)] for k in p.argmax(1)).most_common(2)
        return {"p_genre": float(p[:, ci[genre]].mean()), "p_session": float(p[:, soc == sess_of[genre]].sum(1).mean()),
                "p_other": float(p[:, ci["other"]].mean()) if "other" in ci else 0.0,
                "top": [[g, n] for g, n in top], "windows": int(len(p))}

    def dance_sr(d: Dances, name: str, sample: int, session: int) -> float | None:
        """StyleRetention of the whole dance the window comes from (the scores' cached window probabilities)."""
        if enc is None:
            return None
        try:
            prob, owner = tbl.encoded(d.score, enc)
            sel = owner == d.entry(name, sample)
            return float(prob[sel][:, soc == session].sum(1).mean()) if sel.any() else None
        except Exception:  # noqa: BLE001
            return None

    # ---------------- the forms --------------------------------------------------------------------------------
    rng = np.random.default_rng(a.seed)
    plast_pick = {k: int(rng.integers(2)) for k in range(1, final + 1)}  # which genre of session k, version 1
    clips: dict[str, np.ndarray] = {}
    clip_info: dict[str, dict] = {}
    stage = a.out.with_suffix("")
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "wavs").mkdir(parents=True)

    def new_id() -> str:
        while True:
            c = f"{int(rng.integers(16 ** 8)):08x}"
            if c not in clips:
                return c

    def add_clip(J: np.ndarray, info: dict) -> str:
        cid = new_id()
        clips[cid] = J.astype(np.float32)
        clip_info[cid] = info
        return cid

    versions = {}
    for v in range(1, a.versions + 1):
        used: set[str] = set()
        trials = []
        # retention: every earlier genre after the final session
        for i in range(final):
            for g in sessions[i]:
                if (ours, final, i) not in dances or (base, final, i) not in dances:
                    say(f"v{v} retention {g}: SKIPPED, dances after S{final} on session {i} missing")
                    continue
                trials.append({"condition": "retention", "genre": g, "after": final, "test_session": i})
        # plasticity: one genre of every later session, right after it is learned
        for k in range(1, final + 1):
            if (ours, k, k) not in dances or (base, k, k) not in dances:
                say(f"v{v} plasticity S{k}: SKIPPED, dances after S{k} on session {k} missing")
                continue
            g = sessions[k][(plast_pick[k] + v - 1) % len(sessions[k])]
            trials.append({"condition": "plasticity", "genre": g, "after": k, "test_session": k,
                           "alternative": [h for h in sessions[k] if h != g]})
        built = []
        for tr in trials:
            key = (tr["after"], tr["test_session"])
            got = draw_gen(rng, v, used, tr["genre"], dances[(ours, *key)], dances[(base, *key)])
            if got is None and tr["condition"] == "plasticity" and tr["alternative"]:
                h = tr["alternative"][0]
                say(f"v{v} plasticity S{tr['after']}: no usable music of {tr['genre']}, using {h}")
                tr["genre"] = h
                got = draw_gen(rng, v, used, h, dances[(ours, *key)], dances[(base, *key)])
            if got is None:
                say(f"v{v} {tr['condition']} {tr['genre']}: SKIPPED, no usable test music")
                continue
            tr.update({k2: got[k2] for k2 in ("music", "music_id", "sample", "start_frame")})
            tr["_seg"] = got["seg"]
            built.append(tr)
        # attention checks: reference itself against a real dance two or more sessions away
        genres = [g for gs in sessions for g in gs]
        for c in range(a.checks):
            for _ in range(50):
                g = genres[int(rng.integers(len(genres)))]
                far = [h for h in genres if abs(sess_of[h] - sess_of[g]) >= 2]
                h = far[int(rng.integers(len(far)))]
                r = draw_real(rng, v, used, g, set())
                if r is None:
                    continue
                f = draw_real(rng, v, None, h, {r["music"]})
                if f is None:
                    continue
                built.append({"condition": "check", "genre": g, "foil_genre": h, "after": None, "test_session": None,
                              "music": r["music"], "music_id": r["music_id"], "sample": None,
                              "start_frame": r["start_frame"], "foil_music": f["music"],
                              "foil_start_frame": f["start_frame"], "_seg": {"ref": r["seg"], "same": r["seg"],
                                                                            "far": f["seg"]}})
                break
            else:
                say(f"v{v} check {c + 1}: SKIPPED, no real dance long enough")
        # sides: TMR on A for half of each condition; checks alternate the side of the correct answer
        for cond in ("retention", "plasticity", "check"):
            group = [t for t in built if t["condition"] == cond]
            n_a = len(group) // 2 + (int(rng.integers(2)) if len(group) % 2 else 0)
            on_a = set(rng.permutation(len(group))[:n_a].tolist())
            for k, t in enumerate(group):
                t["_a_first"] = k in on_a
        # order: test trials shuffled, checks spread over the form, never among the first three
        test = [t for t in built if t["condition"] != "check"]
        checks = [t for t in built if t["condition"] == "check"]
        test = [test[k] for k in rng.permutation(len(test))]
        n = len(test) + len(checks)
        slots: set[int] = set()
        if checks:  # one check per equal part of positions 4..n (0-based 3..n-1), never two in a row
            edges = np.linspace(min(3, n - len(checks)), n, len(checks) + 1)
            for c in range(len(checks)):
                last = c == len(checks) - 1
                lo_s = int(edges[c])
                hi_s = max(lo_s + 1, int(edges[c + 1]) - (0 if last else 1))
                free = [p for p in range(lo_s, hi_s) if p not in slots] or [p for p in range(n) if p not in slots]
                slots.add(free[int(rng.integers(len(free)))])
        it_test, it_check = iter(test), iter(checks)
        order = [next(it_check) if pos in slots else next(it_test) for pos in range(n)]
        # clips, music and scores
        out = []
        for pos, t in enumerate(order, 1):
            tid = f"{v}-{pos:02d}"
            seg, g = t.pop("_seg"), t["genre"]
            a_first = t.pop("_a_first")
            gen_info = lambda m: {"role": "gen", "variant": m, "after": t["after"], "test_session": t["test_session"],
                                  "music": t["music"], "sample": t["sample"], "start_frame": t["start_frame"]}
            ref_info = {"role": "ref", "music": t["music"], "start_frame": t["start_frame"]}
            if t["condition"] == "check":
                same = add_clip(seg["same"], dict(ref_info, role="check_same"))
                far = add_clip(seg["far"], {"role": "check_far", "music": t["foil_music"],
                                            "start_frame": t["foil_start_frame"], "genre": t["foil_genre"]})
                A, B = (same, far) if a_first else (far, same)
                key = {"A": "same" if a_first else f"far:{t['foil_genre']}", "B": f"far:{t['foil_genre']}" if a_first else "same"}
                t["expected"] = "A" if a_first else "B"
                scores = {"A": seg_scores(clips[A], g), "B": seg_scores(clips[B], g)}
                extra_ids = {}
            else:
                cid_o, cid_b = add_clip(seg["ours"], gen_info(ours)), add_clip(seg["base"], gen_info(base))
                A, B = (cid_o, cid_b) if a_first else (cid_b, cid_o)
                key = {"A": ours if a_first else base, "B": base if a_first else ours}
                t["expected"] = None
                scores = {"A": seg_scores(clips[A], g), "B": seg_scores(clips[B], g)}
                dk = (t["after"], t["test_session"])
                t["dance_sr"] = {side: dance_sr(dances[(m, *dk)], t["music"], t["sample"], t["test_session"])
                                 for side, m in key.items()}
                t["method_sr"] = {side: dances[(m, *dk)].meta.get("StyleRetention") for side, m in key.items()}
                extra_ids = {}
                for m in extra:  # same music, sample (noise) and window; left out when it cannot be shown
                    d = dances.get((m, *dk))
                    if d is None or t["music"] not in d.index or t["sample"] >= d.samples:
                        continue
                    J = d.get(t["music"], t["sample"])[t["start_frame"]:t["start_frame"] + L]
                    if len(J) == L and not problems(J):
                        extra_ids[m] = add_clip(J, gen_info(m))
                        scores[m] = seg_scores(J, g)
            ref = add_clip(seg["ref"], ref_info)
            scores["ref"] = seg_scores(seg["ref"], g)
            wav = cut_wav(a.wav_dir / f"{t['music']}.wav", stage / "wavs" / f"{tid}.wav", t["start_frame"] / FPS, a.seconds)
            t.pop("alternative", None)
            out.append({"id": tid, "version": v, "position": pos, **t, "genre_name": GENRE.get(g, g),
                        "frames": L, "panels": {"ref": ref, "A": A, "B": B}, "key": key, "extra": extra_ids,
                        "wav": dict(wav, file=f"wavs/{tid}.wav"), "scores": scores})
        versions[str(v)] = out
        cnt = Counter(t["condition"] for t in out)
        say(f"version {v}: {len(out)} trials {dict(cnt)}; genres {[t['genre'] for t in out]}")

    # ---------------- report and write ----------------------------------------------------------------------
    for v, out in versions.items():
        for t in out:
            s = t["scores"]
            sc = "" if s.get("A") is None else (
                f"  P(session) A {s['A']['p_session']:.2f} B {s['B']['p_session']:.2f} ref {s['ref']['p_session']:.2f}")
            when = f"after S{t['after']}" if t["after"] is not None else "real   "
            say(f"  {t['id']} {t['condition']:<10} {t['genre']} {when}  {t['music']} sample {t['sample']} "
                f"@{t['start_frame'] / FPS:5.1f} s  A={t['key']['A']} B={t['key']['B']}{sc}")
    np.savez_compressed(stage / "clips.npz", **clips)
    study = {"kind": "disco_preference", "benchmark": a.benchmark, "results": str(a.results),
             "created": time.strftime("%Y-%m-%d %H:%M"), "fps": FPS, "seconds": a.seconds, "offset": a.offset,
             "seed": a.seed, "args": {k: str(x) for k, x in vars(a).items()}, "reference_dir": str(ref_dir),
             "methods": {"ours": ours, "reference": base, "extra": extra}, "final": final, "sessions": sessions,
             "genres": {g: GENRE.get(g, g) for gs in sessions for g in gs}, "missing": [str(x[3]) for x in missing],
             "versions": versions, "clips": clip_info,
             "style_scores": "mean over the 5 s windows (1 s stride) of each packed window" if enc else None}
    (stage / "study.json").write_text(json.dumps(study, indent=1))
    (stage / "pack_log.txt").write_text("\n".join(lines) + "\n")
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(stage.rglob("*")):
            if f.is_file():
                z.write(f, f"disco_study_pack/{f.relative_to(stage).as_posix()}")
    say(f"wrote {a.out} ({a.out.stat().st_size / 1e6:.1f} MB, {len(clips)} clips) in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
