#!/usr/bin/env python3
"""
Render the two Google Forms user studies on the laptop (the server is headless) and write what the forms need.

  disco  Study 1, the DiscoForcing preference study, from disco_study_pack.zip (scripts/user_study/pack_disco_study.py):
         one 1920x1080 video per trial with the music: the real dance of the test sequence on the left (reference),
         the two models' dances as A and B. Order, sides and windows were fixed by the pack.
  style  Study 2, the StyleRetention validation, from the StyleRetention study pack (scripts/user_study/pack_study.py,
         on the laptop C:/Users/chinnu/3d_cvpr/study_pack): one silent video per clip, two real reference clips of
         the music's genre and the TEST clip, the 5 s window (the unit StyleRetention scores) shown twice.

Study 2 clip selection (fixed rule, seeded, decided before any rating): --clips generated windows per version from the
pack's generated windows (every method and checkpoint), with distinct music, at most 2 per genre and 3 per method.
Among such sets the one is kept whose values under the five compared scores (StyleRetention at genre level, the
handcrafted classifier, 1 - P(other): probabilities, spread over [0, 1]; the kinetic and geometric feature distances:
no scale of their own, spread over their percentiles) cover each score's range most evenly and are least rank-
correlated with one another: symmetric in the metrics, and spread so that human ratings can tell them apart (the
maximum-differentiation idea of Wang & Simoncelli, 2008). Version 2 repeats the rule on the clips and
music version 1 left. Three controls per version: a hidden reference (another real clip of the reference genre), a
real clip of a genre from a distant session, and a figure standing still.

Output per study, under --out: <study>/videos/*.mp4, form_spec.json (videos in order with their titles and the
questions; no labels) and answer_key.csv, which is PRIVATE (what each video shows and every metric's score).

  python scripts/user_study/render_form_studies.py style --pack C:/Users/chinnu/3d_cvpr/study_pack --out C:/Users/chinnu/3d_cvpr/form_studies
  python scripts/user_study/render_form_studies.py disco --pack C:/Users/chinnu/3d_cvpr/disco_study_pack.zip --out C:/Users/chinnu/3d_cvpr/form_studies
needs numpy, opencv, Pillow, matplotlib, imageio-ffmpeg (as scripts/cl/demo_render.py)
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import os
import subprocess
import tempfile
import zipfile
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
FPS, W, H = 30, 1920, 1080
BIG, SMALL = (680, 742), (440, 480)  # panel sizes (w, h); every panel is drawn at BIG (528 x 576 scaled)
INK, MUTED, ACCENT, LINE = (238, 240, 244), (150, 154, 166), (111, 179, 255), (70, 74, 86)
GENRE = {"BR": "Break", "PO": "Pop", "LO": "Lock", "WA": "Waack", "MH": "Middle Hip-hop", "LH": "LA Hip-hop",
         "HO": "House", "KR": "Krump", "JS": "Street Jazz", "JB": "Ballet Jazz"}
DR = None  # scripts/cl/demo_render.py, loaded per process

DISCO_QUESTIONS = [
    {"id": "style", "text": "Style: which dance, A or B, is closer to the STYLE of the real dance on the left?",
     "choices": ["A", "B"]},
    {"id": "music", "text": "Music: which dance, A or B, moves better WITH THE MUSIC (rhythm, beats, energy)?",
     "choices": ["A", "B"]},
    {"id": "natural", "text": "Naturalness: which dance, A or B, looks more NATURAL as human movement?",
     "choices": ["A", "B"]}]
STYLE_QUESTIONS = [
    {"id": "style", "text": "How close is the dance STYLE of the TEST figure to the two REFERENCE dances?",
     "scale": [1, 5], "low": "Completely different style, or not dancing", "high": "Same style"},
    {"id": "dancing", "text": "Is the TEST figure dancing (in any style)?", "choices": ["Yes", "No", "Not sure"]}]


def _load_demo_render():
    global DR
    if DR is None:
        spec = importlib.util.spec_from_file_location("demo_render", REPO / "scripts" / "cl" / "demo_render.py")
        DR = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(DR)
        DR.PW, DR.VH = BIG  # View reads the panel size from these module constants
    return DR


def resolve_ffmpeg() -> str:
    """ffmpeg and its H.264 encoder, found once in the main process (see scripts/user_study/render_study.py)."""
    import imageio_ffmpeg

    os.environ.setdefault("IMAGEIO_FFMPEG_EXE", imageio_ffmpeg.get_ffmpeg_exe())
    try:
        from imageio_ffmpeg._io import get_first_available_h264_encoder

        return get_first_available_h264_encoder()
    except (ImportError, RuntimeError):
        return "libx264"


def done(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 20_000


def signature(*parts) -> str:
    """What a video shows (clip ids, texts, length, music): a rerun re-renders a file whose design changed."""
    import hashlib

    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def current(job: dict) -> bool:
    out = Path(job["out"])
    sig = out.with_suffix(".sig")
    return done(out) and sig.exists() and sig.read_text().strip() == job["sig"]


# ---------------------------------------------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------------------------------------------
def overlay(texts: list, boxes: list) -> np.ndarray:
    """The static part of a frame: background, labels, dashed boxes."""
    from PIL import Image, ImageDraw

    dr = _load_demo_render()
    img = Image.new("RGB", (W, H), dr.BG)
    d = ImageDraw.Draw(img)
    for x0, y0, x1, y1 in boxes:  # dashed rectangle
        for a, b in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
            n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) // 14)
            for k in range(0, n, 2):
                p = (a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n)
                q = (a[0] + (b[0] - a[0]) * (k + 1) / n, a[1] + (b[1] - a[1]) * (k + 1) / n)
                d.line([p, q], fill=LINE, width=3)
    for text, xy, size, bold, fill, anchor in texts:
        d.text(xy, text, font=dr.font(size, bold), fill=fill, anchor=anchor)
    return np.asarray(img).copy()


def render_job(job: dict) -> str:
    """One video: panels (joints, x, y, size) drawn every frame over a static overlay, a time bar, optional music."""
    import cv2
    import imageio_ffmpeg

    out = Path(job["out"])
    if current(job):
        return out.name
    dr = _load_demo_render()
    views = [(dr.View(np.asarray(J, np.float64), job["yaw"], job["elev"], job["dist"]), x, y, size, len(J))
             for J, x, y, size in job["panels"]]
    base = overlay(job["texts"], job["boxes"])
    n = job["frames"]
    tmp = Path(tempfile.mkdtemp()) / "video.mp4"
    params = ["-movflags", "+faststart"] + (["-crf", "18", "-preset", "medium"] if job["codec"] == "libx264" else [])
    w = imageio_ffmpeg.write_frames(str(tmp), (W, H), fps=FPS, codec=job["codec"], macro_block_size=8,
                                    quality=None if job["codec"] == "libx264" else 8, ffmpeg_log_level="error",
                                    output_params=params)
    w.send(None)
    for t in range(n):
        frame = base.copy()
        for view, x, y, size, period in views:
            img = view.draw(t % period)
            if tuple(size) != BIG:
                img = cv2.resize(img, tuple(size), interpolation=cv2.INTER_AREA)
            frame[y:y + size[1], x:x + size[0]] = img
            frame[y - 2:y, x - 2:x + size[0] + 2] = LINE  # thin frame around every panel
            frame[y + size[1]:y + size[1] + 2, x - 2:x + size[0] + 2] = LINE
            frame[y:y + size[1], x - 2:x] = LINE
            frame[y:y + size[1], x + size[0]:x + size[0] + 2] = LINE
        bx0, bx1, by = 40, W - 40, H - 22  # elapsed time
        frame[by:by + 6, bx0:bx1] = LINE
        frame[by:by + 6, bx0:bx0 + int((bx1 - bx0) * (t + 1) / n)] = ACCENT
        w.send(np.ascontiguousarray(frame))
    w.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(".part.mp4")
    if job.get("wav"):
        cmd = [os.environ.get("IMAGEIO_FFMPEG_EXE") or imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
               "-i", str(tmp)]
        if job.get("wav_offset"):
            cmd += ["-ss", f"{job['wav_offset']:.3f}"]
        cmd += ["-t", f"{n / FPS:.3f}", "-i", str(job["wav"]), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest", "-movflags", "+faststart", str(part)]
        subprocess.run(cmd, check=True)
    else:
        part.write_bytes(tmp.read_bytes())
    tmp.unlink()
    if not done(part):
        raise RuntimeError(f"{part} has {part.stat().st_size} bytes: is the disk full?")
    part.replace(out)
    out.with_suffix(".sig").write_text(job["sig"])
    return out.name


def run_jobs(jobs: list[dict], workers: int, only: list[str] | None = None) -> None:
    """Render every video not yet on disk (reruns resume); only: video file names to restrict to (tests, redos)."""
    todo = [j for j in jobs if not current(j) and (not only or Path(j["out"]).name in only)]
    print(f"[form] rendering {len(todo)} of {len(jobs)} videos with {workers} workers ({jobs[0]['codec'] if jobs else ''})",
          flush=True)
    if not todo:
        return
    with Pool(workers, initializer=_load_demo_render) as pool:
        for k, name in enumerate(pool.imap_unordered(render_job, todo), 1):
            print(f"  {k}/{len(todo)} {name}", flush=True)


def insert_positions(n_test: int, n_ctrl: int, rng, first: int = 2) -> set[int]:
    """0-based positions of the controls among n_test + n_ctrl: one per equal part of positions first..end, never two
    in a row."""
    n = n_test + n_ctrl
    edges = np.linspace(min(first, n - n_ctrl), n, n_ctrl + 1)
    slots: set[int] = set()
    for c in range(n_ctrl):
        lo = int(edges[c])
        hi = max(lo + 1, int(edges[c + 1]) - (0 if c == n_ctrl - 1 else 1))
        free = [p for p in range(lo, hi) if p not in slots] or [p for p in range(n) if p not in slots]
        slots.add(free[int(rng.integers(len(free)))])
    return slots


# ---------------------------------------------------------------------------------------------------------------
# Study 1: DiscoForcing preference (TMR+KD against fine-tuning)
# ---------------------------------------------------------------------------------------------------------------
def disco(a) -> int:
    pack = a.pack
    if pack.suffix == ".zip":
        dest = pack.with_suffix("")
        if not (dest / "disco_study_pack" / "study.json").exists():
            with zipfile.ZipFile(pack) as z:
                z.extractall(dest)
        pack = dest / "disco_study_pack"
    elif (pack / "disco_study_pack" / "study.json").exists():
        pack = pack / "disco_study_pack"
    study = json.loads((pack / "study.json").read_text())
    clips = np.load(pack / "clips.npz")
    root = a.out / "study1_disco"
    vdir = root / "videos"
    vdir.mkdir(parents=True, exist_ok=True)
    codec = resolve_ffmpeg()
    versions = [str(v) for v in (a.versions or sorted(study["versions"], key=int))]
    jobs, spec, key_rows = [], {}, []
    n_frames = int(round(study["seconds"] * FPS))
    xs = {"ref": (20, 150 + (BIG[1] - SMALL[1]) // 2 - 60), "A": (510, 150), "B": (1220, 150)}
    for v in versions:
        trials = study["versions"][v]
        spec[v] = []
        for t in trials:
            name = f"study1_v{v}_t{t['position']:02d}.mp4"
            rx, ry = xs["ref"]
            texts = [(f"Trial {t['position']}", (40, 48), 40, True, INK, "lm"),
                     ("sound on", (W - 40, 48), 26, False, MUTED, "rm"),
                     ("Real dance", (rx + SMALL[0] // 2, ry - 52), 32, True, INK, "mm"),
                     ("(reference)", (rx + SMALL[0] // 2, ry - 18), 24, False, MUTED, "mm"),
                     ("A", (xs["A"][0] + BIG[0] // 2, 112), 60, True, ACCENT, "mm"),
                     ("B", (xs["B"][0] + BIG[0] // 2, 112), 60, True, ACCENT, "mm"),
                     ("A and B: computer-generated", (rx + SMALL[0] // 2, ry + SMALL[1] + 34), 22, False, MUTED, "mm"),
                     ("dances for the same music", (rx + SMALL[0] // 2, ry + SMALL[1] + 64), 22, False, MUTED, "mm"),
                     ("Style  ·  Music  ·  Naturalness: answer the three questions below the video",
                      (W // 2, 960), 28, False, MUTED, "mm")]
            panels = [(clips[t["panels"]["ref"]], rx, ry, SMALL), (clips[t["panels"]["A"]], *xs["A"], BIG),
                      (clips[t["panels"]["B"]], *xs["B"], BIG)]
            wav = pack / t["wav"]["file"]
            jobs.append({"out": str(vdir / name), "panels": panels, "texts": texts, "boxes": [], "frames": n_frames,
                         "wav": str(wav) if wav.exists() else None, "wav_offset": t["wav"].get("offset", 0.0),
                         "yaw": a.yaw, "elev": a.elev, "dist": a.dist, "codec": codec,
                         "sig": signature(study["created"], t["panels"], texts, n_frames, t["wav"], a.yaw, a.elev, a.dist)})
            spec[v].append({"position": t["position"], "video": f"videos/{name}", "title": f"Trial {t['position']}",
                            "kind": "check" if t["condition"] == "check" else "test"})
            s = t.get("scores") or {}
            g = lambda side, k: "" if not s.get(side) else f"{s[side][k]:.4f}"
            key_rows.append({"version": v, "position": t["position"], "video": name, "trial": t["id"],
                             "condition": t["condition"], "genre": t["genre"], "after_session": t["after"],
                             "music": t["music"], "sample": t["sample"], "start_s": round(t["start_frame"] / FPS, 2),
                             "A": t["key"]["A"], "B": t["key"]["B"], "expected_style": t.get("expected") or "",
                             "A_p_session": g("A", "p_session"), "B_p_session": g("B", "p_session"),
                             "A_p_genre": g("A", "p_genre"), "B_p_genre": g("B", "p_genre"),
                             "ref_p_session": g("ref", "p_session"),
                             "A_dance_sr": (t.get("dance_sr") or {}).get("A") or "",
                             "B_dance_sr": (t.get("dance_sr") or {}).get("B") or ""})
    if not a.plan_only:
        run_jobs(jobs, a.workers, a.only)
    write_outputs(root, "disco", spec, key_rows, {
        "title": "Dance generation study",
        "minutes": 12, "questions": DISCO_QUESTIONS, "methods": study["methods"], "pack_created": study["created"],
        "seconds": study["seconds"]})
    return 0


# ---------------------------------------------------------------------------------------------------------------
# Study 2: StyleRetention against human style ratings
# ---------------------------------------------------------------------------------------------------------------
DIMS = ["ours_genre", "probe_genre", "kin", "geo", "dance_rate"]


def score_of(c: dict, d: str) -> float:
    s = c["scores"]
    return 1.0 - s["p_other"] if d == "dance_rate" else float(s[d])


def select_style(study: dict, n: int, versions: int, seed: int) -> list[list[str]]:
    """Per version, n generated clips by the fixed rule of the module docstring."""
    clips = study["clips"]
    pool = sorted(c for c, x in clips.items() if x["kind"] == "gen")
    X = np.array([[score_of(clips[c], d) for d in DIMS] for c in pool])
    # each score on its natural scale: probabilities as they are (they live on [0, 1]), the two feature distances,
    # which have no scale of their own, as percentile ranks over the pool
    U = X.copy()
    for j, d in enumerate(DIMS):
        if d in ("kin", "geo"):
            U[:, j] = np.argsort(np.argsort(X[:, j], kind="mergesort"), kind="mergesort") / (len(pool) - 1)
    music = [clips[c]["music"] for c in pool]
    genre = [clips[c]["genre"] for c in pool]
    method = [clips[c]["method"] for c in pool]
    grid = (np.arange(n) + 0.5) / n
    iu = np.triu_indices(len(DIMS), 1)

    def cost(S: list[int]) -> float:
        u = U[S]
        cover = np.abs(np.sort(u, axis=0) - grid[:, None]).mean()
        r = np.argsort(np.argsort(u, axis=0), axis=0).astype(float)
        return float(cover + 0.5 * np.abs(np.corrcoef(r.T)[iu]).mean())

    def feasible(S: list[int]) -> bool:
        return (len({music[k] for k in S}) == len(S) and max(Counter(genre[k] for k in S).values()) <= 2
                and max(Counter(method[k] for k in S).values()) <= 3)

    rng = np.random.default_rng(seed)
    used_music: set[str] = set()
    out = []
    for _ in range(versions):
        avail = [k for k in range(len(pool)) if music[k] not in used_music]

        def random_set() -> list[int] | None:
            S, gm, gg, gmeth = [], set(), Counter(), Counter()
            for k in rng.permutation(avail):
                k = int(k)
                if music[k] in gm or gg[genre[k]] >= 2 or gmeth[method[k]] >= 3:
                    continue
                S.append(k)
                gm.add(music[k])
                gg[genre[k]] += 1
                gmeth[method[k]] += 1
                if len(S) == n:
                    return S
            return None

        starts = sorted((s for s in (random_set() for _ in range(3000)) if s), key=cost)[:4]
        best, best_c = None, np.inf
        for S in starts:  # first-improvement swaps until no swap lowers the cost
            c = cost(S)
            improved = True
            while improved:
                improved = False
                for i in rng.permutation(n):
                    for k in rng.permutation(avail):
                        k = int(k)
                        if k in S:
                            continue
                        T = S[:i] + [k] + S[i + 1:]
                        if feasible(T):
                            ct = cost(T)
                            if ct < c - 1e-12:
                                S, c, improved = T, ct, True
                                break
                    if improved:
                        break
            if c < best_c:
                best, best_c = list(S), c
        used_music |= {music[k] for k in best}
        r = np.argsort(np.argsort(U[best], axis=0), axis=0).astype(float)
        C = np.corrcoef(r.T)
        print(f"[form] style version {len(out) + 1}: cost {best_c:.3f}; rank correlation of SR with "
              + ", ".join(f"{d} {C[0, j]:+.2f}" for j, d in enumerate(DIMS) if j)
              + "; range covered per score (probability, or percentile for kin / geo) "
              + ", ".join(f"{d} {U[best, j].min():.2f}..{U[best, j].max():.2f}" for j, d in enumerate(DIMS)))
        out.append([pool[k] for k in best])
    return out


def static_figure(real: list[np.ndarray], frames: int) -> np.ndarray:
    """A figure standing still (clearly not dancing): over every real clip, the frame that is stillest and closest to
    plain standing (upright, legs straight, wrists below the pelvis, feet under the hips), held, with a 1 cm sway.
    SMPL joints, y up: 0 pelvis, 7/8 ankles, 15 head, 20/21 wrists."""
    best, best_s = None, np.inf
    for J in real:
        sp = np.linalg.norm(np.diff(J, axis=0), axis=-1).mean(1) * FPS
        sp = np.convolve(np.r_[sp, sp[-1]], np.ones(9) / 9, mode="same")
        pel = J[:, 0, 1]
        score = (sp + 2 * np.clip(np.maximum(J[:, 20, 1], J[:, 21, 1]) - pel + 0.05, 0, None)
                 + 2 * np.clip(0.5 - (J[:, 15, 1] - pel), 0, None)
                 + 2 * np.clip(0.8 - (pel - np.minimum(J[:, 7, 1], J[:, 8, 1])), 0, None)
                 + np.abs(np.linalg.norm(J[:, 7, [0, 2]] - J[:, 8, [0, 2]], axis=1) - 0.25))
        k = int(np.argmin(score))
        if score[k] < best_s:
            best, best_s = J[k], float(score[k])
    S = np.repeat(best[None], frames, 0).copy()
    t = np.arange(frames) / FPS
    S[:, :, 0] += 0.01 * np.sin(2 * np.pi * 0.25 * t)[:, None]
    return S


def style(a) -> int:
    pack = a.pack
    if pack.suffix == ".zip":
        with zipfile.ZipFile(pack) as z:
            study = json.loads(z.read("study_pack/study.json"))
            npz = np.load(io.BytesIO(z.read("study_pack/clips.npz")))
            joints = {k: npz[k] for k in npz.files}
    else:
        study = json.loads((pack / "study.json").read_text())
        npz = np.load(pack / "clips.npz")
        joints = {k: npz[k] for k in npz.files}
    clips, refs = study["clips"], study["references"]
    rng = np.random.default_rng(a.seed + 1)
    chosen = select_style(study, a.clips, a.n_versions, a.seed)
    # controls, shared by the versions: hidden reference and distant genre from the pack's attention-check pairs
    # (real clips the encoder recognises as their genre), and a figure standing still
    catch = [t for t in study["trials"] if t["type"] == "catch"]
    c1, c2 = [catch[int(k)] for k in rng.permutation(len(catch))[:2]]
    if c1["genre"] == c2["genre"]:
        c2 = next(t for t in catch if t["genre"] != c1["genre"])
    g3 = str(rng.choice(sorted({g for g in refs if g not in (c1["genre"], c2["genre"])})))
    joints["static"] = static_figure([joints[c] for c, x in clips.items() if x["kind"] == "gt"], 150)
    controls = [{"kind": "control_hidden_reference", "clip": c1["a"], "genre": c1["genre"], "expected": "high, dancing"},
                {"kind": "control_distant_genre", "clip": c2["b"], "genre": c2["genre"],
                 "shown_genre": clips[c2["b"]]["genre"], "expected": "low"},
                {"kind": "control_static", "clip": "static", "genre": g3, "expected": "not dancing, low"}]
    root = a.out / "study2_style"
    vdir = root / "videos"
    vdir.mkdir(parents=True, exist_ok=True)
    codec = resolve_ffmpeg()
    jobs, spec, key_rows = [], {}, []
    loops = 2
    rx = [140, 600]  # reference panels, then the TEST panel
    tx, ty, ry = 1180, 170, 170 + (BIG[1] - SMALL[1]) // 2
    for vi, ids in enumerate(chosen, 1):
        v = str(vi)
        items = [{"kind": "test", "clip": c, "genre": clips[c]["genre"]} for c in ids]
        items = [items[int(k)] for k in rng.permutation(len(items))]
        slots = insert_positions(len(items), len(controls), rng)
        ctrl = iter([controls[int(k)] for k in rng.permutation(len(controls))])
        it = iter(items)
        order = [next(ctrl) if p in slots else next(it) for p in range(len(items) + len(controls))]
        spec[v] = []
        for pos, item in enumerate(order, 1):
            name = f"study2_v{v}_c{pos:02d}.mp4"
            g = item["genre"]
            texts = [(f"Clip {pos}", (40, 48), 40, True, INK, "lm"),
                     ("silent · plays twice", (W - 40, 48), 26, False, MUTED, "rm"),
                     ("REFERENCE STYLE (real dance)", ((rx[0] + rx[1] + SMALL[0]) // 2, ry - 34), 30, True, INK, "mm"),
                     ("TEST", (tx + BIG[0] // 2, ty - 44), 44, True, ACCENT, "mm"),
                     ("How close is the TEST dance's style to the references?   ·   Is the TEST figure dancing?",
                      (W // 2, 975), 28, False, MUTED, "mm")]
            boxes = [(rx[0] - 24, ry - 72, rx[1] + SMALL[0] + 24, ry + SMALL[1] + 24)]
            panels = [(joints[refs[g][0]], rx[0], ry, SMALL), (joints[refs[g][1]], rx[1], ry, SMALL),
                      (joints[item["clip"]], tx, ty, BIG)]
            jobs.append({"out": str(vdir / name), "panels": panels, "texts": texts, "boxes": boxes,
                         "frames": loops * 150, "wav": None, "yaw": a.yaw, "elev": a.elev, "dist": a.dist,
                         "codec": codec, "sig": signature(study["created"], refs[g], item["clip"], texts, boxes,
                                                          loops, a.yaw, a.elev, a.dist)})
            spec[v].append({"position": pos, "video": f"videos/{name}", "title": f"Clip {pos}",
                            "kind": "test" if item["kind"] == "test" else "control"})
            c = clips.get(item["clip"], {"kind": "static", "scores": {}})
            row = {"version": v, "position": pos, "video": name, "kind": item["kind"], "clip": item["clip"],
                   "reference_genre": g, "clip_genre": c.get("genre", ""), "source": c.get("kind", ""),
                   "method": c.get("method", ""), "after_session": c.get("after", ""), "music": c.get("music", ""),
                   "sample": c.get("sample", ""), "window": c.get("window", ""), "expected": item.get("expected", "")}
            row.update({k: (f"{x:.5f}" if isinstance(x, float) else x) for k, x in c.get("scores", {}).items()})
            key_rows.append(row)
    if not a.plan_only:
        run_jobs(jobs, a.workers, a.only)
    write_outputs(root, "style", spec, key_rows, {
        "title": "Dance style study", "minutes": 8, "questions": STYLE_QUESTIONS, "pack_created": study["created"],
        "selection": {"clips_per_version": a.clips, "versions": a.n_versions, "seed": a.seed, "scores": DIMS}})
    return 0


def write_outputs(root: Path, kind: str, spec: dict, key_rows: list[dict], meta: dict) -> None:
    (root / "form_spec.json").write_text(json.dumps({"study": kind, **meta, "versions": spec}, indent=1))
    cols = list(dict.fromkeys(k for r in key_rows for k in r))
    with (root / "answer_key.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(key_rows)
    size = sum(p.stat().st_size for p in (root / "videos").glob("*.mp4")) / 1e6
    print(f"[form] {root}: {sum(len(x) for x in spec.values())} videos ({size:.0f} MB), form_spec.json, "
          f"answer_key.csv (private)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="study", required=True)
    for name in ("disco", "style"):
        p = sub.add_parser(name)
        p.add_argument("--pack", type=Path, required=True)
        p.add_argument("--out", type=Path, default=Path("form_studies"))
        p.add_argument("--workers", type=int, default=4, help="the laptop has ~16 GB: keep 4-5")
        p.add_argument("--yaw", type=float, default=25.0)
        p.add_argument("--elev", type=float, default=12.0)
        p.add_argument("--dist", type=float, default=3.7)
        p.add_argument("--plan_only", action="store_true", help="write form_spec.json / answer_key.csv, render nothing")
        p.add_argument("--only", nargs="*", default=None, help="render only these video file names")
        if name == "disco":
            p.add_argument("--versions", nargs="*", default=None, help="versions to render (default all)")
        else:
            p.add_argument("--clips", type=int, default=12, help="generated clips per version")
            p.add_argument("--n_versions", type=int, default=2)
            p.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    return disco(a) if a.study == "disco" else style(a)


if __name__ == "__main__":
    raise SystemExit(main())
