#!/usr/bin/env python3
"""
Render the forgetting demo from the pack written by scripts/cl/demo_pack.sh (runs anywhere: numpy, opencv, Pillow,
matplotlib for its fonts, imageio-ffmpeg for a bundled ffmpeg; no model or repo imports). The same session 0 test
music danced by several models side by side, with the music. Under each dancer: the style encoder's probability that
the current 5 s are session 0 style (the per-window quantity StyleRetention averages) and the genre it looks most like.

  python scripts/cl/demo_render.py --pack demo_pack.zip --list            # music/sample ranked by the contrast
  python scripts/cl/demo_render.py --pack demo_pack.zip                   # render the best-ranked one
  python scripts/cl/demo_render.py --pack demo_pack.zip --top 5           # the 5 best-ranked, one video each
  python scripts/cl/demo_render.py --pack demo_pack.zip --music <name> --sample 1 --start 5 --seconds 20

--panels <variant>[@<session trained up to>] ... (default: s0_model tmr_joint_neff tmr_joint_neff_kd, i.e. just
learned, TMR replay without distillation, TMR replay with distillation). The ranking scores a (music, sample) by
min(first, last panel) - max(middle panels) of the S0 style probability over the whole dance; every panel uses the
same sample index (same generation seed), and dances with broken poses are flagged and not picked.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

import cv2
import imageio_ffmpeg
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FPS, WIN, STRIDE = 30, 150, 30  # style encoder windows (metrics/style_encoder.py)
GENRE = {"BR": "Break", "PO": "Pop", "LO": "Lock", "WA": "Waack", "MH": "Middle Hip-hop", "LH": "LA Hip-hop",
         "HO": "House", "KR": "Krump", "JS": "Street Jazz", "JB": "Ballet Jazz", "other": "not dance"}
METHOD = {"tmr_joint_neff": "TMR", "tmr": "TMR", "gcr": "GCR", "random": "Random", "random_star": "Random*",
          "ocs": "OCS", "mir": "MIR", "csrel": "CSReL", "tmgm": "TMGM"}
# SMPL 22 joints, y up: (bone, colour) for the left limbs, right limbs and spine
LEFT = [(0, 1), (1, 4), (4, 7), (7, 10), (9, 13), (13, 16), (16, 18), (18, 20)]
RIGHT = [(0, 2), (2, 5), (5, 8), (8, 11), (9, 14), (14, 17), (17, 19), (19, 21)]
SPINE = [(0, 3), (3, 6), (6, 9), (9, 12)]
BONES = [(a, b, (80, 165, 255)) for a, b in LEFT] + [(a, b, (255, 135, 70)) for a, b in RIGHT] \
    + [(a, b, (230, 230, 236)) for a, b in SPINE]
BG, TILE_A, TILE_B, SHADOW = (19, 21, 27), (60, 64, 77), (45, 48, 58), (8, 8, 11)
LOW, HIGH = np.array([235, 85, 70]), np.array([85, 205, 125])  # bar colour at 0 % / 100 % S0 style

PW, VH, TOP, HEAD, FOOT, GAP = 528, 576, 48, 64, 80, 8  # panel width, 3-D view height, bands (pixels)
SUB = 16  # cv2 sub-pixel drawing: coordinates x 16 (shift=4)
FONTS = Path(matplotlib.get_data_path()) / "fonts" / "ttf"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONTS / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")), size)


def smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    r = int(3 * sigma)
    if r < 1 or len(x) < 2:
        return x.copy()
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    xp = np.pad(x, ((r, r), (0, 0)), mode="edge")
    return np.stack([np.convolve(xp[:, i], k / k.sum(), mode="valid") for i in range(x.shape[1])], 1)


def problems(J: np.ndarray) -> str:
    """Why a generated dance is unfit for a demo (empty when it is fine)."""
    if not np.isfinite(J).all():
        return "NaN"
    reach = np.linalg.norm(J - J[:, :1], axis=2).max()
    if reach > 1.5:
        return f"a joint {reach:.1f} m from the pelvis"
    speed = np.linalg.norm(np.diff(J[:, 0, [0, 2]], axis=0), axis=1).max() * FPS if len(J) > 1 else 0.0
    return f"root moves {speed:.0f} m/s" if speed > 6 else ""


def per_frame(prob: np.ndarray, T: int) -> np.ndarray:
    """Window probabilities (windows x classes, 150 frames each, 30-frame stride) -> per frame: mean of the windows
    that contain the frame (frames after the last full window take the last window)."""
    acc, cnt = np.zeros((T, prob.shape[1])), np.zeros(T)
    for k, p in enumerate(prob):
        acc[STRIDE * k: min(T, STRIDE * k + WIN)] += p
        cnt[STRIDE * k: min(T, STRIDE * k + WIN)] += 1
    acc[cnt == 0], cnt[cnt == 0] = prob[-1], 1
    return acc / cnt[:, None]


class Panel:
    """One model's dances for the session 0 test music (a dance file of the pack)."""

    def __init__(self, pack: Path, d: dict, s0: np.ndarray):
        self.d, self.names, self.samples = d, d["names"], d["samples"]
        self.joints = np.load(pack / d["file"])
        st = np.load(pack / d["file"].replace(".joints.npz", ".style.npz"))
        self.prob, self.owner = st["prob"], st["owner"]
        s0p = self.prob[:, s0].sum(1)
        self.mean_s0 = np.bincount(self.owner, weights=s0p) / np.maximum(np.bincount(self.owner), 1)

    def entry(self, music: str, sample: int) -> int:  # joints.npz entry q = sample * n_music + music index
        return sample * len(self.names) + self.names.index(music)

    def label(self) -> tuple[str, str]:
        v, t = self.d["variant"], self.d["after"]
        seed = re.search(r"_seed(\d+)$", v)
        base = re.sub(r"_seed\d+$", "", v)
        kd = base.endswith("_kd")
        m = METHOD.get(base[:-3] if kd else base, base[:-3] if kd else base)
        if v == "s0_model":
            title, sub = "Right after learning S0", "the session 0 model"
        elif base == "vanilla":
            title, sub = f"After S{t}: no replay", "fine-tuned on every new session"
        elif kd:
            title, sub = f"After S{t}: {m} + distillation", f"{m} replay + distillation from the previous model"
        else:
            title, sub = f"After S{t}: {m} replay", f"{m} replay buffer, no distillation"
        return title, sub + (f" (seed {seed.group(1)})" if seed else "")


class View:
    """Perspective camera in front of the dancer (its mean facing direction + yaw), following the smoothed root."""

    def __init__(self, J: np.ndarray, yaw: float, elev: float, dist: float):
        self.J = J
        self.floor = float(np.percentile(J[:, :, 1].min(1), 5))
        across = J[:, 1] - J[:, 2]  # left hip - right hip
        face = np.stack([-across[:, 2], np.zeros(len(J)), across[:, 0]], 1)
        face /= np.linalg.norm(face, axis=1, keepdims=True) + 1e-8
        m = face.sum(0)
        a = (math.atan2(m[0], m[2]) if math.hypot(m[0], m[2]) > 1e-3 * len(J) else 0.0) + math.radians(yaw)
        e = math.radians(elev)
        back = np.array([math.sin(a) * math.cos(e), math.sin(e), math.cos(a) * math.cos(e)])  # target -> camera
        root = smooth(J[:, 0, [0, 2]], 20)
        self.target = np.stack([root[:, 0], np.full(len(J), self.floor + 0.9), root[:, 1]], 1)
        self.cam = self.target + dist * back
        self.fwd = -back
        self.right = np.cross(self.fwd, [0, 1, 0])
        self.right /= np.linalg.norm(self.right)
        self.up = np.cross(self.right, self.fwd)
        self.f = 1.15 * VH

    def project(self, P: np.ndarray, t: int) -> tuple[np.ndarray, np.ndarray]:
        d = P - self.cam[t]
        z = d @ self.fwd
        zc = np.maximum(z, 1e-3)
        xy = np.stack([PW / 2 + self.f * (d @ self.right) / zc, VH / 2 - self.f * (d @ self.up) / zc], -1)
        return xy, z

    def draw(self, t: int) -> np.ndarray:
        img = np.empty((VH, PW, 3), np.uint8)
        img[:] = BG
        # floor: world-fixed 0.5 m checker tiles fading out 3.4 m from the dancer
        R, S = 3.4, 0.5
        tx, tz = self.target[t, 0], self.target[t, 2]
        ii, kk = np.meshgrid(np.arange(math.floor((tx - R) / S), math.ceil((tx + R) / S)),
                             np.arange(math.floor((tz - R) / S), math.ceil((tz + R) / S)), indexing="ij")
        ii, kk = ii.ravel(), kk.ravel()
        dc = np.hypot((ii + 0.5) * S - tx, (kk + 0.5) * S - tz)
        keep = dc < R
        ii, kk, dc = ii[keep], kk[keep], dc[keep]
        corners = np.stack([np.stack([(ii + a) * S, np.full(len(ii), self.floor), (kk + b) * S], -1)
                            for a, b in ((0, 0), (1, 0), (1, 1), (0, 1))], 1)
        xy, z = self.project(corners, t)
        for c in range(len(ii)):
            if z[c].min() < 0.3:
                continue
            fade = min(1.0, (R - dc[c]) / (0.6 * R))
            tile = np.array(TILE_A if (ii[c] + kk[c]) % 2 else TILE_B)
            cv2.fillConvexPoly(img, np.round(xy[c] * SUB).astype(np.int32),
                               tuple(int(x) for x in BG + (tile - BG) * fade), cv2.LINE_AA, 4)
        J = self.J[t]
        px = 0.045 * self.f  # bone thickness in pixels at 1 m
        # soft shadow: the skeleton flattened onto the floor
        sh = J.copy()
        sh[:, 1] = self.floor
        sxy, sz = self.project(sh, t)
        layer = img.copy()
        for a, b, _ in BONES:
            cv2.line(layer, tuple(np.round(sxy[a] * SUB).astype(int)), tuple(np.round(sxy[b] * SUB).astype(int)),
                     SHADOW, max(2, int(px / max(sz[a], 0.3))), cv2.LINE_AA, 4)
        img = cv2.addWeighted(layer, 0.45, img, 0.55, 0)
        # bones far to near, then the head
        xy, z = self.project(J, t)
        q = lambda i: tuple(np.round(xy[i] * SUB).astype(int))
        for a, b, col in sorted(BONES, key=lambda x: -(z[x[0]] + z[x[1]])):
            th = max(2, int(px / max((z[a] + z[b]) / 2, 0.3)))
            cv2.line(img, q(a), q(b), col, th, cv2.LINE_AA, 4)
            cv2.circle(img, q(b), int(th * 0.55 * SUB), (250, 250, 252), -1, cv2.LINE_AA, 4)
        head = J[15] + 0.6 * (J[15] - J[12])
        hxy, hz = self.project(head[None], t)
        r = 0.105 * self.f / max(hz[0], 0.3)
        cv2.line(img, q(12), tuple(np.round(hxy[0] * SUB).astype(int)), (230, 230, 236),
                 max(2, int(px / max(hz[0], 0.3))), cv2.LINE_AA, 4)
        cv2.circle(img, tuple(np.round(hxy[0] * SUB).astype(int)), int(r * SUB), (230, 230, 236), -1, cv2.LINE_AA, 4)
        return img


def rank(panels: list[Panel]) -> list[tuple]:
    """(score, music, sample, per-panel S0 style, problem) for every music/sample all panels have."""
    rows = []
    for music in panels[0].names:
        if not all(music in p.names for p in panels):
            continue
        for k in range(min(p.samples for p in panels)):
            v = [float(p.mean_s0[p.entry(music, k)]) for p in panels]
            score = min(v[0], v[-1]) - max(v[1:-1]) if len(v) > 2 else v[0] - v[-1]
            bad = [f"{p.d['variant']}: {e}" for p in panels if (e := problems(p.joints[str(p.entry(music, k))]))]
            rows.append((score, music, k, v, "; ".join(bad)))
    return sorted(rows, key=lambda r: (bool(r[4]), -r[0]))


def render(panels: list[Panel], titles: list[tuple[str, str]], info: dict, pack: Path, music: str, k: int, args,
           out: Path):
    s0 = np.asarray(info["session_of_class"]) == 0
    classes = info["classes"]
    s0_name = " + ".join(GENRE.get(g, g) for g in info["sessions"][0])
    js = [p.joints[str(p.entry(music, k))] for p in panels]
    T = min(len(j) for j in js)
    a = int(round(args.start * FPS))
    b = T if args.seconds is None else min(T, a + int(round(args.seconds * FPS)))
    if a >= b:
        raise SystemExit(f"--start {args.start} is past the end of the dance ({T / FPS:.1f} s)")
    views = [View(j[:T], args.yaw, args.elev, args.dist) for j in js]
    pf = []
    for p in panels:
        q = p.entry(music, k)
        pf.append(per_frame(p.prob[p.owner == q], T))
    n = len(panels)
    W, H = n * PW + (n - 1) * GAP, TOP + HEAD + VH + FOOT
    f_title, f_sub, f_txt, f_big, f_small = font(21, True), font(14), font(15), font(21, True), font(13)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp()) / "video.mp4"
    writer = imageio_ffmpeg.write_frames(str(tmp), (W, H), fps=FPS, quality=8, macro_block_size=2,
                                         ffmpeg_log_level="error")
    writer.send(None)
    for t in range(a, b):
        canvas = np.empty((H, W, 3), np.uint8)
        canvas[:] = BG
        vals = []
        for i, (v, p) in enumerate(zip(views, pf)):
            x0 = i * (PW + GAP)
            canvas[TOP + HEAD: TOP + HEAD + VH, x0: x0 + PW] = v.draw(t)
            val = float(p[t, s0].sum())
            vals.append((val, int(p[t].argmax())))
            y = TOP + HEAD + VH + 34
            cv2.rectangle(canvas, (x0 + 16, y), (x0 + PW - 16, y + 12), (50, 53, 62), -1)
            cv2.rectangle(canvas, (x0 + 16, y), (x0 + 16 + int((PW - 32) * val), y + 12),
                          tuple(int(c) for c in LOW + (HIGH - LOW) * val), -1)
        img = Image.fromarray(canvas)
        dr = ImageDraw.Draw(img)
        sec = (t - a) / FPS
        dr.text((16, TOP // 2), f"Session 0 test music  ·  {music}", font=f_txt, fill=(200, 202, 210), anchor="lm")
        dr.text((W - 16, TOP // 2), f"sample {k}   {int(sec // 60)}:{sec % 60:04.1f}", font=f_txt,
                fill=(200, 202, 210), anchor="rm")
        for i, ((title, sub), (val, top)) in enumerate(zip(titles, vals)):
            x0 = i * (PW + GAP)
            dr.text((x0 + 16, TOP + 8), title, font=f_title, fill=(245, 245, 248))
            dr.text((x0 + 16, TOP + 38), sub, font=f_sub, fill=(150, 154, 166))
            y = TOP + HEAD + VH
            col = tuple(int(c) for c in LOW + (HIGH - LOW) * val)
            dr.text((x0 + 16, y + 20), f"{s0_name} style (session 0)", font=f_txt, fill=(170, 174, 186), anchor="lm")
            dr.text((x0 + PW - 16, y + 20), f"{val:.0%}", font=f_big, fill=col, anchor="rm")
            g = classes[top]
            dr.text((x0 + 16, y + 62), f"looks like: {GENRE.get(g, g)}", font=f_txt, anchor="lm",
                    fill=tuple(HIGH) if s0[top] else tuple(LOW))
            dr.text((x0 + PW - 16, y + 62), f"whole dance: {panels[i].mean_s0[panels[i].entry(music, k)]:.0%}",
                    font=f_small, fill=(150, 154, 166), anchor="rm")
        writer.send(np.ascontiguousarray(np.asarray(img)))
    writer.close()
    wav = pack / "wavs" / f"{music}.wav"
    if wav.exists():
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(tmp),
                        "-ss", f"{a / FPS:.3f}", "-t", f"{(b - a) / FPS:.3f}", "-i", str(wav), "-map", "0:v:0",
                        "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(out)],
                       check=True)
    else:
        print(f"[render] no music at {wav}: the video is silent")
        out.write_bytes(tmp.read_bytes())
    tmp.unlink()
    print(f"[render] {out}  ({(b - a) / FPS:.1f} s, {music} sample {k})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", type=Path, default=Path("demo_pack.zip"), help="demo_pack.zip or its extracted folder")
    ap.add_argument("--panels", nargs="+", default=["s0_model", "tmr_joint_neff", "tmr_joint_neff_kd"])
    ap.add_argument("--labels", default=None, help='panel titles separated by "|" (default: from the variant names)')
    ap.add_argument("--list", action="store_true", help="only print the music/sample ranking")
    ap.add_argument("--music", default=None)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--top", type=int, default=1, help="render the N best-ranked music/samples")
    ap.add_argument("--start", type=float, default=0.0, help="seconds into the dance")
    ap.add_argument("--seconds", type=float, default=None, help="length (default: to the end)")
    ap.add_argument("--yaw", type=float, default=25.0, help="camera angle from the dancer's front (degrees)")
    ap.add_argument("--elev", type=float, default=12.0, help="camera elevation (degrees)")
    ap.add_argument("--dist", type=float, default=3.7, help="camera distance (m)")
    ap.add_argument("--out", type=Path, default=Path("demo_videos"), help="folder, or an .mp4 path for one video")
    a = ap.parse_args()

    pack = a.pack
    if pack.suffix == ".zip":  # extracted next to the zip, once
        dest = pack.with_suffix("")
        if not ((dest / "info.json").exists() or (dest / "demo_pack" / "info.json").exists()):
            with zipfile.ZipFile(pack) as z:
                z.extractall(dest)
        pack = dest
    if not (pack / "info.json").exists() and (pack / "demo_pack" / "info.json").exists():
        pack = pack / "demo_pack"
    info = json.loads((pack / "info.json").read_text())
    s0 = np.asarray(info["session_of_class"]) == 0

    panels = []
    for spec in a.panels:
        v, _, t = spec.partition("@")
        ds = [d for d in info["dances"] if d["variant"] == v and (not t or d["after"] == int(t))]
        if not ds:
            raise SystemExit(f"{spec} is not in the pack; it has: "
                             + ", ".join(f"{d['variant']}@{d['after']}" for d in info["dances"]))
        panels.append(Panel(pack, max(ds, key=lambda d: d["after"]), s0))
    titles = [p.label() for p in panels]
    if a.labels:
        titles = [(t, s) for t, (_, s) in zip(a.labels.split("|"), titles)]

    print("S0 style over all dances (StyleRetention): "
          + "  ".join(f"{p.d['variant']}@S{p.d['after']} {float(p.mean_s0.mean()):.3f}" for p in panels))
    rows = rank(panels)
    if a.list:
        print(f"\n{'rank':>4}  {'score':>6}  " + "  ".join(f"{p.d['variant'][:18]:>18}" for p in panels)
              + "  music / sample")
        for r, (score, music, k, v, bad) in enumerate(rows, 1):
            print(f"{r:>4}  {score:>6.2f}  " + "  ".join(f"{x:>18.0%}" for x in v) + f"  {music} / {k}"
                  + (f"   UNFIT: {bad}" if bad else ""))
        return 0
    if a.music:  # its samples best-ranked first
        picks = [(m, k) for _, m, k, _, _ in rows if m == a.music and a.sample in (None, k)][: a.top]
        if not picks:
            raise SystemExit(f"music {a.music} (sample {a.sample}) is not in every panel's dances")
    else:
        picks = [(m, k) for _, m, k, _, bad in rows if not bad][: a.top]
    for music, k in picks:
        if a.out.suffix == ".mp4":
            out = a.out if len(picks) == 1 else a.out.with_suffix("") / f"{music}_sample{k}.mp4"
        else:
            out = a.out / f"{music}_sample{k}.mp4"
        render(panels, titles, info, pack, music, k, a, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
