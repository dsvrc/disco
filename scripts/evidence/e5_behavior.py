#!/usr/bin/env python3
"""
E5 -- what forgetting looks like in the generated dances. Nothing is regenerated: the joints saved by
scripts/eval_cl_session.py --out (<results_dir>/after_s{t}_sess{i}.joints.npz) are rescored.

  * style flow: for music of session i, the share of generated 5 s windows that the frozen style encoder assigns to
    each session's genres or to 'other' (not dance); the model after session t gives a matrix P_t[i, s]
  * recency attractor: mass on the newest session's genres for old music, RBI_t = mean_{i<t} P_t[i, t], against the
    same encoder's confusion on real dance (GT), which is what a perfect generator would score
  * conditional style information I_t(music session; danced style) in bits over the sessions seen so far: it is 0
    when which session's music plays no longer changes what is danced; GT and log2(t+1) are the references
  * rhythm vs style: beat-alignment score (BAS, metrics/m2d.py's definition) and mean joint speed of the generated
    dances against real dance, next to StyleRetention -- does the model keep dancing on the beat while it forgets
    the genre?

Usage (repo root):
  python scripts/evidence/e5_behavior.py --results_dir results/aistpp_genre5/vanilla --benchmark aistpp_genre5
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ev_common as E


def eval_names(cfg, benchmark: str, session: int) -> list[str]:
    """Test sequences in the order scripts/eval_cl_session.py generated them (AistPPDataset 'test' filters)."""
    lst = Path(cfg.dirs.raw_data) / "aistpp" / f"cl_{benchmark}_s{session}_test.txt"
    d = lst.parent
    hi = int(cfg.data.max_length)
    out = []
    for n in E.read_list(lst):
        f, t, m = (d / cfg.data.feature_path / f"{n}.npy", d / cfg.data.token_path / f"{n}.npy",
                   d / cfg.data.music_path / f"{n}.npy")
        if f.exists() and t.exists() and m.exists() and np.load(f, mmap_mode="r").shape[0] <= hi \
                and np.load(m, mmap_mode="r").shape[0] <= hi:
            out.append(n)
    return out


def session_matrix(classes: list[str], soc: list[int], K: int) -> np.ndarray:
    """(n_classes, K + 1) map from encoder classes to sessions; column K = 'other' (not dance)."""
    Mx = np.zeros((len(classes), K + 1))
    for c, s in enumerate(soc):
        Mx[c, s if s >= 0 else K] = 1.0
    return Mx


def style_information(P: np.ndarray) -> float:
    """Mutual information (bits) between the music's session (rows, uniform prior) and the danced style (columns)."""
    P = P / P.sum(1, keepdims=True)
    q = P.mean(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(P > 0, P * np.log2(P / q), 0.0)
    return float(t.sum(1).mean())


class Rhythm:
    """BAS exactly as metrics/m2d.py (root-relative joints, velocity minima vs librosa beats), music beats cached and
    cropped to the scored motion length."""

    def __init__(self, music_dir: Path, cache: Path):
        self.music_dir, self.cache = Path(music_dir), Path(cache)
        self.beats = json.loads(self.cache.read_text()) if self.cache.exists() else {}

    def music_beats(self, name: str) -> np.ndarray | None:
        if name not in self.beats:
            wav = self.music_dir / f"{name}.wav"
            if not wav.exists():
                return None
            from metrics.m2d_features.beat_align import get_music_beat_fromwav

            self.beats[name] = np.asarray(get_music_beat_fromwav(str(wav), None)).astype(int).tolist()
        return np.asarray(self.beats[name])

    def bas(self, joints: np.ndarray, name: str) -> float | None:
        from metrics.m2d_features.beat_align import BA, calc_db

        mb = self.music_beats(name)
        if mb is None:
            return None
        mb = mb[mb < len(joints)]
        if len(mb) == 0 or len(joints) < 12:
            return None
        j = joints.astype(np.float64).copy()
        j[:, 1:, :] -= j[:, 0:1, :]
        return float(BA(mb, calc_db(j)[0]))

    def save(self):
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        self.cache.write_text(json.dumps(self.beats))


def speed(j: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(j, axis=0), axis=-1).mean() * 30.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", type=Path, required=True, help="eval_cl_session.py --out JSONs (+ .joints.npz)")
    ap.add_argument("--config", default="configs/cl/ldf_aistpp_genre5_s0.yaml")
    ap.add_argument("--benchmark", default="aistpp_genre5")
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--music_dir", default=None, help="wav dir for BAS (default: manifest music_dir)")
    ap.add_argument("--no_bas", action="store_true")
    a = ap.parse_args()
    a.out_dir = a.out_dir or Path("results/evidence") / a.benchmark
    cfg = E.load_cfg(a.config)
    man = E.manifest(cfg, a.benchmark)
    K = len(man["sessions"])

    from metrics.style_encoder import StyleEncoder, benchmark_session_of_class, ckpt_path
    from utils.motion_process import recover_joint_positions_272

    enc = StyleEncoder(ckpt_path(man["dataset"], -1))
    soc = benchmark_session_of_class(enc.classes, man)
    Mx = session_matrix(enc.classes, soc, K)
    rhythm = None if a.no_bas else Rhythm(a.music_dir or man.get("music_dir", ""), a.out_dir / "cache" / "music_beats.json")
    if rhythm is not None and not rhythm.music_dir.exists():
        E.log(f"no wav dir {rhythm.music_dir}: BAS skipped")
        rhythm = None

    names = {i: eval_names(cfg, a.benchmark, i) for i in range(K)}
    gt = {}
    for i in range(K):
        js = [recover_joint_positions_272(np.load(Path(man["feature_dir"]) / f"{n}.npy"), 22) for n in names[i]]
        if not js:
            continue
        p = enc.encode(js)["prob"] @ Mx
        r = {"dist": p.mean(0), "n": len(js), "speed": float(np.mean([speed(j) for j in js]))}
        if rhythm is not None:
            b = [x for x in (rhythm.bas(j, n) for j, n in zip(js, names[i])) if x is not None]
            r["BAS"] = float(np.mean(b)) if b else None
        gt[i] = r
    E.log("GT style distributions: " + "; ".join(f"S{i}: own {g['dist'][i]:.2f}" for i, g in gt.items()))

    cells = {}
    for jf in sorted(a.results_dir.glob("after_s*_sess*.json")):
        m = re.fullmatch(r"after_s(\d+)_sess(\d+)\.json", jf.name)
        npz = jf.with_suffix(".joints.npz")
        if not m or not npz.exists():
            continue
        t, i = int(m.group(1)), int(m.group(2))
        rep = json.loads(jf.read_text())
        z = np.load(npz)
        js = [z[str(n)] for n in range(len(z.files))]
        p = enc.encode(js)["prob"] @ Mx
        cell = {"dist": p.mean(0), "n_sequences": len(js), "reported": {k: rep.get(k) for k in
                                                                       ("StyleRetention", "DanceRate", "StyleFD", "FID_k", "FID_g")},
                "speed": float(np.mean([speed(j) for j in js]))}
        nm = rep.get("names") or names[i]
        S_ = int(rep.get("samples") or 1)
        if rhythm is not None and len(js) == len(nm) * S_:
            b = [x for x in (rhythm.bas(j, nm[q % len(nm)]) for q, j in enumerate(js)) if x is not None]
            cell["BAS"] = float(np.mean(b)) if b else None
        elif rhythm is not None:
            E.log(f"after_s{t}_sess{i}: {len(js)} sequences != {len(nm)} names x {S_} samples, BAS skipped")
        cells[(t, i)] = cell
        E.log(f"after S{t}, music S{i}: own-style {cell['dist'][i]:.3f} (reported {cell['reported']['StyleRetention']}), "
              f"newest {cell['dist'][t]:.3f}, other {cell['dist'][K]:.3f}, BAS {cell.get('BAS')}")
    if rhythm is not None:
        rhythm.save()
    if not cells:
        raise SystemExit(f"no after_s*_sess*.json with .joints.npz in {a.results_dir}")

    per_t = {}
    for t in sorted({t for t, _ in cells}):
        rows = [i for i in range(t + 1) if (t, i) in cells]
        P = np.stack([cells[(t, i)]["dist"] for i in rows])
        r = {"music_sessions": rows, "P": P,
             "style_information_bits": style_information(P) if len(rows) > 1 else None,
             "max_information_bits": math.log2(len(rows)) if len(rows) > 1 else None}
        if all(i in gt for i in rows) and len(rows) > 1:
            Pg = np.stack([gt[i]["dist"] for i in rows])
            r["gt_style_information_bits"] = style_information(Pg)
        olds = [i for i in rows if i < t]
        if olds:
            r["recency_bias_index"] = float(np.mean([cells[(t, i)]["dist"][t] for i in olds]))
            r["gt_recency_bias_index"] = float(np.mean([gt[i]["dist"][t] for i in olds if i in gt])) if gt else None
            r["old_correct_mass"] = float(np.mean([cells[(t, i)]["dist"][i] for i in olds]))
            r["old_other_mass"] = float(np.mean([cells[(t, i)]["dist"][K] for i in olds]))
            r["old_intermediate_mass"] = float(np.mean([sum(cells[(t, i)]["dist"][s] for s in range(i + 1, t))
                                                        for i in olds]))
        per_t[str(t)] = r

    rhythm_vs_style = []
    for (t, i), c in sorted(cells.items()):
        if i in gt:
            rhythm_vs_style.append({"t": t, "i": i, "old": i < t, "style_ret": c["dist"][i], "gt_style": gt[i]["dist"][i],
                                    "BAS": c.get("BAS"), "gt_BAS": gt[i].get("BAS"), "speed": c["speed"],
                                    "gt_speed": gt[i]["speed"]})
    learned = [r for r in rhythm_vs_style if r["t"] == r["i"]]
    old = [r for r in rhythm_vs_style if r["old"]]
    # the rescored own-genre mass must equal what eval_cl_session.py reported (same encoder, same joints)
    diffs = [abs(c["dist"][i] - c["reported"]["StyleRetention"]) for (t, i), c in cells.items()
             if c["reported"].get("StyleRetention") is not None]
    summary = {"rescoring_max_abs_diff_vs_reported": max(diffs) if diffs else None}
    if diffs and max(diffs) > 0.02:
        E.log(f"WARNING: rescored StyleRetention differs from the reported one by up to {max(diffs):.3f}: "
              "is the style encoder checkpoint the one used by eval_cl_session.py?")
    if old and learned:
        rel = lambda rows, key, ref: float(np.mean([r[key] / r[ref] for r in rows if r.get(key) is not None and r.get(ref)]))
        summary["style_retention_old_vs_learned"] = (np.mean([r["style_ret"] for r in old]), np.mean([r["style_ret"] for r in learned]))
        if all(r.get("BAS") is not None for r in old + learned):
            summary["BAS_old_vs_learned"] = (np.mean([r["BAS"] for r in old]), np.mean([r["BAS"] for r in learned]))
            summary["BAS_old_rel_gt"] = rel(old, "BAS", "gt_BAS")
            summary["style_old_rel_gt"] = rel(old, "style_ret", "gt_style")

    res = {"benchmark": a.benchmark, "K": K, "classes": enc.classes, "session_of_class": soc,
           "gt": {str(i): g for i, g in gt.items()},
           "cells": {f"after_S{t}/music_S{i}": c for (t, i), c in cells.items()},
           "per_t": per_t, "rhythm_vs_style": rhythm_vs_style, "summary": summary}
    E.dump_json(a.out_dir / "e5" / "behavior.json", res)
    figures(res, a.out_dir / "figures", K)
    return 0


def figures(res: dict, out: Path, K: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = sorted(res["per_t"], key=int)
    fig, axes = plt.subplots(1, len(ts) + 1, figsize=(3.4 * (len(ts) + 1), 3.3), squeeze=False)
    labels = [f"S{s}" for s in range(K)] + ["other"]
    colors = [f"C{s}" for s in range(K)] + ["0.6"]
    for ax, t in zip(axes[0], ts):
        r = res["per_t"][t]
        P = np.array(r["P"])
        left = np.zeros(len(P))
        for s in range(K + 1):
            ax.barh([f"music S{i}" for i in r["music_sessions"]], P[:, s], left=left, color=colors[s], label=labels[s])
            left += P[:, s]
        ax.set(xlim=(0, 1), title=f"after S{t}", xlabel="share of windows danced as")
    axes[0][0].legend(fontsize=6, loc="lower left")
    ax = axes[0][-1]
    tt = [int(t) for t in ts if res["per_t"][t].get("style_information_bits") is not None]
    if tt:
        ax.plot(tt, [res["per_t"][str(t)]["style_information_bits"] for t in tt], "o-", label="generated")
        ax.plot(tt, [res["per_t"][str(t)].get("gt_style_information_bits") for t in tt], "s--", label="real dance")
        ax.plot(tt, [res["per_t"][str(t)]["max_information_bits"] for t in tt], ":", color="k", label="$\\log_2(t+1)$")
        ax.plot(tt, [res["per_t"][str(t)].get("recency_bias_index") for t in tt], "^-", label="recency bias index")
        ax.set(xlabel="after session t", title="I(music session; danced style)")
        ax.legend(fontsize=6)
    E.savefig(fig, out / "e5_style_flow")

    rows = [r for r in res["rhythm_vs_style"] if r.get("BAS") is not None]
    if rows:
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
        for r in rows:
            c = "C3" if r["old"] else "C0"
            ax[0].scatter(r["style_ret"] / max(r["gt_style"], 1e-9), r["BAS"] / max(r["gt_BAS"] or 1e-9, 1e-9), c=c, s=20)
        ax[0].axhline(1, color="k", lw=.5)
        ax[0].axvline(1, color="k", lw=.5)
        ax[0].set(xlabel="style retention / real dance", ylabel="BAS / real dance",
                  title="rhythm vs style (red: old sessions)")
        ax[1].scatter([r["style_ret"] for r in rows], [r["speed"] for r in rows],
                      c=["C3" if r["old"] else "C0" for r in rows], s=20)
        ax[1].set(xlabel="StyleRetention", ylabel="mean joint speed (m/s)", title="energy vs style")
        E.savefig(fig, out / "e5_rhythm_vs_style")


if __name__ == "__main__":
    raise SystemExit(main())
