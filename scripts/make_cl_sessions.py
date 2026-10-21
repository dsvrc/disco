#!/usr/bin/env python3
"""
Build continual-learning (CL) session partitions for DiscoForcing.

Benchmarks (written to <out_dir>/<benchmark>/):
  aistpp_genre5         AIST++, 5 sessions x 2 genres, escalating-drift order (main table)
  aistpp_genre10        AIST++, 10 sessions x 1 genre, escalating order, S0 drawn from aistpp_genre5 S0 (long horizon)
  finedance_style4      FineDance, 4 sessions = style families (Street / Classic / Folk / Mix), escalating order

Design:
  * Genres are paired into sessions by minimising the worst, then the total, within-session distance.
    Distance = mean of (normalised) motion and music Frechet distances between genres, train data only.
  * Sessions are ordered as an escalating drift away from every earlier session: the distance from each
    old session to the newly trained one grows monotonically, so vanilla forgetting should compound
    instead of oscillating when a similar genre returns. Ties go to the larger predicted vanilla F_avg
    (the paper's weighting with forgetting(i, t) ~ d(S_i, S_t)), then to the larger session 0.
  * AIST++ test sets are music-disjoint: official test + the unused sequences on each genre's held-out
    music piece (official val/test music); val = official val. Train = crossmodal_train_without_val.
  * FineDance test = official cross_genre test of each family; val = ~10% of train held out per family.
  * Session-0 Mean/Std (repo's grouped 272-dim recipe) are written per benchmark so the frozen motion
    VAE / normalisation never sees future sessions.

Each session directory gets {train,val,test}.txt with sequence names (one per line, no extension), plus
joint_{train,val,test}.txt (upper bound) and manifest.json at the benchmark root.

Usage:
  python scripts/make_cl_sessions.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import linalg

AIST_GENRE_NAMES = {
    "BR": "Break", "PO": "Pop", "LO": "Lock", "WA": "Waack", "MH": "Middle Hip-hop",
    "LH": "LA Hip-hop", "HO": "House", "KR": "Krump", "JS": "Street Jazz", "JB": "Ballet Jazz",
}
AIST_PATTERN = re.compile(r"g(\w\w)_s(\w\w)_c\w+_d(\d\d)_m(\w\w)(\d)_ch(\d\d)")
# FineDance label_json style1 -> session family ("jiewu" = street dance in pinyin)
FINEDANCE_FAMILY = {"Street": "Street", "jiewu": "Street", "Classic": "Classic", "Folk": "Folk", "Mix": "Mix"}


def read_list(path: Path) -> list[str]:
    return [l.strip() for l in path.open() if l.strip()]


def write_list(path: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="\n") as f:
        f.write("\n".join(names) + ("\n" if names else ""))


# ----------------------------------------------------------------------------------------------
# Distances
# ----------------------------------------------------------------------------------------------
def frechet(a: np.ndarray, b: np.ndarray) -> float:
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def pairwise_frechet(feats: dict[str, np.ndarray]) -> dict[tuple[str, str], float]:
    keys = sorted(feats)
    return {(a, b): frechet(feats[a], feats[b]) for a, b in itertools.combinations(keys, 2)}


def normalise(d: dict[tuple[str, str], float]) -> dict[tuple[str, str], float]:
    m = np.mean(list(d.values()))
    return {k: v / m for k, v in d.items()}


def dist(d: dict[tuple[str, str], float], a: str, b: str) -> float:
    return 0.0 if a == b else d[(a, b)] if (a, b) in d else d[(b, a)]


def motion_frames(paths: list[Path], max_frames: int, rng: np.random.Generator) -> np.ndarray:
    x = np.concatenate([np.load(p) for p in paths], axis=0)
    if len(x) > max_frames:
        x = x[rng.choice(len(x), max_frames, replace=False)]
    return x


def music_frames(wavs: list[Path], sr: int = 22050) -> np.ndarray:
    import librosa

    out = []
    for w in wavs:
        y, _ = librosa.load(w, sr=sr, mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        onset = librosa.onset.onset_strength(y=y, sr=sr)[None]
        n = min(mfcc.shape[1], chroma.shape[1], contrast.shape[1], onset.shape[1])
        out.append(np.concatenate([mfcc[:, :n], chroma[:, :n], contrast[:, :n], onset[:, :n]], 0).T)
    return np.concatenate(out, 0)


# ----------------------------------------------------------------------------------------------
# Session construction
# ----------------------------------------------------------------------------------------------
def best_pairing(items: list[str], d) -> list[tuple[str, str]]:
    """Perfect matching minimising the worst within-pair distance, then the total (exhaustive; 945 for 10).

    Min-max first keeps every session coherent: a pure min-sum matching glues the two outlier genres
    (Break, Ballet Jazz) into one session.
    """

    def matchings(xs):
        if not xs:
            yield []
            return
        a = xs[0]
        for i in range(1, len(xs)):
            for rest in matchings(xs[1:i] + xs[i + 1:]):
                yield [(a, xs[i])] + rest

    return min(matchings(sorted(items)),
               key=lambda m: (max(dist(d, a, b) for a, b in m), sum(dist(d, a, b) for a, b in m)))


def group_dist(d, g1, g2) -> float:
    return float(np.mean([dist(d, a, b) for a in g1 for b in g2]))


def order_stats(p, d) -> dict:
    """Forgetting-pattern statistics of a session order p under forgetting(i, t) ~ d(S_i, S_t)."""
    K = len(p)
    pairs = [group_dist(d, p[i], p[t]) <= group_dist(d, p[i], p[t2])
             for i in range(K - 2) for t in range(i + 1, K) for t2 in range(t + 1, K)]
    return {
        "monotone": float(np.mean(pairs)),  # fraction of old-session curves that never decrease
        "pred_F_avg": float(np.mean([np.mean([group_dist(d, p[i], p[t]) for t in range(i + 1, K)])
                                     for i in range(K - 1)])),  # paper's F_avg weighting
        "s0_curve": [round(group_dist(d, p[0], p[t]), 4) for t in range(1, K)],
    }


def escalating_order(groups: list[tuple[str, ...]], d, size_of) -> list[tuple[str, ...]]:
    """Order whose drift away from every earlier session grows monotonically (exhaustive search)."""
    def key(p):
        s = order_stats(p, d)
        return round(s["monotone"], 6), round(s["pred_F_avg"], 6), size_of(p[0])
    return list(max(itertools.permutations(groups), key=key))


def escalating_genre_order(genres: list[str], d, first_candidates) -> list[str]:
    """Long-horizon one-genre-per-session order (vectorised exhaustive search, 9! orders per first genre).

    Requires session 0's distance curve to be monotone, then maximises the monotone fraction over all old
    sessions and the predicted F_avg. Session 0 is restricted to `first_candidates` (the 5-session S0).
    """
    ix = {g: i for i, g in enumerate(genres)}
    D = np.array([[dist(d, a, b) for b in genres] for a in genres])
    K, best, best_key = len(genres), None, None
    for first in first_candidates:
        rest = [ix[g] for g in genres if g != first]
        P = np.array(list(itertools.permutations(rest)), dtype=np.int8)
        P = np.concatenate([np.full((len(P), 1), ix[first], np.int8), P], 1)
        mono, n = np.zeros(len(P)), 0
        for i in range(K - 2):
            for t in range(i + 1, K):
                dt = D[P[:, i], P[:, t]]
                for t2 in range(t + 1, K):
                    mono += dt <= D[P[:, i], P[:, t2]]
                    n += 1
        mono /= n
        pred_f = np.mean([np.mean([D[P[:, i], P[:, t]] for t in range(i + 1, K)], 0) for i in range(K - 1)], 0)
        s0_mono = np.all(np.diff(D[P[:, :1], P[:, 1:]], axis=1) >= 0, axis=1)
        idx = np.where(s0_mono)[0]
        j = idx[np.lexsort((pred_f[idx], mono[idx]))[-1]]
        key = (round(mono[j], 6), round(pred_f[j], 6))
        if best_key is None or key > best_key:
            best, best_key = [genres[k] for k in P[j]], key
    return best


def grouped_std_272(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Same recipe as utils.dataset_process.mean_variance_multi(rep='272')."""
    j = 22
    mean, std = data.mean(0), data.std(0)
    for a, b in [(0, 2), (2, 8), (8, 8 + 3 * j), (8 + 3 * j, 8 + 6 * j), (8 + 6 * j, 8 + 12 * j)]:
        std[a:b] = std[a:b].mean()
    return mean, std


def write_loader_lists(dataset_dir: Path, name: str, sessions: list[dict]) -> None:
    """The repo's dataset classes resolve feature/token/music folders relative to the list file and name the
    dataset after its folder, so training lists must live in the dataset folder (e.g. raw_data/aistpp)."""
    for k, s in enumerate(sessions):
        for split in ("train", "val", "test"):
            write_list(dataset_dir / f"cl_{name}_s{k}_{split}.txt", s[split])


def write_benchmark(root: Path, name: str, sessions: list[dict], feature_dir_of, meta: dict) -> None:
    bdir = root / name
    joint = defaultdict(list)
    write_loader_lists(Path(meta["feature_dir"]).parent, name, sessions)
    for k, s in enumerate(sessions):
        for split in ("train", "val", "test"):
            write_list(bdir / f"s{k}" / f"{split}.txt", s[split])
            joint[split] += s[split]
    for split, names in joint.items():
        write_list(bdir / f"joint_{split}.txt", names)

    # session-0-only normalisation statistics (frozen VAE / feature normalisation)
    s0 = np.concatenate([np.load(feature_dir_of(n) / f"{n}.npy") for n in sessions[0]["train"]], 0)
    mean, std = grouped_std_272(s0)
    np.save(bdir / "s0_Mean_272.npy", mean)
    np.save(bdir / "s0_Std_272.npy", std)

    manifest = dict(meta, benchmark=name, sessions=[
        {k: (v if k not in ("train", "val", "test") else len(v)) for k, v in s.items()} for s in sessions
    ], s0_stats_frames=int(len(s0)))
    (bdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[{name}] " + " -> ".join(f"S{k}:{'+'.join(s['groups'])}({len(s['train'])}/{len(s['val'])}/{len(s['test'])})"
                                      for k, s in enumerate(sessions)))


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", type=Path, default=Path("raw_data"))
    parser.add_argument("--out_dir", type=Path, default=Path("raw_data/cl_sessions"))
    parser.add_argument("--motion_frames_per_group", type=int, default=30000,
                        help="Frames subsampled per genre/family for the motion Frechet distance")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="FineDance: train fraction held out as val")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    R = args.raw_data

    # ---------------- AIST++ ----------------
    a = R / "aistpp"
    a_feat = a / "motions_272_30fps"
    train, val, test = (read_list(a / f) for f in
                        ("crossmodal_train_without_val.txt", "crossmodal_val.txt", "crossmodal_test.txt"))
    ignore = set(read_list(a / "ignore_list.txt"))
    every = sorted(p.stem for p in a_feat.glob("*.npy"))
    info = {n: AIST_PATTERN.match(n).groups() for n in every}  # genre, situation, dancer, mgenre, music_id, ch
    held_music = {info[n][0]: info[n][4] for n in val + test}
    used = set(train) | set(val) | set(test)
    by_genre = defaultdict(lambda: {"train": [], "val": [], "test": []})
    for n in train:
        by_genre[info[n][0]]["train"].append(n)
    for n in val:
        by_genre[info[n][0]]["val"].append(n)
    for n in every:
        g, music = info[n][0], info[n][4]
        if (n in test) or (music == held_music[g] and n not in used and n not in ignore):
            by_genre[g]["test"].append(n)
    genres = sorted(by_genre)

    print("AIST++: computing motion / music distances between genres ...")
    m_feats = {g: motion_frames([a_feat / f"{n}.npy" for n in by_genre[g]["train"]],
                                args.motion_frames_per_group, rng) for g in genres}
    all_m = np.concatenate(list(m_feats.values()))
    mu, sd = all_m.mean(0), all_m.std(0) + 1e-6
    d_motion = normalise(pairwise_frechet({g: (x - mu) / sd for g, x in m_feats.items()}))
    # one wav per unique train music piece of the genre
    pieces = {g: sorted({info[n][3] + info[n][4] for n in by_genre[g]["train"]}) for g in genres}
    wav_of = {}
    for n in train:
        key = info[n][3] + info[n][4]
        if key not in wav_of or info[n][1] == "FM":  # prefer the longer advanced-choreography clips
            wav_of[key] = a / "wavs" / f"{n}.wav"
    a_music = {g: music_frames([wav_of[p] for p in pieces[g]]) for g in genres}
    all_a = np.concatenate(list(a_music.values()))
    amu, asd = all_a.mean(0), all_a.std(0) + 1e-6
    d_music = normalise(pairwise_frechet({g: (x - amu) / asd for g, x in a_music.items()}))
    d_aist = {k: 0.5 * (d_motion[k] + d_music[k]) for k in d_motion}

    pairs = [tuple(p) for p in best_pairing(genres, d_aist)]
    n_train = lambda grp: sum(len(by_genre[g]["train"]) for g in grp)
    order5 = escalating_order(pairs, d_aist, n_train)
    order10 = escalating_genre_order(genres, d_aist, first_candidates=order5[0])

    def aist_sessions(groups):
        out = []
        for grp in groups:
            s = {"groups": list(grp), "group_names": [AIST_GENRE_NAMES[g] for g in grp],
                 "train": [], "val": [], "test": []}
            for g in grp:
                for split in ("train", "val", "test"):
                    s[split] += sorted(by_genre[g][split])
            s["train_minutes"] = round(sum(np.load(a_feat / f"{n}.npy", mmap_mode="r").shape[0]
                                           for n in s["train"]) / 30 / 60, 1)
            out.append(s)
        return out

    a_meta = {
        "dataset": "aistpp", "feature_dir": str(a_feat), "music_dir": str(a / "wavs"),
        "test_protocol": "music-disjoint: official crossmodal test + unused sequences on each genre's held-out "
                         "music piece (ignore_list removed); val = official crossmodal val",
        "held_out_music": held_music,
        "genre_distance": {f"{x}-{y}": round(v, 4) for (x, y), v in sorted(d_aist.items())},
        "motion_distance": {f"{x}-{y}": round(v, 4) for (x, y), v in sorted(d_motion.items())},
        "music_distance": {f"{x}-{y}": round(v, 4) for (x, y), v in sorted(d_music.items())},
    }
    to_dir = lambda n: a_feat
    a_meta["pairing"] = "min-max within-session distance, then min total"
    escalating = "escalating drift: monotone distance from every old session to the new one, then max predicted F_avg"
    write_benchmark(args.out_dir, "aistpp_genre5", aist_sessions(order5), to_dir,
                    dict(a_meta, order=escalating, order_stats=order_stats(order5, d_aist)))
    order10_groups = [(g,) for g in order10]
    write_benchmark(args.out_dir, "aistpp_genre10", aist_sessions(order10_groups), to_dir,
                    dict(a_meta, order="escalating drift over single genres (S0 monotone required), S0 from aistpp_genre5 S0",
                         order_stats=order_stats(order10_groups, d_aist)))

    # ---------------- FineDance ----------------
    f = R / "finedance"
    f_feat = f / "motion_272_30fps"
    labels = {p.stem: json.load(p.open(encoding="utf-8")) for p in (f / "label_json").glob("*.json")}
    f_train, f_test = read_list(f / "train.txt"), read_list(f / "test.txt")
    fam = {n: FINEDANCE_FAMILY[labels[n]["style1"]] for n in f_train + f_test}
    by_fam = defaultdict(lambda: {"train": [], "val": [], "test": []})
    vrng = random.Random(args.seed)
    for family in sorted(set(fam.values())):
        members = sorted(n for n in f_train if fam[n] == family)
        sub_count = defaultdict(int)
        for n in members:
            sub_count[labels[n]["style2"]] += 1
        # hold out val only from sub-genres with >= 3 train sequences, so no sub-genre disappears from train
        pool = [n for n in members if sub_count[labels[n]["style2"]] >= 3]
        n_val = max(1, round(args.val_ratio * len(members)))
        held = set(vrng.sample(pool, n_val))
        by_fam[family]["train"] = [n for n in members if n not in held]
        by_fam[family]["val"] = sorted(held)
        by_fam[family]["test"] = sorted(n for n in f_test if fam[n] == family)
    families = sorted(by_fam)

    print("FineDance: computing motion / music distances between style families ...")
    fm = {x: motion_frames([f_feat / f"{n}.npy" for n in by_fam[x]["train"]], args.motion_frames_per_group, rng)
          for x in families}
    all_fm = np.concatenate(list(fm.values()))
    fmu, fsd = all_fm.mean(0), all_fm.std(0) + 1e-6
    fd_motion = normalise(pairwise_frechet({x: (v - fmu) / fsd for x, v in fm.items()}))
    fa = {x: music_frames([f / "music_wav" / f"{n}.wav" for n in by_fam[x]["train"]]) for x in families}
    all_fa = np.concatenate(list(fa.values()))
    famu, fasd = all_fa.mean(0), all_fa.std(0) + 1e-6
    fd_music = normalise(pairwise_frechet({x: (v - famu) / fasd for x, v in fa.items()}))
    d_fd = {k: 0.5 * (fd_motion[k] + fd_music[k]) for k in fd_motion}
    f_order = escalating_order([(x,) for x in families], d_fd, lambda grp: len(by_fam[grp[0]]["train"]))

    f_sessions = []
    for (x,) in f_order:
        s = {"groups": [x], "sub_genres": sorted({labels[n]["style2"] for n in by_fam[x]["train"]}),
             **{k: v for k, v in by_fam[x].items()}}
        s["train_minutes"] = round(sum(np.load(f_feat / f"{n}.npy", mmap_mode="r").shape[0]
                                       for n in s["train"]) / 30 / 60, 1)
        f_sessions.append(s)
    write_benchmark(args.out_dir, "finedance_style4", f_sessions, lambda n: f_feat, {
        "dataset": "finedance", "feature_dir": str(f_feat), "music_dir": str(f / "music_wav"),
        "test_protocol": "official FineDance cross_genre test per family (long sequences; slice into clips for "
                         f"metrics); val = {args.val_ratio:.0%} of train per family, seed {args.seed}",
        "order": "escalating drift: monotone distance from every old session to the new one, then max predicted F_avg",
        "order_stats": order_stats(f_order, d_fd),
        "family_distance": {f"{x}-{y}": round(v, 4) for (x, y), v in sorted(d_fd.items())},
        "motion_distance": {f"{x}-{y}": round(v, 4) for (x, y), v in sorted(fd_motion.items())},
        "music_distance": {f"{x}-{y}": round(v, 4) for (x, y), v in sorted(fd_music.items())},
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
