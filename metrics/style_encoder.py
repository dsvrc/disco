"""
Learned dance-style encoder for continual-learning evaluation of music-to-dance models (TMR benchmark).

Why: the handcrafted kinetic/geometric genre probe confuses motion quality with style (temporal smoothing with
sigma = 2 frames drops its accuracy 0.86 -> 0.22). This encoder classifies dance style from 5 s windows of joint
positions and is trained to be invariant to generation artefacts (smoothing, jitter, tempo, body scale, root
drift), so a quality defect cannot masquerade as forgetting. Mirroring, frame stutter, low-frequency wobble and
strengths beyond the training ranges are never used in training and serve as unseen-corruption tests.

Input : global joint positions (T, 22, 3), SMPL order, y-up, 30 fps -- what any generator's output recovers to.
        Each window is expressed relative to the pelvis and rotated to the dancer's mean facing direction.
Output: per 5 s window (1 s stride) an L2-normalised 64-d style embedding and temperature-calibrated
        genre probabilities.

CL metrics built on it (see metrics/cl_metrics.py):
  StyleRetention  mean probability mass on the session's genres over generated windows (continuous)
  StyleAcc        hard version: the arg-max genre belongs to the session
  StyleFD         Frechet distance between embeddings and the session's reference windows
  drift           StyleFD between two generated sets (e.g. the model right after session i vs after t)

CLI (run from the repo root):
  python -m metrics.style_encoder train    --dataset aistpp [--fold 0|1|2] [--steps 6000]
  python -m metrics.style_encoder validate --dataset aistpp
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.getcwd())
from utils.motion_process import recover_joint_positions_272

FPS, WIN, STRIDE = 30, 150, 30
L_HIP, R_HIP, L_SHO, R_SHO = 1, 2, 16, 17
LR_PAIRS = [(1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21)]
MIRROR_PERM = list(range(22))
for _a, _b in LR_PAIRS:
    MIRROR_PERM[_a], MIRROR_PERM[_b] = _b, _a

DATASETS = {
    "aistpp": {"feature_dir": "raw_data/aistpp/motions_272_30fps", "benchmark": "aistpp_genre5"},
    "finedance": {"feature_dir": "raw_data/finedance/motion_272_30fps", "benchmark": "finedance_style4"},
}
OUT_ROOT = Path("raw_data/cl_sessions/_style_encoder")

# training-time augmentation ranges (the validation suite marks these on its invariance curves)
AUG = {"smooth": (0.5, 4.0), "jitter_cm": (0.2, 2.0), "tempo": (0.8, 1.25), "scale": (0.9, 1.1), "drift_mps": 0.1}


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------
def _read(p: Path) -> list[str]:
    return [l.strip() for l in p.open() if l.strip()]


def dataset_info(dataset: str) -> dict:
    """Split membership, labels and sessions from the CL benchmark folder."""
    root = Path("raw_data/cl_sessions") / DATASETS[dataset]["benchmark"]
    man = json.loads((root / "manifest.json").read_text())
    K = len(man["sessions"])
    split, sess = {"train": [], "val": [], "test": []}, {}
    for s in range(K):
        for sp in split:
            for n in _read(root / f"s{s}" / f"{sp}.txt"):
                split[sp].append(n)
                sess[n] = s
    if dataset == "aistpp":
        label = {n: n[1:3] for n in sess}  # gLH_sBM_cAll_d17_mLH4_ch02 -> LH
        classes = sorted(set(label.values()))
        dancer = {n: n.split("_")[3] for n in sess}
    else:
        label = {n: man["sessions"][sess[n]]["groups"][0] for n in sess}
        classes = [s["groups"][0] for s in man["sessions"]]
        dancer = None
    session_of_class = [sess[next(n for n in sess if label[n] == c)] for c in classes]
    return {"root": root, "K": K, "split": split, "sess": sess, "label": label, "classes": classes,
            "session_of_class": session_of_class, "dancer": dancer}


def load_joints(dataset: str) -> dict[str, np.ndarray]:
    """Global joints of every sequence in the benchmark splits, cached as one npz."""
    info = dataset_info(dataset)
    names = sorted(info["sess"])
    path = OUT_ROOT / f"joints_{dataset}.npz"
    if path.exists():
        z = np.load(path)
        if set(names) <= set(z.files):
            return {n: z[n] for n in names}
    fdir = Path(DATASETS[dataset]["feature_dir"])
    joints = {n: recover_joint_positions_272(np.load(fdir / f"{n}.npy"), 22).astype(np.float32) for n in names}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **joints)
    return joints


# ----------------------------------------------------------------------------------------------
# Reject class: motion that is none of the benchmark's dance styles
# ----------------------------------------------------------------------------------------------
# Without it the softmax must put every window on some genre, and non-dance motion (an untrained generator,
# static poses) lands on the session-0 genres: an untrained generator scored StyleRetention 0.58 on S0.
OTHER = "other"
# training mixture of negatives; static poses and time-shuffled dance are built on the fly from dance windows
NEG_MIX = {"babel": 0.4, "vae_random": 0.3, "static": 0.15, "shuffled": 0.15}
NEG_VAE = {  # the benchmark's frozen tokenizer, used to decode random latents (generator-like garbage)
    "aistpp": {"config": "configs/cl/vae_aistpp_genre5_alltrain.yaml",
               "ckpt": "outputs/frozen/vae_aistpp_genre5_alltrain_step5000.ckpt",
               "tokens": "raw_data/aistpp/TOKENS_cl_aistpp_genre5_alltrainvae"},
}


def babel_negatives(split: str, max_seqs: int, seed: int = 0) -> list[np.ndarray]:
    """Everyday non-dance motion from BABEL-272 `{split}_stream` as global joints; sequences whose annotations
    mention dance, and sequences shorter than a window, are excluded."""
    path = OUT_ROOT / f"neg_babel_{split}_{max_seqs}_s{seed}.npz"
    if path.exists():
        z = np.load(path)
        return [z[k] for k in z.files]
    root = Path("raw_data/babel_272")
    names = []
    for p in sorted((root / f"{split}_stream").glob("*.npy")):
        t = root / f"{split}_stream_text" / f"{p.stem}.txt"
        if t.exists() and "danc" not in t.read_text(errors="ignore").lower() \
                and np.load(p, mmap_mode="r").shape[0] >= WIN + 10:
            names.append(p.stem)
    pick = np.random.default_rng(seed).choice(names, min(max_seqs, len(names)), replace=False)
    joints = {n: recover_joint_positions_272(np.load(root / f"{split}_stream" / f"{n}.npy").astype(np.float32),
                                             22).astype(np.float32) for n in pick}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **joints)
    return list(joints.values())


def _neg_vae(dataset: str):
    """The benchmark's frozen VAE (EMA weights) and the per-dimension mean / std of its tokens."""
    from torch_ema import ExponentialMovingAverage

    from utils.initialize import Config, instantiate

    spec = NEG_VAE[dataset]
    cfg = Config(spec["config"])
    vae = instantiate(cfg.model.target, cfg=None, hfstyle=False, **dict(cfg.model.params))
    ck = torch.load(spec["ckpt"], map_location="cpu", weights_only=False)
    vae.load_state_dict(ck["state_dict"])
    if "ema_state" in ck:
        ema = ExponentialMovingAverage(vae.parameters(), decay=cfg.model.ema_decay)
        ema.load_state_dict(ck["ema_state"])
        ema.copy_to(vae.parameters())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    toks = np.concatenate([np.load(p) for p in sorted(Path(spec["tokens"]).glob("*.npy"))])
    return vae.to(dev).eval(), dev, toks.mean(0), toks.std(0)


@torch.no_grad()
def vae_latent_motion(dataset: str, kind: str, n: int, seed: int = 0) -> list[np.ndarray]:
    """Joints decoded by the benchmark's frozen VAE from synthetic latent sequences (76 tokens = 10 s):
      random  - token statistics x 0.5-3 noise, half temporally smoothed (training negatives: generator-like garbage)
      constant- one random latent held for all tokens       } never used for training:
      walk    - a slow random walk through latent space     } unseen garbage for validation"""
    path = OUT_ROOT / f"neg_vae_{kind}_{dataset}_{n}_s{seed}.npz"
    if kind == "random" and not path.exists() and (OUT_ROOT / f"neg_vae_random_{dataset}_{n}_s{seed}.npz").exists():
        path = OUT_ROOT / f"neg_vae_random_{dataset}_{n}_s{seed}.npz"
    if path.exists():
        z = np.load(path)
        return [z[k] for k in z.files]
    from utils.motion_process import convert_motion_to_joints

    vae, dev, mu, sd = _neg_vae(dataset)
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(n):
        if kind == "random":
            z = rng.standard_normal((76, len(mu)))
            if rng.random() < 0.5:
                zs = gaussian_filter1d(z, rng.uniform(1, 4), axis=0)
                z = zs / (zs.std(0, keepdims=True) + 1e-6)
            z = z * rng.choice([0.5, 1.0, 2.0, 3.0])
        elif kind == "constant":
            z = np.repeat(rng.standard_normal((1, len(mu))) * rng.uniform(0.5, 2.0), 76, 0)
        elif kind == "walk":
            z = np.cumsum(rng.standard_normal((76, len(mu))) * rng.uniform(0.1, 0.4), 0)
        else:
            raise ValueError(kind)
        feat = vae.decode(torch.from_numpy((mu + sd * z).astype(np.float32))[None].to(dev))[0].float().cpu().numpy()
        out[str(i)] = convert_motion_to_joints(feat, 272).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    return list(out.values())


def vae_random_negatives(dataset: str, n: int, seed: int = 0) -> list[np.ndarray]:
    return vae_latent_motion(dataset, "random", n, seed)


def resample_time(j: np.ndarray, speed: float, length: int | None = None) -> np.ndarray:
    """Linear time resampling; speed > 1 plays the motion faster."""
    n_new = length or max(2, int(round((len(j) - 1) / speed)) + 1)
    t = np.linspace(0, len(j) - 1, n_new)
    lo = np.floor(t).astype(int)
    hi = np.minimum(lo + 1, len(j) - 1)
    w = (t - lo)[:, None, None]
    return j[lo] * (1 - w) + j[hi] * w


def sample_window(seq: np.ndarray, rng: np.random.Generator, aug: bool) -> np.ndarray:
    """One random 150-frame training window, optionally with artefact augmentations."""
    speed = rng.uniform(*AUG["tempo"]) if aug and rng.random() < 0.5 else 1.0
    L = min(len(seq), int(round(WIN * speed)))
    s = rng.integers(0, len(seq) - L + 1)
    w = seq[s:s + L].astype(np.float64)
    if L != WIN:
        w = resample_time(w, 1.0, length=WIN)
    if aug:
        if rng.random() < 0.5:
            w = gaussian_filter1d(w, rng.uniform(*AUG["smooth"]), axis=0, mode="nearest")
        if rng.random() < 0.5:
            w = w + rng.normal(0, rng.uniform(*AUG["jitter_cm"]) / 100.0, w.shape)
        w = w * rng.uniform(*AUG["scale"])
        if rng.random() < 0.3:  # slow root drift (foot sliding)
            v = rng.normal(0, AUG["drift_mps"], 2) / FPS
            w[..., 0] += (np.arange(WIN) * v[0])[:, None]
            w[..., 2] += (np.arange(WIN) * v[1])[:, None]
    return w.astype(np.float32)


def corrupt(j: np.ndarray, kind: str, level: float, rng: np.random.Generator) -> np.ndarray:
    """Quality corruptions of a whole sequence of global joints (used only by the validation suite)."""
    j = j.astype(np.float64).copy()
    if kind == "smooth" and level > 0:
        j = gaussian_filter1d(j, level, axis=0, mode="nearest")
    elif kind == "jitter":
        j += rng.normal(0, level / 100.0, j.shape)
    elif kind == "tempo":
        j = resample_time(j, level)
    elif kind == "mirror":
        j[..., 0] *= -1
        j = j[:, MIRROR_PERM]
    elif kind == "stutter":  # hold every frame twice (15 fps playback)
        j[1::2] = j[0::2][: len(j[1::2])]
    elif kind == "wobble":  # low-frequency joint noise of `level` cm std
        n = gaussian_filter1d(rng.normal(0, 1, j.shape), 8, axis=0)
        j += n / n.std() * (level / 100.0)
    elif kind == "combined":  # smoothing + jitter + tempo together
        j = gaussian_filter1d(j, 2, axis=0, mode="nearest")
        j += rng.normal(0, 0.01, j.shape)
        j = resample_time(j, 1.1)
    return j.astype(np.float32)


def canonical_features(w: torch.Tensor) -> torch.Tensor:
    """(B, T, 22, 3) global joints -> (B, 129, T): pelvis-relative pose in the window's facing frame (63),
    root height (1), root horizontal velocity (2) and pose velocity (63)."""
    root = w[:, :, 0]
    across = ((w[:, :, L_HIP] - w[:, :, R_HIP]) + (w[:, :, L_SHO] - w[:, :, R_SHO])).mean(1)
    theta = torch.atan2(-across[:, 2], across[:, 0])  # yaw of forward = cross(across, up)
    c, s = torch.cos(theta)[:, None, None], torch.sin(theta)[:, None, None]
    rel = w[:, :, 1:] - root[:, :, None]
    rel = torch.stack([rel[..., 0] * c - rel[..., 2] * s, rel[..., 1], rel[..., 0] * s + rel[..., 2] * c], -1)
    c2, s2 = c[..., 0], s[..., 0]
    root_xz = torch.stack([root[..., 0] * c2 - root[..., 2] * s2, root[..., 0] * s2 + root[..., 2] * c2], -1)

    def vel(a):
        d = torch.diff(a, dim=1) * FPS
        return torch.cat([d[:, :1], d], 1)

    feats = torch.cat([rel.flatten(2), root[..., 1:2], vel(root_xz), vel(rel).flatten(2)], -1)
    return feats.transpose(1, 2)


# ----------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, ch: int, dilation: int, stride: int):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, 5, stride=stride, padding=2 * dilation, dilation=dilation)
        self.n1 = nn.GroupNorm(8, ch)
        self.c2 = nn.Conv1d(ch, ch, 5, padding=2, dilation=1)
        self.n2 = nn.GroupNorm(8, ch)
        self.skip = nn.Conv1d(ch, ch, 1, stride=stride) if stride > 1 else nn.Identity()

    def forward(self, x):
        h = F.gelu(self.n1(self.c1(x)))
        return F.gelu(self.n2(self.c2(h)) + self.skip(x))


class StyleNet(nn.Module):
    def __init__(self, in_ch: int, n_classes: int, width: int = 128, emb: int = 64, p_drop: float = 0.2):
        super().__init__()
        self.inp = nn.Sequential(nn.Conv1d(in_ch, width, 5, padding=2), nn.GroupNorm(8, width), nn.GELU())
        self.blocks = nn.Sequential(*[ResBlock(width, d, s) for d, s in
                                      [(1, 1), (2, 2), (4, 1), (8, 2), (1, 1), (2, 2)]])
        self.drop = nn.Dropout(p_drop)
        self.proj = nn.Sequential(nn.Linear(2 * width, 256), nn.GELU(), nn.Linear(256, emb))
        self.head = nn.Linear(emb, n_classes)

    def forward(self, x):
        h = self.blocks(self.inp(x))
        e = self.proj(self.drop(torch.cat([h.mean(-1), h.amax(-1)], 1)))
        return F.normalize(e, dim=1), self.head(e)


def supcon(z: torch.Tensor, y: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al., 2020) on L2-normalised embeddings."""
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    sim = (z @ z.T / tau).masked_fill(eye, -1e9)
    pos = (y[:, None] == y[None, :]) & ~eye
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    return -((log_prob * pos).sum(1) / pos.sum(1).clamp(min=1)).mean()


# ----------------------------------------------------------------------------------------------
# Encoding
# ----------------------------------------------------------------------------------------------
def sequence_windows(joints_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic 5 s windows with 1 s stride (short sequences are padded by holding the last pose)."""
    wins, owner = [], []
    for i, j in enumerate(joints_list):
        if len(j) < WIN:
            j = np.concatenate([j, np.repeat(j[-1:], WIN - len(j), 0)])
        for s in range(0, len(j) - WIN + 1, STRIDE):
            wins.append(j[s:s + WIN])
            owner.append(i)
    return np.stack(wins).astype(np.float32), np.array(owner)


class StyleEncoder:
    """Frozen, calibrated style encoder loaded from a checkpoint written by train()."""

    def __init__(self, ckpt: Path, device: str | None = None):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        # STYLE_ENCODER_DEVICE=cpu is faster on gfx1151 (slow MIOpen 1-D conv path); CUDA GPUs are fine
        self.device = device or os.environ.get("STYLE_ENCODER_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.classes, self.session_of_class = ck["classes"], np.asarray(ck["session_of_class"])
        self.temperature = float(ck["temperature"])
        self.mean = ck["mean"].to(self.device)[None, :, None]
        self.std = ck["std"].to(self.device)[None, :, None]
        self.model = StyleNet(ck["in_ch"], len(self.classes)).to(self.device)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()

    @torch.no_grad()
    def encode_windows(self, W: np.ndarray, batch: int = 256) -> tuple[np.ndarray, np.ndarray]:
        embs, logits = [], []
        for k in range(0, len(W), batch):
            xb = W[k:k + batch]
            n = len(xb)
            if n < batch:  # fixed batch shape avoids re-tuning conv kernels for every remainder
                xb = np.concatenate([xb, np.repeat(xb[-1:], batch - n, 0)])
            x = (canonical_features(torch.from_numpy(xb).to(self.device)) - self.mean) / self.std
            z, lg = self.model(x)
            embs.append(z[:n].float().cpu())
            logits.append(lg[:n].float().cpu())
        return torch.cat(embs).numpy(), torch.cat(logits).numpy()

    def encode(self, joints_list: list[np.ndarray]) -> dict:
        """Global joints (T, 22, 3) per sequence -> per-window embedding, calibrated probabilities, owner index."""
        W, owner = sequence_windows(joints_list)
        emb, logits = self.encode_windows(W)
        prob = torch.softmax(torch.from_numpy(logits) / self.temperature, 1).numpy()
        return {"emb": emb, "prob": prob, "owner": owner}

    def session_scores(self, prob: np.ndarray, session: int, session_of_class=None) -> dict:
        """session_of_class maps encoder classes to the benchmark's sessions; it must come from the benchmark
        being evaluated (benchmark_session_of_class), since the same encoder serves several benchmarks.
        With a reject class, DanceRate = 1 - P(other) says how much of the output is recognised as dance at all."""
        soc = self.session_of_class if session_of_class is None else np.asarray(session_of_class)
        in_session = soc == session
        scores = {"StyleRetention": float(prob[:, in_session].sum(1).mean()),
                  "StyleAcc": float(in_session[prob.argmax(1)].mean())}
        if OTHER in self.classes:
            scores["DanceRate"] = float(1.0 - prob[:, self.classes.index(OTHER)].mean())
        return scores


def benchmark_session_of_class(classes: list[str], manifest: dict) -> list[int]:
    """Session index of every encoder class (AIST++ genre code / FineDance family) in a benchmark manifest;
    the reject class belongs to no session (-1)."""
    return [-1 if c == OTHER else next(k for k, s in enumerate(manifest["sessions"]) if c in s["groups"])
            for c in classes]


# ----------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------
def ckpt_path(dataset: str, fold: int, tag: str = "") -> Path:
    return OUT_ROOT / (f"{dataset}_{tag}" if tag else dataset) / ("encoder.pt" if fold < 0 else f"encoder_fold{fold}.pt")


def negative_window(kind: str, rng: np.random.Generator, pool: dict, dance: list[np.ndarray]) -> np.ndarray:
    """One training window of the reject class."""
    if kind in pool:
        return sample_window(pool[kind][rng.integers(len(pool[kind]))], rng, aug=True)
    w = sample_window(dance[rng.integers(len(dance))], rng, aug=True)
    if kind == "static":
        w = np.repeat(w[rng.integers(WIN)][None], WIN, 0)
        return (w + rng.normal(0, rng.uniform(0, 0.01), w.shape)).astype(np.float32)
    return w[rng.permutation(WIN)]  # time-shuffled dance


def held_out_dancers(info: dict, fold: int) -> dict[str, str]:
    """AIST++: for every genre, the fold-th of its 3 dancers is held out."""
    return {c: sorted({info["dancer"][n] for n in info["sess"] if info["label"][n] == c})[fold] for c in info["classes"]}


def train(dataset: str, fold: int = -1, steps: int = 6000, seed: int = 0, device: str = "cpu", tag: str = "",
          negatives: bool = False) -> Path:
    """On this machine (gfx1151) MIOpen runs these small 1-D convs through a slow GEMM fallback, so the
    CPU (oneDNN) trains ~4x faster than the GPU; run several folds as parallel processes.
    negatives=True adds the reject class OTHER (see NEG_MIX)."""
    info, J = dataset_info(dataset), load_joints(dataset)
    train_names = list(info["split"]["train"])
    if fold >= 0:
        held = held_out_dancers(info, fold)
        train_names = [n for n in train_names if info["dancer"][n] != held[info["label"][n]]]
    genres = info["classes"]
    classes = list(genres) + ([OTHER] if negatives else [])
    session_of_class = list(info["session_of_class"]) + ([-1] if negatives else [])
    by_class = {c: [n for n in train_names if info["label"][n] == c] for c in genres}
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if negatives:
        pool = {"babel": babel_negatives("train", 3000, 0), "vae_random": vae_random_negatives(dataset, 600, 0)}
        babel_val = babel_negatives("val", 600, 0)[::2]  # odd half is held out for validate()
        dance = [J[n] for n in train_names]
        neg_kinds, neg_p = list(NEG_MIX), np.array(list(NEG_MIX.values()))

    clean = np.stack([sample_window(J[train_names[i]], rng, aug=False)
                      for i in rng.integers(0, len(train_names), 4000)])
    with torch.no_grad():
        f = canonical_features(torch.from_numpy(clean).to(device))
        mean, std = f.mean((0, 2)), f.std((0, 2)) + 1e-6
    mu, sd = mean[None, :, None], std[None, :, None]

    model = StyleNet(f.shape[1], len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    warm = 300
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    per_class = max(4, 80 // len(genres))
    n_neg = 2 * per_class if negatives else 0
    n_genre = per_class * len(genres)
    y = torch.cat([torch.arange(len(genres), device=device).repeat_interleave(per_class),
                   torch.full((n_neg,), len(genres), device=device, dtype=torch.long)])

    val_names = info["split"]["val"]
    Wv, ov = sequence_windows([J[n] for n in val_names])
    yv = np.array([classes.index(info["label"][val_names[i]]) for i in ov])
    if negatives:  # calibrate / monitor the reject decision on held-out everyday motion as well
        Wb, _ = sequence_windows(babel_val)
        Wv, yv = np.concatenate([Wv, Wb]), np.concatenate([yv, np.full(len(Wb), len(genres))])

    def val_logits():
        model.eval()
        out = []
        with torch.no_grad():
            for k in range(0, len(Wv), 256):
                xb = Wv[k:k + 256]
                n = len(xb)
                if n < 256:
                    xb = np.concatenate([xb, np.repeat(xb[-1:], 256 - n, 0)])
                out.append(model((canonical_features(torch.from_numpy(xb).to(device)) - mu) / sd)[1][:n].cpu())
        model.train()
        return torch.cat(out)

    t0 = time.time()
    model.train()
    for step in range(steps):
        names = [n for c in genres for n in rng.choice(by_class[c], per_class)]
        windows = [sample_window(J[n], rng, aug=True) for n in names]
        if negatives:
            windows += [negative_window(k, rng, pool, dance) for k in rng.choice(neg_kinds, n_neg, p=neg_p)]
        x = (canonical_features(torch.from_numpy(np.stack(windows).astype(np.float32)).to(device)) - mu) / sd
        z, logits = model(x)
        # the contrastive term shapes the genre geometry only; the heterogeneous reject class gets CE only
        loss = F.cross_entropy(logits, y, label_smoothing=0.1) + 0.5 * supcon(z[:n_genre], y[:n_genre])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if (step + 1) % 500 == 0 or step + 1 == steps:
            pv = val_logits().argmax(1).numpy()
            g = yv < len(genres)
            msg = f"val genre acc {float((pv[g] == yv[g]).mean()):.3f}"
            if negatives:
                msg += f", everyday motion rejected {float((pv[~g] == len(genres)).mean()):.3f}"
            print(f"[{dataset}{'_' + tag if tag else ''} fold {fold}] step {step + 1}/{steps} loss {loss.item():.3f} "
                  f"{msg} ({time.time() - t0:.0f}s)", flush=True)

    # temperature scaling on clean val windows (Guo et al., 2017)
    lv, yt = val_logits(), torch.from_numpy(yv)
    temps = np.linspace(0.3, 5.0, 95)
    temperature = float(temps[np.argmin([F.cross_entropy(lv / t, yt).item() for t in temps])])

    path = ckpt_path(dataset, fold, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "mean": mean.cpu(), "std": std.cpu(), "in_ch": int(f.shape[1]),
                "classes": classes, "session_of_class": session_of_class, "temperature": temperature,
                "fold": fold, "held_out_dancers": held_out_dancers(info, fold) if fold >= 0 else None,
                "train_sequences": len(train_names), "steps": steps, "aug": AUG,
                "negatives": NEG_MIX if negatives else None, "window": WIN, "stride": STRIDE}, path)
    print(f"[{dataset} fold {fold}] temperature {temperature:.2f}; saved {path}", flush=True)
    return path


# ----------------------------------------------------------------------------------------------
# Validation suite
# ----------------------------------------------------------------------------------------------
SEEN = {"smooth": [1, 2, 3, 4, 6], "jitter": [0.5, 1, 2, 3], "tempo": [0.7, 0.8, 0.9, 1.1, 1.25, 1.4]}
UNSEEN = {"mirror": [1], "stutter": [1], "wobble": [3], "combined": [1]}
CRITERIA = {"seen_in_range_max_drop": 0.05, "unseen_max_drop": 0.10,
            "swap_diag_min_retention": 0.8, "swap_offdiag_max_retention": 0.1,
            "reject_max_session_mass": 0.2}  # non-dance probes may put at most this mass on any one session
_JOINTS = {}


def _in_aug_range(kind: str, level: float) -> bool:
    lo, hi = {"smooth": AUG["smooth"], "jitter": AUG["jitter_cm"], "tempo": AUG["tempo"]}[kind]
    return lo <= level <= hi


def _probe_worker(args):
    """Handcrafted kinetic/geometric window features of one corrupted sequence (baseline probe path)."""
    from metrics.cl_metrics import window_features

    dataset, name, kind, level, seed = args
    if dataset not in _JOINTS:
        _JOINTS[dataset] = np.load(OUT_ROOT / f"joints_{dataset}.npz")
    j = corrupt(_JOINTS[dataset][name], kind, level, np.random.default_rng(seed)).astype(np.float64)
    j[:, 1:] -= j[:, 0:1]  # M2DMetrics convention: joints relative to the root
    return window_features(j)


def _ece(prob: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    conf, pred = prob.max(1), prob.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    return float(sum(abs((pred[m] == y[m]).mean() - conf[m].mean()) * m.mean()
                     for lo, hi in zip(edges[:-1], edges[1:]) if (m := (conf > lo) & (conf <= hi)).any()))


def _fd(a: np.ndarray, b: np.ndarray) -> float:
    from metrics.cl_metrics import frozen_fid

    return frozen_fid(a, b, np.zeros(a.shape[1]), np.ones(a.shape[1]))


def validate(dataset: str, tag: str = "", ood_joints: str | None = None) -> dict:
    """ood_joints: optional npz of generated non-dance joints (e.g. an untrained generator's output) that must
    be rejected; never used for training."""
    from multiprocessing import Pool

    from metrics.cl_metrics import GenreProbe, cached_window_features, frozen_fid

    info, J = dataset_info(dataset), load_joints(dataset)
    enc = StyleEncoder(ckpt_path(dataset, -1, tag))
    # genre labels index `classes`; predictions index enc.classes (may end with the reject class -> session -1)
    K, classes, soc = info["K"], info["classes"], np.asarray(enc.session_of_class)
    label, sess_of, test = info["label"], info["sess"], info["split"]["test"]
    fdir, broot = Path(DATASETS[dataset]["feature_dir"]), info["root"]
    probe = GenreProbe.load(broot / "eval" / "probe.npz")
    refs = np.load(broot / "eval" / "refs.npz")
    rng = np.random.default_rng(0)
    rep = {"dataset": dataset, "classes": classes, "session_of_class": soc.tolist(), "aug_ranges": AUG,
           "criteria": CRITERIA, "window": WIN, "stride": STRIDE}
    probe_cls = lambda X: np.argmax(((X - probe.mu) / probe.sd) @ probe.W.T + probe.b, 1)

    # 1. clean, music-disjoint test windows ------------------------------------------------------
    out = enc.encode([J[n] for n in test])
    win_s = np.array([sess_of[test[i]] for i in out["owner"]])
    win_c = np.array([classes.index(label[test[i]]) for i in out["owner"]])
    pred = out["prob"].argmax(1)
    Xh = np.concatenate([np.concatenate(cached_window_features(fdir, n), 1) for n in test])
    ph = probe_cls(Xh)
    rep["clean"] = {
        "n_windows": int(len(pred)),
        "encoder": {"genre_acc": float((pred == win_c).mean()), "session_acc": float((soc[pred] == win_s).mean()),
                    "per_session": [enc.session_scores(out["prob"][win_s == s], s) for s in range(K)],
                    "ece": _ece(out["prob"], win_c), "temperature": enc.temperature},
        "probe": {"genre_acc": float((ph == win_c).mean()) if dataset == "aistpp" else None,
                  "session_acc": float((probe.session_of_class[ph] == win_s).mean())},
    }
    print(f"[clean] encoder genre {rep['clean']['encoder']['genre_acc']:.3f} session "
          f"{rep['clean']['encoder']['session_acc']:.3f} | probe session {rep['clean']['probe']['session_acc']:.3f}",
          flush=True)

    # 2. held-out dancers (AIST++: each genre's 3 dancers are exclusive to it) ---------------------
    if dataset == "aistpp":
        rep["held_out_dancers"] = []
        for f in range(3):
            if not ckpt_path(dataset, f, tag).exists():
                train(dataset, f, tag=tag, negatives=OTHER in enc.classes)
            ef, held = StyleEncoder(ckpt_path(dataset, f, tag)), held_out_dancers(info, f)
            is_held = lambda n: info["dancer"][n] == held[label[n]]
            tr = [n for n in info["split"]["train"] if not is_held(n)]
            Xtr = [np.concatenate(cached_window_features(fdir, n), 1) for n in tr]
            ytr = np.concatenate([[classes.index(label[n])] * len(x) for n, x in zip(tr, Xtr)])
            Xtr = np.concatenate(Xtr)
            keep = np.concatenate([rng.permutation(np.where(ytr == c)[0])[:4000] for c in range(len(classes))])
            pf = GenreProbe.fit(Xtr[keep], ytr[keep], soc)
            row = {"fold": f, "held_out": held}
            for subset, names in (("all_sequences", [n for n in info["split"]["train"] + test if is_held(n)]),
                                  ("music_disjoint", [n for n in test if is_held(n)])):
                e = ef.encode([J[n] for n in names])
                yc = np.array([classes.index(label[names[i]]) for i in e["owner"]])
                pe = e["prob"].argmax(1)
                Xp = np.concatenate([np.concatenate(cached_window_features(fdir, n), 1) for n in names])
                yp = np.concatenate([[classes.index(label[n])] * len(cached_window_features(fdir, n)[0]) for n in names])
                pp = np.argmax(((Xp - pf.mu) / pf.sd) @ pf.W.T + pf.b, 1)
                row[subset] = {"n_sequences": len(names), "n_windows": int(len(pe)),
                               "encoder_genre_acc": float((pe == yc).mean()),
                               "encoder_session_acc": float((soc[pe] == soc[yc]).mean()),
                               "probe_genre_acc": float((pp == yp).mean()),
                               "probe_session_acc": float((soc[pp] == soc[yp]).mean())}
            rep["held_out_dancers"].append(row)
            a = row["all_sequences"]
            print(f"[dancers fold {f}] encoder genre {a['encoder_genre_acc']:.3f} session {a['encoder_session_acc']:.3f}"
                  f" | probe genre {a['probe_genre_acc']:.3f} session {a['probe_session_acc']:.3f}", flush=True)

    # 3. invariance: quality corruptions that keep the genre --------------------------------------
    clean_emb = {s: out["emb"][win_s == s] for s in range(K)}
    settings = [("clean", 0)] + [(k, l) for k, ls in SEEN.items() for l in ls] + \
               [(k, l) for k, ls in UNSEEN.items() for l in ls]
    rep["invariance"] = []
    # every worker imports torch/scipy (~1-2 GB committed each); too many exhaust the Windows page file
    with Pool(max(1, min(int(os.environ.get("VALIDATE_WORKERS", 6)), (os.cpu_count() or 2) - 2))) as pool:
        for kind, level in settings:
            cj = [corrupt(J[n], kind, level, np.random.default_rng(i)) for i, n in enumerate(test)]
            e = enc.encode(cj)
            ws = np.array([sess_of[test[i]] for i in e["owner"]])
            wc = np.array([classes.index(label[test[i]]) for i in e["owner"]])
            pe = e["prob"].argmax(1)
            hand = pool.map(_probe_worker, [(dataset, n, kind, level, i) for i, n in enumerate(test)], chunksize=2)
            hk = np.concatenate([h[0] for h in hand])
            hs = np.concatenate([[sess_of[n]] * len(h[0]) for n, h in zip(test, hand)])
            ph = probe.session_of_class[probe_cls(np.concatenate([np.concatenate(h, 1) for h in hand]))]
            row = {"kind": kind, "level": level,
                   "seen_in_training": kind in SEEN and _in_aug_range(kind, level),
                   "encoder_session_acc": float((soc[pe] == ws).mean()),
                   "encoder_genre_acc": float((pe == wc).mean()),
                   "encoder_retention": float(np.mean([enc.session_scores(e["prob"][ws == s], s)["StyleRetention"]
                                                       for s in range(K)])),
                   "style_fd_to_clean": float(np.mean([_fd(clean_emb[s], e["emb"][ws == s]) for s in range(K)])),
                   "probe_session_acc": float((ph == hs).mean()),
                   "fid_k": float(np.mean([frozen_fid(refs[f"s{s}_kin"], hk[hs == s], refs["kin_mean"], refs["kin_std"])
                                           for s in range(K)]))}
            rep["invariance"].append(row)
            print(f"[{kind:8s} {level:>5}] encoder session {row['encoder_session_acc']:.3f} retention "
                  f"{row['encoder_retention']:.3f} StyleFD {row['style_fd_to_clean']:6.3f} | probe session "
                  f"{row['probe_session_acc']:.3f} | FID_k {row['fid_k']:9.1f}", flush=True)

    # 4. genre swap: score each session's reference against every session's real dance -------------
    prob_s = {s: out["prob"][win_s == s] for s in range(K)}
    own = {s: out["owner"][win_s == s] for s in range(K)}
    ret = [[enc.session_scores(prob_s[j], i)["StyleRetention"] for j in range(K)] for i in range(K)]
    acc = [[enc.session_scores(prob_s[j], i)["StyleAcc"] for j in range(K)] for i in range(K)]
    fdm = [[_fd(clean_emb[i], clean_emb[j]) if i != j else 0.0 for j in range(K)] for i in range(K)]
    noise = []
    for s in range(K):
        seqs = np.unique(own[s])
        fd_n, ret_n = [], []
        for _ in range(30):
            h = rng.permutation(seqs)[: len(seqs) // 2]
            m = np.isin(own[s], h)
            fd_n.append(_fd(clean_emb[s][m], clean_emb[s][~m]))
            ret_n.append(enc.session_scores(prob_s[s][~m], s)["StyleRetention"])
        off_ret = max(ret[s][j] for j in range(K) if j != s)
        off_fd = min(fdm[s][j] for j in range(K) if j != s)
        noise.append({"style_fd_noise_mean": float(np.mean(fd_n)), "style_fd_noise_std": float(np.std(fd_n)),
                      "retention_halfset_mean": float(np.mean(ret_n)), "retention_halfset_std": float(np.std(ret_n)),
                      "snr_style_fd": float((off_fd - np.mean(fd_n)) / (np.std(fd_n) + 1e-9)),
                      "snr_retention": float((np.mean(ret_n) - off_ret) / (np.std(ret_n) + 1e-9))})
    rep["genre_swap"] = {"retention": ret, "style_acc": acc, "style_fd": fdm, "noise": noise}
    print("[swap] retention matrix (row = session scored, col = whose real dance):", flush=True)
    for i in range(K):
        print("   " + " ".join(f"{v:5.2f}" for v in ret[i]) + f"   SNR retention {noise[i]['snr_retention']:6.1f}"
              f" | SNR StyleFD {noise[i]['snr_style_fd']:6.1f}", flush=True)

    # 6. rejection: motion that is not dance must not count as any session's style -------------------
    if OTHER in enc.classes:
        from utils.motion_process import convert_motion_to_joints

        oi = enc.classes.index(OTHER)
        gt10 = [J[n][:301] for n in test]
        must_reject = {}
        if ood_joints:
            z = np.load(ood_joints)
            must_reject["untrained generator"] = [z[k] for k in sorted(z.files, key=int)]
        must_reject["static pose"] = [np.repeat(j[:1], len(j), 0) for j in gt10]
        must_reject["time-shuffled dance"] = [j[rng.permutation(len(j))] for j in gt10]
        if dataset in NEG_VAE:
            must_reject["random VAE latents (unseen seed)"] = vae_random_negatives(dataset, 100, 12345)
            must_reject["unseen: constant latent"] = vae_latent_motion(dataset, "constant", 40, 777)
            must_reject["unseen: latent random walk"] = vae_latent_motion(dataset, "walk", 40, 778)
        must_reject["everyday motion (BABEL, held out)"] = babel_negatives("val", 600, 0)[1::2]
        report_only = {"dance smoothed sigma=12": [gaussian_filter1d(j, 12, axis=0, mode="nearest") for j in gt10]}
        if dataset in NEG_VAE:
            rec = Path(NEG_VAE[dataset]["tokens"].replace("TOKENS_", "RECOVERED_"))
            report_only["VAE reconstructions of test dance"] = [
                convert_motion_to_joints(np.load(rec / f"{n}.npy"), 272) for n in test]
        rep["rejection"] = []
        for name, Jp in list(must_reject.items()) + list(report_only.items()):
            e = enc.encode(Jp)
            p = e["prob"]
            mass = [float(p[:, soc == s].sum(1).mean()) for s in range(K)]
            row = {"probe": name, "must_reject": name in must_reject, "n_windows": int(len(p)),
                   "p_other": float(p[:, oi].mean()), "session_mass": mass, "max_session_mass": float(max(mass))}
            if name in report_only:  # still dance: report how much of its own session's style survives
                own = np.array([sess_of[test[i]] for i in e["owner"]])
                row["own_session_retention"] = float(np.mean([p[k, soc == own[k]].sum() for k in range(len(p))]))
            rep["rejection"].append(row)
            print(f"[reject] {name:36s} P(other) {row['p_other']:.3f} | max session mass {row['max_session_mass']:.3f}"
                  + (f" | own-session retention {row['own_session_retention']:.3f}" if name in report_only else ""),
                  flush=True)

    # 5. criteria -----------------------------------------------------------------------------------
    clean_acc = rep["invariance"][0]["encoder_session_acc"]
    clean_probe = rep["invariance"][0]["probe_session_acc"]
    seen = [r for r in rep["invariance"][1:] if r["seen_in_training"]]
    unseen = [r for r in rep["invariance"][1:] if r["kind"] in UNSEEN or not r["seen_in_training"]]
    rep["verdict"] = {
        "seen_in_range_worst_drop": float(max(clean_acc - r["encoder_session_acc"] for r in seen)),
        "unseen_or_out_of_range_worst_drop": float(max(clean_acc - r["encoder_session_acc"] for r in unseen)),
        "probe_seen_in_range_worst_drop": float(max(clean_probe - r["probe_session_acc"] for r in seen)),
        "swap_min_diag": float(min(ret[i][i] for i in range(K))),
        "swap_max_offdiag": float(max(ret[i][j] for i in range(K) for j in range(K) if i != j)),
    }
    v = rep["verdict"]
    v["pass_seen"] = v["seen_in_range_worst_drop"] <= CRITERIA["seen_in_range_max_drop"]
    v["pass_unseen"] = v["unseen_or_out_of_range_worst_drop"] <= CRITERIA["unseen_max_drop"]
    v["pass_swap"] = (v["swap_min_diag"] >= CRITERIA["swap_diag_min_retention"]
                      and v["swap_max_offdiag"] <= CRITERIA["swap_offdiag_max_retention"])
    if dataset == "aistpp":
        hd = rep["held_out_dancers"]
        v["held_out_encoder_genre_acc"] = float(np.mean([r["all_sequences"]["encoder_genre_acc"] for r in hd]))
        v["held_out_probe_genre_acc"] = float(np.mean([r["all_sequences"]["probe_genre_acc"] for r in hd]))
        v["pass_dancers"] = v["held_out_encoder_genre_acc"] > v["held_out_probe_genre_acc"]
    if "rejection" in rep:
        rows = [r for r in rep["rejection"] if r["must_reject"]]
        v["reject_worst_session_mass"] = float(max(r["max_session_mass"] for r in rows))
        v["pass_reject"] = v["reject_worst_session_mass"] <= CRITERIA["reject_max_session_mass"]
    path = ckpt_path(dataset, -1, tag).parent / "validation.json"
    path.write_text(json.dumps(rep, indent=1))
    print(f"[verdict] {json.dumps(v)}\nwrote {path}", flush=True)
    return rep


def plot(dataset: str, tag: str = "") -> Path:
    """Six-panel validation figure from validation.json (static PNG for the paper appendix)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import transforms
    from matplotlib.colors import LinearSegmentedColormap

    out_dir = ckpt_path(dataset, -1, tag).parent
    rep = json.loads((out_dir / "validation.json").read_text())
    info = dataset_info(dataset)
    groups = ["+".join(s["groups"]) for s in json.loads((info["root"] / "manifest.json").read_text())["sessions"]]
    C = {"enc": "#2a78d6", "probe": "#eb6834", "ret": "#1baf7a", "surface": "#fcfcfb", "ink": "#0b0b0b",
         "ink2": "#52514e", "grid": "#e4e3df", "band": "#ecebe7"}
    blues = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelcolor": C["ink2"],
                         "xtick.color": C["ink2"], "ytick.color": C["ink2"], "text.color": C["ink"]})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.8), dpi=200)
    fig.patch.set_facecolor(C["surface"])
    for ax in axes.flat:
        ax.set_facecolor(C["surface"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(C["grid"])
        ax.grid(axis="y", color=C["grid"], linewidth=0.6)
        ax.set_axisbelow(True)

    inv = rep["invariance"]
    clean = inv[0]
    line_kw = dict(linewidth=1.6, marker="o", markersize=5.5, markeredgecolor=C["surface"], markeredgewidth=1.2)
    series = [("encoder_session_acc", C["enc"], "Encoder: session accuracy"),
              ("encoder_retention", C["ret"], "Encoder: style retention"),
              ("probe_session_acc", C["probe"], "Handcrafted probe: session accuracy")]
    panels = [("smooth", "Temporal smoothing σ (frames)", 0.0, AUG["smooth"]),
              ("jitter", "Joint jitter σ (cm)", 0.0, AUG["jitter_cm"]),
              ("tempo", "Playback speed (×)", 1.0, AUG["tempo"])]
    for ax, (kind, xlabel, x0, rng_) in zip(axes[0], panels):
        rows = sorted([r for r in inv[1:] if r["kind"] == kind] + [dict(clean, level=x0)], key=lambda r: r["level"])
        xs = [r["level"] for r in rows]
        ax.axvspan(*rng_, color=C["band"], zorder=0)
        ax.text(np.mean(rng_), 0.04, "trained range", ha="center", color=C["ink2"], fontsize=8)
        ends = []
        for key, col, name in series:
            ys = [r[key] for r in rows]
            ax.plot(xs, ys, color=col, label=name, **line_kw)
            ends.append(ys[-1])
        # end-of-line value labels, nudged apart vertically when two lines finish close together
        placed = []
        for y in sorted(ends):
            placed.append(max(y, placed[-1] + 0.05) if placed else y)
        shift = transforms.offset_copy(ax.transData, fig=fig, x=6, y=0, units="points")
        for y_true, y_lab in zip(sorted(ends), placed):
            ax.text(xs[-1], y_lab, f"{y_true:.2f}", transform=shift, va="center", color=C["ink"], fontsize=8)
        ax.set_ylim(0, 1.04)
        ax.set_xlabel(xlabel)
        ax.margins(x=0.08)
    axes[0, 0].set_ylabel("Accuracy / retention (test windows)")
    titles = ["a  Smoothing (seen in training up to σ=4)", "b  Jitter (seen up to 2 cm)", "c  Tempo (seen 0.8-1.25×)"]
    for ax, t in zip(axes[0], titles):
        ax.set_title(t, loc="left", color=C["ink"])

    # d: unseen corruptions and out-of-range strengths
    ax = axes[1, 0]
    picks = [("clean", 0, "clean")] + [(k, l, k) for k, ls in UNSEEN.items() for l in ls] + \
            [("smooth", 6, "smooth σ=6"), ("jitter", 3, "jitter 3 cm"), ("tempo", 0.7, "tempo 0.7×"), ("tempo", 1.4, "tempo 1.4×")]
    lookup = {(r["kind"], r["level"]): r for r in inv}
    x = np.arange(len(picks))
    w = 0.38
    for off, key, col, name in ((-w / 2, "encoder_session_acc", C["enc"], "Encoder"),
                                (w / 2, "probe_session_acc", C["probe"], "Handcrafted probe")):
        vals = [lookup[(k, l)][key] for k, l, _ in picks]
        ax.bar(x + off, vals, width=w - 0.04, color=col, label=name)
    ax.set_xticks(x, [p[2] for p in picks], rotation=35, ha="right")
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("Session accuracy")
    ax.set_title("d  Corruptions never seen in training", loc="left", color=C["ink"])
    ax.set_xlabel("blue = encoder, orange = handcrafted probe")

    # e: held-out dancers (AIST++) or per-session clean retention (FineDance)
    ax = axes[1, 1]
    if dataset == "aistpp":
        hd = rep["held_out_dancers"]
        labels = [f"fold {r['fold']}" for r in hd] + ["mean"]
        enc_v = [r["all_sequences"]["encoder_genre_acc"] for r in hd]
        pr_v = [r["all_sequences"]["probe_genre_acc"] for r in hd]
        enc_v.append(float(np.mean(enc_v)))
        pr_v.append(float(np.mean(pr_v)))
        x = np.arange(len(labels))
        ax.bar(x - w / 2, enc_v, width=w - 0.04, color=C["enc"], label="Encoder")
        ax.bar(x + w / 2, pr_v, width=w - 0.04, color=C["probe"], label="Handcrafted probe")
        for xi, (a, b) in enumerate(zip(enc_v, pr_v)):
            ax.text(xi - w / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=8)
            ax.text(xi + w / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=8)
        ax.axhline(1 / len(rep["classes"]), color=C["ink2"], linewidth=0.8)
        ax.text(len(labels) - 0.5, 1 / len(rep["classes"]) + 0.02, "chance", ha="right", color=C["ink2"], fontsize=8)
        ax.set_xticks(x, labels)
        ax.set_ylabel("10-genre accuracy on an unseen dancer")
        ax.set_title("e  Held-out dancers (genre, not identity)", loc="left", color=C["ink"])
        ax.legend(frameon=False, loc="upper right", fontsize=8)
    else:
        per = rep["clean"]["encoder"]["per_session"]
        x = np.arange(len(per))
        ax.bar(x, [p["StyleRetention"] for p in per], width=0.5, color=C["enc"])
        ax.set_xticks(x, groups)
        ax.set_ylabel("Style retention on real test dance")
        ax.set_title("e  Clean retention per session", loc="left", color=C["ink"])
    ax.set_ylim(0, 1.04)

    # f: genre swap matrix
    ax = axes[1, 2]
    M = np.array(rep["genre_swap"]["retention"])
    ax.imshow(M, cmap=blues, vmin=0, vmax=1)
    for i in range(len(M)):
        for j in range(len(M)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if M[i, j] > 0.55 else C["ink"])
    ax.set_xticks(range(len(M)), groups, rotation=35, ha="right")
    ax.set_yticks(range(len(M)), groups)
    ax.set_xlabel("Real dance scored (whose test set)")
    ax.set_ylabel("Session whose style is measured")
    ax.grid(False)
    ax.set_title("f  Genre swap: retention matrix", loc="left", color=C["ink"])

    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = out_dir / "validation.png"
    fig.savefig(path, facecolor=C["surface"])
    plt.close(fig)
    print(f"wrote {path}")
    return path


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "validate", "plot"])
    ap.add_argument("--dataset", default="aistpp", choices=sorted(DATASETS))
    ap.add_argument("--fold", type=int, default=-1, help="AIST++ held-out-dancer fold (0-2); -1 = all dancers")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="training device (cpu is faster on gfx1151, see train())")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch default)")
    ap.add_argument("--tag", default="", help="checkpoint/report folder suffix (e.g. 'rej' -> _style_encoder/aistpp_rej)")
    ap.add_argument("--negatives", action="store_true", help="train with the reject class (see NEG_MIX)")
    ap.add_argument("--ood_joints", default=None, help="validate: npz of generated non-dance joints to reject")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.cmd == "train":
        train(a.dataset, a.fold, a.steps, a.seed, a.device, a.tag, a.negatives)
    elif a.cmd == "validate":
        validate(a.dataset, a.tag, a.ood_joints)
    else:
        plot(a.dataset, a.tag)
