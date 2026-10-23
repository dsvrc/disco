"""
Shared utilities for the TMR evidence scripts (scripts/evidence/e*.py).

Everything works in the frozen latent space of the DiscoForcing CL benchmark. A "sample" is a fixed-length window of
VAE tokens (T latent frames, 4 motion frames each) with its token-aligned causal music windows, i.e. exactly what
DiffForcingWanModel sees in training (datasets/aistpp.py stream mode with train_max_tokens crops). The audio encoder
of assumption (A1) is WanModel.music_embedding (35 -> 1024 -> 1024), whose output is the cross-attention context;
phi_i in R^{T x D} is that output pooled over the causal music window (CL_TMR_PLAN.md, Sec. 3).

Losses use frozen diffusion noise (common random numbers): every window gets R stratified diffusion-forcing times and
Gaussian draws from a seeded generator, so L_j(theta) is a deterministic, differentiable function of theta -- the
object of the paper's descent-lemma analysis -- and loss differences between checkpoints carry no Monte-Carlo noise.

Run the scripts from the repo root, like scripts/eval_cl_session.py.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.getcwd())

AIST_RE = re.compile(r"g(\w\w)_s(\w\w)_c\w+_d(\d\d)_m(\w\w)(\d)_ch(\d\d)")
TOKEN_FPS = 7.5  # 30 fps motion, one VAE token per 4 frames
LOSS_BINS = (0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0 + 1e-6)  # per-token noise-level bins for loss profiles


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# ----------------------------------------------------------------------------------------------
# Config / model / checkpoints
# ----------------------------------------------------------------------------------------------
def load_cfg(config_path: str, overrides: dict | None = None):
    """Same resolution as utils.initialize.Config (paths.yaml first), without importing lightning."""
    paths = Path("configs/paths.yaml")
    if not paths.exists():
        paths = Path("configs/paths_default.yaml")
    cfg = OmegaConf.merge(OmegaConf.load(paths), OmegaConf.load(config_path))
    for k, v in (overrides or {}).items():
        OmegaConf.update(cfg, k, v)
    OmegaConf.resolve(cfg)
    return cfg


def instantiate(target: str, **kw):
    module, name = target.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)(**kw)


def _fp32_varlen_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.0, softmax_scale=None, q_scale=None,
                           causal=False, window_size=(-1, -1), deterministic=False, dtype=None, version=None):
    """flash_attn_varlen_func semantics (keys beyond k_lens ignored, bottom-right causal / window masks) in fp32.
    The stock kernels cast q/k/v to bf16, whose rounding swamps the small loss differences measured here."""
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype
    q, k, v = q.float(), k.float(), v.float()
    if q_scale is not None:
        q = q * q_scale
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    if k.size(1) != q.size(1):
        rep = q.size(1) // k.size(1)
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    mask = empty = None
    if q_lens is not None or k_lens is not None or causal or tuple(window_size) != (-1, -1):
        dev = q.device
        q_lens = torch.full((b,), lq, device=dev) if q_lens is None else q_lens.to(dev)
        k_lens = torch.full((b,), lk, device=dev) if k_lens is None else k_lens.to(dev)
        i = torch.arange(lq, device=dev).view(1, lq, 1)
        j = torch.arange(lk, device=dev).view(1, 1, lk)
        diag = i + (k_lens - q_lens).view(b, 1, 1)
        mask = (j < k_lens.view(b, 1, 1)) & (i < q_lens.view(b, 1, 1))
        left, right = window_size
        if causal:
            right = 0
        if right >= 0:
            mask &= j <= diag + right
        if left >= 0:
            mask &= j >= diag - left
        mask = mask.unsqueeze(1)
        empty = ~mask.any(-1, keepdim=True)
        mask = mask | empty
    x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p,
                                                         scale=softmax_scale)
    if empty is not None:
        x = x.masked_fill(empty, 0.0)
    return x.transpose(1, 2).contiguous().type(out_dtype)


def set_precision(fp32_attention: bool = True) -> None:
    """Exact fp32 everywhere: no TF32 matmuls, and (by default) fp32 attention instead of the bf16 kernels."""
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if fp32_attention:
        import models.tools.wan_model as wan_model  # imports flash_attention into its own namespace

        wan_model.flash_attention = _fp32_varlen_attention


def build_model(cfg, device: str):
    return instantiate(cfg.model.target, **OmegaConf.to_container(cfg.model.params, resolve=True)).to(device).eval()


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


@torch.no_grad()
def load_weights(model, ckpt: str | Path, ema: bool = True) -> dict:
    """Load a training checkpoint into `model`; ema=True copies the EMA shadow weights, i.e. the model that
    scripts/eval_cl_session.py evaluates (torch_ema keeps shadow_params in model.parameters() order)."""
    ck = _torch_load(ckpt)
    model.load_state_dict(ck["state_dict"], strict=True)
    used_ema = bool(ema and "ema_state" in ck)
    if used_ema:
        shadow = ck["ema_state"]["shadow_params"]
        params = list(model.parameters())
        assert len(shadow) == len(params), f"{ckpt}: {len(shadow)} EMA tensors for {len(params)} parameters"
        for p, s in zip(params, shadow):
            p.copy_(s.to(p.device, p.dtype))
    return {"path": str(ckpt), "global_step": ck.get("global_step"), "ema": used_ema}


def find_run_dirs(outputs: str | Path, exp_prefix: str, overrides: list[str] | None = None,
                  max_steps: int | None = None) -> dict[int, Path]:
    """Session k -> newest run dir outputs/<time>_<exp_prefix>_s<k> (what `ls -d ... | tail -1` picks).
    overrides: ["k=path", ...] to point a session at a specific run dir. With max_steps, a session counts only once
    its final step checkpoint exists: Lightning rewrites last.ckpt every save, so a session still training would
    otherwise pass as finished."""
    runs = {}
    for p in sorted(Path(outputs).glob(f"*_{exp_prefix}_s*")):
        m = re.fullmatch(rf".*_{re.escape(exp_prefix)}_s(\d+)", p.name)
        if m and (p / "last.ckpt").exists():
            runs[int(m.group(1))] = p
    for o in overrides or []:
        k, v = o.split("=", 1)
        runs[int(k)] = Path(v)
    if max_steps:
        for k, p in list(runs.items()):
            done = max((s for s, _ in step_ckpts(p)), default=0)
            if done < max_steps:
                log(f"session {k}: {p} has reached step {done} < {max_steps}, treated as unfinished and left out "
                    "(--allow_incomplete to include it)")
                del runs[k]
    return dict(sorted(runs.items()))


def step_ckpts(run_dir: Path) -> list[tuple[int, Path]]:
    """Intermediate checkpoints of one session run, sorted by step (Lightning names them step_step=2500.ckpt)."""
    out = []
    for p in run_dir.glob("*.ckpt"):
        m = re.search(r"step=(\d+)", p.name) or re.fullmatch(r"step_(\d+)\.ckpt", p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def session_path(run_dirs: dict[int, Path], k: int, every: int = 1) -> list[tuple[str, Path]]:
    """Checkpoint path of session k: theta_{k-1} (previous session's last.ckpt), the intermediate checkpoints of
    session k (every `every`-th), then session k's last.ckpt (= theta_k)."""
    path = [(f"S{k - 1}:end", run_dirs[k - 1] / "last.ckpt")] if k > 0 else []
    steps = step_ckpts(run_dirs[k])
    keep = steps[every - 1::every] if every > 1 else steps
    path += [(f"S{k}:{s}", p) for s, p in keep]
    if steps and keep and keep[-1][0] == steps[-1][0]:
        path[-1] = (f"S{k}:end", run_dirs[k] / "last.ckpt")  # last.ckpt holds the final step; use it as theta_k
    else:
        path.append((f"S{k}:end", run_dirs[k] / "last.ckpt"))
    return path


# ----------------------------------------------------------------------------------------------
# Benchmark data: fixed windows of tokens + token-aligned music
# ----------------------------------------------------------------------------------------------
def read_list(p: Path) -> list[str]:
    return [l.strip() for l in Path(p).open() if l.strip()]


def benchmark_root(cfg, benchmark: str) -> Path:
    return Path(cfg.dirs.raw_data) / "cl_sessions" / benchmark


def manifest(cfg, benchmark: str) -> dict:
    return json.loads((benchmark_root(cfg, benchmark) / "manifest.json").read_text())


def parse_meta(name: str) -> dict:
    """AIST++ sequence name -> genre / situation / dancer / music piece (FineDance names give {})."""
    m = AIST_RE.match(name)
    if not m:
        return {}
    g, sit, dancer, mg, mid, ch = m.groups()
    return {"genre": g, "situation": sit, "dancer": dancer, "music": mg + mid, "music_id": int(mid), "chor": ch}


def token_music_index(n_tokens: int, n_music_frames: int) -> list[int]:
    """Music frame used for each VAE token, as in AistPPDataset._process (stream mode): token 0 -> frame 0, token
    i >= 1 -> frame 1 + 4 (i - 1); tokens whose frame is past the music end get no music."""
    idx = [0] if n_music_frames > 0 else []
    for i in range(1, n_tokens):
        f = 1 + (i - 1) * 4
        if f >= n_music_frames:
            break
        idx.append(f)
    return idx


@dataclass
class Pool:
    names: list[str]
    session: np.ndarray  # (N,) session of each window
    start: np.ndarray  # (N,) token offset of the window inside its sequence
    tokens: np.ndarray  # (N, T, C)
    music: np.ndarray  # (N, T, W, F)
    meta: list[dict] = field(default_factory=list)

    def __len__(self):
        return len(self.names)

    def subset(self, idx) -> "Pool":
        idx = np.asarray(idx, dtype=int)
        return Pool([self.names[i] for i in idx], self.session[idx], self.start[idx], self.tokens[idx],
                    self.music[idx], [self.meta[i] for i in idx])

    @staticmethod
    def concat(pools: list["Pool"]) -> "Pool":
        pools = [p for p in pools if len(p)]
        return Pool(sum((p.names for p in pools), []), np.concatenate([p.session for p in pools]),
                    np.concatenate([p.start for p in pools]), np.concatenate([p.tokens for p in pools]),
                    np.concatenate([p.music for p in pools]), sum((p.meta for p in pools), []))


def build_pool(cfg, benchmark: str, sessions, split: str, T: int, stride: int | None = None,
               max_per_session: int | None = None, seed: int = 0) -> Pool:
    """Non-overlapping (stride = T) T-token windows of every sequence of `split` in `sessions`, with the length
    filters of AistPPDataset (train/val: min_length..max_length frames; test: 0..max_length)."""
    stride = stride or T
    man = manifest(cfg, benchmark)
    data_dir = Path(man["feature_dir"]).parent
    tok_dir, mus_dir = data_dir / cfg.data.token_path, data_dir / cfg.data.music_path
    feat_dir = data_dir / cfg.data.feature_path if cfg.data.get("feature_path") else None
    lo = 0 if split == "test" else int(cfg.data.min_length)
    hi = int(cfg.data.max_length)
    rng = np.random.default_rng(seed)
    pools = []
    for s in sessions:
        names, sess, starts, toks, mus, metas, skipped = [], [], [], [], [], [], 0
        for n in read_list(benchmark_root(cfg, benchmark) / f"s{s}" / f"{split}.txt"):
            tp, mp = tok_dir / f"{n}.npy", mus_dir / f"{n}.npy"
            if not (tp.exists() and mp.exists()):
                skipped += 1
                continue
            if feat_dir is not None:
                fp = feat_dir / f"{n}.npy"
                if not fp.exists() or not lo <= np.load(fp, mmap_mode="r").shape[0] <= hi:
                    skipped += 1
                    continue
            music = np.load(mp, mmap_mode="r")
            if not lo <= music.shape[0] <= hi:
                skipped += 1
                continue
            tok = np.load(tp).astype(np.float32)
            idx = token_music_index(len(tok), music.shape[0])
            L = min(len(tok), len(idx))
            for st in range(0, L - T + 1, stride):
                names.append(n)
                sess.append(s)
                starts.append(st)
                toks.append(tok[st:st + T])
                mus.append(np.asarray(music[idx[st:st + T]], dtype=np.float32))
                metas.append(parse_meta(n))
        if max_per_session and len(names) > max_per_session:
            keep = np.sort(rng.choice(len(names), max_per_session, replace=False))
            names, sess, starts = [names[i] for i in keep], [sess[i] for i in keep], [starts[i] for i in keep]
            toks, mus, metas = [toks[i] for i in keep], [mus[i] for i in keep], [metas[i] for i in keep]
        if names:
            pools.append(Pool(names, np.array(sess), np.array(starts), np.stack(toks), np.stack(mus), metas))
        log(f"pool {benchmark} S{s}/{split}: {len(names)} windows of {T} tokens"
            + (f" ({skipped} sequences skipped: missing / out of length range)" if skipped else ""))
    if not pools:
        raise SystemExit(f"no {split} windows for sessions {list(sessions)}: check {tok_dir} and {mus_dir}")
    return Pool.concat(pools)


def noise_bank(n: int, T: int, C: int, R: int, chunk: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Frozen diffusion-forcing noise: R stratified times t in [0, T / chunk) and Gaussian draws per window."""
    t = np.empty((n, R), np.float32)
    eps = np.empty((n, R, T, C), np.float32)
    for i in range(n):
        g = torch.Generator().manual_seed(seed * 1_000_003 + i)
        u = torch.rand(R, generator=g).numpy()
        t[i] = (np.arange(R) + u) / R * (T / chunk)
        eps[i] = torch.randn(R, T, C, generator=g).numpy()
    return t, eps


# ----------------------------------------------------------------------------------------------
# Loss (DiffForcingWanModel.forward in eval mode with explicit noise) and audio trajectories
# ----------------------------------------------------------------------------------------------
def dfw_loss(model, tokens: torch.Tensor, music: torch.Tensor, t: torch.Tensor, eps: torch.Tensor,
             null: bool = False, per_token: bool = False):
    """Per-entry diffusion-forcing loss of DiffForcingWanModel.forward (active chunk + noisy history), with the
    diffusion times t (B,) and noise eps (B, T, C) given instead of sampled. null=True zeroes the music, which is
    the null condition of CFG training. per_token=True also returns, per entry, (mse per token, noise level)."""
    m = model
    B, T, C = tokens.shape
    nl = m._get_noise_levels(tokens.device, T, t)
    noisy = tokens * (1 - nl[..., None]) + nl[..., None] * eps
    feat, noisy_p, eps_p = m.preprocess(tokens), m.preprocess(noisy), m.preprocess(eps)
    ends = [min(T, int(m.chunk_size * float(t[b])) + 1) for b in range(B)]
    ctx = torch.zeros_like(music) if null else music
    ctx = ctx.reshape(B * T, music.shape[2], music.shape[3])
    pred = m.model([noisy_p[b, :, :ends[b]] for b in range(B)], nl * m.time_embedding_scale, ctx, T, y=None)
    losses, tok = [], []
    for b in range(B):
        e = ends[b]
        if m.prediction_type == "vel":
            target = feat[b, :, :e] - eps_p[b, :, :e]
        elif m.prediction_type == "x0":
            target = feat[b, :, :e]
        else:
            target = eps_p[b, :, :e]
        mse = ((pred[b] - target) ** 2).mean(dim=(0, 2, 3))
        a = min(m.chunk_size, e)
        loss = mse[e - a:].mean()
        hist = nl[b, : e - a] > 1e-5
        if hist.any():
            loss = loss + m.history_loss_weight * mse[: e - a][hist].mean()
        losses.append(loss)
        if per_token:
            used = torch.zeros(e, dtype=torch.bool, device=mse.device)
            used[e - a:] = True
            used[: e - a] = hist
            tok.append((mse.detach()[used], nl[b, :e].detach()[used]))
    out = torch.stack(losses)
    return (out, tok) if per_token else out


def _batches(n_entries: int, bs: int):
    for s in range(0, n_entries, bs):
        yield s, min(n_entries, s + bs)


def _entries(idx, R: int):
    """(window, draw) pairs of the windows idx, flattened."""
    idx = np.asarray(idx, dtype=int)
    return np.repeat(idx, R), np.tile(np.arange(R), len(idx))


def _to(x, device, dtype=torch.float32):
    return torch.from_numpy(np.ascontiguousarray(x)).to(device=device, dtype=dtype)


@torch.no_grad()
def pool_losses(model, pool: Pool, bank, device, idx=None, bs: int = 64, null: bool = False,
                bins: tuple = LOSS_BINS) -> dict:
    """Frozen-noise loss of every window (mean over its R draws) and per-token MSE summed per noise-level bin."""
    t_all, eps_all = bank
    R = t_all.shape[1]
    idx = np.arange(len(pool)) if idx is None else np.asarray(idx, dtype=int)
    wi, ri = _entries(idx, R)
    per_entry = np.empty(len(wi))
    bsum, bcnt = np.zeros(len(bins) - 1), np.zeros(len(bins) - 1)
    edges = torch.tensor(bins, device=device)
    for s, e in _batches(len(wi), bs):
        w, r = wi[s:e], ri[s:e]
        loss, tok = dfw_loss(model, _to(pool.tokens[w], device), _to(pool.music[w], device), _to(t_all[w, r], device),
                             _to(eps_all[w, r], device), null=null, per_token=True)
        per_entry[s:e] = loss.double().cpu().numpy()
        mse = torch.cat([a for a, _ in tok])
        lev = torch.cat([b for _, b in tok])
        k = torch.bucketize(lev, edges, right=True) - 1
        bsum += torch.zeros(len(bins) - 1, device=device, dtype=torch.float64).index_add_(0, k, mse.double()).cpu().numpy()
        bcnt += torch.bincount(k, minlength=len(bins) - 1).double().cpu().numpy()
    per_window = per_entry.reshape(len(idx), R).mean(1)
    return {"per_window": per_window, "bin_sum": bsum, "bin_count": bcnt}


@torch.no_grad()
def trajectories(model, pool: Pool, device, mode: str = "mean", bs: int = 8) -> np.ndarray:
    """phi_i = WanModel.music_embedding(music windows) pooled over the causal window W: (N, T, D) float32.
    mode='mean' averages the W frames (CL_TMR_PLAN.md); mode='last' keeps the current frame's embedding."""
    emb = model.model.music_embedding
    out = []
    for s in range(0, len(pool), bs):
        e = emb(_to(pool.music[s:s + bs], device))
        out.append((e.mean(2) if mode == "mean" else e[:, :, -1]).float().cpu().numpy())
    return np.concatenate(out)


def phi_cached(a, model, ckpt: Path, pool: Pool, tag: str) -> np.ndarray:
    """trajectories() at checkpoint ckpt, cached under <out_dir>/cache (keyed by checkpoint file, pool, settings)."""
    key = file_tag(Path(ckpt), a.T, a.pool_mode, a.max_windows, a.benchmark, tag, len(pool), pool.names[0], pool.names[-1])
    path = Path(a.out_dir) / "cache" / f"phi_{key}.npy"
    if path.exists():
        return np.load(path)
    load_weights(model, ckpt)
    phi = trajectories(model, pool, a.device, a.pool_mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, phi)
    return phi


def pca2(X: np.ndarray) -> np.ndarray:
    """2-D PCA scores of the rows of X via the N x N Gram matrix (cheap when the feature dimension is large)."""
    X = X.reshape(len(X), -1).astype(np.float64)
    X = X - X.mean(0)
    lam, V = np.linalg.eigh(X @ X.T)
    return V[:, ::-1][:, :2] * np.sqrt(np.clip(lam[::-1][:2], 0, None))


# ----------------------------------------------------------------------------------------------
# Parameters, blocks, gradients, sketches
# ----------------------------------------------------------------------------------------------
BLOCKS = ("audio_encoder", "cross_attn_kv", "cross_attn_qo", "self_attn", "ffn", "modulation", "time_embed", "io")


def param_block(name: str) -> str | None:
    """Functional block of a DiffForcingWanModel parameter. The top-level music_embedding is unused by the model
    (it conditions through WanModel.music_embedding) and gets no block."""
    if not name.startswith("model."):
        return None
    n = name[len("model."):]
    if n.startswith("music_embedding."):
        return "audio_encoder"
    m = re.match(r"blocks\.\d+\.(\w+)(?:\.(\w+))?", n)
    if m:
        sub, leaf = m.group(1), m.group(2)
        if sub == "cross_attn":
            return "cross_attn_kv" if leaf in ("k", "v", "norm_k") else "cross_attn_qo"
        if sub == "norm3":
            return "cross_attn_qo"
        if sub in ("self_attn", "ffn", "modulation"):
            return sub
        return "io"
    if n.startswith(("time_embedding", "time_projection")):
        return "time_embed"
    if n.startswith("head.modulation"):
        return "modulation"
    return "io"


def layer_of(name: str) -> str:
    m = re.match(r"model\.blocks\.(\d+)\.", name)
    return f"layer{int(m.group(1))}" if m else "embed/head"


@dataclass
class Params:
    names: list[str]
    tensors: list[torch.Tensor]
    blocks: list[str]
    offsets: np.ndarray  # start offset of each tensor in the flat vector

    @property
    def numel(self) -> int:
        return int(self.offsets[-1])

    def flat(self) -> torch.Tensor:
        return torch.cat([p.detach().reshape(-1) for p in self.tensors])

    @torch.no_grad()
    def assign(self, vec: torch.Tensor) -> None:
        for p, a, b in zip(self.tensors, self.offsets[:-1], self.offsets[1:]):
            p.copy_(vec[a:b].view_as(p))

    def block_numel(self) -> dict[str, int]:
        out = {}
        for p, blk in zip(self.tensors, self.blocks):
            out[blk] = out.get(blk, 0) + p.numel()
        return out

    def block_dots(self, a: torch.Tensor, b: torch.Tensor, by: str = "block") -> dict[str, float]:
        """<a, b> restricted to every block (by='block') or transformer layer (by='layer')."""
        sizes = [int(e - s) for s, e in zip(self.offsets[:-1], self.offsets[1:])]
        dots = torch.stack([(x.double() * y.double()).sum() for x, y in zip(a.split(sizes), b.split(sizes))]).cpu()
        out = {}
        for n, blk, d in zip(self.names, self.blocks, dots.tolist()):
            key = blk if by == "block" else layer_of(n)
            out[key] = out.get(key, 0.0) + d
        return out


def model_params(model) -> Params:
    names, tensors, blocks = [], [], []
    for n, p in model.named_parameters():
        blk = param_block(n)
        if blk is None or not p.requires_grad:
            continue
        names.append(n)
        tensors.append(p)
        blocks.append(blk)
    offsets = np.concatenate([[0], np.cumsum([p.numel() for p in tensors])])
    return Params(names, tensors, blocks, offsets)


def flat_grad(loss: torch.Tensor, P: Params) -> torch.Tensor:
    grads = torch.autograd.grad(loss, P.tensors, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(grads, P.tensors)])


def mean_grad(model, P: Params, pool: Pool, bank, device, idx, bs: int = 32, null: bool = False,
              weights=None) -> tuple[torch.Tensor, float]:
    """Gradient of sum_i w_i L_i over windows idx (default w = 1/|idx|), L_i the frozen-noise loss of window i
    (mean over its R draws). Returns (flat gradient, loss value)."""
    t_all, eps_all = bank
    R = t_all.shape[1]
    idx = np.asarray(idx, dtype=int)
    w_win = np.full(len(idx), 1.0 / len(idx)) if weights is None else np.asarray(weights, dtype=np.float64)
    wi, ri = _entries(idx, R)
    w_ent = np.repeat(w_win, R) / R
    g = torch.zeros(P.numel, device=device)
    total = 0.0
    for s, e in _batches(len(wi), bs):
        w, r = wi[s:e], ri[s:e]
        loss = dfw_loss(model, _to(pool.tokens[w], device), _to(pool.music[w], device), _to(t_all[w, r], device),
                        _to(eps_all[w, r], device), null=null)
        obj = (loss * _to(w_ent[s:e], device)).sum()
        g += flat_grad(obj, P)
        total += float(obj)
    return g, total


def per_sample_grads(model, P: Params, pool: Pool, bank, device, idx, fn, null: bool = False,
                     draws: slice = slice(None)) -> None:
    """fn(position, window index, flat gradient of L_i, L_i) for every window i in idx (one backward per window,
    its draws batched). draws selects a subset of the bank's R draws (e.g. two independent halves)."""
    t_all, eps_all = bank
    for pos, i in enumerate(np.asarray(idx, dtype=int)):
        t_i, e_i = t_all[i][draws], eps_all[i][draws]
        w = np.full(len(t_i), i)
        loss = dfw_loss(model, _to(pool.tokens[w], device), _to(pool.music[w], device), _to(t_i, device),
                        _to(e_i, device), null=null).mean()
        fn(pos, i, flat_grad(loss, P), float(loss))


class CountSketch:
    """Linear sketch S: R^p -> R^m (random bucket + sign per coordinate). For any fixed vector x, ||Sx|| = ||x||
    up to O(1/sqrt(m)) relative error, so norms of *differences* of averaged gradients -- the buffer gradient error
    delta(B, w) -- are preserved without storing p-dimensional per-sample gradients."""

    def __init__(self, numel: int, m: int, seed: int, device):
        g = torch.Generator().manual_seed(seed)
        self.m = m
        self.bucket = torch.randint(0, m, (numel,), generator=g).to(device)
        self.sign = (torch.randint(0, 2, (numel,), generator=g, dtype=torch.int8) * 2 - 1).to(device)

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        return torch.zeros(self.m, device=v.device, dtype=torch.float64).index_add_(
            0, self.bucket, (v * self.sign).double())


# ----------------------------------------------------------------------------------------------
# Output helpers
# ----------------------------------------------------------------------------------------------
def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return o


def dump_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(obj), indent=1))
    log(f"wrote {path}")


def savefig(fig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    log(f"wrote {path.with_suffix('.png')}")


def file_tag(*parts) -> str:
    """Short stable hash of checkpoint paths / settings, used for cache file names."""
    h = hashlib.sha1()
    for p in parts:
        if isinstance(p, Path) and p.exists():
            st = p.stat()
            h.update(f"{p.resolve()}:{st.st_size}:{int(st.st_mtime)}".encode())
        else:
            h.update(str(p).encode())
    return h.hexdigest()[:12]


def common_args(ap, need_runs: bool = True):
    """Arguments shared by e1-e4."""
    ap.add_argument("--config", default="configs/cl/ldf_aistpp_genre5_s0.yaml",
                    help="any session's LDF config (model / data paths are identical across sessions)")
    ap.add_argument("--benchmark", default="aistpp_genre5")
    ap.add_argument("--out_dir", type=Path, default=None, help="default: results/evidence/<benchmark>")
    if need_runs:
        ap.add_argument("--outputs", default="outputs", help="where the session run dirs live")
        ap.add_argument("--exp_prefix", default=None, help="default: cl_ldf_<benchmark>")
        ap.add_argument("--run", action="append", default=[], help="k=<run dir> to override session k's run dir")
        ap.add_argument("--allow_incomplete", action="store_true",
                        help="also use sessions whose last step checkpoint is below trainer.max_steps")
    ap.add_argument("--T", type=int, default=60, help="window length in VAE tokens (60 = 240 frames = 8 s)")
    ap.add_argument("--pool_mode", choices=["mean", "last"], default="mean",
                    help="pooling of music_embedding over the causal music window")
    ap.add_argument("--max_windows", type=int, default=None, help="cap on windows per session (default: all)")
    ap.add_argument("--R", type=int, default=2, help="frozen noise draws per window")
    ap.add_argument("--bs", type=int, default=64, help="(window, draw) entries per forward pass")
    ap.add_argument("--budget", type=float, default=0.10, help="replay quota per past session (fraction)")
    ap.add_argument("--bf16_attention", action="store_true", help="use the stock bf16 attention kernels (faster)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap


def finish_args(a):
    a.out_dir = a.out_dir or Path("results/evidence") / a.benchmark
    if hasattr(a, "exp_prefix"):
        a.exp_prefix = a.exp_prefix or f"cl_ldf_{a.benchmark}"
    return a


def runs_of(a, cfg) -> dict[int, Path]:
    """Finished session run dirs (see find_run_dirs)."""
    return find_run_dirs(a.outputs, a.exp_prefix, a.run,
                         max_steps=None if a.allow_incomplete else int(cfg.trainer.max_steps))
