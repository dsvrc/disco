"""
Replay for continual learning on DiscoForcing (TMR benchmark, CL_TMR_PLAN.md stage D).

Unit of replay: a clip = a window of up to `clip_len` VAE tokens (default data.train_max_tokens = 76, the training
window, ~10 s) of one training sequence, with its aligned causal music windows. The candidates of a session tile
every training sequence with non-overlapping clips (plus one end-aligned clip when the remainder is at least half a
clip); a buffer keeps `ratio` (10 %) of them, chosen by scripts/build_replay_buffer.py, and is stored as JSON
(clip references + importance weights), not tensors.

Training (train_ldf.py, config block `replay`): session k replays the union of the buffers of sessions 0..k-1.
Every step adds  replay.weight * mean_i(w_i * l_i)  over a buffer batch (w_i = GCR / TMR weights, else 1), plus
the STAR regulariser when replay.star.enable is set.
"""

from __future__ import annotations

import json
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage

from datasets.aistpp import AistPPDataset, collate_fn
from utils.initialize import instantiate


# ----------------------------------------------------------------------------------------------
# Clips
# ----------------------------------------------------------------------------------------------
def tile_clips(length: int, clip_len: int) -> list[tuple[int, int]]:
    """(start, length) windows covering a sequence of `length` tokens."""
    if length <= clip_len:
        return [(0, length)]
    starts = list(range(0, length - clip_len + 1, clip_len))
    if length - (starts[-1] + clip_len) >= clip_len // 2:
        starts.append(length - clip_len)
    return [(s, clip_len) for s in starts]


class ClipDataset(AistPPDataset):
    """Fixed token windows of training sequences with their aligned music (no random cropping).

    clips: [{name, start, length, weight?, session?}]; None -> tile every sequence of the lists (candidates)."""

    def __init__(self, cfg, list_paths, clips=None, clip_len=None):
        self.cfg = cfg
        self.split = "clips"  # not "train": _process takes no random token window
        self.stream_mode = cfg.data.get("stream_mode", False)
        self.file_list = list(list_paths)
        self.min_length = cfg.data.min_length
        self.max_length = cfg.data.max_length
        self.window_length = cfg.data.max_length
        self.random_length = 0
        self.feature_path = cfg.data.get("feature_path", None)
        self.token_path = cfg.data.get("token_path", None)
        self.music_path = cfg.data.get("music_path", None)
        self.is_vae = False
        self.name_filter = None if clips is None else {c["name"] for c in clips}
        self.dataset = self._load_file_list()
        self.by_name = {d["name"]: d for d in self.dataset}
        if clips is None:
            clip_len = clip_len or cfg.data.train_max_tokens
            clips = [{"name": n, "start": s, "length": l}
                     for n in sorted(self.by_name) for s, l in tile_clips(self.by_name[n]["token_length"], clip_len)]
        missing = sorted({c["name"] for c in clips} - set(self.by_name))
        if missing:
            raise FileNotFoundError(f"{len(missing)} clip sequences not loadable (tokens / music missing?): {missing[:5]}")
        self.clips = clips

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        c = self.clips[idx]
        out = self._process(self.by_name[c["name"]])
        s, n = c["start"], c["length"]
        out["token"] = out["token"][s : s + n]
        out["token_length"] = out["token"].shape[0]
        out["music"] = out["music"][s : s + n]
        out["music_length"] = out["music"].shape[0]
        out.pop("feature", None)  # motion features are not used by the diffusion loss
        out.pop("feature_length", None)
        out["weight"] = float(c.get("weight", 1.0))
        out["clip_index"] = idx
        return out


def model_batch(batch, device=None):
    """Collated clip batch -> the diffusion model's input (tokens are the features), optionally on `device`."""
    b = dict(batch, feature=batch["token"], feature_length=batch["token_length"])
    if device is not None:
        b = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in b.items()}
    return b


def clip_loader(dataset, batch_size=16, shuffle=False, num_workers=0, drop_last=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=collate_fn, drop_last=drop_last, persistent_workers=num_workers > 0)


# ----------------------------------------------------------------------------------------------
# Buffers
# ----------------------------------------------------------------------------------------------
def save_buffer(path, clips, **meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**meta, "clips": clips}, indent=1))


def load_buffers(buffer_dir, session):
    """Clips and train lists of the buffers of sessions 0..session-1 (buffer_dir/s{j}.json)."""
    clips, lists = [], []
    for j in range(session):
        p = Path(buffer_dir) / f"s{j}.json"
        if not p.exists():
            raise FileNotFoundError(f"replay buffer of session {j} missing: {p} (scripts/build_replay_buffer.py)")
        b = json.loads(p.read_text())
        clips += [dict(c, session=j) for c in b["clips"]]
        lists.append(b["list_path"])
    return clips, lists


class ReplayStream:
    """Endless shuffled batches of the buffered clips of all earlier sessions."""

    def __init__(self, cfg, buffer_dir, session, batch_size, num_workers=2):
        clips, lists = load_buffers(buffer_dir, session)
        self.dataset = ClipDataset(cfg, lists, clips=clips)
        self.loader = clip_loader(self.dataset, batch_size=min(batch_size, len(self.dataset)), shuffle=True,
                                  num_workers=num_workers, drop_last=True)
        self._it = None
        per_session = {j: sum(c["session"] == j for c in clips) for j in range(session)}
        self.summary = f"{len(clips)} clips from {buffer_dir} (per session: {per_session})"

    def next(self, device):
        if self._it is None:
            self._it = iter(self.loader)
        try:
            batch = next(self._it)
        except StopIteration:
            self._it = iter(self.loader)
            batch = next(self._it)
        return model_batch(batch, device)


# ----------------------------------------------------------------------------------------------
# Models, losses and features used by the buffer builders
# ----------------------------------------------------------------------------------------------
def load_model(cfg, ckpt, device, use_ema=True):
    """Diffusion model of a Lightning checkpoint (EMA weights by default, as evaluated); ckpt None -> fresh init."""
    if ckpt is None:
        torch.manual_seed(cfg.seed)
    model = instantiate(target=cfg.model.target, cfg=None, hfstyle=False, **dict(cfg.model.params))
    if ckpt is not None:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"], strict=True)
        if use_ema and "ema_state" in ck:
            ema = ExponentialMovingAverage(model.parameters(), decay=cfg.model.ema_decay)
            ema.load_state_dict(ck["ema_state"])
            ema.copy_to(model.parameters())
    return model.to(device)


@contextmanager
def fixed_rng(seed):
    """Same seed -> same diffusion time steps, noise and condition-dropout draws (torch CPU/CUDA, numpy, random);
    the global random streams are restored afterwards."""
    np_state, py_state = np.random.get_state(), random.getstate()
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        np.random.seed(seed % 2**32)
        random.seed(seed)
        try:
            yield
        finally:
            np.random.set_state(np_state)
            random.setstate(py_state)


@torch.no_grad()
def clip_losses(model, loader, device, seeds=(0, 1, 2, 3)):
    """Per-clip diffusion loss averaged over fixed noise draws, in eval mode (no condition dropout), so two models
    scored with the same loader and seeds see identical time steps and noise."""
    was_training = model.training
    model.eval()
    out = []
    for batch in loader:
        mb = model_batch(batch, device)
        losses = []
        for s in seeds:
            with fixed_rng(s):
                losses.append(model(mb, per_sample=True)["per_sample"].float())
        out.append(torch.stack(losses).mean(0).cpu())
    model.train(was_training)
    return torch.cat(out)


@torch.no_grad()
def clip_music_features(model, loader, device):
    """Temporal music features of every clip: the WanModel music_embedding output (the cross-attention context,
    assumption A1 of the plan) mean-pooled over the causal window W at the token rate -> (N, T, D), zero-padded
    to the longest clip, its valid mask (N, T), and the clip tokens (N, T, C)."""
    model.eval()
    feats, masks, toks = [], [], []
    for batch in loader:
        music = batch["music"].to(device).float()  # (B, T, W, 35)
        f = model.model.music_embedding(music).mean(2)  # (B, T, D)
        lengths = batch["token_length"]
        T = f.shape[1]
        m = torch.arange(T)[None] < lengths[:, None]
        tok = batch["token"][:, :T].float()
        for i in range(f.shape[0]):
            feats.append(f[i].cpu())
            masks.append(m[i].float())
            toks.append(tok[i])
    T = max(x.shape[0] for x in feats)
    pad = lambda xs: torch.stack([torch.nn.functional.pad(x, (0, 0, 0, T - x.shape[0])) if x.dim() == 2
                                  else torch.nn.functional.pad(x, (0, T - x.shape[0])) for x in xs])
    return pad(feats), pad(masks).bool(), pad(toks)


def pooled_clip_features(feats, mask, tokens):
    """Clip-level features for OCS / GCR (as the reference builders: pooled condition features + latent mean and
    std); each block is scaled to unit mean norm so the 1024-d music block does not swamp the 4-d latent stats."""
    m = mask[..., None].float()
    n = m.sum(1).clamp(min=1)
    music = (feats * m).sum(1) / n
    tok_mean = (tokens * m).sum(1) / n
    tok_std = (((tokens - tok_mean[:, None]) ** 2 * m).sum(1) / n).sqrt()
    blocks = [music, torch.cat([tok_mean, tok_std], 1)]
    return torch.cat([b / (b.norm(dim=1).mean() + 1e-8) for b in blocks], 1)


# ----------------------------------------------------------------------------------------------
# Orthogonal matching pursuit (GCR / TMR)
# ----------------------------------------------------------------------------------------------
def omp_select(A, b, budget, lam=0.5, positive=True, tol=1e-4):
    """Regularised orthogonal matching pursuit, as GCR's `orthogonalmp_reg_parallel` (CRAIG / GradMatch):
    greedily add the column of A (dim x N) most aligned with the residual, refit ridge-regularised weights on the
    selected columns (dropping any that turn negative when positive=True). Returns weights (N,), zero if unselected."""
    A, b = A.double(), b.double()
    AT = A.T
    resid, normb = b.clone(), b.norm()
    idx, x = [], None
    for _ in range(budget):
        if resid.norm() / normb < tol:
            break
        proj = AT @ resid
        j = int(torch.argmax(proj if positive else proj.abs()))
        if j in idx:
            break
        idx.append(j)
        while True:
            Ai = AT[idx]
            x = torch.linalg.solve(Ai @ Ai.T + lam * torch.eye(len(idx), dtype=A.dtype, device=A.device), Ai @ b)
            if not positive or len(idx) == 1 or x.min() >= 0:
                break
            idx.pop(int(torch.argmin(x)))
        resid = b - AT[idx].T @ x
    w = torch.zeros(A.shape[1], dtype=torch.float64, device=A.device)
    if idx:
        w[idx] = x
    return w.float().cpu()


def weights_to_selection(w, budget, rng):
    """OMP weights -> (indices, weights normalised to mean 1); random clips with weight 1 fill an incomplete budget."""
    idx = torch.nonzero(w > 0).view(-1).tolist()
    wts = w[idx].tolist()
    if len(idx) < budget:
        rest = sorted(set(range(len(w))) - set(idx))
        extra = rng.choice(rest, size=budget - len(idx), replace=False).tolist()
        idx += extra
        wts += [1.0] * len(extra)
    s = sum(wts)
    return idx, [v / s * len(wts) for v in wts]


# ----------------------------------------------------------------------------------------------
# STAR
# ----------------------------------------------------------------------------------------------
class STAR:
    """STAR: Stability-Inducing Weight Perturbation for Continual Learning (Eskandar et al., ICLR 2025,
    arXiv:2503.01595; reference code github.com/Gnomy17/STAR_CL, models/utils/star_perturber.py), for a
    diffusion model:

        L_STAR = lam * D(theta, theta + delta) on buffer samples,   delta = argmax_{||delta|| small} D
        delta^(l) = delta_0^(l) + gamma * ||W^(l)|| * g^(l) / ||g^(l)||,  g = grad of D at theta + delta_0
        (delta_0^(l): random direction of norm eps * ||W^(l)||; only weight matrices are perturbed)

    and the gradient of L_STAR is taken at theta + delta (SAM-style), which here falls out of evaluating the model
    with functional_call(params + delta.detach()). Divergence D: the classifier's KL between predictive
    distributions becomes the squared difference between the model's predictions and its own unperturbed
    predictions on identical noisy inputs (the KL of fixed-variance Gaussians), under the model's loss masks.
    Regression has no "correctly classified" filter, so every buffer clip is used."""

    def __init__(self, module, lam=0.1, gamma=0.01, eps=1e-3, steps=1):
        self.lam, self.gamma, self.eps, self.steps = lam, gamma, eps, steps
        self.names = [n for n, p in module.named_parameters() if p.requires_grad and p.dim() > 1 and "weight" in n]

    def loss(self, module, mb):
        params = dict(module.named_parameters())
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())

        def run(override, **kw):
            with fixed_rng(seed):  # identical time steps / noise / dropout in every pass
                if override is None:
                    return module(mb, **kw)
                return functional_call(module, override, (mb,), kw)

        with torch.no_grad():
            ref = [p.detach() for p in run(None, return_prediction=True)["prediction"]]
        delta = {}
        for n in self.names:
            w = params[n].detach()
            z = torch.randn_like(w)
            delta[n] = z * (self.eps * w.norm() / (z.norm() + 1e-12))
        for _ in range(self.steps):
            leaves = {n: (params[n].detach() + delta[n]).requires_grad_(True) for n in self.names}
            div = run(leaves, targets=ref)["total"]
            grads = torch.autograd.grad(div, [leaves[n] for n in self.names], allow_unused=True)
            for n, g in zip(self.names, grads):
                if g is not None:
                    delta[n] = delta[n] + (self.gamma / self.steps) * params[n].detach().norm() * g / (g.norm() + 1e-12)
        perturbed = {n: params[n] + delta[n].detach() for n in self.names}
        return self.lam * run(perturbed, targets=ref)["total"]
