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

import collections
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

    clips: [{name, start, length, weight?, session?}]; None -> tile every sequence of the lists (candidates).
    drop_missing: silently drop clips whose sequence the loader filters out (length limits) instead of failing."""

    def __init__(self, cfg, list_paths, clips=None, clip_len=None, drop_missing=False):
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
        if missing and drop_missing:
            clips = [c for c in clips if c["name"] in self.by_name]
        elif missing:
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
        out["session"] = int(c.get("session", -1))
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


class SessionBalancedBatches(torch.utils.data.Sampler):
    """Endless batches with the same number of clips from every buffered session (the remainder slots rotate over the
    sessions); each session's clips are drawn without replacement and reshuffled once exhausted."""

    def __init__(self, sessions, batch_size, seed=0):
        self.by_session = {s: [i for i, x in enumerate(sessions) if x == s] for s in sorted(set(sessions))}
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        order = list(self.by_session)
        pools = {s: [] for s in order}
        offset = 0
        while True:
            batch = []
            for k in range(self.batch_size):
                s = order[(offset + k) % len(order)]
                if not pools[s]:
                    pools[s] = self.rng.permutation(self.by_session[s]).tolist()
                batch.append(pools[s].pop())
            offset = (offset + self.batch_size) % len(order)
            yield batch


class ReplayStream:
    """Endless batches of the buffered clips of all earlier sessions: balance='uniform' shuffles the union (sessions
    appear in proportion to their buffer sizes), balance='session' puts equally many clips of every session in each
    batch."""

    def __init__(self, cfg, buffer_dir, session, batch_size, num_workers=2, balance="uniform"):
        clips, lists = load_buffers(buffer_dir, session)
        self.buffer_dir, self.session, self.balance = buffer_dir, session, balance
        self.dataset = ClipDataset(cfg, lists, clips=clips)
        self.batch_size = min(batch_size, len(self.dataset))
        if balance == "session":
            self.loader = DataLoader(self.dataset, batch_sampler=SessionBalancedBatches(
                [c["session"] for c in self.dataset.clips], self.batch_size), num_workers=num_workers,
                collate_fn=collate_fn, persistent_workers=num_workers > 0)
        elif balance == "uniform":
            self.loader = clip_loader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers,
                                      drop_last=True)
        else:
            raise ValueError(f"replay.balance must be 'uniform' or 'session', got {balance}")
        self._it = None
        per_session = {j: sum(c["session"] == j for c in clips) for j in range(session)}
        self.summary = f"{len(clips)} clips from {buffer_dir} (per session: {per_session}), balance={balance}"

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
# Replay diagnostics (read-only: no effect on training randomness or gradients)
# ----------------------------------------------------------------------------------------------
@contextmanager
def preserve_rng():
    """Run diagnostics without moving the training run's random streams (torch CPU / CUDA, numpy, random)."""
    np_state, py_state = np.random.get_state(), random.getstate()
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        try:
            yield
        finally:
            np.random.set_state(np_state)
            random.setstate(py_state)


def _tag(lines):
    """Every line of a diagnostics block starts with [replay-diag] (grep-able even when the progress bar shares the
    line), and the block starts on a fresh line."""
    return "\n" + "\n".join(l if l.startswith("[replay-diag]") else f"[replay-diag] {l}" for l in lines)


def _sequence_lengths(cfg, list_path):
    """Token length of every sequence of a loader list (memory-mapped, no data loading)."""
    d = Path(str(list_path)).parent
    tok_dir = d / cfg.data.token_path
    out = {}
    for n in Path(str(list_path)).read_text().split():
        p = tok_dir / f"{n}.npy"
        if p.exists():
            out[n] = int(np.load(p, mmap_mode="r").shape[0])
    return out


class ReplayDiagnostics:
    """Decisive replay logs for one continual-learning session. Every `every` optimizer steps (and on the first
    `first` steps) it prints a [replay-diag] block:

      batch now       current-session clips vs replayed clips per previous session in this step's batch
      samples so far  cumulative clips seen per session, the current : replay ratio, each previous session's share of
                      the replay against its share of the buffer
      loss            window averages of the current loss and the replay loss per previous session, the replay weights
                      used, and each term's share of the objective (per-sample loss weights)
      gradients       norms of the gradient of every loss term separately and the cosine between the current and the
                      replay gradient (is replay propagating, and does it conflict with the new session?)
      coverage        buffer clips replayed at least once
      probe           fixed-noise diffusion loss (raw weights, same clips and noise every time) on: current-session
                      clips, buffered clips and NON-buffered training clips of every previous session, with the change
                      since the session started. Buffered down + non-buffered up = the buffer is memorised but the
                      session is still forgotten; both flat or down = replay protects the session.
    """

    def __init__(self, cfg, stream: ReplayStream, every=1000, first=3, probe_clips=16, seed=0):
        self.cfg, self.stream, self.every, self.first = cfg, stream, every, first
        self.k = stream.session
        self.weight = float(cfg.replay.get("weight", 1.0))
        self.use_w = bool(cfg.replay.get("use_sample_weights", True))
        self.probe_clips, self.seed = probe_clips, seed
        self.n_cur = 0
        self.n_rep = collections.Counter()
        self.seen = set()
        self.win = collections.defaultdict(list)
        self.probe_sets, self.probe_start = None, None
        clips = stream.dataset.clips
        self.buffer_share = {j: sum(c["session"] == j for c in clips) / len(clips) for j in range(self.k)}

    def due(self, step):
        return step < self.first or step % self.every == 0

    def startup(self):
        clips = self.stream.dataset.clips
        lines = [f"[replay-diag] session S{self.k}: replaying S0..S{self.k - 1} from {Path(self.stream.buffer_dir).resolve()}"]
        problems = []
        for j in range(self.k):
            cj = [c for c in clips if c["session"] == j]
            w = np.array([c.get("weight", 1.0) for c in cj], float)
            zero = int((w <= 1e-8).sum())
            ess = w.sum() ** 2 / max((w ** 2).sum(), 1e-12) if len(w) else 0.0
            lines.append(f"  S{j}: {len(cj)} clips, {sum(c['length'] for c in cj)} tokens, "
                         f"{len({c['name'] for c in cj})} sequences | weights mean {w.mean():.3f} min {w.min():.3f} "
                         f"max {w.max():.3f}, zero {zero}, effective size {ess:.1f}")
            if not cj:
                problems.append(f"S{j} has no clips in the replay set")
            if zero and self.use_w:
                problems.append(f"S{j} has {zero} zero-weight clips: they take replay slots but add no loss")
        bc = int(self.cfg.data.train_bs)
        br = self.stream.batch_size
        lines.append(f"  each step: {bc} current clips (random {self.cfg.data.get('train_max_tokens')}-token windows of S{self.k}) "
                     f"+ {br} replay clips (balance={self.stream.balance}); loss = L_current + {self.weight} * "
                     f"mean_i({'w_i * ' if self.use_w else ''}l_i)")
        mean_w = float(np.mean([c.get("weight", 1.0) for c in clips])) if self.use_w else 1.0
        lines.append(f"  per-sample loss weight: current 1/{bc} = {1 / bc:.4f}, replay {self.weight}*w/{br} = "
                     f"{self.weight / br:.4f} x w  -> current : replay objective weight 1 : {self.weight * mean_w:.2f} "
                     f"(the {self.k} previous session(s) share the replay part)")
        lines += [f"  CHECK FAIL: {p}" for p in problems] or ["  CHECK OK: every previous session is in the replay set"]
        return _tag(lines)

    def record(self, batch, replay_batch, per_sample, weights, loss_cur, star=None, replay_term=None, kd=None):
        """per_sample: ground-truth replay loss per clip; replay_term: the replay part of the objective as optimised
        (ground truth and / or distillation, weighted); kd: mean distillation loss of the batch (if enabled)."""
        self.n_cur += len(batch["name"])
        for s, ci, l, w in zip(replay_batch["session"], replay_batch["clip_index"], per_sample.tolist(), weights.tolist()):
            self.n_rep[int(s)] += 1
            self.seen.add(int(ci))
            self.win[f"rep{int(s)}"].append(l)
            self.win["w"].append(w)
            self.win["wl"].append(w * l)
        self.win["cur"].append(float(loss_cur))
        if replay_term is not None:
            self.win["term"].append(float(replay_term))
        if kd is not None:
            self.win["kd"].append(float(kd))
        if star is not None:
            self.win["star"].append(float(star))

    def _probe_sets(self):
        """Fixed clip sets: current session, and for each previous session its buffered clips and clips of training
        sequences that have no buffered clip at all."""
        rng = np.random.default_rng(self.seed)
        cfg, n = self.cfg, self.probe_clips
        clip_len = int(cfg.data.train_max_tokens)
        clips = self.stream.dataset.clips
        _, lists = load_buffers(self.stream.buffer_dir, self.k)
        sets = {}

        def unbuffered(list_path, exclude):
            lengths = _sequence_lengths(cfg, list_path)
            names = [m for m in sorted(lengths) if m not in exclude]
            pick = rng.permutation(names)[: 2 * n].tolist()
            out = [{"name": m, "start": s, "length": l} for m in pick for s, l in tile_clips(lengths[m], clip_len)]
            return [out[i] for i in sorted(rng.permutation(len(out))[:n])]

        for j in range(self.k):
            cj = [c for c in clips if c["session"] == j]
            sets[f"S{j} buffered"] = (lists[j], [cj[i] for i in sorted(rng.permutation(len(cj))[:n])])
            sets[f"S{j} unbuffered"] = (lists[j], unbuffered(lists[j], {c["name"] for c in cj}))
        cur_list = str(cfg.data.train_meta_paths[0])
        sets[f"S{self.k} current"] = (cur_list, unbuffered(cur_list, set()))
        out = {}
        for name, (lst, cl) in sets.items():
            ds = ClipDataset(cfg, [lst], clips=cl, drop_missing=True)
            if len(ds):
                out[name] = clip_loader(ds, batch_size=8)
        return out

    def probe(self, model, device):
        with preserve_rng():
            if self.probe_sets is None:
                self.probe_sets = self._probe_sets()
            res = {k: float(clip_losses(model, loader, device, seeds=(0, 1)).mean()) for k, loader in self.probe_sets.items()}
        if self.probe_start is None:
            self.probe_start = dict(res)
        return res

    @staticmethod
    def grad_stats(params, terms):
        """Norm of the gradient of every loss term on its own, and the cosine between the current and replay terms."""
        flat = {}
        for name, loss in terms.items():
            g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            flat[name] = torch.cat([(x if x is not None else torch.zeros_like(p)).reshape(-1).float() for x, p in zip(g, params)])
        out = {f"|g_{k}|": float(v.norm()) for k, v in flat.items()}
        if "current" in flat and "replay" in flat:
            out["cos(current,replay)"] = float(torch.nn.functional.cosine_similarity(flat["current"], flat["replay"], dim=0))
        return out

    def report(self, step, replay_batch, grads, probe):
        mean = lambda xs: float(np.mean(xs)) if xs else float("nan")
        k = self.k
        now = collections.Counter(int(s) for s in replay_batch["session"])
        tot_rep = sum(self.n_rep.values())
        lines = [f"[replay-diag] step {step} (S{k}, replaying S0..S{k - 1})"]
        lines.append(f"  batch now      : current S{k} x{int(self.cfg.data.train_bs)} | replay "
                     + ", ".join(f"S{j} x{now.get(j, 0)}" for j in range(k))
                     + f" (clips {sorted(int(c) for c in replay_batch['clip_index'])})")
        share = {j: self.n_rep[j] / max(tot_rep, 1) for j in range(k)}
        dev = max(abs(share[j] - self.buffer_share[j]) for j in range(k)) if tot_rep else 0.0
        verdict = "OK" if tot_rep < 20 * k or dev < 0.05 else "WARN sessions replayed out of proportion to the buffer"
        lines.append(f"  samples so far : current {self.n_cur} | replay {tot_rep} (current : replay = 1 : "
                     f"{tot_rep / max(self.n_cur, 1):.2f}) = " + ", ".join(
                         f"S{j} {self.n_rep[j]} ({share[j]:.1%} vs {self.buffer_share[j]:.1%} of buffer)" for j in range(k))
                     + f" -> {verdict}")
        cur, rep_w = mean(self.win["cur"]), mean(self.win["wl"])
        rep_terms = ", ".join(f"S{j} {mean(self.win[f'rep{j}']):.3f}" for j in range(k))
        obj_cur = cur
        obj_rep = mean(self.win["term"]) if self.win["term"] else self.weight * rep_w
        star = mean(self.win["star"])
        lines.append(f"  loss (window)  : current {cur:.3f} | replay (ground truth) per session {rep_terms} | weights "
                     f"mean {mean(self.win['w']):.3f} | weighted replay {rep_w:.3f}"
                     + (f" | distillation {mean(self.win['kd']):.4f}" if self.win["kd"] else "")
                     + f" | replay term in the objective {obj_rep:.3f}"
                     + (f" | STAR {star:.5f}" if self.win["star"] else ""))
        frac = obj_rep / max(obj_cur + obj_rep, 1e-12)
        lines.append(f"  loss share     : current {1 - frac:.1%} : replay {frac:.1%} of the objective -> "
                     + ("OK" if 0.25 <= frac <= 0.75 else "WARN one term dominates the objective"))
        if grads:
            gc, gr = grads.get("|g_current|", float("nan")), grads.get("|g_replay|", float("nan"))
            kd_zero = bool(self.win["kd"]) and max(self.win["kd"]) < 1e-6
            g_verdict = ("OK distillation is zero until the model drifts from the previous session" if gr < 0.1 * gc
                         and kd_zero else "WARN replay gradient is negligible" if gr < 0.1 * gc else
                         "WARN replay gradient dominates" if gr > 10 * gc else "OK replay gradient propagates")
            lines.append("  gradients      : " + " ".join(f"{k_}={v:.4g}" for k_, v in grads.items())
                         + f" | ratio replay/current {gr / max(gc, 1e-12):.2f} -> {g_verdict}")
        lines.append(f"  coverage       : {len(self.seen)}/{len(self.stream.dataset)} buffer clips replayed at least once")
        if probe:
            parts, verdicts = [], []
            for name, v in probe.items():
                parts.append(f"{name} {v:.3f} (d{v - self.probe_start[name]:+.3f})")
            for j in range(k):
                b, u = probe.get(f"S{j} buffered"), probe.get(f"S{j} unbuffered")
                if b is None or u is None or step == 0:
                    continue
                b0, u0 = self.probe_start[f"S{j} buffered"], self.probe_start[f"S{j} unbuffered"]
                rb, ru = (b - b0) / max(abs(b0), 1e-8), (u - u0) / max(abs(u0), 1e-8)  # relative change
                if ru > 0.03 and rb < 0:
                    verdicts.append(f"S{j} MEMORISING buffer (buffered {rb:+.1%}, unbuffered {ru:+.1%})")
                elif ru > 0.03:
                    verdicts.append(f"S{j} FORGETTING despite replay (buffered {rb:+.1%}, unbuffered {ru:+.1%})")
                else:
                    verdicts.append(f"S{j} retained (buffered {rb:+.1%}, unbuffered {ru:+.1%})")
            lines.append("  probe          : " + " | ".join(parts))
            if verdicts:
                lines.append("  verdict        : " + "; ".join(verdicts))
        self.win.clear()
        return _tag(lines)


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


@torch.no_grad()
def clip_response_features(model, loader, device, seed=0):
    """Motion-response trajectories of every clip: the last transformer block's hidden states (T, dim) while the
    denoiser sees the whole clip with its music (diffusion-forcing time = length / chunk_size: clean history, only the
    last chunk noisy), zero-padded to the longest clip. Unlike the music trajectory, this differs between dances
    performed to the same song."""
    was_training = model.training
    model.eval()
    store = {}
    hook = model.model.blocks[-1].register_forward_hook(lambda mod, inp, out: store.__setitem__("x", out))
    feats = []
    try:
        for batch in loader:
            mb = model_batch(batch, device)
            with fixed_rng(seed):
                model(mb, time_steps=mb["feature_length"].float() / model.chunk_size)
            x = store["x"].float().cpu()  # (B, seq_len, dim)
            for i in range(x.shape[0]):
                feats.append(x[i, : int(batch["token_length"][i])])
    finally:
        hook.remove()
        model.train(was_training)
    T = max(f.shape[0] for f in feats)
    return torch.stack([torch.nn.functional.pad(f, (0, 0, 0, T - f.shape[0])) for f in feats])


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
