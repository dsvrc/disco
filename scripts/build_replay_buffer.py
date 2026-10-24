#!/usr/bin/env python3
"""
Build the replay buffer of one continual-learning session (TMR benchmark, DiscoForcing).

Run right after session j has been trained, with that session's config and the model after session j. Candidates
are the session's training clips (utils/cl_replay.py: windows of up to data.train_max_tokens tokens tiling every
training sequence); the buffer keeps --ratio (10 %) of them. Methods (adapted from the GestureLSM buffer builders):

  random  uniform sample (ER)
  ocs     coreset by Similarity + Diversity of pooled clip features, scored per minibatch (OCS-style)
  gcr     GCR, feature-based: OMP on pooled clip features towards their sum; OMP weights weight the replay loss
  tmr     TMR (ours), paper Algorithm 1 via scripts/evidence/tmr_select.py: temporal music trajectories phi_i
          (WanModel.music_embedding mean-pooled over the causal window, at the token rate) -> temporal operator ->
          M modes by PPCA BIC -> whitened modal atoms -> simplex-constrained OMP; its weights weight the replay loss
  tmgm    the same idea as built by the earlier TMGM builder (build_replay_buffer_tmgm.py), on the same trajectories:
          its own BIC, whitening, random projection to proj_dim, regularised OMP towards the sum of the normalised
          atoms, random fill to the budget; its weights weight the replay loss
  tmr_dance  TMR for dance models: the same core (temporal operator, BIC rank, whitened modal atoms, simplex OMP),
          with three changes for music-to-dance data, where many dances share one song (AIST++: ~19 per song) so music
          trajectories cannot tell them apart: (1) motion-aware trajectories, the music trajectory next to the
          denoiser's hidden-state trajectory on the clip, each block scaled to equal weight; (2) at most one clip per
          dance (per-sequence OMP quota), filled by k-center coverage if OMP stops early; (3) keep the modes that
          explain 90 % of the temporal variance (tmr's BIC with N_eff = #clips x dims keeps nearly every mode, so
          whitening amplifies noise directions) and uniform replay weights. Pair it with
          distillation replay at training time (replay.distill). Each change has a flag for ablations (--dance_*).
  mir     MIR as a buffer criterion: positive interference (loss increase of a clip after one virtual update on
          a training batch of the same session), averaged over several virtual updates per clip
  csrel   CSReL (reducible loss): t_out rounds; each round trains the pre-session model on the current coreset
          and adds the clips with the largest  loss(theta_S) - loss(theta_D)

Losses are compared on identical diffusion time steps and noise (fixed seeds, eval mode). Output:
buffers/<benchmark>/<method>/s{j}.json, read by train_ldf.py through replay.buffer_dir.

  python scripts/build_replay_buffer.py --method tmr --config configs/cl/ldf_aistpp_genre5_s0.yaml \
      --ckpt <session-0 last.ckpt> --session 0 --out buffers/aistpp_genre5/tmr/s0.json
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import Subset

sys.path.insert(0, os.getcwd())
from utils.cl_replay import (ClipDataset, clip_loader, clip_losses, clip_music_features, clip_response_features,
                             fixed_rng, load_model, model_batch, omp_select, pooled_clip_features, save_buffer,
                             weights_to_selection)
from utils.initialize import load_config


# ----------------------------------------------------------------------------------------------
# Selection methods: return (indices into the candidates, per-clip weights, per-clip score or None)
# ----------------------------------------------------------------------------------------------
def select_random(N, budget, rng, **_):
    return sorted(rng.choice(N, budget, replace=False).tolist()), None, None


def select_ocs(feats, budget, rng, batch=128, **_):
    """Similarity to the minibatch centroid + diversity (negative mean similarity to the rest of the minibatch),
    on L2-normalised features, minibatches of `batch` shuffled clips; top-`budget` by score."""
    f = torch.nn.functional.normalize(feats, dim=1)
    score = torch.zeros(len(f))
    order = torch.from_numpy(rng.permutation(len(f)))
    for chunk in order.split(batch):
        x = f[chunk]
        S = x @ torch.nn.functional.normalize(x.mean(0, keepdim=True), dim=1).T
        if len(x) > 1:
            sim = x @ x.T
            V = -(sim.sum(1) - sim.diagonal()) / (len(x) - 1)
        else:
            V = torch.zeros(len(x))
        score[chunk] = S.view(-1) + V
    return sorted(torch.topk(score, budget).indices.tolist()), None, score


def select_gcr(feats, budget, rng, lam=0.5, **_):
    g = torch.nn.functional.normalize(feats, dim=1)
    w = omp_select(g.T, g.sum(0), budget, lam=lam)
    idx, wts = weights_to_selection(w, budget, rng)
    return idx, wts, None


def temporal_operator_masked(phi, mask):
    """C = 1/(N D) sum_i phi~_i phi~_i^T (T x T) for zero-padded clips: every entry (t, s) is averaged over the clips
    valid at both t and s (identical to tmr_select.temporal_operator when no clip is padded)."""
    g, m = phi.double(), mask.double()
    D = g.shape[2]
    bar = (g * m[..., None]).sum(0) / m.sum(0).clamp(min=1)[:, None]  # mean trajectory (T, D)
    tilde = (g - bar) * m[..., None]
    return torch.einsum("ntd,nsd->ts", tilde, tilde) / ((m.T @ m).clamp(min=1) * D)


def select_tmr(temporal, mask, budget, rng, stop="fw", **_):
    """TMR, paper Algorithm 1 (OMP variant), from scripts/evidence/tmr_select.py: temporal operator of the uncentred
    trajectories phi_i, rank M by the PPCA BIC (n_eff = N D), whitened modal atoms gamma_i = Lambda^{-1/2} V^T phi_i,
    simplex-constrained OMP towards the mean atom; it stops early when no atom lowers the modal error (stop='fw')."""
    sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "evidence"))
    import tmr_select

    phi = (temporal * mask[..., None]).float()
    N, T, D = phi.shape
    lam_, U = tmr_select.eig_desc(temporal_operator_masked(phi, mask).numpy())
    Z, info = tmr_select.modal_dictionary(phi.numpy(), "temporal", ops={"lam": lam_, "U": U})
    idx, w, hist = tmr_select.omp_simplex(Z, budget, stop=stop, trace=True)
    M = info["M"]
    print(f"[tmr] T={T} D={D}: M={M} modes by PPCA BIC ({lam_[:M].sum() / lam_.sum():.1%} of temporal variance); "
          f"OMP picked {len(idx)}/{budget} clips, modal error {hist[0]:.3f} -> {hist[-1]:.3f}")
    if len(idx) < budget:
        print(f"[tmr] stopped early: no further clip lowers the modal error (Algorithm 1); buffer has {len(idx)} clips")
    return idx.tolist(), (np.asarray(w) * len(idx)).tolist(), None


def _pad_to(x, T):
    return torch.nn.functional.pad(x, (0, 0, 0, T - x.shape[1])) if x.dim() == 3 else torch.nn.functional.pad(x, (0, T - x.shape[1]))


def balanced_blocks(blocks, mask):
    """Concatenate trajectory blocks (N, T, D_b) along channels, each scaled to unit mean squared deviation per valid
    token so every block carries the same weight in the temporal operator."""
    m = mask.float()
    out = []
    for g in blocks:
        g = g.float()
        bar = (g * m[..., None]).sum(0) / m.sum(0).clamp(min=1)[:, None]
        dev = (((g - bar) ** 2).sum(-1) * m).sum() / m.sum().clamp(min=1)
        out.append(g / dev.sqrt().clamp(min=1e-12))
    return torch.cat(out, -1) * m[..., None]


def select_tmr_dance(blocks, mask, groups, budget, rng, per_dance=1, rank="energy", energy=0.9, weights="uniform",
                     stop="fw", **_):
    """TMR core (tmr_select: temporal operator, rank, whitened modal atoms, simplex-constrained OMP) on motion-aware
    trajectories, with at most `per_dance` clips per dance (0 = no limit) and k-center fill to the budget.
    rank: energy = fewest modes explaining `energy` of the temporal variance | bic_clips = BIC with N_eff = #clips |
    bic_dims = BIC with N_eff = #clips x channels (tmr's choice, which keeps nearly every mode)."""
    sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "evidence"))
    import tmr_select

    phi = balanced_blocks(blocks, mask)
    N, T, D = phi.shape
    lam_, U = tmr_select.eig_desc(temporal_operator_masked(phi, mask).numpy())
    M_clips, M_dims = tmr_select.bic_rank(lam_, N)[0], tmr_select.bic_rank(lam_, N * D)[0]
    M_energy = int(np.searchsorted(np.cumsum(lam_) / lam_.sum(), energy) + 1)
    M = {"energy": M_energy, "bic_clips": M_clips, "bic_dims": M_dims}[rank]
    Z, _ = tmr_select.modal_dictionary(phi.numpy(), "temporal", ops={"lam": lam_, "U": U}, M=M)
    groups = np.asarray(groups)
    quotas = {int(g): per_dance for g in np.unique(groups)} if per_dance > 0 else None
    idx, w, hist = tmr_select.omp_simplex(Z, budget, groups if quotas else None, quotas, stop=stop, trace=True)
    idx, n_omp = [int(i) for i in idx], len(idx)
    modal_w = dict(zip(idx, (np.asarray(w) * max(len(idx), 1)).tolist()))
    if len(idx) < budget:  # k-center fill: the clip farthest (in modal space) from the buffer, from an unused dance
        Zd = Z.astype(np.float64)
        dmin = np.full(N, np.inf)
        for s in idx:
            dmin = np.minimum(dmin, np.linalg.norm(Zd - Zd[s], axis=1))
        count = {}
        for i in idx:
            count[groups[i]] = count.get(groups[i], 0) + 1
        while len(idx) < budget:
            ok = np.array([i not in modal_w and (per_dance <= 0 or count.get(groups[i], 0) < per_dance) for i in range(N)])
            if not ok.any():
                break
            j = int(np.argmax(np.where(ok, dmin if idx else 0.0, -np.inf)))
            idx.append(j)
            modal_w[j] = 0.0
            count[groups[j]] = count.get(groups[j], 0) + 1
            dmin = np.minimum(dmin, np.linalg.norm(Zd - Zd[j], axis=1))
    wts = [1.0] * len(idx) if weights == "uniform" else [max(modal_w[i], 1e-3) for i in idx]
    s = sum(wts)
    wts = [v / s * len(wts) for v in wts]
    print(f"[tmr_dance] T={T} channels={D} (blocks {[b.shape[-1] for b in blocks]}): M={M} modes by {rank} "
          f"({energy:.0%} energy: {M_energy}, BIC N_eff=N: {M_clips}, BIC N_eff=N*D: {M_dims}; "
          f"{lam_[:M].sum() / lam_.sum():.1%} of temporal variance); OMP picked {n_omp}/{budget}"
          + (f", k-center fill {len(idx) - n_omp}" if len(idx) > n_omp else "")
          + f"; modal error {hist[0]:.3f} -> {hist[-1]:.3f}; {len(set(groups[idx].tolist()))} dances; weights {weights}")
    score = torch.full((N,), float("nan"))
    for i in idx:
        score[i] = modal_w[i]  # stored as each clip's "score": its OMP (simplex) weight, 0 for k-center fill
    return idx, wts, score


def bic_tmgm(lam_, N, T):
    """Number of modes by the BIC of the reference TMGM builder (build_replay_buffer_tmgm.py)."""
    trace_C = lam_.sum()
    total_var = trace_C / T
    bic = []
    for p in range(1, T + 1):
        sigma = torch.clamp((trace_C - lam_[:p].sum()) / T, min=1e-8)
        penalty = math.log(N) * p * float((lam_[p:] ** 2).sum() / total_var)
        bic.append(float(torch.log(sigma)) + penalty)
    return int(np.argmin(bic)) + 1


def select_tmgm(temporal, mask, budget, rng, lam=0.5, proj_dim=4096, seed=0, **_):
    """The reference TMGM builder: temporal covariance -> M by its BIC -> whitened modal coefficients -> random
    projection to proj_dim -> regularised OMP towards the sum of the normalised atoms (random fill to the budget)."""
    N, T, D = temporal.shape
    C = temporal_operator_masked(temporal, mask)
    lam_, U = torch.linalg.eigh(C)
    lam_, U = lam_.flip(0).clamp(min=0), U.flip(1)
    M = bic_tmgm(lam_, N, T)
    if M >= 0.9 * T:
        print(f"[tmgm] WARNING: its BIC keeps M={M} of T={T} modes; whitening then amplifies near-zero modes "
              f"(smallest kept eigenvalue {float(lam_[M - 1]):.2e} vs largest {float(lam_[0]):.2e})")
    beta = torch.einsum("tm,ntd->nmd", U[:, :M], (temporal * mask[..., None]).double())  # modal coefficients
    flat = (beta / (lam_[:M].sqrt() + 1e-8).view(1, M, 1)).reshape(N, M * D)  # whitened
    if flat.shape[1] > proj_dim:
        gen = torch.Generator().manual_seed(seed)
        flat = flat @ (torch.randn(flat.shape[1], proj_dim, generator=gen, dtype=flat.dtype) / math.sqrt(proj_dim))
    flat = torch.nn.functional.normalize(flat, dim=1)
    w = omp_select(flat.T, flat.sum(0), budget, lam=lam)
    idx, wts = weights_to_selection(w, budget, rng)
    print(f"[tmgm] T={T} D={D}: M={M} modes ({float(lam_[:M].sum() / lam_.sum()):.1%} of temporal variance), "
          f"OMP picked {int((w > 0).sum())}/{budget} before the random fill")
    return idx, wts, None


def select_mir(model, cands, train_set, budget, rng, device, lr, cand_size=50, vbatch=16, evals=3, update="adam",
               seeds=(0, 1, 2, 3), **_):
    N = len(cands)
    # every clip is scored under exactly `evals` virtual updates: each pass over a random permutation in chunks
    chunks = [c for _ in range(evals) for c in np.array_split(rng.permutation(N), math.ceil(N / cand_size))]
    n_updates = len(chunks)
    total, count = torch.zeros(N), torch.zeros(N)
    vloader = clip_loader(train_set, batch_size=vbatch, shuffle=True, drop_last=True)
    vit = iter(vloader)
    for u, cand in enumerate(chunks):
        try:
            vb = next(vit)
        except StopIteration:
            vit = iter(vloader)
            vb = next(vit)
        model.train()
        model.zero_grad(set_to_none=True)
        with fixed_rng(10_000 + u):
            model(model_batch(vb, device))["total"].backward()
        future = copy.deepcopy(model)
        with torch.no_grad():  # virtual step: first AdamW step (= lr * sign(g)) or plain SGD
            for p, fp in zip(model.parameters(), future.parameters()):
                if p.grad is not None:
                    fp -= lr * (torch.sign(p.grad) if update == "adam" else p.grad)
        model.zero_grad(set_to_none=True)
        loader = clip_loader(Subset(cands, cand.tolist()), batch_size=16)
        interference = clip_losses(future, loader, device, seeds) - clip_losses(model, loader, device, seeds)
        total[cand] += interference.clamp(min=0)
        count[cand] += 1
        del future
        if u == 0:
            print(f"[mir] interference of the first update: mean {interference.mean():.3e}, "
                  f"positive {int((interference > 0).sum())}/{len(cand)}")
    score = total / count.clamp(min=1)
    print(f"[mir] {n_updates} virtual updates, every clip scored {int(count.min())}-{int(count.max())} times")
    return sorted(torch.topk(score, budget).indices.tolist()), None, score


def select_csrel(model, init_model, cands, budget, rng, device, rounds=5, steps=100, lr=1e-4, seeds=(0, 1, 2, 3), **_):
    N = len(cands)
    all_loader = clip_loader(cands, batch_size=16)
    ref = clip_losses(model, all_loader, device, seeds)  # theta_D: trained on the whole session
    selected, rel_of = [], torch.full((N,), float("nan"))
    per_round = math.ceil(budget / rounds)
    for r in range(rounds):
        sub = copy.deepcopy(init_model)  # theta_S: the pre-session model trained on the current coreset only
        if selected:
            opt = torch.optim.Adam(sub.parameters(), lr=lr)
            loader = clip_loader(Subset(cands, selected), batch_size=min(16, len(selected)), shuffle=True)
            it, step = iter(loader), 0
            sub.train()
            while step < steps:
                try:
                    b = next(it)
                except StopIteration:
                    it = iter(loader)
                    b = next(it)
                opt.zero_grad(set_to_none=True)
                sub(model_batch(b, device))["total"].backward()
                torch.nn.utils.clip_grad_norm_(sub.parameters(), 1.0)
                opt.step()
                step += 1
        remaining = [i for i in range(N) if i not in set(selected)]
        rel = clip_losses(sub, clip_loader(Subset(cands, remaining), batch_size=16), device, seeds) - ref[remaining]
        take = torch.topk(rel, min(per_round, budget - len(selected))).indices.tolist()
        for t in take:
            selected.append(remaining[t])
            rel_of[remaining[t]] = rel[t]
        print(f"[csrel] round {r + 1}/{rounds}: coreset {len(selected)}, top ReL {rel[take[0]]:.4f}, "
              f"median ReL {rel.median():.4f}")
        del sub
        if len(selected) >= budget:
            break
    return sorted(selected), None, rel_of


# ----------------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["random", "ocs", "gcr", "tmr", "tmr_dance", "tmgm", "mir", "csrel"])
    ap.add_argument("--config", required=True, help="config of the session whose buffer is built (its train list)")
    ap.add_argument("--ckpt", help="model after this session (not needed for random)")
    ap.add_argument("--prev_ckpt", default=None, help="csrel: model before this session (default: fresh init)")
    ap.add_argument("--session", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ratio", type=float, default=0.10, help="buffer size as a fraction of the session's clips")
    ap.add_argument("--clip_len", type=int, default=None, help="tokens per clip (default data.train_max_tokens)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--omp_lam", type=float, default=0.5)
    ap.add_argument("--tmr_stop", choices=["fw", "positive"], default="fw", help="tmr: OMP stopping rule (tmr_select)")
    ap.add_argument("--proj_dim", type=int, default=4096, help="tmgm: random projection of the modal coefficients")
    ap.add_argument("--dance_traj", choices=["response", "latent", "music"], default="response",
                    help="tmr_dance trajectories: music + denoiser hidden states | music + VAE latents | music only")
    ap.add_argument("--dance_per_dance", type=int, default=1, help="tmr_dance: max clips per dance (0 = no limit)")
    ap.add_argument("--dance_rank", choices=["energy", "bic_clips", "bic_dims"], default="energy",
                    help="tmr_dance modes: explaining --dance_energy of the temporal variance | BIC with N_eff = #clips"
                         " | BIC with N_eff = #clips x channels (tmr's choice)")
    ap.add_argument("--dance_energy", type=float, default=0.9)
    ap.add_argument("--dance_weights", choices=["uniform", "modal"], default="uniform",
                    help="tmr_dance replay weights: uniform | the OMP simplex weights")
    ap.add_argument("--ocs_batch", type=int, default=128)
    ap.add_argument("--mir_cand", type=int, default=50)
    ap.add_argument("--mir_vbatch", type=int, default=16)
    ap.add_argument("--mir_evals", type=int, default=3, help="virtual updates each clip is scored under (on average)")
    ap.add_argument("--mir_update", choices=["adam", "sgd"], default="adam")
    ap.add_argument("--csrel_rounds", type=int, default=5)
    ap.add_argument("--csrel_steps", type=int, default=100)
    ap.add_argument("--csrel_lr", type=float, default=1e-4)
    ap.add_argument("--override", action="append", default=[], help="config overrides key=value (repeatable)")
    a = ap.parse_args()
    method = a.method

    t0 = time.time()
    cfg = load_config(a.config, dict(o.split("=", 1) for o in a.override))
    list_path = str(cfg.data.train_meta_paths[0])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(a.seed * 1000 + a.session)
    torch.manual_seed(a.seed)
    cands = ClipDataset(cfg.config, [list_path], clip_len=a.clip_len)
    N = len(cands)
    budget = max(1, round(a.ratio * N))
    print(f"session {a.session}: {N} candidate clips from {list_path}, budget {budget} ({a.ratio:.0%})")

    model = None
    if method != "random":
        if not a.ckpt:
            ap.error(f"--ckpt is required for {method}")
        model = load_model(cfg, a.ckpt, device)
    loader = clip_loader(cands, batch_size=16)
    kw = dict(budget=budget, rng=rng, device=device, seed=a.seed)
    if method == "random":
        idx, wts, score = select_random(N, **kw)
    elif method in ("ocs", "gcr"):
        feats = pooled_clip_features(*clip_music_features(model, loader, device))
        idx, wts, score = (select_ocs(feats, batch=a.ocs_batch, **kw) if method == "ocs"
                           else select_gcr(feats, lam=a.omp_lam, **kw))
    elif method == "tmr":
        temporal, mask, _ = clip_music_features(model, loader, device)
        idx, wts, score = select_tmr(temporal, mask, stop=a.tmr_stop, **kw)
    elif method == "tmgm":
        temporal, mask, _ = clip_music_features(model, loader, device)
        idx, wts, score = select_tmgm(temporal, mask, lam=a.omp_lam, proj_dim=a.proj_dim, **kw)
    elif method == "tmr_dance":
        music, mask, tokens = clip_music_features(model, loader, device)
        blocks = [music]
        if a.dance_traj == "response":
            blocks.append(clip_response_features(model, loader, device, seed=a.seed))
        elif a.dance_traj == "latent":
            blocks.append(tokens)
        T = max(b.shape[1] for b in blocks)
        blocks, mask = [_pad_to(b, T) for b in blocks], _pad_to(mask.float(), T).bool()
        dance_id = {n: i for i, n in enumerate(sorted({c["name"] for c in cands.clips}))}
        idx, wts, score = select_tmr_dance(blocks, mask, [dance_id[c["name"]] for c in cands.clips],
                                           per_dance=a.dance_per_dance, rank=a.dance_rank, energy=a.dance_energy,
                                           weights=a.dance_weights, stop=a.tmr_stop, **kw)
    elif method == "mir":
        from datasets.aistpp import AistPPDataset
        train_set = AistPPDataset(cfg.config, split="train")  # random training windows, as in training
        idx, wts, score = select_mir(model, cands, train_set, lr=float(cfg.optimizer.params.lr), cand_size=a.mir_cand,
                                     vbatch=a.mir_vbatch, evals=a.mir_evals, update=a.mir_update, **kw)
    else:
        init_model = load_model(cfg, a.prev_ckpt, device)  # prev_ckpt None -> fresh init (session 0)
        idx, wts, score = select_csrel(model, init_model, cands, rounds=a.csrel_rounds, steps=a.csrel_steps,
                                       lr=a.csrel_lr, **kw)

    wts = wts or [1.0] * len(idx)
    clips = []
    for i, w in zip(idx, wts):
        c = dict(cands.clips[i], weight=float(w))
        if score is not None and not torch.isnan(score[i]):
            c["score"] = float(score[i])
        clips.append(c)
    cand_tokens = sum(c["length"] for c in cands.clips)
    buf_tokens = sum(c["length"] for c in clips)
    save_buffer(a.out, clips, method=method, session=a.session, ratio=a.ratio, list_path=list_path,
                n_candidates=N, budget=budget, candidate_tokens=cand_tokens, buffer_tokens=buf_tokens,
                n_sequences=len({c["name"] for c in clips}), ckpt=a.ckpt, prev_ckpt=a.prev_ckpt, seed=a.seed,
                params={k: v for k, v in vars(a).items() if k.split("_")[0] in ("omp", "proj", "ocs", "tmr", "dance", "mir", "csrel")},
                seconds=round(time.time() - t0, 1))
    w = np.array(wts)
    print(f"saved {a.out}: {len(clips)} clips from {len({c['name'] for c in clips})} sequences, "
          f"{buf_tokens}/{cand_tokens} tokens ({buf_tokens / cand_tokens:.1%}); weights mean {w.mean():.3f} "
          f"min {w.min():.3f} max {w.max():.3f}; {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
