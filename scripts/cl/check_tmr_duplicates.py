#!/usr/bin/env python3
"""
Do dances to the same song give TMR identical trajectories, and does that explain TMR's buffers?

For one session's candidate clips (the same clips the buffer builder sees) this measures:
  same-window pairs   clips of different dances with the same song and the same start offset, vs all other pairs:
                      relative distance of their music trajectories (TMR's phi). ~0 means TMR cannot tell them apart.
  duplicate groups    clips whose music trajectories are within --eps of each other (relative to the mean norm)
  TMR buffer          (--buffer) how many picks sit in duplicate groups, how often the pick is the first clip of its
                      group by name (the OMP tie-break), dances and dancers covered
  motion-aware        the same distances on TMR-dance trajectories (music + denoiser hidden states): same-window pairs
                      should no longer be near zero

  python scripts/cl/check_tmr_duplicates.py --session 0 --ckpt <model after S0> \
      --buffer buffers/aistpp_genre5/tmr/s0.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
from build_replay_buffer import balanced_blocks
from utils.cl_replay import ClipDataset, clip_loader, clip_music_features, clip_response_features, load_model
from utils.initialize import load_config

AIST_RE = re.compile(r"g(\w\w)_s(\w\w)_c\w+_d(\d\d)_m(\w\w\d)_ch(\d\d)")


def meta(name):
    m = AIST_RE.match(name)
    return {"genre": m.group(1), "dancer": m.group(3), "song": m.group(4)} if m else {"genre": "?", "dancer": "?", "song": name}


def rel_distances(x, mask):
    """Pairwise distances of masked, flattened trajectories, relative to the mean trajectory norm."""
    flat = (x * mask[..., None]).reshape(len(x), -1).double()
    d = torch.cdist(flat, flat)
    return d / flat.norm(dim=1).mean().clamp(min=1e-12)


def describe(d, same, other, eps, label):
    iu = torch.triu_indices(len(d), len(d), 1)
    dv, sv, ov = d[iu[0], iu[1]], same[iu[0], iu[1]], other[iu[0], iu[1]]
    q = lambda v: (f"median {v.median():.4f}, 90th pct {v.quantile(0.9):.4f}" if len(v) else "none")
    near = int((dv[sv] < eps).sum()) if sv.any() else 0
    print(f"[{label}] same song + same offset, different dance: {int(sv.sum())} pairs, {q(dv[sv])}; "
          f"{near} below eps={eps}")
    print(f"[{label}] all other pairs                         : {int(ov.sum())} pairs, {q(dv[ov])}")
    return dv[sv].median().item() if sv.any() else float("nan"), dv[ov].median().item()


def groups_below(d, eps):
    parent = list(range(len(d)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    ii, jj = torch.nonzero(torch.triu(d < eps, 1), as_tuple=True)
    for i, j in zip(ii.tolist(), jj.tolist()):
        parent[find(i)] = find(j)
    g = collections.defaultdict(list)
    for i in range(len(d)):
        g[find(i)].append(i)
    return list(g.values())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="aistpp_genre5")
    ap.add_argument("--session", type=int, required=True)
    ap.add_argument("--ckpt", required=True, help="model after this session (the one the buffer was built with)")
    ap.add_argument("--buffer", default=None, help="a TMR buffer of this session to analyse")
    ap.add_argument("--eps", type=float, default=0.02, help="near-duplicate threshold (relative distance)")
    ap.add_argument("--override", action="append", default=[], help="config overrides key=value (repeatable)")
    a = ap.parse_args()

    cfg = load_config(f"configs/cl/ldf_{a.benchmark}_s{a.session}.yaml", dict(o.split("=", 1) for o in a.override))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cands = ClipDataset(cfg.config, [str(cfg.data.train_meta_paths[0])])
    clips = cands.clips
    N = len(clips)
    md = [meta(c["name"]) for c in clips]
    print(f"session {a.session}: {N} candidate clips, {len({c['name'] for c in clips})} dances, "
          f"{len({m['song'] for m in md})} songs, {len({m['dancer'] for m in md})} dancers")

    model = load_model(cfg, a.ckpt, device)
    loader = clip_loader(cands, batch_size=16)
    music, mask, _ = clip_music_features(model, loader, device)
    response = clip_response_features(model, loader, device)
    T = max(music.shape[1], response.shape[1])
    pad = lambda x: torch.nn.functional.pad(x, (0, 0, 0, T - x.shape[1]))
    music, response = pad(music), pad(response)
    mask = torch.nn.functional.pad(mask.float(), (0, T - mask.shape[1]))

    song = np.array([m["song"] for m in md])
    start = np.array([c["start"] for c in clips])
    name = np.array([c["name"] for c in clips])
    same = torch.from_numpy((song[:, None] == song[None, :]) & (start[:, None] == start[None, :])
                            & (name[:, None] != name[None, :]))
    other = torch.from_numpy(~same.numpy() & (name[:, None] != name[None, :]))

    d_music = rel_distances(music.double(), mask)
    s_med, o_med = describe(d_music, same, other, a.eps, "music trajectories (TMR)")
    groups = groups_below(d_music, a.eps)
    sizes = sorted((len(g) for g in groups), reverse=True)
    dup = sum(s for s in sizes if s > 1)
    print(f"[music trajectories (TMR)] {len(groups)} distinct trajectories among {N} clips; {dup} clips "
          f"({dup / N:.0%}) have a near-duplicate; largest groups {sizes[:8]}")
    if s_med < 0.1 * o_med:
        print("VERDICT: dances to the same song give (near-)identical music trajectories -> TMR cannot tell them apart")
    else:
        print("VERDICT: same-song dances are NOT near-identical in music space -> the duplicate explanation does not hold")

    joint = balanced_blocks([music, response], mask.bool())
    d_joint = rel_distances(joint.double(), mask)
    j_med, jo_med = describe(d_joint, same, other, a.eps, "music + response (TMR-dance)")
    print(f"[music + response (TMR-dance)] same-window pairs now at {j_med / max(jo_med, 1e-12):.0%} of the typical "
          f"distance (music only: {s_med / max(o_med, 1e-12):.0%})")

    if a.buffer:
        b = json.loads(open(a.buffer).read())
        key = {(c["name"], c["start"]): i for i, c in enumerate(clips)}
        picks = [key[(c["name"], c["start"])] for c in b["clips"] if (c["name"], c["start"]) in key]
        group_of = {i: g for g in groups for i in g}
        in_dup = sum(len(group_of[i]) > 1 for i in picks)
        first = sum(len(group_of[i]) > 1 and i == min(group_of[i]) for i in picks)
        dancers = collections.Counter(md[i]["dancer"] for i in picks)
        pool_dancers = collections.Counter(m["dancer"] for m in md)
        print(f"[buffer {a.buffer}] {len(picks)} clips, {len({clips[i]['name'] for i in picks})} dances; "
              f"{in_dup} picks have near-duplicates, {first} of them are the first clip of their group by name "
              f"(the OMP tie-break)")
        print(f"[buffer] dancers picked {dict(sorted(dancers.items()))} vs candidate clips per dancer "
              f"{dict(sorted(pool_dancers.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
