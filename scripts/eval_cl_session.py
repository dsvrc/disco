#!/usr/bin/env python3
"""
Score one session of a continual-learning benchmark (TMR / DiscoForcing) with the validated style metric.

Two modes:
  * model (default): generate dances for the session's test music with a diffusion checkpoint (EMA weights),
    decode with the frozen VAE and score -> one m_{i,t} entry (session i, checkpoint after session t)
  * --vae_recon: score the frozen VAE's reconstructions of the session's ground truth (no generation); this is
    the ceiling a generator using that VAE can reach, and checks that the VAE preserves each genre's style

Metrics come from metrics.cl_metrics.CLEvaluator: StyleRetention (primary), StyleAcc, StyleFD, FID_k, FID_g.

Examples:
  python scripts/eval_cl_session.py --config configs/cl/ldf_aistpp_genre5_s0.yaml --vae_ckpt <vae.ckpt> \
      --ckpt <ldf.ckpt> --session 0 --samples 3
  python scripts/eval_cl_session.py --config configs/cl/ldf_aistpp_genre5_s0.yaml --vae_ckpt <vae.ckpt> \
      --vae_recon --session 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage

sys.path.insert(0, os.getcwd())
from metrics.cl_metrics import CLEvaluator
from utils.initialize import get_function, instantiate, load_config
from utils.motion_process import convert_motion_to_joints


def load_with_ema(target: str, params: dict, ckpt_path: str, ema_decay: float, device: str):
    model = instantiate(target=target, cfg=None, hfstyle=False, **params)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"], strict=True)
    if "ema_state" in ck:
        ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)
        ema.load_state_dict(ck["ema_state"])
        ema.copy_to(model.parameters())
    return model.to(device).eval()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="the session's LDF config (data / model / VAE settings)")
    ap.add_argument("--vae_ckpt", required=True)
    ap.add_argument("--ckpt", help="LDF checkpoint (model mode)")
    ap.add_argument("--benchmark", default="aistpp_genre5")
    ap.add_argument("--session", type=int, required=True, help="session whose test set is scored")
    ap.add_argument("--samples", type=int, default=3, help="generated samples per test music")
    ap.add_argument("--vae_recon", action="store_true", help="score VAE reconstructions of GT instead")
    ap.add_argument("--untrained", action="store_true",
                    help="randomly initialised generator: the floor that shows the score comes from the generator")
    ap.add_argument("--gt", action="store_true", help="score the real dance on the same windows (reference)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--override", action="append", default=[], help="config overrides key=value (repeatable)")
    ap.add_argument("--max_tokens", type=int, default=None,
                    help="score only the first N tokens (4 frames each) of every test music; streaming generation "
                         "cost grows with length, so e.g. 76 (~10 s) gives fast checkpoint checks")
    a = ap.parse_args()
    if not (a.vae_recon or a.untrained or a.gt or a.ckpt):
        ap.error("--ckpt is required unless --vae_recon, --untrained or --gt")
    a.vae_recon = a.vae_recon or a.gt  # both skip generation

    device = "cuda" if torch.cuda.is_available() else "cpu"
    overrides = dict(o.split("=", 1) for o in a.override)
    cfg = load_config(a.config, {"test_vae_ckpt": a.vae_ckpt, **overrides})
    test_list = f"{cfg.get('dirs.raw_data')}/aistpp/cl_{a.benchmark}_s{a.session}_test.txt"
    OmegaConf.update(cfg.config, "data.test_meta_paths", [test_list])
    dataset = instantiate(cfg.data.target, cfg=cfg.config, split="test")
    if len(dataset) == 0:
        raise SystemExit(f"no test sequences loaded from {test_list}: check data.token_path / music_path "
                         f"({cfg.data.token_path}, {cfg.data.music_path}) exist for this VAE")
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=get_function(cfg.data.collate_fn))

    vae = load_with_ema(cfg.test_vae.target, dict(cfg.test_vae.params), a.vae_ckpt, cfg.test_vae.ema_decay, device)
    if a.vae_recon:
        model = None
    elif a.untrained:
        torch.manual_seed(a.seed)
        model = instantiate(target=cfg.model.target, cfg=None, hfstyle=False, **dict(cfg.model.params)).to(device).eval()
    else:
        model = load_with_ema(cfg.model.target, dict(cfg.model.params), a.ckpt, cfg.model.ema_decay, device)

    joints, t0 = [], time.time()
    for sample in range(1 if a.vae_recon else a.samples):
        torch.manual_seed(a.seed + sample)
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            if a.max_tokens:
                batch["token"] = batch["token"][:, : a.max_tokens]
                batch["token_length"] = batch["token_length"].clamp(max=a.max_tokens)
                if torch.is_tensor(batch.get("music")):
                    batch["music"] = batch["music"][:, : a.max_tokens]
            with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=model is not None):
                if a.gt:  # real dance, cropped to the frames the tokens cover (first token = 1 frame, then 4 each)
                    for i in range(len(batch["name"])):
                        n_frames = 1 + 4 * (int(batch["token_length"][i]) - 1)
                        feat = batch["feature"][i][: min(n_frames, int(batch["feature_length"][i]))]
                        joints.append(convert_motion_to_joints(feat.float().cpu().numpy(), 272))
                    continue
                if model is None:
                    tokens = [batch["token"][i][: batch["token_length"][i]] for i in range(len(batch["name"]))]
                else:
                    mb = dict(batch, feature=batch["token"], feature_length=batch["token_length"])
                    tokens = model.generate(mb)["generated"]
                for i, tok in enumerate(tokens):
                    feat = vae.decode(tok[None].float().to(device))[0].float().cpu().numpy()
                    joints.append(convert_motion_to_joints(feat, 272))
    gen_time = time.time() - t0

    root = Path("raw_data/cl_sessions") / a.benchmark
    metrics = CLEvaluator(root).session_metrics(a.session, joints)
    mode = "gt" if a.gt else "vae_recon" if a.vae_recon else "untrained" if a.untrained else "model"
    metrics.update({"session": a.session, "mode": mode,
                    "ckpt": a.ckpt, "vae_ckpt": a.vae_ckpt, "samples": 1 if a.vae_recon else a.samples,
                    "n_sequences": len(joints), "max_tokens": a.max_tokens, "generation_seconds": round(gen_time, 1),
                    # joints.npz entry q belongs to names[q % len(names)] (sample-major order)
                    "names": [d["name"] for d in dataset.dataset]})
    noise = json.loads((root / "eval" / "noise.json").read_text())[f"s{a.session}"]
    metrics["gt_reference"] = {k: noise[k] for k in ("StyleRetention_gt", "StyleRetention_complete_forgetting",
                                                      "StyleFD_noise_mean", "StyleFD_complete_forgetting")}
    print(json.dumps(metrics, indent=1))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(metrics, indent=1))
        # keep the generated joints so they can be re-scored (e.g. with another style encoder) without regenerating
        np.savez_compressed(a.out.with_suffix(".joints.npz"), **{str(i): j.astype(np.float32) for i, j in enumerate(joints)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
