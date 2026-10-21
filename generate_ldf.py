"""
Generate motion from audio using streaming diffusion.

Audio preprocessing is auto-detected from the config's model.params.music_dim:
  music_dim == 16  ->  vqpae   (T, 180, 16)
  music_dim == 35  ->  librosa (T, window_size, 35)
Use --audio_type to override if needed.

Usage:
    # VQ-PAE (auto-detected from config)
    python generate_ldf.py --config configs/stream_272_pae.yaml --audio_root tmp/demo

    # librosa (auto-detected from config)
    python generate_ldf.py --config configs/stream_hg.yaml --audio_paths a.wav b.wav

    # explicit override
    python generate_ldf.py --config configs/stream_272_pae.yaml --audio_type vqpae --audio_paths a.wav b.wav
"""

import argparse
import os
import time
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

# GPU selection: honour CUDA_VISIBLE_DEVICES if already set, otherwise default to "0".
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))

import torch
import numpy as np
from lightning import seed_everything
from torch_ema import ExponentialMovingAverage

from utils.initialize import compare_statedict_and_parameters, instantiate, load_config
from utils.motion_process import StreamMotionRecovery272
from utils.visualize import render_single_video

# Set tokenizers parallelism to false to avoid warnings in multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_model_from_config(config_path: str, override_args: Optional[Dict[str, Any]] = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    # IMPORTANT: pass config_path explicitly to avoid utils.initialize parsing sys.argv
    cfg = load_config(config_path=config_path, override_args=override_args)
    seed_everything(cfg.seed)

    vae = instantiate(
        target=cfg.test_vae.target,
        cfg=None,
        hfstyle=False,
        **cfg.test_vae.params,
    )
    vae_ckpt = torch.load(cfg.test_vae_ckpt, map_location="cpu", weights_only=False)
    if "ema_state" in vae_ckpt:
        vae.load_state_dict(vae_ckpt["state_dict"], strict=True)
        vae_ema = ExponentialMovingAverage(
            vae.parameters(), decay=cfg.test_vae.ema_decay
        )
        vae_ema.load_state_dict(vae_ckpt["ema_state"])
        vae_ema.copy_to(vae.parameters())
        print(f"Loaded VAE model from {cfg.test_vae_ckpt} with EMA")
    else:
        vae.load_state_dict(vae_ckpt["state_dict"], strict=True)
        print(f"Loaded VAE model from {cfg.test_vae_ckpt} w/o EMA")

    compare_statedict_and_parameters(
        state_dict=vae.state_dict(),
        named_parameters=vae.named_parameters(),
        named_buffers=vae.named_buffers(),
    )
    vae.to(device)
    vae.eval()

    # model
    model = instantiate(
        target=cfg.model.target, cfg=None, hfstyle=False, **cfg.model.params
    )
    checkpoint = torch.load(cfg.test_ckpt, map_location="cpu", weights_only=False)

    state_dict = checkpoint["state_dict"]
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        # Compatibility: some older checkpoints may not have the top-level
        # DiffForcingWanModel.music_embedding.* params.
        msg = str(e)
        expected_missing = (
            "music_embedding.0.weight",
            "music_embedding.0.bias",
            "music_embedding.2.weight",
            "music_embedding.2.bias",
        )
        if all(k in msg for k in expected_missing):
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(
                "[warn] Loaded model with strict=False due to missing music_embedding.* keys "
                f"(likely ckpt/model version mismatch). Missing={len(missing)}, Unexpected={len(unexpected)}"
            )
        else:
            raise

    if "ema_state" in checkpoint:
        ema = ExponentialMovingAverage(model.parameters(), decay=cfg.model.ema_decay)
        try:
            ema.load_state_dict(checkpoint["ema_state"])
            ema.copy_to(model.parameters())
            print(f"Loaded model from {cfg.test_ckpt} with EMA")
        except ValueError as e:
            # EMA state is sensitive to exact parameter ordering/count.
            # If checkpoint/model mismatch (e.g., strict=False load above), fall back to non-EMA.
            print(f"[warn] EMA state incompatible, falling back to non-EMA weights: {e}")
    else:
        print(f"Loaded model from {cfg.test_ckpt} w/o EMA")

    compare_statedict_and_parameters(
        state_dict=model.state_dict(),
        named_parameters=model.named_parameters(),
        named_buffers=model.named_buffers(),
    )
    model.to(device)
    model.eval()

    return cfg, vae, model


def _infer_audio_type(cfg) -> str:
    """Infer audio preprocessing type from config's model.params.music_dim.
    music_dim==16 -> vqpae, music_dim==35 (default) -> librosa.
    """
    music_dim = cfg.get("model.params.music_dim", 35)
    if int(music_dim) == 16:
        return "vqpae"
    return "librosa"


def _parse_override_list(overrides: Optional[List[str]]) -> Optional[Dict[str, Any]]:
    if not overrides:
        return None
    out: Dict[str, Any] = {}
    for override in overrides:
        key, value = override.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _collect_wavs(audio_root: str) -> List[str]:
    root = Path(audio_root)
    if not root.exists():
        raise FileNotFoundError(f"audio_root does not exist: {audio_root}")
    wavs = list(root.rglob("*.wav")) + list(root.rglob("*.WAV"))
    wavs = [str(p) for p in wavs if p.is_file()]
    wavs.sort()
    return wavs


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate motion from wav(s) using streaming diffusion.")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument(
        "--audio_type",
        type=str,
        default=None,
        choices=["vqpae", "librosa"],
        help="Audio preprocessing backend. Auto-detected from config music_dim if omitted.",
    )
    parser.add_argument(
        "--override",
        type=str,
        action="append",
        default=None,
        help="Override config values (key=value). Can be repeated.",
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        default=None,
        help="(compat) Single wav path. Equivalent to --audio_paths <audio_path>.",
    )
    parser.add_argument(
        "--audio_paths",
        type=str,
        nargs="*",
        default=None,
        help="One or more wav paths. If omitted, will scan --audio_root recursively for wav files.",
    )
    parser.add_argument(
        "--audio_root",
        type=str,
        default="tmp/demo",
        help="Root dir to scan recursively for wav files when --audio_paths not provided.",
    )
    parser.add_argument(
        "--vqpae_ckpt",
        type=str,
        default="PAE_training/DeepPhase4Audio-main/output/20260111_023242_VQ_PAE-aistpp_1000_eps-8emb/checkpoints/step_epoch=199.ckpt",
        help="VQ-PAE checkpoint path (used when --audio_type vqpae).",
    )
    parser.add_argument(
        "--librosa_window",
        type=int,
        default=64,
        help="Librosa sliding window size (used when --audio_type librosa).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tmp/output",
        help="Directory to save outputs (mp4 + npy).",
    )
    parser.add_argument(
        "--concat",
        action="store_true",
        help="If set and multiple --audio_paths are provided, concatenate audio features and generate one 'concatenated' output.",
    )
    return parser.parse_args()


def _init_smpl_store() -> Dict[str, List[np.ndarray]]:
    # Per-frame SMPL-related outputs from StreamMotionRecovery272.process_frame()
    return {"pose_aa": [], "root_rot": [], "root_trans": [], "joints": []}


def _append_smpl_store(store: Dict[str, List[np.ndarray]], frame_out: Any):
    """
    frame_out is expected to be the dict returned by StreamMotionRecovery272.process_frame()
    when enable_smpl=True (default).
    """
    if isinstance(frame_out, dict):
        store["pose_aa"].append(frame_out["pose_aa"])
        store["root_rot"].append(frame_out["root_rot"])
        store["root_trans"].append(frame_out["root_trans"])
        store["joints"].append(frame_out["joints"])
    else:
        # Fallback: if enable_smpl=False, process_frame returns joints only.
        store["joints"].append(frame_out)


# ---------------------------------------------------------------------------
# Audio feature extraction helpers
# ---------------------------------------------------------------------------

def _extract_vqpae(audio_paths: List[str], vqpae_ckpt: str, device: str) -> Optional[np.ndarray]:
    """Extract VQ-PAE features from wav files. Returns (T, 180, 16) or None."""
    from utils.VQPAE_preprocess import load_vqpae_model, extract_vqpae_features

    print(f"Loading VQ-PAE model from {vqpae_ckpt}")
    vqpae_model = load_vqpae_model(vqpae_ckpt)
    vqpae_model.to(device)
    vqpae_model.eval()
    print("VQ-PAE model loaded")

    all_features = []
    for idx, audio_path in enumerate(audio_paths):
        print(f"  [{idx+1}/{len(audio_paths)}] Processing: {audio_path}")
        start_time = time.time()
        features, _ = extract_vqpae_features(
            file_path=audio_path,
            model=vqpae_model,
            device=device,
            skip_completed=False,
            dest_dir="tmp",
            output_window_size=180,
            stride=90,
        )
        if features is not None:
            print(f"    Extracted: {features.shape} in {time.time()-start_time:.2f}s")
            all_features.append(features)

    if not all_features:
        return None
    return all_features[0] if len(all_features) == 1 else np.concatenate(all_features, axis=0)


def _extract_librosa(audio_paths: List[str], window_size: int) -> Optional[np.ndarray]:
    """Extract librosa features from wav files. Returns (T, window_size, 35) or None."""
    from utils.librosa_preprocess import extract_audio_features

    all_features = []
    for idx, audio_path in enumerate(audio_paths):
        print(f"  [{idx+1}/{len(audio_paths)}] Processing: {audio_path}")
        start_time = time.time()
        features, _ = extract_audio_features(
            file_path=audio_path,
            skip_completed=False,
            dest_dir="tmp",
            slice_length=window_size,
        )
        if features is not None:
            print(f"    Extracted: {features.shape} in {time.time()-start_time:.2f}s")
            all_features.append(features)

    if not all_features:
        return None
    return all_features[0] if len(all_features) == 1 else np.concatenate(all_features, axis=0)


if __name__ == "__main__":
    # Ensure tmp directory exists
    os.makedirs("tmp", exist_ok=True)
    args = _parse_args()
    override_args = _parse_override_list(args.override)

    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve input wav list
    if args.audio_paths and len(args.audio_paths) > 0:
        audio_paths = args.audio_paths
    elif args.audio_path:
        audio_paths = [args.audio_path]
    else:
        audio_paths = _collect_wavs(args.audio_root)
    if len(audio_paths) == 0:
        raise RuntimeError(f"No wav files found. audio_root={args.audio_root!r}, audio_paths={args.audio_paths!r}")

    ## IMPORTANT: Load VAE and LDF models FIRST, before importing VQPAE
    ## This is because VQPAE imports modify sys.path and can interfere
    print("Loading VAE and LDF models...")
    cfg, vae, model = load_model_from_config(config_path=args.config, override_args=override_args)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve audio_type: explicit flag > auto-detect from config music_dim
    audio_type = args.audio_type or _infer_audio_type(cfg)
    print(f"Audio type: {audio_type}")

    # Determine generation batches
    if args.concat and len(audio_paths) > 1:
        batches = [("concatenated", audio_paths)]
    else:
        batches = [(Path(p).stem, [p]) for p in audio_paths]

    for batch_idx, (audio_name, batch_audio_paths) in enumerate(batches, start=1):
        print(f"\n=== [{batch_idx}/{len(batches)}] Generating for: {audio_name} ===")

        ## Extract features based on audio_type
        print(f"Extracting {audio_type} features from {len(batch_audio_paths)} audio file(s)")
        if audio_type == "vqpae":
            audio_features = _extract_vqpae(batch_audio_paths, args.vqpae_ckpt, device)
        else:
            audio_features = _extract_librosa(batch_audio_paths, args.librosa_window)

        if audio_features is None:
            print(f"[warn] No features extracted for {audio_name}, skipping.")
            continue

        print(f"Final audio features shape: {audio_features.shape}")

        ## Streaming generation setup
        target_motion_frames = audio_features.shape[0]
        vae_scale_factor = 4
        latent_steps = math.ceil((target_motion_frames - 1) / vae_scale_factor)

        print(f"\nStarting MUSIC-to-MOTION generation...")
        print(f"Target Motion Frames: {target_motion_frames}")
        print(f"VAE Scale Factor: {vae_scale_factor}")
        print(f"Latent Steps: {latent_steps}")

        # streaming generate step
        vae.clear_cache()
        model.init_generated(60, batch_size=1)

        first_chunk = True
        stream_decoded_g = []

        # Initialize stream joint recovery
        stream_recovery_no_smooth = StreamMotionRecovery272(joints_num=22, smoothing_alpha=1.0)
        stream_recovery_smooth = StreamMotionRecovery272(joints_num=22, smoothing_alpha=0.5)
        stream_joints_no_smooth = _init_smpl_store()
        stream_joints_smooth = _init_smpl_store()

        scene_mode = "causal"
        print(f"Using Scene Mode: {scene_mode}")

        # Current Motion Frame Index (for audio alignment)
        current_motion_idx = 0

        # Per-step timing
        step_durations_s = []
        diffusion_durations_s = []
        vae_decode_durations_s = []
        total_stream_start = time.perf_counter()

        # We loop over LATENT steps
        for i in range(latent_steps):
            step_start = time.perf_counter()

            center_audio_idx = current_motion_idx + 1
            if center_audio_idx >= target_motion_frames:
                center_audio_idx = target_motion_frames - 1

            # Prepare context: (1, ...)
            audio_ctx = audio_features[center_audio_idx:center_audio_idx + 1]

            x = {}
            x["music"] = [audio_ctx]

            # Step Model (Generates 1 Latent Frame)
            t_diff_start = time.perf_counter()
            output = model.stream_generate_step(x, first_chunk=first_chunk)
            t_diff_end = time.perf_counter()
            latent_out = output["generated"]

            # Decode (1 Latent Frame -> 4 Motion Frames)
            t_vae_start = time.perf_counter()
            decoded_g = vae.stream_decode(
                latent_out[0][None, :], first_chunk=first_chunk
            )[0]
            t_vae_end = time.perf_counter()

            first_chunk = False
            stream_decoded_g.append(decoded_g)

            # Recover Joints
            decoded_g_np = decoded_g.cpu().numpy()

            if decoded_g_np.ndim == 1:
                frame_joints_no_smooth = stream_recovery_no_smooth.process_frame(decoded_g_np)
                frame_joints_smooth = stream_recovery_smooth.process_frame(decoded_g_np)
                _append_smpl_store(stream_joints_no_smooth, frame_joints_no_smooth)
                _append_smpl_store(stream_joints_smooth, frame_joints_smooth)
            else:
                for frame_idx in range(decoded_g_np.shape[0]):
                    frame_data = decoded_g_np[frame_idx]
                    frame_joints_no_smooth = stream_recovery_no_smooth.process_frame(frame_data)
                    frame_joints_smooth = stream_recovery_smooth.process_frame(frame_data)
                    _append_smpl_store(stream_joints_no_smooth, frame_joints_no_smooth)
                    _append_smpl_store(stream_joints_smooth, frame_joints_smooth)

            step_dur_s = time.perf_counter() - step_start
            diff_dur_s = t_diff_end - t_diff_start
            vae_dur_s = t_vae_end - t_vae_start
            step_durations_s.append(step_dur_s)
            diffusion_durations_s.append(diff_dur_s)
            vae_decode_durations_s.append(vae_dur_s)
            print(
                f"Latent Step {i+1}/{latent_steps}: "
                f"diffusion {diff_dur_s * 1000.0:.3f}ms | "
                f"vae_decode {vae_dur_s * 1000.0:.3f}ms | "
                f"total {step_dur_s * 1000.0:.3f}ms | "
                f"Generated {decoded_g.shape[0]} motion frames"
            )
            current_motion_idx += vae_scale_factor

        total_stream_s = time.perf_counter() - total_stream_start
        if step_durations_s:
            avg_step_s = sum(step_durations_s) / len(step_durations_s)
            avg_diff_s = sum(diffusion_durations_s) / len(diffusion_durations_s) if diffusion_durations_s else 0.0
            avg_vae_s = sum(vae_decode_durations_s) / len(vae_decode_durations_s) if vae_decode_durations_s else 0.0
            print(f"Avg latent-step time: {avg_step_s * 1000.0:.3f}ms over {len(step_durations_s)} steps")
            print(f"Avg diffusion time: {avg_diff_s * 1000.0:.3f}ms over {len(step_durations_s)} steps")
            print(f"Avg VAE decode time: {avg_vae_s * 1000.0:.3f}ms over {len(step_durations_s)} steps")
            if len(step_durations_s) > 1:
                avg_wo_warmup_s = sum(step_durations_s[1:]) / (len(step_durations_s) - 1)
                avg_diff_wo_warmup_s = sum(diffusion_durations_s[1:]) / (len(diffusion_durations_s) - 1)
                avg_vae_wo_warmup_s = sum(vae_decode_durations_s[1:]) / (len(vae_decode_durations_s) - 1)
                print(f"Avg latent-step time (excluding first step): {avg_wo_warmup_s * 1000.0:.3f}ms")
                print(f"Avg diffusion time (excluding first step): {avg_diff_wo_warmup_s * 1000.0:.3f}ms")
                print(f"Avg VAE decode time (excluding first step): {avg_vae_wo_warmup_s * 1000.0:.3f}ms")
            print(f"Total streaming loop time: {total_stream_s * 1000.0:.3f}ms")

        vae.clear_cache()
        stream_decoded_g = torch.cat(stream_decoded_g, dim=0)

        print(f"Final generated shape: {stream_decoded_g.shape}")

        # Stack recovered SMPL outputs
        joints_no_smooth = np.stack(stream_joints_no_smooth["joints"], axis=0) if stream_joints_no_smooth["joints"] else np.zeros((0, 22, 3), dtype=np.float32)
        joints_smooth = np.stack(stream_joints_smooth["joints"], axis=0) if stream_joints_smooth["joints"] else np.zeros((0, 22, 3), dtype=np.float32)
        print(f"stream_joints shape: {joints_no_smooth.shape}")

        # Save results
        motion_272 = stream_decoded_g.cpu().numpy()
        music_path = batch_audio_paths[0]
        render_single_video(
            motion=motion_272,
            save_path=os.path.join(args.output_dir, f"{audio_name}_motion.mp4"),
            dim=272,
            render_setting={"music_path": music_path, "music_name": audio_name},
        )

        # Save 272-dim motion as npy
        np.save(os.path.join(args.output_dir, f"{audio_name}_motion.npy"), motion_272)

        # Keep existing joints outputs as pure (T,22,3) arrays
        np.save(os.path.join(args.output_dir, f"{audio_name}_joints_no_smooth.npy"), joints_no_smooth)
        np.save(os.path.join(args.output_dir, f"{audio_name}_joints_smooth.npy"), joints_smooth)

        # Save full SMPL params (pose_aa/root_rot/root_trans/joints)
        if stream_joints_no_smooth["pose_aa"]:
            np.savez(
                os.path.join(args.output_dir, f"{audio_name}_smpl_no_smooth.npz"),
                pose_aa=np.stack(stream_joints_no_smooth["pose_aa"], axis=0),
                root_rot=np.stack(stream_joints_no_smooth["root_rot"], axis=0),
                root_trans=np.stack(stream_joints_no_smooth["root_trans"], axis=0),
                joints=joints_no_smooth,
            )
        if stream_joints_smooth["pose_aa"]:
            np.savez(
                os.path.join(args.output_dir, f"{audio_name}_smpl_smooth.npz"),
                pose_aa=np.stack(stream_joints_smooth["pose_aa"], axis=0),
                root_rot=np.stack(stream_joints_smooth["root_rot"], axis=0),
                root_trans=np.stack(stream_joints_smooth["root_trans"], axis=0),
                joints=joints_smooth,
            )
        print(f"Saved outputs to {args.output_dir} ({audio_name}_*.mp4/.npy)")
