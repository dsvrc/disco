"""
conda activate music2dance && \
python utils/VQPAE_preprocess.py --checkpoint PAE_training/DeepPhase4Audio-main/output/20260111_023242_VQ_PAE-aistpp_1000_eps-8emb/checkpoints/step_epoch=199.ckpt --stride 90
"""

import os
import sys
from functools import partial
from pathlib import Path
import time

import librosa
import numpy as np
import torch
from tqdm import tqdm

# Add PAE path to sys.path
pae_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../PAE_training/DeepPhase4Audio-main/src'))
if pae_src_path not in sys.path:
    sys.path.insert(0, pae_src_path)

# Also add parent directory for "src" imports
pae_parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../PAE_training/DeepPhase4Audio-main'))
if pae_parent_path not in sys.path:
    sys.path.insert(0, pae_parent_path)

# Constants
FPS = 30  ## Motion frame rate
SAMPLE_RATE = 16000  ## VQ-PAE target sample rate
WINDOW_LENGTH = 6.0  ## seconds - VQPAE training window (6 seconds for direct 180 frames output)
TIME_RANGE = 96000  ## 6 seconds @ 16kHz
EMBEDDING_CHANNELS = 8  ## VQ-PAE embedding channels


def load_vqpae_model(checkpoint_path):
    """Load pretrained VQ-PAE model (original, not 30FPS version)"""
    import sys
    
    # Add BOTH PAE paths (parent for 'src' imports, src for 'models' imports)
    pae_parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../PAE_training/DeepPhase4Audio-main'))
    pae_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../PAE_training/DeepPhase4Audio-main/src'))
    
    if pae_parent_path not in sys.path:
        sys.path.insert(0, pae_parent_path)
    if pae_src_path not in sys.path:
        sys.path.insert(0, pae_src_path)
    
    # Backup and clear 'models' module cache to avoid conflict with project's models/ directory
    models_backup = sys.modules.pop('models', None)
    
    try:
        # Import VQ_PAE from PAE's models package
        from models.VQ_PAE import VQ_AE  ## Import from PAE's models
    finally:
        # Restore original project's models module
        if models_backup is not None:
            sys.modules['models'] = models_backup
    
    # Mock config matching training setup
    class Config:
        input_channels = 1
        embedding_channels = EMBEDDING_CHANNELS
        time_range = TIME_RANGE
        window = WINDOW_LENGTH
        use_fft_mlp = False
        vq_only = False  ## Original model uses both VQ and phase
        intermediate_channels = 64
        enc_dilation_rates = [9, 3, 1]
        dec_dilation_rates = [1, 3, 9]
        enc_kernel_sizes = [7, 3, 3]
        dec_kernel_sizes = [3, 3, 7]
        activation = 'elu'
        combiner = 'concat'
        
        class vq:
            @staticmethod
            def to_dict():
                return {
                    'input_dim': EMBEDDING_CHANNELS,
                    'n_codebooks': 9,
                    'codebook_size': 1024,
                    'codebook_dim': 8,
                    'quantizer_dropout': 0.5,
                }
    
    cfg = Config()
    model = VQ_AE(cfg)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Remove "model." prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[k[6:]] = v  ## Remove "model." prefix
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)  ## Use strict=False to ignore missing keys
    model.eval()
    return model


def extract_vqpae_features(file_path, model, device='cuda', skip_completed=True, dest_dir="vqpae_feats", 
                           output_window_size=180, stride=90, profile=False):
    """
    Extract VQ-PAE features from audio file using sliding window
    
    Args:
        file_path: Path to audio file
        model: VQ-PAE model
        device: Device to run model on
        skip_completed: Skip if output already exists
        dest_dir: Destination directory
        output_window_size: Output window size in frames @ 30fps (default: 180 = 6 seconds)
        stride: Sliding window stride (not used currently, kept for future)
    
    Returns:
        features: (n_frames, output_window_size, channels) array
        save_path: Path where features will be saved
    """
    os.makedirs(dest_dir, exist_ok=True)
    audio_name = Path(file_path).stem
    save_path = os.path.join(dest_dir, audio_name + ".npy")

    if os.path.exists(save_path) and skip_completed:
        return None, save_path

    try:
        # Load audio at 16kHz (VQ-PAE target rate)
        data, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None, None

    # Calculate number of frames at 30fps
    audio_duration = len(data) / SAMPLE_RATE
    n_frames = int(audio_duration * FPS)

    if n_frames <= 0:
        print(f"Warning: {audio_name} has 0 frames, skipping.")
        return None, None

    t_total_start = time.perf_counter()
    
    # print(f"Processing {audio_name}: {n_frames} frames, audio duration: {audio_duration:.2f}s")
    # print(f"  VQ-PAE window: {WINDOW_LENGTH}s ({TIME_RANGE} samples) -> {output_window_size} frames/window")
    # print(f"  Output: ({n_frames}, {output_window_size}, 16)")
    
    # For output_window_size frames output, VQ-PAE directly outputs this many frames
    # Strategy: For each frame t, extract 6-second window ending at t
    
    # Calculate padding needed for frames without sufficient history
    # For frame t, we need audio history: [t-(output_window_size-1), t] frames @ 30fps
    # = output_window_size frames = 6 seconds = 96000 samples
    # Padding strategy: pad zeros at the beginning for early frames
    pad_start = (output_window_size - 1) * (SAMPLE_RATE // FPS)  ## e.g., 179*533 samples for 180 frames
    
    # Pad at end to reach full n_frames
    total_samples_needed = n_frames * (SAMPLE_RATE // FPS)
    pad_end = max(0, total_samples_needed - len(data))
    
    data_padded = np.pad(data, (pad_start, pad_end), mode='constant')
    # print(f"  Padded audio: {len(data_padded)} samples (pad_start={pad_start} for early frames, pad_end={pad_end})")
    
    # Extract features using sliding window with caching for efficiency
    # Strategy: Process with stride, cache results, then assemble per-frame contexts
    model = model.to(device)
    
    # Step 1: Extract features at stride intervals and cache
    # print(f"  Step 1: Extracting features with stride={stride}")
    cache = {}  ## cache[start_frame] = [180, 16] feature array

    t_step1_start = time.perf_counter()
    n_forward_step1 = 0
    
    with torch.no_grad():
        for start_frame in range(0, n_frames, stride):
            # Extract 6-second window starting from start_frame
            audio_start_pos = start_frame * (SAMPLE_RATE // FPS) + pad_start
            audio_end_pos = audio_start_pos + TIME_RANGE
            
            if audio_end_pos > len(data_padded):
                audio_end_pos = len(data_padded)
                audio_start_pos = max(0, audio_end_pos - TIME_RANGE)
            
            audio_window = data_padded[audio_start_pos:audio_end_pos]
            
            # Ensure exactly TIME_RANGE samples
            if len(audio_window) < TIME_RANGE:
                audio_window = np.pad(audio_window, (0, TIME_RANGE - len(audio_window)), mode='constant')
            elif len(audio_window) > TIME_RANGE:
                audio_window = audio_window[:TIME_RANGE]
            
            # VQ-PAE forward pass
            audio_tensor = torch.from_numpy(audio_window).float().unsqueeze(0).to(device)
            output = model(audio_tensor)
            n_forward_step1 += 1
            
            z_pae = output['z_pae']  ## [1, 8, 96000]
            z_downsampled = torch.nn.functional.adaptive_avg_pool1d(z_pae, output_size=output_window_size)  ## [1, 8, 180]
            
            z_vq_full = output['z_vq']  ## [1, 8, 96000]
            z_vq = torch.nn.functional.adaptive_avg_pool1d(z_vq_full, output_size=output_window_size)  ## [1, 8, 180]
            
            z_concat = torch.cat([z_vq, z_downsampled], dim=1)  ## [1, 16, 180]
            z_concat = z_concat.squeeze(0).permute(1, 0).cpu().numpy()  ## [180, 16]
            
            cache[start_frame] = z_concat
            
            # if start_frame % 300 == 0:
            #     print(f"    Cached window starting at frame {start_frame}/{n_frames}")
    
    t_step1_end = time.perf_counter()

    # Step 2: Assemble per-frame contexts from cache
    # print(f"  Step 2: Assembling per-frame contexts from cache")
    features_list = []

    t_step2_start = time.perf_counter()
    n_forward_recompute = 0
    
    for frame_idx in range(n_frames):
        # Find which cached window(s) cover this frame's context [frame_idx-179, frame_idx]
        context_start = max(0, frame_idx - output_window_size + 1)
        context_end = frame_idx + 1
        
        # Find the cached window that best covers this context
        # Use the window whose start_frame <= context_start and is closest
        best_cache_start = None
        for cache_start in sorted(cache.keys(), reverse=True):
            if cache_start <= context_start:
                best_cache_start = cache_start
                break
        
        if best_cache_start is None:
            best_cache_start = 0  ## Use first cached window
        
        # Extract the relevant portion from cache
        offset_in_cache = context_start - best_cache_start
        end_offset_in_cache = offset_in_cache + output_window_size
        
        if end_offset_in_cache <= output_window_size:
            # Context fully within one cached window
            frame_feature = cache[best_cache_start][offset_in_cache:end_offset_in_cache]
        else:
            # Context spans multiple cached windows - need to concatenate
            # For simplicity, re-compute this frame (rare case at boundaries)
            audio_end_pos = (frame_idx + 1) * (SAMPLE_RATE // FPS) + pad_start
            audio_start_pos = audio_end_pos - TIME_RANGE
            
            if audio_start_pos < 0:
                audio_start_pos = 0
                audio_end_pos = TIME_RANGE
            
            audio_window = data_padded[audio_start_pos:audio_end_pos]
            if len(audio_window) < TIME_RANGE:
                audio_window = np.pad(audio_window, (0, TIME_RANGE - len(audio_window)), mode='constant')
            elif len(audio_window) > TIME_RANGE:
                audio_window = audio_window[:TIME_RANGE]
            
            with torch.no_grad():
                audio_tensor = torch.from_numpy(audio_window).float().unsqueeze(0).to(device)
                output = model(audio_tensor)
                n_forward_recompute += 1
                
                z_pae = output['z_pae']
                z_downsampled = torch.nn.functional.adaptive_avg_pool1d(z_pae, output_size=output_window_size)
                
                z_vq_full = output['z_vq']
                z_vq = torch.nn.functional.adaptive_avg_pool1d(z_vq_full, output_size=output_window_size)
                
                z_concat = torch.cat([z_vq, z_downsampled], dim=1)
                frame_feature = z_concat.squeeze(0).permute(1, 0).cpu().numpy()
        
        features_list.append(frame_feature)
    
    t_step2_end = time.perf_counter()

    # Stack all features: (n_frames, 180, 16)
    features = np.stack(features_list, axis=0)
    # print(f"  Final shape: {features.shape} (n_frames, window_size, channels)")

    t_total_end = time.perf_counter()

    if profile:
        total_s = t_total_end - t_total_start
        step1_s = t_step1_end - t_step1_start
        step2_s = t_step2_end - t_step2_start
        total_forward = n_forward_step1 + n_forward_recompute
        ms_per_frame = (total_s / n_frames) * 1000.0
        step1_ms_per_frame = (step1_s / n_frames) * 1000.0
        step2_ms_per_frame = (step2_s / n_frames) * 1000.0
        ms_per_forward = (total_s / max(1, total_forward)) * 1000.0
        print(
            f"[profile] {audio_name}: "
            f"frames={n_frames}, stride={stride}, cached_windows={len(cache)}, recompute={n_forward_recompute}, "
            f"total={total_s:.3f}s ({ms_per_frame:.3f} ms/frame), "
            f"step1={step1_s:.3f}s ({step1_ms_per_frame:.3f} ms/frame), "
            f"step2={step2_s:.3f}s ({step2_ms_per_frame:.3f} ms/frame), "
            f"forward={total_forward} ({ms_per_forward:.3f} ms/forward)"
        )
    
    return features, save_path


def process_audio_directory(src_dir, dest_dir, checkpoint_path, device='cuda', debug=False, 
                           output_window_size=180, stride=90, profile=False):
    """Process all audio files in the directory."""
    file_paths = sorted(Path(src_dir).glob("*.wav"))
    if not file_paths:
        file_paths = sorted(Path(src_dir).glob("*.mp3"))
    
    # print(f"Found {len(file_paths)} audio files")
    
    # Load model once
    # print(f"Loading VQ-PAE model from {checkpoint_path}")
    model = load_vqpae_model(checkpoint_path)
    # print(f"Model loaded successfully")
    
    if debug:
        # Process only first file for debugging
        file_paths = file_paths[:1]
        # print(f"DEBUG MODE: Processing only first file: {file_paths[0]}")
    
    extract_features = partial(
        extract_vqpae_features, 
        model=model, 
        device=device,
        skip_completed=True, 
        dest_dir=dest_dir,
        output_window_size=output_window_size,
        stride=stride,
        profile=profile,
    )
    
    for file_path in tqdm(file_paths, desc="Processing audio files"):
        feats, path = extract_features(file_path)
        if feats is not None:
            np.save(path, feats)
            # print(f"Saved: {path}, shape: {feats.shape}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract VQ-PAE features from audio files.")
    parser.add_argument("--src", default="raw_data/aistpp/wavs", help="Source directory containing audio files")
    parser.add_argument("--dest", default="raw_data/aistpp/vqpae_feats", help="Destination directory for extracted features")
    parser.add_argument("--checkpoint", default="PAE_training/DeepPhase4Audio-main/output/20260111_023242_VQ_PAE-aistpp_1000_eps-8emb/checkpoints/step_epoch=199.ckpt", 
                        help="Path to VQ-PAE checkpoint")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to run model on")
    parser.add_argument("--debug", action="store_true", help="Debug mode: process only first file")
    parser.add_argument("--output_window_size", type=int, default=180, 
                        help="Output window size in frames @ 30fps (default: 180 = 6 seconds)")
    parser.add_argument("--stride", type=int, default=90,
                        help="Sliding window stride in frames (default: 90 = 3 seconds, currently not used)")
    parser.add_argument("--profile", action="store_true", help="Print per-frame timing (ms/frame)")
    args = parser.parse_args()

    process_audio_directory(args.src, args.dest, args.checkpoint, device=args.device, debug=args.debug,
                           output_window_size=args.output_window_size, stride=args.stride, profile=args.profile)


# python utils/VQPAE_preprocess.py \
#   --src tmp/test_wav \
#   --dest tmp/test_vqpae_feats \
#   --checkpoint PAE_training/DeepPhase4Audio-main/output/20260111_023242_VQ_PAE-aistpp_1000_eps-8emb/checkpoints/step_epoch=199.ckpt \
#   --device cuda \
#   --stride 90 \
#   --profile