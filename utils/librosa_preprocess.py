import os
from functools import partial
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm
import time

# Constants for audio processing
FPS = 30
HOP_LENGTH = 512
SAMPLE_RATE = FPS * HOP_LENGTH

def extract_audio_features(
    file_path,
    skip_completed=True,
    dest_dir="baseline_feats",
    slice_length=32,
    profile=False,
):
    """Extract audio features from a file and save them to a directory."""
    os.makedirs(dest_dir, exist_ok=True)
    audio_name = Path(file_path).stem
    save_path = os.path.join(dest_dir, audio_name + ".npy")

    if os.path.exists(save_path) and skip_completed:
        return None, save_path

    try:
        data, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None, None

    # Get reference number of frames from the full audio
    ref_envelope = librosa.onset.onset_strength(y=data, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    n_frames = len(ref_envelope)

    if n_frames <= 0:
        print(f"Warning: {audio_name} has 0 frames, skipping.")
        return None, None

    t_total_start = time.perf_counter()
    t_vec_start = time.perf_counter()
    
    # Pre-pad audio to handle the first slices
    # Slice for frame t should cover frames [t - slice_length + 1, t]
    pad_start = (slice_length - 1) * HOP_LENGTH
    total_samples_needed = (n_frames + slice_length - 1) * HOP_LENGTH
    pad_end = max(0, total_samples_needed - (len(data) + pad_start))
    data_padded = np.pad(data, (pad_start, pad_end), mode='constant')
    
    # Frame the audio into slices
    y_frames = librosa.util.frame(data_padded, frame_length=slice_length * HOP_LENGTH, hop_length=HOP_LENGTH)
    y_frames = y_frames[:, :n_frames].T  # Shape: (n_frames, slice_length * HOP_LENGTH)
    
    print(f"Processing {audio_name}: {n_frames} frames, vectorized extraction...")
    
    # Vectorized extraction for MFCC, Envelope, and Chroma
    # MFCC: (n_frames, 20, slice_length+1) -> (n_frames, slice_length, 20)
    mfcc = librosa.feature.mfcc(y=y_frames, sr=SAMPLE_RATE, n_mfcc=20, hop_length=HOP_LENGTH)[:, :, :slice_length]
    mfcc = np.transpose(mfcc, (0, 2, 1))
    
    # Envelope: (n_frames, slice_length+1) -> (n_frames, slice_length)
    envelope = librosa.onset.onset_strength(y=y_frames, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)[:, :slice_length]
    
    # Chroma: (n_frames, 12, slice_length+1) -> (n_frames, slice_length, 12)
    chroma = librosa.feature.chroma_cens(y=y_frames, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, n_chroma=12)[:, :, :slice_length]
    chroma = np.transpose(chroma, (0, 2, 1))

    t_vec_end = time.perf_counter()
    
    # Non-vectorized features: Peak and Beat detection per slice
    peak_onehots = np.zeros((n_frames, slice_length), dtype=np.float32)
    beat_onehots = np.zeros((n_frames, slice_length), dtype=np.float32)
    
    # Get global tempo for better local beat tracking
    # start_bpm = librosa.beat.tempo(y=data, sr=SAMPLE_RATE)[0]
    start_bpm = 120.0  # set a common bpm

    t_loop_start = time.perf_counter()
        
    for i in range(n_frames):
        # Peak detection on local envelope
        peaks = librosa.onset.onset_detect(onset_envelope=envelope[i], sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
        if len(peaks) > 0:
            peak_onehots[i, peaks] = 1.0
            
        # Beat tracking on local envelope (needs numba < 0.68: numba 0.68's build of librosa's beat tracker
        # segfaults on Windows, which silently kills multiprocessing workers)
        try:
            _, beats = librosa.beat.beat_track(
                onset_envelope=envelope[i],
                sr=SAMPLE_RATE,
                hop_length=HOP_LENGTH,
                start_bpm=start_bpm,
                tightness=100
            )
            if len(beats) > 0:
                beat_onehots[i, beats] = 1.0
        except:
            pass

    t_loop_end = time.perf_counter()

    # Concatenate all features into (n_frames, slice_length, 35)
    features = np.concatenate(
        [
            envelope[:, :, None],      # (n_frames, slice_length, 1)
            mfcc,                      # (n_frames, slice_length, 20)
            chroma,                    # (n_frames, slice_length, 12)
            peak_onehots[:, :, None],  # (n_frames, slice_length, 1)
            beat_onehots[:, :, None]   # (n_frames, slice_length, 1)
        ],
        axis=-1
    )

    t_total_end = time.perf_counter()

    if profile:
        total_s = t_total_end - t_total_start
        vec_s = t_vec_end - t_vec_start
        loop_s = t_loop_end - t_loop_start
        ms_per_frame = (total_s / n_frames) * 1000.0
        vec_ms_per_frame = (vec_s / n_frames) * 1000.0
        loop_ms_per_frame = (loop_s / n_frames) * 1000.0
        print(
            f"[profile] {audio_name}: "
            f"total={total_s:.3f}s ({ms_per_frame:.3f} ms/frame), "
            f"vec={vec_s:.3f}s ({vec_ms_per_frame:.3f} ms/frame), "
            f"loop={loop_s:.3f}s ({loop_ms_per_frame:.3f} ms/frame)"
        )

    return features, save_path

def process_audio_directory(src_dir, dest_dir, slice_length=32, profile=False):
    """Process all audio files in the directory."""
    file_paths = sorted(Path(src_dir).glob("*.wav"))
    if not file_paths:
        file_paths = sorted(Path(src_dir).glob("*.mp3"))
    
    extract_features = partial(
        extract_audio_features,
        skip_completed=True,
        dest_dir=dest_dir,
        slice_length=slice_length,
        profile=profile,
    )
    for file_path in tqdm(file_paths):
        feats, path = extract_features(file_path)
        if feats is not None:
            np.save(path, feats)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract audio features from PopDanceSet.")
    parser.add_argument("--src", required=True, help="Source directory containing audio files")
    parser.add_argument("--dest", required=True, help="Destination directory for extracted features")
    parser.add_argument("--window_size", type=int, default=32, help="Length of the audio window (default: 32)")
    parser.add_argument("--profile", action="store_true", help="Print per-frame timing (ms/frame)")
    args = parser.parse_args()

    process_audio_directory(args.src, args.dest, slice_length=args.window_size, profile=args.profile)

