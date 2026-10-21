"""
Extract music features for PAE training
Based on the original librosa_preprocess.py implementation

Output format: .npy files with shape (T, 35) where T is number of frames
Features: [envelope(1), mfcc(20), chroma(12), peak(1), beat(1)]
"""

import os
from pathlib import Path
import argparse

import librosa
import numpy as np
from tqdm import tqdm

# Constants (matching the original implementation)
FPS = 30
HOP_LENGTH = 512
SAMPLE_RATE = FPS * HOP_LENGTH  # 15360 Hz


def extract_music_features_sequential(audio_path, window_size=150):
    """
    Extract music features for PAE training (sequential, not windowed)
    
    Args:
        audio_path: Path to audio file
        window_size: Window size for local feature extraction (used for peak/beat detection)
    
    Returns:
        features: (T, 35) array where T is number of frames
    """
    try:
        data, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return None
    
    # Get reference number of frames
    ref_envelope = librosa.onset.onset_strength(y=data, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    n_frames = len(ref_envelope)
    
    print(f"Processing {Path(audio_path).stem}: {n_frames} frames...")
    
    # ===== Extract frame-level features (not windowed) =====
    
    # 1. Envelope (onset strength): (n_frames,)
    envelope = librosa.onset.onset_strength(y=data, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    
    # 2. MFCC: (20, n_frames) -> (n_frames, 20)
    mfcc = librosa.feature.mfcc(y=data, sr=SAMPLE_RATE, n_mfcc=20, hop_length=HOP_LENGTH)
    mfcc = mfcc.T
    
    # 3. Chroma: (12, n_frames) -> (n_frames, 12)
    chroma = librosa.feature.chroma_cens(y=data, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, n_chroma=12)
    chroma = chroma.T
    
    # Ensure all have same length
    min_len = min(len(envelope), mfcc.shape[0], chroma.shape[0])
    envelope = envelope[:min_len]
    mfcc = mfcc[:min_len]
    chroma = chroma[:min_len]
    n_frames = min_len
    
    # ===== Extract peak and beat (using local windows) =====
    # For PAE training, we still compute these frame-by-frame but use local context
    
    peak_onehots = np.zeros(n_frames, dtype=np.float32)
    beat_onehots = np.zeros(n_frames, dtype=np.float32)
    
    # Pad audio for windowed peak/beat detection
    pad_start = (window_size - 1) * HOP_LENGTH
    total_samples_needed = (n_frames + window_size - 1) * HOP_LENGTH
    pad_end = max(0, total_samples_needed - (len(data) + pad_start))
    data_padded = np.pad(data, (pad_start, pad_end), mode='constant')
    
    # Get global tempo
    start_bpm = 120.0
    
    # Process in windows for peak/beat detection
    for i in tqdm(range(n_frames), desc="Detecting peaks/beats", leave=False):
        # Extract window [i - window_size + 1, i]
        window_start = i * HOP_LENGTH
        window_end = window_start + window_size * HOP_LENGTH
        y_window = data_padded[window_start:window_end]
        
        # Local envelope
        local_envelope = librosa.onset.onset_strength(
            y=y_window, sr=SAMPLE_RATE, hop_length=HOP_LENGTH
        )
        
        # Peak detection
        peaks = librosa.onset.onset_detect(
            onset_envelope=local_envelope, 
            sr=SAMPLE_RATE, 
            hop_length=HOP_LENGTH
        )
        # Check if current frame (last in window) is a peak
        if len(peaks) > 0 and (window_size - 1) in peaks:
            peak_onehots[i] = 1.0
        
        # Beat detection
        try:
            _, beats = librosa.beat.beat_track(
                onset_envelope=local_envelope,
                sr=SAMPLE_RATE,
                hop_length=HOP_LENGTH,
                start_bpm=start_bpm,
                tightness=100
            )
            # Check if current frame is a beat
            if len(beats) > 0 and (window_size - 1) in beats:
                beat_onehots[i] = 1.0
        except:
            pass
    
    # ===== Concatenate all features =====
    features = np.concatenate([
        envelope[:, None],      # (n_frames, 1)
        mfcc,                   # (n_frames, 20)
        chroma,                 # (n_frames, 12)
        peak_onehots[:, None],  # (n_frames, 1)
        beat_onehots[:, None]   # (n_frames, 1)
    ], axis=-1)  # (n_frames, 35)
    
    return features


def process_audio_directory(src_dir, dest_dir, window_size=150):
    """
    Process all audio files in a directory
    
    Args:
        src_dir: Directory containing audio files (.wav or .mp3)
        dest_dir: Directory to save extracted features
        window_size: Window size for local feature computation
    """
    os.makedirs(dest_dir, exist_ok=True)
    
    # Find audio files
    audio_files = list(Path(src_dir).glob("*.wav")) + list(Path(src_dir).glob("*.mp3"))
    audio_files = sorted(audio_files)
    
    if len(audio_files) == 0:
        print(f"No audio files found in {src_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files")
    
    for audio_path in tqdm(audio_files, desc="Processing audio files"):
        audio_name = audio_path.stem
        save_path = os.path.join(dest_dir, f"{audio_name}.npy")
        
        # Skip if already processed
        if os.path.exists(save_path):
            print(f"Skipping {audio_name} (already exists)")
            continue
        
        # Extract features
        features = extract_music_features_sequential(audio_path, window_size=window_size)
        
        if features is not None:
            # Save features
            np.save(save_path, features)
            print(f"Saved {audio_name}: shape {features.shape}")
        else:
            print(f"Failed to process {audio_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract music features for PAE training"
    )
    parser.add_argument(
        "--src", 
        required=True, 
        help="Source directory containing audio files"
    )
    parser.add_argument(
        "--dest", 
        required=True, 
        help="Destination directory for extracted features"
    )
    parser.add_argument(
        "--window_size", 
        type=int, 
        default=150,
        help="Window size for local peak/beat detection (default: 150)"
    )
    
    args = parser.parse_args()
    
    process_audio_directory(args.src, args.dest, window_size=args.window_size)
    print("Done!")


if __name__ == "__main__":
    main()
