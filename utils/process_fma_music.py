"""
Process FMA music dataset:
1. Extract 35-dim Librosa features
2. Randomly concatenate 3 music segments

Usage:
    python utils/process_fma_music.py --input_dir raw_data/fma/fma_medium --output_dir raw_data/fma/processed --num_concat 5000

    python utils/process_fma_music.py --input_dir raw_data/fma/fma_medium --output_dir raw_data/fma/debug --debug
"""

import argparse
import os
import random
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm


def extract_music_features(audio_path, sr=22050, hop_length=512, target_fps=30):
    """
    Extract 35-dim music features:
    - MFCC: 13 dims
    - Chroma: 12 dims
    - Tempogram: 8 dims
    - RMS: 1 dim
    - Spectral Centroid: 1 dim
    
    Args:
        audio_path: path to audio file
        sr: sample rate
        hop_length: hop length for feature extraction
        target_fps: target frame rate (default: 30 fps)
        
    Returns:
        features: (T, 35) numpy array at target_fps
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
        tempogram = librosa.feature.tempogram(y=y, sr=sr, hop_length=hop_length)[:8]
        rms = librosa.feature.rms(y=y, hop_length=hop_length)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
        
        features = np.vstack([mfcc, chroma, tempogram, rms, spectral_centroid])
        features = features.T
        
        original_fps = sr / hop_length
        if abs(original_fps - target_fps) > 0.1:
            from scipy.interpolate import interp1d
            original_time = np.arange(features.shape[0])
            target_length = int(features.shape[0] * target_fps / original_fps)
            target_time = np.linspace(0, features.shape[0] - 1, target_length)
            
            interpolator = interp1d(original_time, features, axis=0, kind='linear')
            features = interpolator(target_time)
        
        return features
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


def collect_audio_files(input_dir):
    """Collect all audio files from input directory"""
    audio_extensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(Path(input_dir).rglob(f'*{ext}'))
    
    return [str(f) for f in audio_files]


def process_single_files(audio_files, output_dir, max_files=None):
    """
    Process individual audio files and save features
    
    Args:
        audio_files: list of audio file paths
        output_dir: output directory for features
        max_files: maximum number of files to process (for debugging)
    """
    single_output_dir = os.path.join(output_dir, 'single')
    os.makedirs(single_output_dir, exist_ok=True)
    
    processed_files = []
    
    if max_files:
        audio_files = audio_files[:max_files]
        print(f"DEBUG MODE: Processing only {len(audio_files)} files...")
    else:
        print("Processing individual audio files...")
    
    for audio_path in tqdm(audio_files):
        features = extract_music_features(audio_path)
        
        if features is not None:
            filename = Path(audio_path).stem
            output_path = os.path.join(single_output_dir, f'{filename}.npy')
            np.save(output_path, features)
            processed_files.append((filename, features.shape[0]))
    
    print(f"Processed {len(processed_files)} files")
    return processed_files


def concatenate_random_segments(processed_files, output_dir, num_concat=1000, segments_per_concat=3):
    """
    Randomly concatenate 3 music segments
    
    Args:
        processed_files: list of (filename, length) tuples
        output_dir: output directory
        num_concat: number of concatenated files to generate
        segments_per_concat: number of segments to concatenate (default: 3)
    """
    concat_output_dir = os.path.join(output_dir, 'concatenated')
    os.makedirs(concat_output_dir, exist_ok=True)
    
    single_output_dir = os.path.join(output_dir, 'single')
    
    print(f"\nGenerating {num_concat} concatenated music files...")
    for i in tqdm(range(num_concat)):
        # Randomly select 3 files
        selected = random.sample(processed_files, segments_per_concat)
        
        # Load and concatenate features
        concat_features = []
        concat_names = []
        
        for filename, _ in selected:
            feature_path = os.path.join(single_output_dir, f'{filename}.npy')
            features = np.load(feature_path)
            concat_features.append(features)
            concat_names.append(filename)
        
        # Concatenate along time dimension
        final_features = np.concatenate(concat_features, axis=0)
        
        # Create output filename from concatenated names
        output_name = '_'.join(concat_names)
        output_path = os.path.join(concat_output_dir, f'{output_name}.npy')
        
        np.save(output_path, final_features)
    
    print(f"Generated {num_concat} concatenated files")


def create_file_list(output_dir):
    """Create train/val/test file lists"""
    concat_dir = os.path.join(output_dir, 'concatenated')
    files = [f.stem for f in Path(concat_dir).glob('*.npy')]
    
    # Shuffle files
    random.shuffle(files)
    
    # Split: 80% train, 10% val, 10% test
    n = len(files)
    train_split = int(0.8 * n)
    val_split = int(0.9 * n)
    
    train_files = files[:train_split]
    val_files = files[train_split:val_split]
    test_files = files[val_split:]
    
    # Write to text files
    with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_files))
    
    with open(os.path.join(output_dir, 'val.txt'), 'w') as f:
        f.write('\n'.join(val_files))
    
    with open(os.path.join(output_dir, 'test.txt'), 'w') as f:
        f.write('\n'.join(test_files))
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_files)} files")
    print(f"  Val: {len(val_files)} files")
    print(f"  Test: {len(test_files)} files")


def compute_statistics(output_dir):
    """Compute mean and std statistics for normalization"""
    concat_dir = os.path.join(output_dir, 'concatenated')
    files = list(Path(concat_dir).glob('*.npy'))
    
    print("\nComputing dataset statistics...")
    all_features = []
    for f in tqdm(files[:1000]):  # Use subset for efficiency
        features = np.load(f)
        all_features.append(features)
    
    all_features = np.concatenate(all_features, axis=0)
    
    mean = np.mean(all_features, axis=0)
    std = np.std(all_features, axis=0)
    
    np.save(os.path.join(output_dir, 'Mean_35.npy'), mean)
    np.save(os.path.join(output_dir, 'Std_35.npy'), std)
    
    print(f"Statistics saved: Mean shape {mean.shape}, Std shape {std.shape}")


def main():
    parser = argparse.ArgumentParser(description='Process FMA music dataset')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing audio files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for processed features')
    parser.add_argument('--num_concat', type=int, default=1000,
                        help='Number of concatenated files to generate')
    parser.add_argument('--segments_per_concat', type=int, default=3,
                        help='Number of segments to concatenate per file')
    parser.add_argument('--sr', type=int, default=22050,
                        help='Sample rate for audio loading')
    parser.add_argument('--hop_length', type=int, default=512,
                        help='Hop length for feature extraction')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode: process only 20 files and generate 50 concatenations')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Debug mode overrides
    if args.debug:
        print("=" * 50)
        print("DEBUG MODE ENABLED")
        print("=" * 50)
        max_files = 20
        num_concat = 50
        print(f"Will process {max_files} audio files and generate {num_concat} concatenations\n")
    else:
        max_files = None
        num_concat = args.num_concat
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Collect audio files
    print(f"Scanning audio files in {args.input_dir}...")
    audio_files = collect_audio_files(args.input_dir)
    print(f"Found {len(audio_files)} audio files")
    
    if len(audio_files) == 0:
        print("No audio files found! Check your input directory.")
        return
    
    # Step 2: Process individual files
    processed_files = process_single_files(audio_files, args.output_dir, max_files=max_files)
    
    if len(processed_files) < args.segments_per_concat:
        print(f"Not enough processed files ({len(processed_files)}) for concatenation (need at least {args.segments_per_concat})")
        return
    
    # Step 3: Concatenate random segments
    concatenate_random_segments(
        processed_files, 
        args.output_dir, 
        num_concat=num_concat,
        segments_per_concat=args.segments_per_concat
    )
    
    # Step 4: Create train/val/test splits
    create_file_list(args.output_dir)
    
    # Step 5: Compute statistics
    compute_statistics(args.output_dir)
    
    print("\n✅ Processing complete!")
    print(f"Output directory: {args.output_dir}")
    print(f"  - single/: Individual music features")
    print(f"  - concatenated/: Concatenated music features")
    print(f"  - train.txt, val.txt, test.txt: Dataset splits")
    print(f"  - Mean_35.npy, Std_35.npy: Normalization statistics")


if __name__ == '__main__':
    main()
