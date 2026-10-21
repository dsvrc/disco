import numpy as np
from tqdm import tqdm
import os
import soundfile as sf
import subprocess
import json

def slice_motion(motion_path, output_path, sample_times=50, max_length=300):
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    motion_files = [
        f for f in os.listdir(motion_path) if f.endswith('.npy')
    ]
    for motion_file in tqdm(motion_files, desc="Slicing motions"):
        motion_path = os.path.join(motion_path, motion_file)
        motion_data = np.load(motion_path, allow_pickle=True)

        length = motion_data.shape[0]
        if length <= max_length:
            # If the motion is shorter than max_length, save it as is
            np.save(os.path.join(output_path, motion_file), motion_data)
        else:
            # Randomly sample segments
            for idx in range(sample_times):
                start_idx = np.random.randint(0, length - max_length + 1)
                segment = motion_data[start_idx:start_idx + max_length]
                segment_file = f"{os.path.splitext(motion_file)[0]}_{idx}.npy"
                np.save(os.path.join(output_path, segment_file), segment)
    print(f"Sliced motions saved to {output_path}")

def slice_motion_music(motion_path, music_path, output_motion_path, output_music_path, sample_times=50, max_length=300):
    if not os.path.exists(output_motion_path):
        os.makedirs(output_motion_path, exist_ok=True)
    if not os.path.exists(output_music_path):
        os.makedirs(output_music_path, exist_ok=True)
    
    motion_files = [
        f for f in os.listdir(motion_path) if f.endswith('.npy')
    ]


    for motion_file in tqdm(motion_files, desc="Slicing motion-music pairs"):
        motion_path_full = os.path.join(motion_path, motion_file)
        motion_data = np.load(motion_path_full, allow_pickle=True)

        music_file = motion_file  # Assuming music file has the same name as motion file
        music_path_full = os.path.join(music_path, music_file)
        if not os.path.exists(music_path_full):
            continue  # Skip if corresponding music file does not exist
        music_data = np.load(music_path_full, allow_pickle=True)
        length = motion_data.shape[0]
        if length <= max_length:
            # If the motion is shorter than max_length, save it as is
            np.save(os.path.join(output_motion_path, motion_file), motion_data)
            np.save(os.path.join(output_music_path, music_file), music_data)
        else:
            # Randomly sample segments
            for idx in range(sample_times):
                start_idx = np.random.randint(0, length - max_length + 1)
                motion_segment = motion_data[start_idx:start_idx + max_length]
                music_segment = music_data[start_idx:start_idx + max_length]
                
                segment_motion_file = f"{os.path.splitext(motion_file)[0]}_{idx}.npy"
                segment_music_file = f"{os.path.splitext(music_file)[0]}_{idx}.npy"
                
                np.save(os.path.join(output_motion_path, segment_motion_file), motion_segment)
                np.save(os.path.join(output_music_path, segment_music_file), music_segment)    

def slice_motion_music_wav(motion_path, music_wav_path, output_motion_path, output_music_wav_path, sample_times=50, max_length=300, motion_fps=30):
    """
    Slice motion and corresponding music wav files synchronously
    
    Args:
        motion_path: Path to motion .npy files
        music_wav_path: Path to music .wav files
        output_motion_path: Output path for sliced motion files
        output_music_wav_path: Output path for sliced music wav files
        sample_times: Number of random samples to take for long sequences
        max_length: Maximum length in frames for motion
        motion_fps: Motion frame rate (default 30fps)
    """
    if not os.path.exists(output_motion_path):
        os.makedirs(output_motion_path, exist_ok=True)
    if not os.path.exists(output_music_wav_path):
        os.makedirs(output_music_wav_path, exist_ok=True)
    
    motion_files = [
        f for f in os.listdir(motion_path) if f.endswith('.npy')
    ]

    for motion_file in tqdm(motion_files, desc="Slicing motion-music wav pairs"):
        motion_path_full = os.path.join(motion_path, motion_file)
        motion_data = np.load(motion_path_full, allow_pickle=True)

        # Find corresponding music file
        base_name = os.path.splitext(motion_file)[0]
        music_file = base_name + ".wav"
        music_path_full = os.path.join(music_wav_path, music_file)
        
        if not os.path.exists(music_path_full):
            print(f"Warning: Music file not found: {music_path_full}")
            continue
        
        # Load audio file
        audio_data, sample_rate = sf.read(music_path_full)
        
        motion_length = motion_data.shape[0]
        audio_length = len(audio_data)
        
        # Calculate audio samples per motion frame
        audio_samples_per_frame = sample_rate / motion_fps
        
        if motion_length <= max_length:
            # If the motion is shorter than max_length, save it as is
            np.save(os.path.join(output_motion_path, motion_file), motion_data)
            sf.write(os.path.join(output_music_wav_path, music_file), audio_data, sample_rate)
        else:
            # Randomly sample segments
            for idx in range(sample_times):
                # Random start frame for motion
                start_frame = np.random.randint(0, motion_length - max_length + 1)
                end_frame = start_frame + max_length
                
                # Calculate corresponding audio segment
                start_audio_sample = int(start_frame * audio_samples_per_frame)
                end_audio_sample = int(end_frame * audio_samples_per_frame)
                
                # Make sure we don't exceed audio length
                end_audio_sample = min(end_audio_sample, audio_length)
                
                # Extract segments
                motion_segment = motion_data[start_frame:end_frame]
                audio_segment = audio_data[start_audio_sample:end_audio_sample]
                
                # Save segments
                segment_motion_file = f"{base_name}_{idx}.npy"
                segment_music_file = f"{base_name}_{idx}.wav"
                
                np.save(os.path.join(output_motion_path, segment_motion_file), motion_segment)
                sf.write(os.path.join(output_music_wav_path, segment_music_file), audio_segment, sample_rate)
    
    print(f"Sliced motions saved to {output_motion_path}")
    print(f"Sliced music wavs saved to {output_music_wav_path}")

def slice_videos(video_path, output_video_path, sample_times=50, max_duration=10.0, video_fps=30):
    """
    Slice video files synchronously with motion data
    
    Args:
        video_path: Path to video .mp4 files
        output_video_path: Output path for sliced video files
        sample_times: Number of random samples to take for long sequences
        max_duration: Maximum duration in seconds for videos
        video_fps: Video frame rate (default 30fps)
    """
    if not os.path.exists(output_video_path):
        os.makedirs(output_video_path, exist_ok=True)
    
    video_files = [
        f for f in os.listdir(video_path) if f.endswith('.mp4')
    ]

    for video_file in tqdm(video_files, desc="Slicing videos"):
        video_path_full = os.path.join(video_path, video_file)
        base_name = os.path.splitext(video_file)[0]
        
        # Get video duration using ffprobe
        cmd = [
            'ffprobe', '-v', 'error', '-show_entries',
            'format=duration', '-of',
            'default=noprint_wrappers=1:nokey=1', video_path_full
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        try:
            duration = float(result.stdout.strip())
        except ValueError:
            print(f"Warning: Could not get duration for {video_file}, skipping")
            continue
        
        if duration <= max_duration:
            # If video is shorter than max_duration, copy it as is
            output_file = os.path.join(output_video_path, video_file)
            subprocess.run(['cp', video_path_full, output_file], check=True)
        else:
            # Randomly sample segments
            for idx in range(sample_times):
                # Random start time
                start_time = np.random.uniform(0, duration - max_duration)
                
                # Output file name
                segment_video_file = f"{base_name}_{idx}.mp4"
                output_file = os.path.join(output_video_path, segment_video_file)
                
                # Use ffmpeg to extract segment
                cmd = [
                    'ffmpeg', '-hide_banner', '-loglevel', 'error',
                    '-ss', str(start_time),
                    '-i', video_path_full,
                    '-t', str(max_duration),
                    '-c', 'copy',
                    '-y', output_file
                ]
                subprocess.run(cmd, check=True)
    
    print(f"Sliced videos saved to {output_video_path}")

def slice_motion_music_wav_video(
    motion_path, 
    output_motion_path, 
    music_wav_path=None, 
    output_music_wav_path=None,
    video_path=None, 
    output_video_path=None,
    music_feat_paths=None, 
    output_music_feat_paths=None,
    sample_times=50, 
    max_length=300, 
    motion_fps=30,
    slice_info_file=None
):
    """
    Slice motion, music wav, video, and music features files synchronously.
    If output_motion_path already contains sliced data, it will load the slice info
    and skip re-slicing motion, only slicing other optional modalities.
    
    Args:
        motion_path: Path to motion .npy files (required)
        output_motion_path: Output path for sliced motion files (required)
        music_wav_path: Path to music .wav files (optional, None to skip)
        output_music_wav_path: Output path for sliced music wav files (optional, None to skip)
        video_path: Path to video .mp4 files (optional, None to skip)
        output_video_path: Output path for sliced video files (optional, None to skip)
        music_feat_paths: List of paths to music feature .npy files (optional, None to skip)
        output_music_feat_paths: List of output paths for sliced music feature files (optional, None to skip)
        sample_times: Number of random samples to take for long sequences
        max_length: Maximum length in frames for motion
        motion_fps: Motion frame rate (default 30fps)
        slice_info_file: Path to JSON file storing slice information (default: output_motion_path/slice_info.json)
    """
    # Validate paired inputs
    if (music_wav_path is None) != (output_music_wav_path is None):
        raise ValueError("music_wav_path and output_music_wav_path must be both provided or both None")
    if (video_path is None) != (output_video_path is None):
        raise ValueError("video_path and output_video_path must be both provided or both None")
    if (music_feat_paths is None) != (output_music_feat_paths is None):
        raise ValueError("music_feat_paths and output_music_feat_paths must be both provided or both None")
    if music_feat_paths is not None and len(music_feat_paths) != len(output_music_feat_paths):
        raise ValueError("music_feat_paths and output_music_feat_paths must have the same length")
    
    # Create output directories
    if not output_motion_path:
        raise ValueError("output_motion_path must be a non-empty path")
    os.makedirs(output_motion_path, exist_ok=True)
    if output_music_wav_path:
        os.makedirs(output_music_wav_path, exist_ok=True)
    if output_video_path:
        os.makedirs(output_video_path, exist_ok=True)
    if output_music_feat_paths:
        for out_path in output_music_feat_paths:
            os.makedirs(out_path, exist_ok=True)
    
    # Determine slice info file path
    if slice_info_file is None:
        slice_info_file = os.path.join(output_motion_path, "slice_info.json")
    
    # Check if motion is already sliced and load slice info
    sliced_motion_files = [f for f in os.listdir(output_motion_path) if f.endswith('.npy')]
    has_existing_sliced_motion = len(sliced_motion_files) > 0

    slice_info_to_use = {}
    use_existing_slices = False

    if os.path.exists(slice_info_file):
        # Load slice_info.json even if output_motion_path currently has no sliced .npy files
        print(f"Loading slice info from {slice_info_file}")
        with open(slice_info_file, 'r') as f:
            slice_info_to_use = json.load(f)
        use_existing_slices = True
        if has_existing_sliced_motion:
            print(f"Found existing sliced motion data in {output_motion_path}")
        print(f"Will use existing slice information for {len(slice_info_to_use)} original files")
    elif has_existing_sliced_motion:
        print(f"Found existing sliced motion data in {output_motion_path} but no slice_info.json")
        print("Attempting to infer slice information from existing sliced files...")
        slice_info_to_use = _infer_slice_info_from_files(motion_path, output_motion_path, max_length)
        # Save inferred slice info
        with open(slice_info_file, 'w') as f:
            json.dump(slice_info_to_use, f, indent=2)
        use_existing_slices = True
        print(f"Inferred and saved slice information for {len(slice_info_to_use)} original files")
    
    # Get list of source motion files
    motion_files = [f for f in os.listdir(motion_path) if f.endswith('.npy')]
    
    for motion_file in tqdm(motion_files, desc="Processing motion-music-video-feat"):
        motion_path_full = os.path.join(motion_path, motion_file)
        base_name = os.path.splitext(motion_file)[0]
        
        # Determine slice ranges
        if use_existing_slices and base_name in slice_info_to_use:
            # Use existing slice information
            file_slice_info = slice_info_to_use[base_name]
            motion_length = file_slice_info['motion_length']
            slices = file_slice_info['slices']  # List of (start_frame, end_frame) tuples

            # If we have slice info but no existing sliced motion files in output_motion_path,
            # slice and save motion according to the provided slices.
            if not has_existing_sliced_motion:
                motion_data = np.load(motion_path_full, allow_pickle=True)
                actual_motion_length = motion_data.shape[0]
                if actual_motion_length != motion_length:
                    print(
                        f"Warning: motion_length mismatch for {base_name}: "
                        f"slice_info={motion_length}, actual={actual_motion_length}. "
                        f"Using actual length."
                    )
                    motion_length = actual_motion_length
                    file_slice_info['motion_length'] = actual_motion_length

                for idx, (start_frame, end_frame) in enumerate(slices):
                    motion_segment = motion_data[start_frame:end_frame]
                    if (
                        len(slices) == 1
                        and motion_length <= max_length
                        and start_frame == 0
                        and end_frame == motion_length
                    ):
                        segment_motion_file = motion_file
                    else:
                        segment_motion_file = f"{base_name}_{idx}.npy"
                    np.save(os.path.join(output_motion_path, segment_motion_file), motion_segment)
        else:
            # Create new slices
            motion_data = np.load(motion_path_full, allow_pickle=True)
            motion_length = motion_data.shape[0]
            
            if motion_length <= max_length:
                slices = [(0, motion_length)]  # Single slice with full data
            else:
                # Generate random slices
                slices = []
                for idx in range(sample_times):
                    start_frame = np.random.randint(0, motion_length - max_length + 1)
                    end_frame = start_frame + max_length
                    slices.append((start_frame, end_frame))
            
            # Save slice info for this file
            slice_info_to_use[base_name] = {
                'motion_length': motion_length,
                'slices': slices
            }
            
            # Save motion slices (only if not using existing)
            for idx, (start_frame, end_frame) in enumerate(slices):
                motion_segment = motion_data[start_frame:end_frame]
                if len(slices) == 1 and motion_length <= max_length:
                    segment_motion_file = motion_file
                else:
                    segment_motion_file = f"{base_name}_{idx}.npy"
                np.save(os.path.join(output_motion_path, segment_motion_file), motion_segment)
        
        # Process optional modalities
        _process_music_wav(music_wav_path, output_music_wav_path, base_name, slices, motion_length, motion_fps)
        _process_video(video_path, output_video_path, base_name, slices, motion_length, motion_fps)
        _process_music_feats(music_feat_paths, output_music_feat_paths, base_name, slices, motion_length)
    
    # Save slice info
    with open(slice_info_file, 'w') as f:
        json.dump(slice_info_to_use, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Slicing completed!")
    print(f"Slice info saved to {slice_info_file}")
    if not use_existing_slices:
        print(f"Sliced motions saved to {output_motion_path}")
    if output_music_wav_path:
        print(f"Sliced music wavs saved to {output_music_wav_path}")
    if output_video_path:
        print(f"Sliced videos saved to {output_video_path}")
    if output_music_feat_paths:
        for out_path in output_music_feat_paths:
            print(f"Sliced music features saved to {out_path}")
    print(f"{'='*50}\n")


def _infer_slice_info_from_files(motion_path, output_motion_path, max_length):
    """
    Infer slice information from existing sliced motion files.
    """
    slice_info = {}
    
    # Get all sliced files
    sliced_files = [f for f in os.listdir(output_motion_path) if f.endswith('.npy')]
    
    # Group by base name
    from collections import defaultdict
    file_groups = defaultdict(list)
    
    for sliced_file in sliced_files:
        # Parse filename: base_name_idx.npy or base_name.npy
        name_without_ext = os.path.splitext(sliced_file)[0]
        parts = name_without_ext.rsplit('_', 1)
        
        if len(parts) == 2 and parts[1].isdigit():
            base_name = parts[0]
            idx = int(parts[1])
        else:
            base_name = name_without_ext
            idx = None
        
        file_groups[base_name].append((idx, sliced_file))
    
    # For each base file, infer slices
    for base_name, sliced_files_list in file_groups.items():
        original_file = os.path.join(motion_path, f"{base_name}.npy")
        if not os.path.exists(original_file):
            print(f"Warning: Original motion file not found: {original_file}, skipping")
            continue
        
        original_data = np.load(original_file, allow_pickle=True)
        motion_length = original_data.shape[0]
        
        slices = []
        sliced_files_list.sort(key=lambda x: x[0] if x[0] is not None else -1)
        
        for idx, sliced_file in sliced_files_list:
            sliced_path = os.path.join(output_motion_path, sliced_file)
            sliced_data = np.load(sliced_path, allow_pickle=True)
            slice_length = sliced_data.shape[0]
            
            # Try to find matching segment in original data
            if idx is None and motion_length <= max_length:
                # Unsliced file
                slices.append((0, motion_length))
            else:
                # Search for matching segment
                start_frame = _find_matching_segment(original_data, sliced_data)
                if start_frame >= 0:
                    slices.append((start_frame, start_frame + slice_length))
                else:
                    print(f"Warning: Could not find matching segment for {sliced_file}")
        
        slice_info[base_name] = {
            'motion_length': motion_length,
            'slices': slices
        }
    
    return slice_info


def _find_matching_segment(original_data, sliced_data):
    """
    Find the start frame of sliced_data in original_data by comparing content.
    Returns -1 if not found.
    """
    slice_length = sliced_data.shape[0]
    original_length = original_data.shape[0]
    
    if slice_length > original_length:
        return -1
    
    # Try to find exact match
    for start_idx in range(original_length - slice_length + 1):
        segment = original_data[start_idx:start_idx + slice_length]
        if np.allclose(segment, sliced_data, rtol=1e-5, atol=1e-8):
            return start_idx
    
    return -1


def _process_music_wav(music_wav_path, output_music_wav_path, base_name, slices, motion_length, motion_fps):
    """Process music wav files according to slice information."""
    if music_wav_path is None or output_music_wav_path is None:
        return
    
    music_file = base_name + ".wav"
    music_path_full = os.path.join(music_wav_path, music_file)
    
    if not os.path.exists(music_path_full):
        print(f"Warning: Music file not found: {music_path_full}")
        return
    
    # Load audio file
    audio_data, sample_rate = sf.read(music_path_full)
    audio_samples_per_frame = sample_rate / motion_fps
    
    for idx, (start_frame, end_frame) in enumerate(slices):
        # Calculate corresponding audio segment
        start_audio_sample = int(start_frame * audio_samples_per_frame)
        end_audio_sample = int(end_frame * audio_samples_per_frame)
        end_audio_sample = min(end_audio_sample, len(audio_data))
        
        audio_segment = audio_data[start_audio_sample:end_audio_sample]
        
        if len(slices) == 1 and motion_length <= end_frame - start_frame:
            segment_music_file = music_file
        else:
            segment_music_file = f"{base_name}_{idx}.wav"
        
        sf.write(os.path.join(output_music_wav_path, segment_music_file), audio_segment, sample_rate)


def _process_video(video_path, output_video_path, base_name, slices, motion_length, motion_fps):
    """Process video files according to slice information."""
    if video_path is None or output_video_path is None:
        return
    
    video_file = base_name + ".mp4"
    video_path_full = os.path.join(video_path, video_file)
    
    if not os.path.exists(video_path_full):
        print(f"Warning: Video file not found: {video_path_full}")
        return
    
    for idx, (start_frame, end_frame) in enumerate(slices):
        start_time = start_frame / motion_fps
        duration = (end_frame - start_frame) / motion_fps
        
        if len(slices) == 1 and motion_length <= end_frame - start_frame:
            segment_video_file = video_file
            # Just copy the file
            subprocess.run(['cp', video_path_full, os.path.join(output_video_path, segment_video_file)], check=True)
        else:
            segment_video_file = f"{base_name}_{idx}.mp4"
            output_video_file = os.path.join(output_video_path, segment_video_file)
            
            # Extract video segment using ffmpeg
            cmd = [
                'ffmpeg', '-hide_banner', '-loglevel', 'error',
                '-ss', str(start_time),
                '-i', video_path_full,
                '-t', str(duration),
                '-c', 'copy',
                '-y', output_video_file
            ]
            subprocess.run(cmd, check=True)


def _process_music_feats(music_feat_paths, output_music_feat_paths, base_name, slices, motion_length):
    """Process music feature files according to slice information."""
    if music_feat_paths is None or output_music_feat_paths is None:
        return
    
    for feat_path, output_feat_path in zip(music_feat_paths, output_music_feat_paths):
        feat_file = base_name + ".npy"
        feat_path_full = os.path.join(feat_path, feat_file)
        
        if not os.path.exists(feat_path_full):
            print(f"Warning: Music feat file not found: {feat_path_full}")
            continue
        
        feat_data = np.load(feat_path_full, allow_pickle=True)
        
        for idx, (start_frame, end_frame) in enumerate(slices):
            feat_segment = feat_data[start_frame:end_frame]
            
            if len(slices) == 1 and motion_length <= end_frame - start_frame:
                segment_feat_file = feat_file
            else:
                segment_feat_file = f"{base_name}_{idx}.npy"
            
            np.save(os.path.join(output_feat_path, segment_feat_file), feat_segment)


def split_new_dataset_from_txt(data_path, origin_txt, output_txt):
    """
    >>> origin_txt: 001, 004, 010, 192, ...
    >>> output_txt(same with sliced motion names): 001_1, 001_2, 001_3, ..., 004_1, 004_2, ..., 010_1, ..., 192_1, 192_2, ...
    """
    with open(origin_txt, "r") as f:
        names = [line.strip() for line in f if line.strip()]
        
    
    motion_files = [
        f for f in os.listdir(data_path) if f.endswith('.npy')
    ]
    new_names = []
    for name in names:
        count = 0
        for motion_file in motion_files:
            if motion_file.startswith(name + "_") or motion_file == name + ".npy":
                new_names.append(os.path.splitext(motion_file)[0])
                count += 1
        if count == 0:
            print(f"Warning: No sliced files found for {name}")

    with open(output_txt, "w") as f:
        for new_name in new_names:
            f.write(new_name + "\n")


if __name__ == "__main__":
    # # Slice motion, music wav, video, and music features synchronously
    slice_motion_music_wav_video(
        motion_path="raw_data/finedance/motion_272_30fps",
        output_motion_path="raw_data/finedance/motion_272_30fps_sliced",
        # motion_path="raw_data/finedance/motion_263_30fps",
        # output_motion_path="raw_data/finedance/motion_263_30fps_sliced",
        # music_wav_path="raw_data/finedance/music_wav",
        # output_music_wav_path="raw_data/finedance/music_wav_sliced",
        # video_path="raw_data/finedance/animations_272_30fps",
        # output_video_path="raw_data/finedance/animations_272_30fps_sliced",
        # music_feat_paths=["raw_data/finedance/baseline_feats_window_32", "raw_data/finedance/baseline_feats_window_64", "raw_data/finedance/vqpae_feats"],
        # output_music_feat_path=["raw_data/finedance/baseline_feats_window_32_sliced", "raw_data/finedance/baseline_feats_window_64_sliced", "raw_data/finedance/vqpae_feats_sliced"],
        # music_feat_paths=["raw_data/finedance/baseline_feats_window_32", "raw_data/finedance/baseline_feats_window_64"],
        # output_music_feat_paths=["raw_data/finedance/baseline_feats_window_32_sliced", "raw_data/finedance/baseline_feats_window_64_sliced"],
        music_feat_paths=["raw_data/finedance/vqpae_feats"],
        output_music_feat_paths=["raw_data/finedance/vqpae_feats_sliced"],
        sample_times=50,
        max_length=300,
        motion_fps=30,

    )

    # Split txt files based on sliced data
    # split_new_dataset_from_txt(
    #     data_path="raw_data/finedance/motion_272_30fps_sliced",
    #     origin_txt="raw_data/finedance/train.txt",
    #     output_txt="raw_data/finedance/train_sliced.txt"
    # )

    # split_new_dataset_from_txt(
    #     data_path="raw_data/finedance/motion_272_30fps_sliced",
    #     origin_txt="raw_data/finedance/val.txt",
    #     output_txt="raw_data/finedance/val_sliced.txt"
    # )

    # split_new_dataset_from_txt(
    #     data_path="raw_data/finedance/motion_272_30fps_sliced",
    #     origin_txt="raw_data/finedance/test.txt",
    #     output_txt="raw_data/finedance/test_sliced.txt"
    # )
    # slice_motion_music_wav_video(
    #     motion_path="raw_data/aistpp/motions_272_30fps",
    #     output_motion_path="raw_data/aistpp/motions_272_30fps_sliced",
    #     # music_wav_path="raw_data/aistpp/wavs",
    #     # output_music_wav_path="raw_data/aistpp/wavs_272_sliced",
    #     # video_path="raw_data/aistpp/animations_272_30fps",
    #     # output_video_path="raw_data/aistpp/animations_272_30fps_sliced",
    #     # music_feat_paths=["raw_data/aistpp/baseline_feats_window_64", "raw_data/aistpp/baseline_feats_window_64"],
    #     # output_music_feat_paths=["raw_data/aistpp/baseline_feats_272_window_32_sliced", "raw_data/aistpp/baseline_feats_272_window_64_sliced"],
    #     music_feat_paths=["raw_data/aistpp/vqpae_feats"],
    #     output_music_feat_paths=["raw_data/aistpp/vqpae_feats_sliced"],
    #     sample_times=10,
    #     max_length=300,
    #     motion_fps=30
    # )

    # # Split txt files based on sliced data
    # split_new_dataset_from_txt(
    #     data_path="raw_data/aistpp/motions_272_30fps_sliced",
    #     origin_txt="raw_data/aistpp/crossmodal_train_without_val.txt",
    #     output_txt="raw_data/aistpp/crossmodal_train_without_val_272_sliced.txt"
    # )

    # split_new_dataset_from_txt(
    #     data_path="raw_data/aistpp/motions_272_30fps_sliced",
    #     origin_txt="raw_data/aistpp/crossmodal_val.txt",
    #     output_txt="raw_data/aistpp/crossmodal_val_272_sliced.txt"
    # )

    # split_new_dataset_from_txt(
    #     data_path="raw_data/aistpp/motions_272_30fps_sliced",
    #     origin_txt="raw_data/aistpp/crossmodal_test.txt",
    #     output_txt="raw_data/aistpp/crossmodal_test_272_sliced.txt"
    # )