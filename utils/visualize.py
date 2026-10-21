import os
import pickle
import subprocess
from pathlib import Path
from typing import List

import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from utils.motion_process import convert_motion_to_joints
from utils.render_skeleton import (
    get_humanml3d_chains,
    render_simple_skeleton_video,
    render_skeleton_video,
)


def _get_video_encoder():
    """Detect available H.264 encoder, prefer libx264, fallback to libopenh264."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-encoders"],
            capture_output=True,
            text=True,
            timeout=5
        )
        encoders = result.stdout
        if "libx264" in encoders:
            return "libx264"
        elif "libopenh264" in encoders:
            return "libopenh264"
        else:
            return "libx264"
    except Exception:
        return "libx264"


VIDEO_ENCODER = _get_video_encoder()


def render_single_video(
    motion: np.ndarray,
    save_path: str,
    dim: int,
    render_setting: dict = {},
    mean_np: np.ndarray = None,
    std_np: np.ndarray = None,
    frames: np.ndarray = None,
    pkl_save_path: str = None,
):
    chains = get_humanml3d_chains()
    joint_positions = convert_motion_to_joints(motion, dim, mean_np, std_np)
    
    # Save joint positions to pickle file if path is provided
    if pkl_save_path is not None:
        pkl_dir = os.path.dirname(pkl_save_path)
        if pkl_dir:
            os.makedirs(pkl_dir, exist_ok=True)
        
        pkl_data = {
            'full_pose': joint_positions  # shape: (frames, 22, 3)
        }
        with open(pkl_save_path, 'wb') as f:
            pickle.dump(pkl_data, f)
    
    if render_setting.get("simple", True):
        render_simple_skeleton_video(
            data=joint_positions,
            chains=chains,
            out_path=save_path,
            fps=render_setting.get("fps", 30),
            frames=frames,
            text=render_setting.get("music_name", None),
        )
    else:
        render_skeleton_video(
            data=joint_positions,
            chains=chains,
            out_path=save_path,
            fps=render_setting.get("fps", 30),
            frames=frames,
        )
    
    if render_setting.get("music_path", None) is not None:
        # Add audio to video using ffmpeg
        wave_path = render_setting["music_path"]
        temp_output = save_path.replace(".mp4", "_temp.mp4")
        os.rename(save_path, temp_output)
        cmd = f'ffmpeg -hide_banner -loglevel error -y -i "{temp_output}" -i "{wave_path}" -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 -shortest "{save_path}"'
        os.system(cmd)
        os.remove(temp_output)


def render_video(
    motion_dir: str,
    save_dir: str,
    render_setting,
    frames_dir: str = None,
    pkl_save_dir: str = None,
    music_dir: str = None,
):
    os.makedirs(save_dir, exist_ok=True)
    motion_path = Path(motion_dir)
    npy_files = list(motion_path.glob("*.npy"))

    chains = get_humanml3d_chains()
    motion_count = 0
    error_count = 0

    if render_setting.get("mean_path") is not None:
        mean_np = np.load(render_setting["mean_path"])
        std_np = np.load(render_setting["std_path"])
    else:
        mean_np = np.zeros(render_setting["recover_dim"])
        std_np = np.ones(render_setting["recover_dim"])

    for npy_file in tqdm(npy_files, desc="Rendering"):
        motion_data = np.load(npy_file)
        output_filename = npy_file.stem + ".mp4"
        output_path = os.path.join(save_dir, output_filename)

        if frames_dir is not None and os.path.exists(frames_dir):
            frames_path = os.path.join(frames_dir, npy_file.name)
            frames = np.load(frames_path)
        else:
            frames = None

        # Construct pickle file path if pkl_save_dir is provided
        pkl_path = None
        if pkl_save_dir is not None:
            pkl_filename = f"{motion_count:d}_{npy_file.stem}.pkl"
            pkl_path = os.path.join(pkl_save_dir, pkl_filename)

        # Construct music path if music_dir is provided
        music_path = None
        if music_dir is not None:
            music_path = os.path.join(music_dir, npy_file.stem + ".wav")
            if not os.path.exists(music_path):
                print(f"Warning: Music file not found: {music_path}")
                music_path = None

        # Update render_setting with music_path
        current_render_setting = render_setting.copy()
        if music_path is not None:
            current_render_setting["music_path"] = music_path

        try:
            render_single_video(
                motion=motion_data,
                save_path=output_path,
                dim=render_setting["recover_dim"],
                render_setting=current_render_setting,
                mean_np=mean_np,
                std_np=std_np,
                frames=frames,
                pkl_save_path=pkl_path,
            )
        except Exception as e:
            print(f"Error rendering {npy_file}: {e}")
            error_count += 1
            continue
        motion_count += 1

    print(
        f"{motion_count} motion clips rendered. {error_count} errors. Saved to {save_dir}"
    )


def render_text_bar(
    text, width, padding=20, font_size=28, bg_color=(0, 0, 0), fg_color=(255, 255, 255)
):
    """Renders a text bar with automatic wrapping using PIL, returns np.uint8(H,W,3)."""
    try:
        font_path = font_manager.findfont("DejaVu Sans")
    except Exception:
        font_path = font_manager.findfont("Arial")
    font = ImageFont.truetype(font_path, font_size)

    # Split text by separator if present
    segments = []
    if "//////////" in text:
        parts = text.split("//////////")
        for part in parts:
            if part.strip():
                segments.append(part.strip())
    else:
        segments.append(text)

    # Define tint colors matching render_skeleton.py
    # Using slightly more saturated colors for text to ensure readability against black background
    tint_colors = [
        (255, 180, 180),  # Reddish
        (180, 255, 180),  # Greenish
        (180, 180, 255),  # Blueish
    ]

    dummy = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    max_w = width - 2 * padding

    all_lines = []  # List of (text_content, color) tuples

    for i, segment in enumerate(segments):
        color = fg_color
        if len(segments) > 1:
            color = tint_colors[i % len(tint_colors)]

        # Wrap text for this segment
        cur = ""
        first_word = True
        for w in segment.split():
            test = (cur + " " + w).strip() if cur else w
            if dummy.textlength(test, font=font) <= max_w:
                cur = test
            else:
                all_lines.append((cur, color))
                cur = w
        if cur:
            all_lines.append((cur, color))

    _, top, _, bottom = font.getbbox("A")
    line_h = bottom - top + 4
    bar_h = padding * 2 + line_h * len(all_lines)

    # Ensure height is even for H.264 encoding
    if bar_h % 2 != 0:
        bar_h += 1

    img = Image.new("RGB", (width, bar_h), bg_color)
    draw = ImageDraw.Draw(img)
    y = padding
    for line_text, line_color in all_lines:
        draw.text((padding, y), line_text, font=font, fill=line_color)
        y += line_h
    return np.array(img)


def render_aligned_title_bar(
    total_width,
    widths,
    titles,
    font_size=32,
    bg_color=(255, 255, 255),
    fg_color=(0, 0, 0),
    padding=10,
):
    """Renders a title bar with centered titles aligned to video sections."""
    try:
        font_path = font_manager.findfont("DejaVu Sans")
    except Exception:
        font_path = font_manager.findfont("Arial")
    font = ImageFont.truetype(font_path, font_size)

    # Calculate title bar height
    _, top, _, bottom = font.getbbox("A")
    bar_height = bottom - top + 2 * padding

    # Ensure height is even for H.264 encoding
    if bar_height % 2 != 0:
        bar_height += 1

    # Create image
    img = Image.new("RGB", (total_width, bar_height), bg_color)
    draw = ImageDraw.Draw(img)

    # Calculate positions and draw titles
    x_offset = 0
    for i, (title, width) in enumerate(zip(titles, widths)):
        # Calculate center position for this section
        text_width = draw.textlength(title, font=font)
        x_center = x_offset + width // 2
        x_pos = x_center - text_width // 2
        y_pos = padding

        draw.text((x_pos, y_pos), title, font=font, fill=fg_color)
        x_offset += width

    return np.array(img)


def _get_video_info(video_path: str):
    """Get video width, height, and duration using ffprobe."""
    import subprocess

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,duration",
        "-of",
        "csv=p=0",
        video_path,
    ]
    output = subprocess.check_output(cmd, text=True).strip().split(",")
    return int(output[0]), int(output[1]), float(output[2])


def _get_fps(video_path: str):
    """Get video frame rate using ffprobe."""
    import subprocess

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "csv=p=0",
        video_path,
    ]
    fps_str = subprocess.check_output(cmd, text=True).strip()
    num, den = map(int, fps_str.split("/"))
    return num / den


def _build_video_filter(
    input_idx: int,
    video_idx: int,
    target_height: int,
    duration: float,
    max_duration: float,
    target_width: int = -2,
):
    """Build ffmpeg filter for a single video stream."""
    filters = []

    # Scale to target height, ensure even dimensions
    # filters.append(f"[{input_idx}:v]scale=-2:{target_height}[v{video_idx}_scaled]")
    filters.append(
        f"[{input_idx}:v]scale={target_width}:{target_height}[v{video_idx}_scaled]"
    )

    # Handle duration and graying
    if duration < max_duration:
        pad_duration = max_duration - duration
        filters.append(
            f"[v{video_idx}_scaled]tpad=stop_mode=clone:stop_duration={pad_duration}[v{video_idx}_padded]"
        )
        # Apply gray effect after original duration using hue filter with enable
        # Note: eq filter might not be available in some ffmpeg builds
        filters.append(
            f"[v{video_idx}_padded]hue=b=-0.5:s=0:enable='gte(t,{duration})'[v{video_idx}]"
        )
    else:
        # Video is already long enough, just use scaled version
        filters.append(f"[v{video_idx}_scaled]null[v{video_idx}]")

    return filters


def make_composite_compare_videos(
    result_folder: str,
    save_dir: str,
    music_name_folder: str = None,
    music_dir: str = None,
    # text_folder: str = None,
    compare_folders: list = None,
    compare_names: list = None,
):
    """Generates composite videos of (result | compare_folders) with captions and music.

    Args:
        result_folder: Folder containing result videos (base for comparison)
        save_dir: Directory to save composite videos
        music_name_folder: Folder containing music name txt files (optional)
        music_dir: Directory containing music wav files (optional)
        # text_folder: Folder containing text descriptions (optional)
        compare_folders: List of folders to compare with result (optional)
        compare_names: List of names for compare folders (optional)

    Uses the longest video duration. Missing videos show black screen.
    Videos that end early show their last frame grayed out.
    Optimized version using ffmpeg directly for much faster processing.
    """
    import subprocess
    import tempfile

    os.makedirs(save_dir, exist_ok=True)
    video_files = list(Path(result_folder).glob("*.mp4"))

    # Handle empty or non-existent compare folders
    if compare_folders is None:
        compare_folders = []
    if compare_names is None:
        compare_names = []

    # Filter out non-existent compare folders
    valid_compare_folders = []
    valid_compare_names = []
    for i, folder in enumerate(compare_folders):
        if folder and os.path.exists(folder):
            valid_compare_folders.append(folder)
            if i < len(compare_names):
                valid_compare_names.append(compare_names[i])
            else:
                valid_compare_names.append(f"Compare {i + 1}")

    compare_folders = valid_compare_folders
    compare_names = valid_compare_names

    for video_file in tqdm(video_files, desc="Creating composite videos"):
        video_id = video_file.stem

        # Prepare video paths - start with result, then add compare folders
        video_paths = [str(video_file)]
        video_names = ["Ours"]

        # Add compare folder videos
        for folder, name in zip(compare_folders, compare_names):
            compare_path = os.path.join(folder, f"{video_id}.mp4")
            video_paths.append(compare_path)
            video_names.append(name)

        video_exists = [os.path.exists(p) for p in video_paths]

        # Load music name
        music_name = None
        music_path = None
        if music_name_folder:
            music_name_file = os.path.join(music_name_folder, f"{video_id}.txt")
            if os.path.exists(music_name_file):
                music_name = Path(music_name_file).read_text().strip()
            else:
                music_name = f"Music: {video_id}"
        else:
            music_name = f"Music: {video_id}"
        
        # Get music wav path
        if music_dir and music_name:
            music_path = os.path.join(music_dir, f"{music_name}.wav")
            if not os.path.exists(music_path):
                print(f"Warning: Music file not found: {music_path}")
                music_path = None

        # # Load text description
        # if text_folder:
        #     text_file = os.path.join(text_folder, f"{video_id}.txt")
        #     text_content = (
        #         Path(text_file).read_text().strip()
        #         if os.path.exists(text_file)
        #         else f"Motion: {video_id}"
        #     )
        # else:
        #     text_content = f"Motion: {video_id}"

        # Find reference video for properties (should always have result video)
        reference_video = str(video_file)
        if not os.path.exists(reference_video):
            print(f"Error: Result video not found for {video_id}, skipping")
            continue

        # Get video properties
        try:
            fps = _get_fps(reference_video)
            ref_width, ref_height, _ = _get_video_info(reference_video)
        except Exception as e:
            print(f"Error probing {video_id}: {e}, skipping")
            continue

        # Collect dimensions and durations for all videos
        widths, heights, durations = [], [], []
        for path, exists, name in zip(video_paths, video_exists, video_names):
            if exists:
                try:
                    w, h, d = _get_video_info(path)
                    widths.append(w)
                    heights.append(h)
                    durations.append(d)
                except Exception as e:
                    print(
                        f"Error probing {name} video for {video_id}: {e}, will use black screen"
                    )
                    widths.append(ref_width)
                    heights.append(ref_height)
                    durations.append(0)
            else:
                print(
                    f"Warning: {name} video missing for {video_id}, will use black screen"
                )
                widths.append(ref_width)
                heights.append(ref_height)
                durations.append(0)

        max_duration = max(durations)
        if max_duration == 0:
            print(f"Warning: All videos for {video_id} have zero duration, skipping")
            continue

        target_height = min(h for h in heights if h > 0)
        # Ensure target height is even
        if target_height % 2 != 0:
            target_height += 1

        # Re-calculate widths based on target_height scaling
        new_widths = []
        for w, h in zip(widths, heights):
            if h > 0:
                # Calculate scaled width maintaining aspect ratio
                aspect_ratio = w / h
                scaled_w = int(target_height * aspect_ratio)
                # Ensure even width
                if scaled_w % 2 != 0:
                    scaled_w += 1
                new_widths.append(scaled_w)
            else:
                new_widths.append(w)

        widths = new_widths
        total_width = sum(widths)

        # print(f"DEBUG: video_id={video_id}")
        # print(f"DEBUG: original widths={widths} (after update), heights={heights}")
        # print(f"DEBUG: target_height={target_height}")
        # print(f"DEBUG: total_width={total_width}")

        # Create and save title/music bars
        title_bar = render_aligned_title_bar(
            total_width, widths, video_names, font_size=32
        )
        # Use music_name as the bottom bar content
        music_bar = render_text_bar(music_name, width=total_width, font_size=24)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            title_path = f.name
            Image.fromarray(title_bar).save(title_path)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            music_bar_path = f.name
            Image.fromarray(music_bar).save(music_bar_path)

        # Build ffmpeg command
        inputs = []
        filter_parts = []
        input_count = 0

        # Add video inputs and filters
        for i, (path, exists, width, duration) in enumerate(
            zip(video_paths, video_exists, widths, durations)
        ):
            if exists:
                inputs.extend(["-i", path])
                filter_parts.extend(
                    _build_video_filter(
                        input_count,
                        i,
                        target_height,
                        duration,
                        max_duration,
                        target_width=width,
                    )
                )
            else:
                inputs.extend(
                    [
                        "-f",
                        "lavfi",
                        "-i",
                        f"color=c=black:s={width}x{target_height}:d={max_duration}:r={int(fps)}",
                    ]
                )
                filter_parts.append(f"[{input_count}:v]null[v{i}]")
            input_count += 1

        # Add title and music bar images
        inputs.extend(["-loop", "1", "-i", title_path])
        title_idx = input_count
        input_count += 1

        inputs.extend(["-loop", "1", "-i", music_bar_path])
        music_bar_idx = input_count

        # Compose final layout - horizontally stack all videos
        num_videos = len(video_paths)
        if num_videos == 1:
            filter_parts.append("[v0]null[videos]")
        else:
            video_inputs = "".join([f"[v{i}]" for i in range(num_videos)])
            filter_parts.append(f"{video_inputs}hstack=inputs={num_videos}[videos]")

        filter_parts.append(
            f"[{title_idx}:v][videos][{music_bar_idx}:v]vstack=inputs=3[out]"
        )

        # Execute ffmpeg to create video without audio first
        output_path = os.path.join(save_dir, f"{video_id}_composite.mp4")
        cmd = [
            "ffmpeg",
            "-y",
            *inputs,
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "[out]",
            "-t",
            str(max_duration),
            "-r",
            str(int(fps)),
            "-c:v",
            VIDEO_ENCODER,
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            output_path,
        ]

        try:
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error processing {video_id}: Return code {result.returncode}")
                print(f"Command: {' '.join(cmd)}")
                print(f"Stderr: {result.stderr}")
            elif os.path.exists(output_path) and os.path.getsize(output_path) == 0:
                print(f"Warning: Generated video {output_path} is empty!")
                print(f"Command: {' '.join(cmd)}")
                print(f"Stderr: {result.stderr}")
            else:
                # Add audio to video if music_path is available
                if music_path:
                    temp_output = output_path.replace(".mp4", "_temp.mp4")
                    os.rename(output_path, temp_output)
                    audio_cmd = f'ffmpeg -hide_banner -loglevel error -y -i "{temp_output}" -i "{music_path}" -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 -shortest "{output_path}"'
                    os.system(audio_cmd)
                    os.remove(temp_output)
        except Exception as e:
            print(f"Unexpected error processing {video_id}: {e}")
        finally:
            for path in [title_path, music_bar_path]:
                try:
                    os.unlink(path)
                except Exception:
                    pass
                

    print(f"Composite videos saved to {save_dir}")
