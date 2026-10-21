import argparse
import os

from utils.visualize import render_video


def main():
    parser = argparse.ArgumentParser(description="Render motion feature files to videos.")
    parser.add_argument(
        "--feature_dir",
        type=str,
        required=True,
        help="Directory containing .npy motion feature files.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save rendered videos.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=263,
        help="Motion feature dimension (e.g. 263 or 272).",
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    render_setting = {}
    render_setting["recover_dim"] = args.dim
    render_setting["simple"] = True

    print(f"Rendering {args.feature_dir} to {args.save_dir}")
    render_video(
        motion_dir=args.feature_dir,
        save_dir=args.save_dir,
        render_setting=render_setting,
    )


if __name__ == "__main__":
    main()
