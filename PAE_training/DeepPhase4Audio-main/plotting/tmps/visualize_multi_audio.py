"""
Visualize manifolds from multiple audio files

Usage:
python tmp/visualize_multi_audio.py \
    --feature_dir tmp/manifold_exp_011 \
    --num_audios 5 \
    --output tmp/manifold_exp_011/final_visualization.png
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from pathlib import Path

class Plotter:
    def __init__(self) -> None:
        self.pca = PCA(n_components=2)

    def plot_pca(self, ax, manifold_collection, title=None, point_color=None, draw_line=True, draw_arrow=True):
        ax.cla()

        point_alpha = 0.3
        line_alpha = 0.2
        arrow_alpha = 1.0
        arrow_step = 50  ## Draw arrow every 50 points
        arrow_size = 0.015
        arrow_power = 1.0
        arrow_color = (0.25, 0.25, 0.5)

        num_steps = [len(manifold) for manifold in manifold_collection]
        manifolds = torch.cat(manifold_collection, dim=0).cpu()
        
        manifolds_pca = torch.tensor(self.pca.fit_transform(manifolds)).split(num_steps, dim=0)

        for i, manifold in enumerate(manifolds_pca):
            if draw_line:
                ax.plot(manifold[:, 0], manifold[:, 1], color=(0.0, 0.0, 0.0), alpha=line_alpha)
            if point_color is None:
                ax.scatter(manifold[:, 0], manifold[:, 1], alpha=point_alpha, label=i)
            else:
                ax.scatter(manifold[:, 0], manifold[:, 1], color=point_color[i], alpha=point_alpha, label=i)
            if draw_arrow:
                for j in range(0, len(manifold) - arrow_step, arrow_step):
                    d = torch.norm(manifold[j, :], dim=-1)
                    d = torch.pow(d, arrow_power)
                    ax.arrow(
                        manifold[j, 0],
                        manifold[j, 1],
                        manifold[j + 1, 0] - manifold[j, 0],  ## Arrow to next point
                        manifold[j + 1, 1] - manifold[j, 1],
                        alpha=arrow_alpha, width=d * arrow_size, color=arrow_color
                    )
        ax.legend()
        ax.set_axis_off()
        if title != None:
            ax.set_title(title)


def main(feature_dir, num_audios, output_path):
    """Load and visualize multiple audio features"""
    
    print(f"\n{'='*60}")
    print(f"Visualizing {num_audios} audio files")
    print(f"Feature directory: {feature_dir}")
    print(f"{'='*60}\n")
    
    ## Find all audio files
    npy_files = sorted(Path(feature_dir).glob("*_original.npy"))
    audio_names = [f.stem.replace("_original", "") for f in npy_files[:num_audios]]
    
    print(f"Found {len(audio_names)} audio files:")
    for name in audio_names:
        print(f"  - {name}")
    
    ## Load features for each audio
    original_manifolds = []
    librosa_manifolds = []
    pae_manifolds = []
    vqpae_manifolds = []
    
    for audio_name in audio_names:
        print(f"\nLoading: {audio_name}")
        try:
            original = np.load(os.path.join(feature_dir, f"{audio_name}_original.npy"))
            librosa_feat = np.load(os.path.join(feature_dir, f"{audio_name}_librosa.npy"))
            pae = np.load(os.path.join(feature_dir, f"{audio_name}_pae.npy"))
            vqpae = np.load(os.path.join(feature_dir, f"{audio_name}_vqpae.npy"))
            
            ## Downsample to same length
            min_len = min(len(original), len(librosa_feat), len(pae), len(vqpae))
            
            original_manifolds.append(torch.from_numpy(original[:min_len]).float())
            librosa_manifolds.append(torch.from_numpy(librosa_feat[:min_len]).float())
            pae_manifolds.append(torch.from_numpy(pae[:min_len]).float())
            vqpae_manifolds.append(torch.from_numpy(vqpae[:min_len]).float())
            
            print(f"  ✓ Loaded (min_len={min_len})")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue
    
    if len(original_manifolds) == 0:
        print("\n[ERROR] No valid features found!")
        return
    
    print(f"\n{'='*60}")
    print(f"Creating visualization with {len(original_manifolds)} audio files...")
    print(f"{'='*60}\n")
    
    ## Create colors
    colors = list(mcolors.TABLEAU_COLORS.values())[:len(original_manifolds)]
    
    ## Create plot
    plotter = Plotter()
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    ## Plot each method
    print("Plotting manifolds...")
    plotter.plot_pca(axes[0], original_manifolds, title="(a) Original", point_color=colors)
    plotter.plot_pca(axes[1], librosa_manifolds, title="(b) Librosa", point_color=colors)
    plotter.plot_pca(axes[2], pae_manifolds, title="(c) PAE", point_color=colors)
    plotter.plot_pca(axes[3], vqpae_manifolds, title="(d) VQPAE", point_color=colors)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    print(f"\n{'='*60}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize multiple audio manifolds")
    parser.add_argument("--feature_dir", required=True, help="Directory with .npy files")
    parser.add_argument("--num_audios", type=int, default=5, help="Number of audio files")
    parser.add_argument("--output", required=True, help="Output visualization path")
    
    args = parser.parse_args()
    
    main(args.feature_dir, args.num_audios, args.output)
