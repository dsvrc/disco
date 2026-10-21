"""
Visualize audio manifolds from pre-extracted .npy features

Usage:
conda activate music2dance && \
python tmp/visualize_from_npy.py \
    --feature_dir tmp/manifold_exp_011 \
    --audio_name gBR_sBM_cAll_d04_mBR0_ch01 \
    --output tmp/manifold_exp_011/visualization_v2.png
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class Plotter:
    def __init__(self) -> None:
        self.pca = PCA(n_components=2)

    def plot_pca(self, ax, manifold_collection, title=None, point_color=None, draw_line=True, draw_arrow=True):
        ax.cla()

        point_alpha = 0.3
        line_alpha = 0.2
        arrow_alpha = 1.0
        arrow_step = 50  ## Original value for cleaner visualization
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
                        manifold[j + 1, 0] - manifold[j, 0],  ## Fixed: j+1 not j+arrow_step
                        manifold[j + 1, 1] - manifold[j, 1],
                        alpha=arrow_alpha, width=d * arrow_size, color=arrow_color
                    )
        ax.legend()
        ax.set_axis_off()
        if title != None:
            ax.set_title(title)


def main(feature_dir, audio_name, output_path):
    """Load pre-extracted features and visualize"""
    
    print(f"\n{'='*60}")
    print(f"Visualizing from pre-extracted features")
    print(f"Feature directory: {feature_dir}")
    print(f"Audio name: {audio_name}")
    print(f"{'='*60}\n")
    
    ## Load features
    print("Loading features...")
    original_feats = np.load(os.path.join(feature_dir, f"{audio_name}_original.npy"))
    librosa_feats = np.load(os.path.join(feature_dir, f"{audio_name}_librosa.npy"))
    pae_feats = np.load(os.path.join(feature_dir, f"{audio_name}_pae.npy"))
    vqpae_feats = np.load(os.path.join(feature_dir, f"{audio_name}_vqpae.npy"))
    
    print(f"  Original: {original_feats.shape}")
    print(f"  Librosa:  {librosa_feats.shape}")
    print(f"  PAE:      {pae_feats.shape}")
    print(f"  VQPAE:    {vqpae_feats.shape}")
    
    ## Downsample to same length
    min_len = min(len(original_feats), len(librosa_feats), len(pae_feats), len(vqpae_feats))
    print(f"\nDownsampling to min_len={min_len} frames")
    
    original_feats = original_feats[:min_len]
    librosa_feats = librosa_feats[:min_len]
    pae_feats = pae_feats[:min_len]
    vqpae_feats = vqpae_feats[:min_len]
    
    ## Create visualization
    print("\nCreating visualization...")
    plotter = Plotter()
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    ## Plot each method as single continuous trajectory
    plotter.plot_pca(axes[0], [torch.from_numpy(original_feats)], title="(a) Original")
    plotter.plot_pca(axes[1], [torch.from_numpy(librosa_feats)], title="(b) Librosa")
    plotter.plot_pca(axes[2], [torch.from_numpy(pae_feats)], title="(c) PAE")
    plotter.plot_pca(axes[3], [torch.from_numpy(vqpae_feats)], title="(d) VQPAE")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    print(f"\n{'='*60}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize from pre-extracted features")
    parser.add_argument("--feature_dir", required=True, help="Directory containing .npy feature files")
    parser.add_argument("--audio_name", required=True, help="Audio file basename (without extension)")
    parser.add_argument("--output", required=True, help="Output visualization path")
    
    args = parser.parse_args()
    
    main(args.feature_dir, args.audio_name, args.output)
