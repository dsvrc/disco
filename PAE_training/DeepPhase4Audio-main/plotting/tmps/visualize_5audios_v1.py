"""
Visualize audio manifolds for 5 audio files - Version 1
Reference: correct dense visualization with multiple colored trajectories

Usage:
conda activate music2dance && \
python tmp/visualize_5audios_v1.py \
    --audio_dir raw_data/aistpp/wavs \
    --num_samples 5 \
    --output tmp/audio_manifolds_5audios_v1.png
"""

import os
import sys
from pathlib import Path

import numpy as np
import librosa
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA

## Add PAE path
pae_parent_path = Path(__file__).parent.parent / "PAE_training" / "DeepPhase4Audio-main"
sys.path.insert(0, str(pae_parent_path))


class Plotter:
    def __init__(self) -> None:
        self.pca = PCA(n_components=2)

    def plot_pca(self, ax, manifold_collection, title=None, point_color=None, draw_line=True, draw_arrow=True):
        ax.cla()

        point_alpha = 0.3
        line_alpha = 0.2
        arrow_alpha = 1.0
        arrow_step = 50
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
                        manifold[j + 1, 0] - manifold[j, 0],
                        manifold[j + 1, 1] - manifold[j, 1],
                        alpha=arrow_alpha, width=d * arrow_size, color=arrow_color
                    )
        ax.legend()
        ax.set_axis_off()
        if title != None:
            ax.set_title(title)


def extract_original_features(audio_path, sr=16000):
    """Extract original mel spectrogram features - DENSE sampling"""
    y, sr = librosa.load(audio_path, sr=sr)
    
    ## Use same hop_length as reference: 512 samples = 0.032 seconds at 16kHz
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    features = mel_spec_db.T  ## Shape: (time, n_mels=128)
    return features


def extract_librosa_features(audio_path, sr=16000):
    """Extract librosa hand-crafted features - DENSE sampling"""
    y, sr = librosa.load(audio_path, sr=sr)
    
    ## Use same hop_length as reference: 512 samples
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=512).T
    chroma = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=512, n_chroma=12).T
    
    features = np.concatenate([mfcc, chroma], axis=1)  ## (time, 32)
    return features


def load_pae_model(checkpoint_path, config_path):
    """Load PAE model"""
    from src.utils.config import load_config
    from src.utils.helpers import resolve_model_class
    
    cfg = load_config(config_path, flatten=False)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model = resolve_model_class(cfg.model_config.model_name, cfg.model_config)
    
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for key, val in state_dict.items():
        if key.startswith('model.'):
            new_state_dict[key[6:]] = val
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    return model, cfg


def extract_pae_features(audio_path, model, cfg, sr=16000):
    """Extract PAE latent features - DENSE sampling"""
    y, sr = librosa.load(audio_path, sr=sr)
    
    window_size = 32000  ## 2 seconds @ 16kHz
    hop_size = 512  ## DENSE: 0.032 seconds for high resolution
    
    features = []
    with torch.no_grad():
        for i in range(0, len(y) - window_size + 1, hop_size):
            chunk = y[i:i + window_size]
            chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0).unsqueeze(0)
            
            _, latent, _, _ = model(chunk_tensor)
            latent_avg = latent.mean(dim=2).squeeze(0)
            features.append(latent_avg.cpu().numpy())
    
    if len(features) == 0:
        chunk_tensor = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0)
        if chunk_tensor.shape[2] < window_size:
            pad_size = window_size - chunk_tensor.shape[2]
            chunk_tensor = torch.nn.functional.pad(chunk_tensor, (0, pad_size))
        _, latent, _, _ = model(chunk_tensor)
        latent_avg = latent.mean(dim=2).squeeze(0)
        features.append(latent_avg.cpu().numpy())
    
    features = np.stack(features, axis=0)
    return features


def load_vqpae_model(checkpoint_path, config_path):
    """Load VQ-PAE model"""
    ## Add src path for models import
    pae_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../PAE_training/DeepPhase4Audio-main/src'))
    if pae_src_path not in sys.path:
        sys.path.insert(0, pae_src_path)
    
    models_backup = sys.modules.pop('models', None)
    
    try:
        from models.VQ_PAE import VQ_AE
    finally:
        if models_backup is not None:
            sys.modules['models'] = models_backup
    
    from src.utils.config import load_config
    
    cfg = load_config(config_path, flatten=False)
    
    model = VQ_AE(cfg.model_config)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model, cfg


def extract_vqpae_features(audio_path, model, sr=16000, output_window_size=180):
    """Extract VQPAE features - MODERATE sampling (still slower)"""
    y, sr = librosa.load(audio_path, sr=sr)
    
    window_size = 96000  ## 6 seconds @ 16kHz
    hop_size = 3200  ## 0.2 seconds - balance between density and speed
    
    features = []
    with torch.no_grad():
        for i in range(0, len(y) - window_size + 1, hop_size):
            chunk = y[i:i + window_size]
            chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0)
            
            output = model(chunk_tensor)
            z_pae = output['z_pae']
            z_vq = output['z_vq']
            
            z_pae_down = torch.nn.functional.adaptive_avg_pool1d(z_pae, output_size=output_window_size)
            z_vq_down = torch.nn.functional.adaptive_avg_pool1d(z_vq, output_size=output_window_size)
            
            z_concat = torch.cat([z_vq_down, z_pae_down], dim=1)
            z_concat = z_concat.squeeze(0).permute(1, 0).cpu().numpy()
            
            z_avg = z_concat.mean(axis=0)
            features.append(z_avg)
    
    if len(features) == 0:
        chunk_tensor = torch.from_numpy(y).float().unsqueeze(0)
        if chunk_tensor.shape[1] < window_size:
            pad_size = window_size - chunk_tensor.shape[1]
            chunk_tensor = torch.nn.functional.pad(chunk_tensor, (0, pad_size))
        
        output = model(chunk_tensor)
        z_pae = output['z_pae']
        z_vq = output['z_vq']
        z_pae_down = torch.nn.functional.adaptive_avg_pool1d(z_pae, output_size=output_window_size)
        z_vq_down = torch.nn.functional.adaptive_avg_pool1d(z_vq, output_size=output_window_size)
        z_concat = torch.cat([z_vq_down, z_pae_down], dim=1)
        z_concat = z_concat.squeeze(0).permute(1, 0).cpu().numpy()
        z_avg = z_concat.mean(axis=0)
        features.append(z_avg)
    
    features = np.stack(features, axis=0)
    return features


def main(audio_dir, num_samples, output_path, 
         pae_checkpoint="PAE_training/DeepPhase4Audio-main/output/20260103_213749_paeflat_twinkle_200eps/checkpoints/step_epoch=1999.ckpt",
         pae_config="PAE_training/DeepPhase4Audio-main/configs/paeflat200.yaml",
         vqpae_checkpoint="PAE_training/DeepPhase4Audio-main/output/20260111_023242_VQ_PAE-aistpp_1000_eps-8emb/checkpoints/step_epoch=199.ckpt",
         vqpae_config="PAE_training/DeepPhase4Audio-main/configs/vqpae_aistpp_1000_eps_8emb.yaml"):
    """
    Main function to extract and visualize features from multiple audio files
    """
    
    ## Load audio files
    audio_files = sorted(Path(audio_dir).glob("*.wav"))[:num_samples]
    if len(audio_files) == 0:
        audio_files = sorted(Path(audio_dir).glob("*.mp3"))[:num_samples]
    
    print(f"Found {len(audio_files)} audio files\n")
    
    ## Load models
    print("Loading models...")
    pae_model, pae_cfg = load_pae_model(pae_checkpoint, pae_config)
    print(f"  PAE model loaded (embedding_channels={pae_cfg.model_config.embedding_channels})")
    
    vqpae_model, vqpae_cfg = load_vqpae_model(vqpae_checkpoint, vqpae_config)
    print(f"  VQPAE model loaded (embedding_channels={vqpae_cfg.model_config.embedding_channels})")
    print()
    
    ## Extract features
    original_manifolds = []
    librosa_manifolds = []
    pae_manifolds = []
    vqpae_manifolds = []
    
    for i, audio_file in enumerate(audio_files):
        print(f"[{i+1}/{len(audio_files)}] Processing: {audio_file.name}")
        
        ## Original
        original_feat = extract_original_features(str(audio_file))
        original_manifolds.append(torch.from_numpy(original_feat).float())
        print(f"  Original: {original_feat.shape}")
        
        ## Librosa
        librosa_feat = extract_librosa_features(str(audio_file))
        librosa_manifolds.append(torch.from_numpy(librosa_feat).float())
        print(f"  Librosa: {librosa_feat.shape}")
        
        ## PAE
        pae_feat = extract_pae_features(str(audio_file), pae_model, pae_cfg)
        pae_manifolds.append(torch.from_numpy(pae_feat).float())
        print(f"  PAE: {pae_feat.shape}")
        
        ## VQPAE
        vqpae_feat = extract_vqpae_features(str(audio_file), vqpae_model)
        vqpae_manifolds.append(torch.from_numpy(vqpae_feat).float())
        print(f"  VQPAE: {vqpae_feat.shape}")
        print()
    
    ## Visualize
    print("Creating visualization...")
    plotter = Plotter()
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    colors = list(mcolors.TABLEAU_COLORS.values())[:len(audio_files)]
    
    plotter.plot_pca(axes[0], original_manifolds, title="(a) Original", point_color=colors)
    plotter.plot_pca(axes[1], librosa_manifolds, title="(b) Librosa", point_color=colors)
    plotter.plot_pca(axes[2], pae_manifolds, title="(c) PAE", point_color=colors)
    plotter.plot_pca(axes[3], vqpae_manifolds, title="(d) VQPAE", point_color=colors)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize 5 audio manifolds - Version 1")
    parser.add_argument("--audio_dir", required=True, help="Directory containing audio files")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of audio samples to process")
    parser.add_argument("--output", default="tmp/audio_manifolds_5audios_v1.png", help="Output visualization path")
    
    args = parser.parse_args()
    
    main(args.audio_dir, args.num_samples, args.output)
