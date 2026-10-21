"""
Audio Manifold Visualization Experiment

Usage:
conda activate music2dance && \
python tmp/audio_manifold_experiment.py \
    --audio raw_data/aistpp/wavs/gBR_sBM_cAll_d04_mBR0_ch01.wav \
    --output_dir tmp/manifold_exp_001

This script:
1. Extracts features from audio using 4 methods: Original, Librosa, PAE, VQPAE
2. Saves features as .npy files
3. Visualizes manifolds using PCA
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import librosa
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

## Add PAE path
pae_parent_path = Path(__file__).parent.parent / "PAE_training" / "DeepPhase4Audio-main"
sys.path.insert(0, str(pae_parent_path))

## Plotter class for visualization
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
                        manifold[j + arrow_step, 0] - manifold[j, 0],
                        manifold[j + arrow_step, 1] - manifold[j, 1],
                        alpha=arrow_alpha, width=d * arrow_size, color=arrow_color
                    )
        ax.legend()
        ax.set_axis_off()
        if title != None:
            ax.set_title(title)


## Feature extraction functions

def extract_original_features(audio_path, sr=16000):
    """Extract original mel spectrogram features"""
    print(f"  [Original] Loading audio...")
    y, sr = librosa.load(audio_path, sr=sr)
    
    print(f"  [Original] Computing mel spectrogram...")
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    features = mel_spec_db.T  ## Shape: (time, n_mels=128)
    print(f"  [Original] Features shape: {features.shape}")
    return features


def extract_librosa_features(audio_path, sr=16000):
    """Extract librosa hand-crafted features"""
    print(f"  [Librosa] Loading audio...")
    y, sr = librosa.load(audio_path, sr=sr)
    
    print(f"  [Librosa] Extracting features...")
    ## MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=160).T  ## Reduced hop
    
    ## Chroma
    chroma = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=160, n_chroma=12).T  ## Reduced hop
    
    ## Concatenate
    features = np.concatenate([mfcc, chroma], axis=1)  ## (time, 32)
    print(f"  [Librosa] Features shape: {features.shape}")
    return features


def load_pae_model(checkpoint_path, config_path):
    """Load PAE model using original config file"""
    from src.utils.config import load_config
    from src.utils.helpers import resolve_model_class
    
    ## Load config from original config file
    cfg = load_config(config_path, flatten=False)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"  Using config: {config_path}")
    print(f"  Model: {cfg.model_config.model_name}")
    print(f"  Embedding channels: {cfg.model_config.embedding_channels}")
    print(f"  Time range: {cfg.model_config.time_range}")
    
    ## Use resolve_model_class to get the right model
    model = resolve_model_class(cfg.model_config.model_name, cfg.model_config)
    
    ## Load state dict (remove 'model.' prefix)
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for key, val in state_dict.items():
        if key.startswith('model.'):
            new_state_dict[key[6:]] = val
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    return model, cfg
    
    ## Load state dict (remove 'model.' prefix)
    new_state_dict = {}
    for key, val in state_dict.items():
        if key.startswith('model.'):
            new_state_dict[key[6:]] = val
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    return model, cfg


def extract_pae_features(audio_path, model, cfg, sr=16000):
    """Extract PAE latent features"""
    print(f"  [PAE] Loading audio...")
    y, sr = librosa.load(audio_path, sr=sr)
    
    window_size = 32000  ## 2 seconds @ 16kHz
    hop_size = 1600  ## 0.1 second hop (balanced speed and quality)
    
    print(f"  [PAE] Extracting latent features...")
    features = []
    with torch.no_grad():
        for i in range(0, len(y) - window_size + 1, hop_size):
            chunk = y[i:i + window_size]
            chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0).unsqueeze(0)
            
            _, latent, _, _ = model(chunk_tensor)  ## (1, embedding_channels, time)
            latent_avg = latent.mean(dim=2).squeeze(0)  ## (embedding_channels,)
            features.append(latent_avg.cpu().numpy())
    
    if len(features) == 0:
        ## Audio too short, process whole thing
        chunk_tensor = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0)
        if chunk_tensor.shape[2] < window_size:
            pad_size = window_size - chunk_tensor.shape[2]
            chunk_tensor = torch.nn.functional.pad(chunk_tensor, (0, pad_size))
        _, latent, _, _ = model(chunk_tensor)
        latent_avg = latent.mean(dim=2).squeeze(0)
        features.append(latent_avg.cpu().numpy())
    
    features = np.stack(features, axis=0)  ## (n_frames, embedding_channels)
    print(f"  [PAE] Features shape: {features.shape}")
    return features


def load_vqpae_model(checkpoint_path, config_path):
    """Load VQ-PAE model using original config file (reference: VQPAE_preprocess.py)"""
    import sys
    
    ## Add PAE src path for models import
    pae_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../PAE_training/DeepPhase4Audio-main/src'))
    if pae_src_path not in sys.path:
        sys.path.insert(0, pae_src_path)
    
    ## Backup 'models' module to avoid conflict
    models_backup = sys.modules.pop('models', None)
    
    try:
        from models.VQ_PAE import VQ_AE  ## Import from PAE's models
    finally:
        ## Restore project's models module
        if models_backup is not None:
            sys.modules['models'] = models_backup
    
    from src.utils.config import load_config
    
    ## Load config from original config file
    cfg = load_config(config_path, flatten=False)
    
    print(f"  Using config: {config_path}")
    print(f"  Model: {cfg.model_config.model_name}")
    print(f"  Embedding channels: {cfg.model_config.embedding_channels}")
    print(f"  Time range: {cfg.model_config.time_range}")
    
    ## Initialize model with config
    model = VQ_AE(cfg.model_config)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    ## Remove "model." prefix
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
    """Extract VQPAE features (reference: VQPAE_preprocess.py)"""
    print(f"  [VQPAE] Loading audio...")
    y, sr = librosa.load(audio_path, sr=sr)
    
    window_size = 96000  ## 6 seconds @ 16kHz
    hop_size = 3200  ## 0.2 second hop (balanced speed and quality)
    
    print(f"  [VQPAE] Extracting VQ-PAE latent features...")
    features = []
    with torch.no_grad():
        for i in range(0, len(y) - window_size + 1, hop_size):
            chunk = y[i:i + window_size]
            chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0)
            
            output = model(chunk_tensor)
            z_pae = output['z_pae']  ## (1, 8, 96000)
            z_vq = output['z_vq']  ## (1, 8, 96000)
            
            ## Downsample to output_window_size
            z_pae_down = torch.nn.functional.adaptive_avg_pool1d(z_pae, output_size=output_window_size)
            z_vq_down = torch.nn.functional.adaptive_avg_pool1d(z_vq, output_size=output_window_size)
            
            ## Concatenate: (1, 16, output_window_size)
            z_concat = torch.cat([z_vq_down, z_pae_down], dim=1)
            z_concat = z_concat.squeeze(0).permute(1, 0).cpu().numpy()  ## (output_window_size, 16)
            
            ## Take average over time for this window
            z_avg = z_concat.mean(axis=0)  ## (16,)
            features.append(z_avg)
    
    if len(features) == 0:
        ## Audio too short
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
    
    features = np.stack(features, axis=0)  ## (n_frames, 16)
    print(f"  [VQPAE] Features shape: {features.shape}")
    return features


def main(audio_paths, output_dir, pae_checkpoint, vqpae_checkpoint, pae_config, vqpae_config):
    """Main experiment pipeline - support multiple audio files"""
    ## Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Audio Manifold Experiment: {len(audio_paths)} audio files")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    ## Step 1: Extract features for all audio files
    print("Step 1: Extracting features...")
    print("-" * 60)
    
    all_original_feats = []
    all_librosa_feats = []
    all_pae_feats = []
    all_vqpae_feats = []
    
    ## Load models once
    print("\nLoading models...")
    try:
        pae_model, pae_cfg = load_pae_model(pae_checkpoint, pae_config)
        print(f"  PAE model loaded (embedding_channels={pae_cfg.model_config.embedding_channels})")
    except Exception as e:
        print(f"  [ERROR] PAE model loading failed: {e}")
        pae_model = None
    
    try:
        vqpae_model, vqpae_cfg = load_vqpae_model(vqpae_checkpoint, vqpae_config)
        print(f"  VQPAE model loaded (embedding_channels={vqpae_cfg.model_config.embedding_channels})")
    except Exception as e:
        print(f"  [ERROR] VQPAE model loading failed: {e}")
        vqpae_model = None
    
    ## Extract features for each audio
    for idx, audio_path in enumerate(audio_paths):
        audio_name = Path(audio_path).stem
        print(f"\n[{idx+1}/{len(audio_paths)}] Processing: {audio_name}")
        
        ## Original
        print("  [Original]", end=" ")
        original_feats = extract_original_features(audio_path)
        np.save(os.path.join(output_dir, f"{audio_name}_original.npy"), original_feats)
        all_original_feats.append(original_feats)
        print(f"✓ {original_feats.shape}")
        
        ## Librosa
        print("  [Librosa]", end=" ")
        librosa_feats = extract_librosa_features(audio_path)
        np.save(os.path.join(output_dir, f"{audio_name}_librosa.npy"), librosa_feats)
        all_librosa_feats.append(librosa_feats)
        print(f"✓ {librosa_feats.shape}")
        
        ## PAE
        if pae_model is not None:
            print("  [PAE]", end=" ")
            pae_feats = extract_pae_features(audio_path, pae_model, pae_cfg)
            np.save(os.path.join(output_dir, f"{audio_name}_pae.npy"), pae_feats)
            all_pae_feats.append(pae_feats)
            print(f"✓ {pae_feats.shape}")
        
        ## VQPAE
        if vqpae_model is not None:
            print("  [VQPAE]", end=" ")
            vqpae_feats = extract_vqpae_features(audio_path, vqpae_model)
            np.save(os.path.join(output_dir, f"{audio_name}_vqpae.npy"), vqpae_feats)
            all_vqpae_feats.append(vqpae_feats)
            print(f"✓ {vqpae_feats.shape}")
    
    ## Step 2: Visualize manifolds
    print("\n" + "="*60)
    print("Step 2: Visualizing manifolds...")
    print("-" * 60)
    
    ## Downsample all features to same length
    min_len = min([len(f) for f in all_original_feats + all_librosa_feats + all_pae_feats + all_vqpae_feats])
    print(f"\nDownsampling to min_len={min_len} frames")
    
    ## Prepare manifolds for visualization
    original_manifolds = [torch.from_numpy(f[:min_len]) for f in all_original_feats]
    librosa_manifolds = [torch.from_numpy(f[:min_len]) for f in all_librosa_feats]
    pae_manifolds = [torch.from_numpy(f[:min_len]) for f in all_pae_feats] if all_pae_feats else None
    vqpae_manifolds = [torch.from_numpy(f[:min_len]) for f in all_vqpae_feats] if all_vqpae_feats else None
    
    ## Create plot
    plotter = Plotter()
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    ## Plot each method with multiple audio files
    print("\nPlotting PCA manifolds...")
    plotter.plot_pca(axes[0], original_manifolds, title="(a) Original")
    plotter.plot_pca(axes[1], librosa_manifolds, title="(b) Librosa")
    if pae_manifolds:
        plotter.plot_pca(axes[2], pae_manifolds, title="(c) PAE")
    if vqpae_manifolds:
        plotter.plot_pca(axes[3], vqpae_manifolds, title="(d) VQPAE")
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "visualization.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization: {output_path}")
    
    print("\n" + "="*60)
    print("Experiment completed!")
    print(f"Results saved to: {output_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Audio manifold visualization experiment")
    parser.add_argument("--audio_dir", required=True, help="Directory containing audio files")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of audio samples to process")
    parser.add_argument("--output_dir", default="tmp/manifold_exp_012", help="Output directory for results")
    parser.add_argument("--pae_checkpoint", 
                       default="PAE_training/DeepPhase4Audio-main/output/20260103_213749_paeflat_twinkle_200eps/checkpoints/step_epoch=1999.ckpt",
                       help="Path to PAE checkpoint")
    parser.add_argument("--vqpae_checkpoint",
                       default="PAE_training/DeepPhase4Audio-main/output/20260111_023242_VQ_PAE-aistpp_1000_eps-8emb/checkpoints/step_epoch=199.ckpt",
                       help="Path to VQPAE checkpoint")
    parser.add_argument("--pae_config",
                       default="PAE_training/DeepPhase4Audio-main/configs/paeflat200.yaml",
                       help="Path to PAE config")
    parser.add_argument("--vqpae_config",
                       default="PAE_training/DeepPhase4Audio-main/configs/vqpae_aistpp_1000_eps_8emb.yaml",
                       help="Path to VQPAE config")
    
    args = parser.parse_args()
    
    ## Load audio files
    audio_files = sorted(Path(args.audio_dir).glob("*.wav"))[:args.num_samples]
    if len(audio_files) == 0:
        audio_files = sorted(Path(args.audio_dir).glob("*.mp3"))[:args.num_samples]
    
    print(f"Found {len(audio_files)} audio files")
    
    main([str(f) for f in audio_files], args.output_dir, args.pae_checkpoint, args.vqpae_checkpoint, args.pae_config, args.vqpae_config)
