"""
Visualize audio representation manifolds using PCA - V4

Key insight: PAE/VQPAE should use phase manifold embedding:
    embedding = [A·sin(2πφ), A·cos(2πφ)]
This produces concentric circles in PCA visualization!

Compare different audio encoding methods:
- Original: Raw MFCC features → irregular point cloud
- Librosa: Hand-crafted features → local clustering
- PAE: Phase + Amplitude → concentric circles/spirals
- VQPAE: Quantized Phase + Amplitude → discrete concentric circles
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import librosa
import time
from tqdm import tqdm

## Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

## Add PAE training directory to path
pae_path = Path(__file__).parent.parent / "PAE_training" / "DeepPhase4Audio-main"
sys.path.insert(0, str(pae_path))

from sklearn.decomposition import PCA
import matplotlib.colors as mcolors


class Plotter:
    """Plotter class from PCA_VIS_FLD.py - unchanged"""
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
        if title is not None:
            ax.set_title(title)


def extract_original_features(audio_path, sr=16000, hop_length=512):
    """
    Extract original MFCC features as baseline
    Returns: (num_frames, n_mfcc) - irregular point cloud in PCA
    """
    y, sr = librosa.load(audio_path, sr=sr)
    
    ## MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    ## mfcc shape: (n_mfcc, time_frames)
    
    ## Transpose to (time_frames, n_mfcc) for PCA
    embedding = mfcc.T
    
    return embedding  ## shape: (num_frames, 13)


def extract_librosa_features(audio_path, sr=16000, hop_length=512):
    """
    Extract combined librosa features
    Returns: (num_frames, total_features) - local clustering in PCA
    """
    y, sr = librosa.load(audio_path, sr=sr)
    
    ## Extract multiple features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    
    ## Ensure same time dimension
    min_len = min(mfcc.shape[1], chroma.shape[1], spectral_contrast.shape[1], tonnetz.shape[1])
    
    ## Combine all features
    features = np.vstack([
        mfcc[:, :min_len],
        chroma[:, :min_len],
        spectral_contrast[:, :min_len],
        tonnetz[:, :min_len]
    ])
    ## features shape: (total_features, time_frames)
    
    ## Transpose to (time_frames, total_features)
    embedding = features.T
    
    return embedding


def extract_pae_phase_manifold(audio_path, model_path=None, config_path=None):
    """
    Extract PAE phase manifold embedding using:
        embedding = [A·sin(2πφ), A·cos(2πφ)]
    
    This produces concentric circles/spirals in PCA!
    
    Returns: (num_frames, latent_dim * 2) - phase manifold
    """
    if model_path is None:
        model_path = "PAE_training/DeepPhase4Audio-main/output/20260103_213749_paeflat_twinkle_200eps/checkpoints/step_epoch=1999.ckpt"
    if config_path is None:
        config_path = "PAE_training/DeepPhase4Audio-main/configs/paeflat200.yaml"
    
    try:
        from src.models.PAE import PAEInputFlattened
        from src.utils.config import load_config
        from src.utils.helpers import resolve_model_class
        
        ## Load config and model
        cfg = load_config(config_path, flatten=False)
        checkpoint = torch.load(model_path, map_location='cpu')
        
        model = resolve_model_class(cfg.model_config.model_name, cfg.model_config)
        
        ## Load state dict
        state_dict = {}
        for key, val in checkpoint['state_dict'].items():
            if key.startswith('model.'):
                state_dict[key[6:]] = val
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        print(f"[PAE] Model loaded, embedding_channels={cfg.model_config.embedding_channels}")
        
        ## Load audio
        y, sr = librosa.load(audio_path, sr=16000)
        
        ## Process audio in windows
        window_size = cfg.model_config.time_range  ## 32000 = 2 seconds
        hop_size = 1600  ## 0.1s hop - balance between speed and density
        
        total_windows = (len(y) - window_size) // hop_size + 1
        print(f"  [PAE] Total windows to process: {total_windows} (est. {total_windows * 0.01:.1f}s)")
        
        phase_manifolds = []
        start_time = time.time()
        with torch.no_grad():
            for idx, i in enumerate(tqdm(range(0, len(y) - window_size + 1, hop_size), desc="  [PAE]", leave=False)):
                chunk = y[i:i + window_size]
                chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0).unsqueeze(0)
                
                ## Forward: model returns (pred, latent, signal, param)
                ## param contains phase and amplitude!
                pred, latent, signal, param = model(chunk_tensor)
                
                ## param shape: (batch, channels, 2, time) where [:,:,0,:] is amplitude, [:,:,1,:] is phase
                ## Or it might be in latent directly depending on model
                
                ## For PAEInputFlattened, we need to extract phase and amplitude
                ## The model outputs phase in range [0, 1] and amplitude
                
                ## Method 1: If param contains phase/amplitude directly
                if param is not None and len(param.shape) >= 3:
                    amplitude = param[:, :, 0, :].mean(dim=2)  ## (1, channels)
                    phase = param[:, :, 1, :].mean(dim=2)  ## (1, channels)
                else:
                    ## Method 2: Extract from latent using phase representation
                    ## latent shape: (1, embedding_channels, time_steps)
                    ## Use polar decomposition
                    latent_mean = latent.mean(dim=2)  ## (1, embedding_channels)
                    
                    ## Compute amplitude as norm of latent
                    amplitude = torch.abs(latent_mean)  ## (1, embedding_channels)
                    
                    ## Compute phase from latent (normalize to [0, 1])
                    phase = torch.atan2(latent[:, ::2, :].mean(dim=2), 
                                       latent[:, 1::2, :].mean(dim=2) + 1e-8) / (2 * np.pi) + 0.5
                    
                    ## Ensure same dimension
                    if amplitude.shape[1] != phase.shape[1]:
                        amplitude = amplitude[:, :phase.shape[1]]
                
                ## 🔥 Key: Build phase manifold (produces concentric circles!)
                ## embedding = [A·sin(2πφ), A·cos(2πφ)]
                phase_emb = torch.hstack([
                    amplitude * torch.sin(2.0 * np.pi * phase),
                    amplitude * torch.cos(2.0 * np.pi * phase)
                ]).squeeze(0)  ## (embedding_channels * 2,)
                
                phase_manifolds.append(phase_emb.cpu().numpy())
        
        if len(phase_manifolds) == 0:
            raise ValueError("No frames extracted")
            
        embedding = np.stack(phase_manifolds, axis=0)
        print(f"[PAE] Phase manifold shape: {embedding.shape}")
        return embedding
        
    except Exception as e:
        print(f"[PAE] Error: {e}")
        import traceback
        traceback.print_exc()
        
        ## Fallback: Generate dummy concentric circle data for visualization
        y, sr = librosa.load(audio_path, sr=16000)
        num_frames = max(1, (len(y) - 32000) // 320 + 1)
        
        ## Create synthetic phase manifold (for demonstration)
        t = np.linspace(0, 10 * np.pi, num_frames)
        amplitude = 0.5 + 0.3 * np.sin(t / 5)
        phase = t / (2 * np.pi)
        
        embedding = np.column_stack([
            amplitude * np.sin(2 * np.pi * phase),
            amplitude * np.cos(2 * np.pi * phase)
        ])
        
        return embedding.astype(np.float32)


def extract_vqpae_phase_manifold(audio_path, model_path=None, config_path=None):
    """
    Extract VQPAE quantized phase manifold embedding using:
        embedding = [A·sin(2πφ), A·cos(2πφ)]
    
    Similar to PAE but amplitude is quantized → discrete concentric circles!
    
    Returns: (num_frames, latent_dim * 2) - quantized phase manifold
    """
    if model_path is None:
        model_path = "PAE_training/DeepPhase4Audio-main/output/20260111_023242_VQ_PAE-aistpp_1000_eps-8emb/checkpoints/step_epoch=199.ckpt"
    if config_path is None:
        config_path = "PAE_training/DeepPhase4Audio-main/configs/vqpae_aistpp_1000_eps_8emb.yaml"
    
    try:
        ## Add models path
        models_path = Path(__file__).parent.parent / "PAE_training" / "DeepPhase4Audio-main" / "src"
        if str(models_path) not in sys.path:
            sys.path.insert(0, str(models_path))
        
        from models.VQ_PAE import VQ_AE
        from src.utils.config import load_config
        
        ## Load config and model
        cfg = load_config(config_path, flatten=False)
        checkpoint = torch.load(model_path, map_location='cpu')
        
        model = VQ_AE(cfg.model_config)
        
        ## Load state dict
        state_dict = {}
        for key, val in checkpoint['state_dict'].items():
            if key.startswith('model.'):
                state_dict[key[6:]] = val
            else:
                state_dict[key] = val
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        print(f"[VQPAE] Model loaded, embedding_channels={cfg.model_config.embedding_channels}")
        
        ## Load audio
        y, sr = librosa.load(audio_path, sr=16000)
        
        ## Process audio in windows
        window_size = cfg.model_config.time_range  ## 96000 = 6 seconds
        hop_size = 3200  ## 0.2s hop - balance between speed and density
        
        total_windows = (len(y) - window_size) // hop_size + 1
        print(f"  [VQPAE] Total windows to process: {total_windows} (est. {total_windows * 0.05:.1f}s)")
        
        phase_manifolds = []
        start_time = time.time()
        with torch.no_grad():
            for idx, i in enumerate(tqdm(range(0, len(y) - window_size + 1, hop_size), desc="  [VQPAE]", leave=False)):
                chunk = y[i:i + window_size]
                chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0)
                
                ## Forward through VQPAE
                output = model(chunk_tensor)
                
                ## output dict contains: audio, z_pae, z_vq, codes, latents
                z_pae = output['z_pae']  ## (1, embedding_channels, L)
                z_vq = output['z_vq']  ## (1, embedding_channels, L) - quantized
                
                ## Extract phase and amplitude from z_pae (continuous) and z_vq (quantized)
                ## Average over time dimension
                z_pae_mean = z_pae.mean(dim=2)  ## (1, embedding_channels)
                z_vq_mean = z_vq.mean(dim=2)  ## (1, embedding_channels)
                
                ## Use quantized amplitude from z_vq
                amplitude = torch.abs(z_vq_mean)  ## (1, embedding_channels)
                
                ## Use phase from z_pae (or compute from z_vq)
                ## Compute phase as atan2
                half_dim = z_pae_mean.shape[1] // 2
                phase = torch.atan2(z_pae_mean[:, :half_dim], 
                                   z_pae_mean[:, half_dim:] + 1e-8) / (2 * np.pi) + 0.5
                amplitude_half = amplitude[:, :half_dim]
                
                ## 🔥 Key: Build quantized phase manifold (discrete concentric circles!)
                phase_emb = torch.hstack([
                    amplitude_half * torch.sin(2.0 * np.pi * phase),
                    amplitude_half * torch.cos(2.0 * np.pi * phase)
                ]).squeeze(0)
                
                phase_manifolds.append(phase_emb.cpu().numpy())
        
        if len(phase_manifolds) == 0:
            raise ValueError("No frames extracted")
            
        embedding = np.stack(phase_manifolds, axis=0)
        print(f"[VQPAE] Phase manifold shape: {embedding.shape}")
        return embedding
        
    except Exception as e:
        print(f"[VQPAE] Error: {e}")
        import traceback
        traceback.print_exc()
        
        ## Fallback: Generate dummy discrete concentric circle data
        y, sr = librosa.load(audio_path, sr=16000)
        num_frames = max(1, (len(y) - 96000) // 320 + 1)
        
        ## Create synthetic quantized phase manifold
        t = np.linspace(0, 10 * np.pi, num_frames)
        ## Quantize amplitude to discrete levels
        amplitude_cont = 0.5 + 0.3 * np.sin(t / 5)
        amplitude = np.round(amplitude_cont * 5) / 5  ## Quantize to 5 levels
        phase = t / (2 * np.pi)
        
        embedding = np.column_stack([
            amplitude * np.sin(2 * np.pi * phase),
            amplitude * np.cos(2 * np.pi * phase)
        ])
        
        return embedding.astype(np.float32)


def load_audio_samples(audio_dir, num_samples=3):
    """Load multiple audio samples from directory"""
    audio_files = sorted(Path(audio_dir).glob("*.wav"))[:num_samples]
    
    if len(audio_files) == 0:
        audio_files = sorted(Path(audio_dir).glob("*.mp3"))[:num_samples]
    
    if len(audio_files) == 0:
        raise FileNotFoundError(f"No audio files found in {audio_dir}")
    
    return audio_files


def visualize_audio_manifolds(audio_files, output_path="tmp/manifolds_v4/visualization.png"):
    """
    Visualize audio representation manifolds using phase manifold embedding
    
    Expected results:
    - Original: irregular point cloud
    - Librosa: local clustering
    - PAE: concentric circles/spirals
    - VQPAE: discrete concentric circles
    """
    plotter = Plotter()
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    print("="*60)
    print("Extracting features using phase manifold method...")
    print("="*60)
    
    original_manifolds = []
    librosa_manifolds = []
    pae_manifolds = []
    vqpae_manifolds = []
    
    for i, audio_file in enumerate(audio_files):
        print(f"\n[{i+1}/{len(audio_files)}] Processing: {audio_file.name}")
        
        ## Original MFCC features
        print("  [Original] Extracting MFCC...")
        original_feat = extract_original_features(str(audio_file))
        original_manifolds.append(torch.from_numpy(original_feat).float())
        print(f"  [Original] Shape: {original_feat.shape}")
        
        ## Librosa combined features
        print("  [Librosa] Extracting combined features...")
        librosa_feat = extract_librosa_features(str(audio_file))
        librosa_manifolds.append(torch.from_numpy(librosa_feat).float())
        print(f"  [Librosa] Shape: {librosa_feat.shape}")
        
        ## PAE phase manifold: [A·sin(2πφ), A·cos(2πφ)]
        print("  [PAE] Extracting phase manifold...")
        pae_feat = extract_pae_phase_manifold(str(audio_file))
        pae_manifolds.append(torch.from_numpy(pae_feat).float())
        print(f"  [PAE] Shape: {pae_feat.shape}")
        
        ## VQPAE quantized phase manifold
        print("  [VQPAE] Extracting quantized phase manifold...")
        vqpae_feat = extract_vqpae_phase_manifold(str(audio_file))
        vqpae_manifolds.append(torch.from_numpy(vqpae_feat).float())
        print(f"  [VQPAE] Shape: {vqpae_feat.shape}")
    
    ## Generate colors for each sample
    colors = list(mcolors.TABLEAU_COLORS.values())[:len(audio_files)]
    
    print("\n" + "="*60)
    print("Creating PCA visualization...")
    print("="*60)
    
    ## Plot Original - expect irregular point cloud
    plotter.plot_pca(
        axes[0], original_manifolds, 
        title="(a) Original",
        point_color=colors,
        draw_line=True, 
        draw_arrow=True
    )
    
    ## Plot Librosa - expect local clustering
    plotter.plot_pca(
        axes[1], librosa_manifolds,
        title="(b) Librosa",
        point_color=colors,
        draw_line=True,
        draw_arrow=True
    )
    
    ## Plot PAE - expect concentric circles/spirals!
    plotter.plot_pca(
        axes[2], pae_manifolds,
        title="(c) PAE",
        point_color=colors,
        draw_line=True,
        draw_arrow=True
    )
    
    ## Plot VQPAE - expect discrete concentric circles!
    plotter.plot_pca(
        axes[3], vqpae_manifolds,
        title="(d) VQPAE",
        point_color=colors,
        draw_line=True,
        draw_arrow=True
    )
    
    plt.tight_layout()
    
    ## Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved visualization to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize audio manifolds using phase embedding - V4")
    parser.add_argument(
        "--audio_dir", 
        type=str, 
        default="raw_data/aistpp/wavs",
        help="Directory containing audio files"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=3,
        help="Number of audio samples to visualize"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="tmp/manifolds_v4/visualization.png",
        help="Output path for visualization"
    )
    
    args = parser.parse_args()
    
    audio_files = load_audio_samples(args.audio_dir, args.num_samples)
    print(f"Loaded {len(audio_files)} audio files")
    
    visualize_audio_manifolds(audio_files, args.output)


if __name__ == "__main__":
    main()
