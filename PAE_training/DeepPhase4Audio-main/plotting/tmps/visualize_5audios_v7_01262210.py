"""
Visualize audio representation manifolds using PCA - V7

Key improvements:
1. Original: Uses raw waveform chunks (completely different from librosa)
2. No connecting lines (draw_line=False)
3. VQPAE: Fixed amplitude normalization to match PAE (concentric circles)
4. Denser sampling for better periodicity visualization

python PAE_training/DeepPhase4Audio-main/plotting/tmps/visualize_5audios_v7_01262210.py \
--audio_dir PAE_training/DeepPhase4Audio-main/plotting/wav \
--num_samples 5 \
--output PAE_training/DeepPhase4Audio-main/plotting/manifolds_v7_01262210/visualization.png
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import librosa
import time
from tqdm import tqdm

## Add parent directory to path ##
sys.path.append(str(Path(__file__).parent.parent))

## Add PAE training directory to path - fix absolute path ##
pae_base = Path("PAE_training/DeepPhase4Audio-main")
if str(pae_base) not in sys.path:
    sys.path.insert(0, str(pae_base))

from sklearn.decomposition import PCA
import matplotlib.colors as mcolors


class Plotter:
    """Plotter class - Always use PCA for dimensionality reduction"""
    def __init__(self) -> None:
        self.pca = PCA(n_components=2)

    def plot_pca(self, ax, manifold_collection, title=None, point_color=None, draw_line=True, draw_arrow=False):
        """
        PCA visualization with configurable line drawing
        """
        ax.cla()

        point_alpha = 0.6
        line_alpha = 0.3
        arrow_alpha = 1.0
        arrow_step = 50
        arrow_size = 0.015
        arrow_power = 1.0
        arrow_color = (0.25, 0.25, 0.5)

        num_steps = [len(manifold) for manifold in manifold_collection]
        manifolds = torch.cat(manifold_collection, dim=0).cpu()
        
        ## PCA to 2D ##
        manifolds_pca = torch.tensor(self.pca.fit_transform(manifolds)).split(num_steps, dim=0)

        for i, manifold in enumerate(manifolds_pca):
            if draw_line:
                color = point_color[i] if point_color else (0.0, 0.0, 0.0)
                ax.plot(manifold[:, 0], manifold[:, 1], color=color, alpha=line_alpha)
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


def extract_original_features(audio_path, sr=16000, frame_length=2048, hop_length=512):
    """
    Extract raw waveform chunks - completely different from librosa spectral features
    Returns: (num_frames, frame_length) - raw audio samples
    """
    y, sr = librosa.load(audio_path, sr=sr)
    
    ## Extract raw waveform chunks ##
    frames = []
    for i in range(0, len(y) - frame_length, hop_length):
        chunk = y[i:i + frame_length]
        frames.append(chunk)
    
    embedding = np.array(frames)
    
    return embedding  ## shape: (num_frames, frame_length) ##


def extract_librosa_features(audio_path, sr=16000, hop_length=512):
    """
    Extract combined librosa features - spectral only (no MFCC)
    Returns: (num_frames, total_features) - local clustering in PCA
    """
    y, sr = librosa.load(audio_path, sr=sr)
    
    ## Extract spectral features (different from raw waveform) ##
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    
    ## Ensure same time dimension ##
    min_len = min(chroma.shape[1], spectral_contrast.shape[1], spectral_centroid.shape[1],
                   spectral_bandwidth.shape[1], spectral_rolloff.shape[1], zcr.shape[1], tonnetz.shape[1])
    
    ## Combine all spectral features ##
    features = np.vstack([
        chroma[:, :min_len],
        spectral_contrast[:, :min_len],
        spectral_centroid[:, :min_len],
        spectral_bandwidth[:, :min_len],
        spectral_rolloff[:, :min_len],
        zcr[:, :min_len],
        tonnetz[:, :min_len]
    ])
    
    embedding = features.T
    
    return embedding


def extract_pae_phase_manifold(audio_path, model_path=None, config_path=None):
    """
    Extract PAE phase manifold with dense sampling
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
        
        ## Load config and model ##
        cfg = load_config(config_path, flatten=False)
        checkpoint = torch.load(model_path, map_location='cpu')
        
        model = resolve_model_class(cfg.model_config.model_name, cfg.model_config)
        
        ## Load state dict ##
        state_dict = {}
        for key, val in checkpoint['state_dict'].items():
            if key.startswith('model.'):
                state_dict[key[6:]] = val
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        print(f"[PAE] Model loaded, embedding_channels={cfg.model_config.embedding_channels}")
        
        ## Load audio ##
        y, sr = librosa.load(audio_path, sr=16000)
        
        ## Dense sampling for better periodicity ##
        window_size = cfg.model_config.time_range  ## 32000 = 2 seconds ##
        hop_size = 1600  ## 0.1s hop - 5x denser ##
        
        total_windows = (len(y) - window_size) // hop_size + 1
        print(f"  [PAE] Total windows: {total_windows} (dense sampling)")
        
        phase_manifolds = []
        with torch.no_grad():
            for idx, i in enumerate(tqdm(range(0, len(y) - window_size + 1, hop_size), desc="  [PAE]", leave=False)):
                chunk = y[i:i + window_size]
                chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0).unsqueeze(0)
                
                pred, latent, signal, param = model(chunk_tensor)
                
                if param is not None and isinstance(param, list) and len(param) == 4:
                    p, f, a, b = param
                    phase = (p.squeeze(2) + 0.5) % 1.0  ## [0, 1] with wrap ##
                    amplitude = torch.clamp(a.squeeze(2), min=0.0)  ## Positive only ##
                else:
                    latent_mean = latent.mean(dim=2)
                    amplitude = torch.abs(latent_mean)
                    phase = (torch.atan2(latent[:, ::2, :].mean(dim=2), 
                                        latent[:, 1::2, :].mean(dim=2) + 1e-8) / (2 * np.pi)) % 1.0
                    if amplitude.shape[1] != phase.shape[1]:
                        amplitude = amplitude[:, :phase.shape[1]]
                
                if idx == 0:
                    print(f"    [PAE Debug] phase: [{phase.min():.3f}, {phase.max():.3f}], amplitude: [{amplitude.min():.3f}, {amplitude.max():.3f}]")
                
                ## Build phase manifold ##
                phase_emb = torch.hstack([
                    amplitude * torch.sin(2.0 * np.pi * phase),
                    amplitude * torch.cos(2.0 * np.pi * phase)
                ]).squeeze(0)
                
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
        raise


def extract_vqpae_phase_manifold(audio_path, model_path=None, config_path=None):
    """
    Extract VQPAE quantized phase manifold
    Fixed: Use same amplitude handling as PAE (no per-frame normalization)
    Returns: (num_frames, latent_dim * 2) - quantized phase manifold
    """
    if model_path is None:
        model_path = "PAE_training/DeepPhase4Audio-main/output/20260111_023242_VQ_PAE-aistpp_1000_eps-8emb/checkpoints/step_epoch=199.ckpt"
    if config_path is None:
        config_path = "PAE_training/DeepPhase4Audio-main/configs/vqpae_aistpp_1000_eps_8emb.yaml"
    
    try:
        from src.models.VQ_PAE import VQ_AE
        from src.utils.config import load_config
        
        ## Load config and model ##
        cfg = load_config(config_path, flatten=False)
        checkpoint = torch.load(model_path, map_location='cpu')
        
        model = VQ_AE(cfg.model_config)
        
        ## Load state dict ##
        state_dict = {}
        for key, val in checkpoint['state_dict'].items():
            if key.startswith('model.'):
                state_dict[key[6:]] = val
            else:
                state_dict[key] = val
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        print(f"[VQPAE] Model loaded, embedding_channels={cfg.model_config.embedding_channels}")
        
        ## Load audio ##
        y, sr = librosa.load(audio_path, sr=16000)
        
        window_size = cfg.model_config.time_range  ## 96000 = 6 seconds ##
        hop_size = 16000  ## 1.0s hop ##
        
        total_windows = (len(y) - window_size) // hop_size + 1
        print(f"  [VQPAE] Total windows: {total_windows}")
        
        phase_manifolds = []
        with torch.no_grad():
            for idx, i in enumerate(tqdm(range(0, len(y) - window_size + 1, hop_size), desc="  [VQPAE]", leave=False)):
                chunk = y[i:i + window_size]
                chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0)
                
                output = model(chunk_tensor)
                
                phase_params = output.get('phase_params', None)
                p, f, a, b = phase_params
                
                phase = (p.squeeze(2) + 0.5) % 1.0  ## [0, 1] with wrap ##
                ## KEY FIX: Use clamp like PAE, no per-frame normalization ##
                amplitude = torch.clamp(a.squeeze(2), min=0.0)  ## Same as PAE ##
                
                if idx == 0:
                    print(f"    [VQPAE Debug] phase: [{phase.min():.3f}, {phase.max():.3f}], amplitude: [{amplitude.min():.3f}, {amplitude.max():.3f}]")
                
                ## Build phase manifold (should be concentric now) ##
                phase_emb = torch.hstack([
                    amplitude * torch.sin(2.0 * np.pi * phase),
                    amplitude * torch.cos(2.0 * np.pi * phase)
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
        raise


def load_audio_samples(audio_dir, num_samples=3):
    """Load multiple audio samples from directory"""
    audio_files = sorted(Path(audio_dir).glob("*.wav"))[:num_samples]
    
    if len(audio_files) == 0:
        audio_files = sorted(Path(audio_dir).glob("*.mp3"))[:num_samples]
    
    if len(audio_files) == 0:
        raise FileNotFoundError(f"No audio files found in {audio_dir}")
    
    return audio_files


def visualize_audio_manifolds(audio_files, output_path="tmp/manifolds_v7/visualization.png"):
    """
    Visualize audio representation manifolds using phase manifold embedding
    
    V7 improvements:
    - Original: raw waveform (very different from librosa)
    - No connecting lines (cleaner visualization)
    - VQPAE: fixed amplitude normalization (concentric circles like PAE)
    """
    plotter = Plotter()
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    print("="*60)
    print("Extracting features (V7 - improved)...")
    print("="*60)
    
    original_manifolds = []
    librosa_manifolds = []
    pae_manifolds = []
    vqpae_manifolds = []
    
    for i, audio_file in enumerate(audio_files):
        print(f"\n[{i+1}/{len(audio_files)}] Processing: {audio_file.name}")
        
        ## Original: raw waveform ##
        print("  [Original] Extracting raw waveform...")
        original_feat = extract_original_features(str(audio_file))
        original_manifolds.append(torch.from_numpy(original_feat).float())
        print(f"  [Original] Shape: {original_feat.shape}")
        
        ## Librosa: spectral features ##
        print("  [Librosa] Extracting spectral features...")
        librosa_feat = extract_librosa_features(str(audio_file))
        librosa_manifolds.append(torch.from_numpy(librosa_feat).float())
        print(f"  [Librosa] Shape: {librosa_feat.shape}")
        
        ## PAE: phase manifold ##
        print("  [PAE] Extracting phase manifold...")
        pae_feat = extract_pae_phase_manifold(str(audio_file))
        pae_manifolds.append(torch.from_numpy(pae_feat).float())
        print(f"  [PAE] Shape: {pae_feat.shape}")
        
        ## VQPAE: quantized phase manifold ##
        print("  [VQPAE] Extracting quantized phase manifold...")
        vqpae_feat = extract_vqpae_phase_manifold(str(audio_file))
        vqpae_manifolds.append(torch.from_numpy(vqpae_feat).float())
        print(f"  [VQPAE] Shape: {vqpae_feat.shape}")
    
    ## Generate colors ##
    colors = list(mcolors.TABLEAU_COLORS.values())[:len(audio_files)]
    
    print("\n" + "="*60)
    print("Creating PCA visualization (no connecting lines)...")
    print("="*60)
    
    ## Plot Original - raw waveform ##
    plotter.plot_pca(
        axes[0], original_manifolds, 
        title="(a) Original",
        point_color=colors,
        draw_line=False,  ## No lines ##
        draw_arrow=False
    )
    
    ## Plot Librosa - spectral features ##
    plotter.plot_pca(
        axes[1], librosa_manifolds,
        title="(b) Librosa",
        point_color=colors,
        draw_line=False,  ## No lines ##
        draw_arrow=False
    )
    
    ## Plot PAE - concentric circles ##
    plotter.plot_pca(
        axes[2], pae_manifolds,
        title="(c) PAE",
        point_color=colors,
        draw_line=False,  ## No lines ##
        draw_arrow=False
    )
    
    ## Plot VQPAE - discrete concentric circles ##
    plotter.plot_pca(
        axes[3], vqpae_manifolds,
        title="(d) VQPAE",
        point_color=colors,
        draw_line=False,  ## No lines ##
        draw_arrow=False
    )
    
    plt.tight_layout()
    
    ## Save figure ##
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved visualization to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize audio manifolds - V7 (improved)")
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
        default="tmp/manifolds_v7/visualization.png",
        help="Output path for visualization"
    )
    
    args = parser.parse_args()
    
    audio_files = load_audio_samples(args.audio_dir, args.num_samples)
    print(f"Loaded {len(audio_files)} audio files")
    
    visualize_audio_manifolds(audio_files, args.output)


if __name__ == "__main__":
    main()
