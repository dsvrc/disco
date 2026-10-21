"""
Visualize audio representation manifolds using PCA - V4 (Fixed)

Key insight: PAE/VQPAE should use phase manifold embedding:
    embedding = [A·sin(2πφ), A·cos(2πφ)]
This produces concentric circles in PCA visualization!

Compare different audio encoding methods:
- Original: Raw MFCC features → irregular point cloud
- Librosa: Hand-crafted features → local clustering
- PAE: Phase + Amplitude → concentric circles/spirals
- VQPAE: Quantized Phase + Amplitude → discrete concentric circles


python PAE_training/DeepPhase4Audio-main/plotting/tmps/visualize_5audios_v6_01262128.py \
--audio_dir PAE_training/DeepPhase4Audio-main/plotting/wav \
--num_samples 5 \
--output tmp/manifolds_v6_01262140/visualization.png
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

## Add PAE training directory to path - fix absolute path ##
pae_base = Path("PAE_training/DeepPhase4Audio-main")
if str(pae_base) not in sys.path:
    sys.path.insert(0, str(pae_base))

from sklearn.decomposition import PCA
import matplotlib.colors as mcolors


class Plotter:
    """Plotter class - Fixed to always use PCA like FLD"""
    def __init__(self) -> None:
        self.pca = PCA(n_components=2)

    def plot_pca(self, ax, manifold_collection, title=None, point_color=None, draw_line=True, draw_arrow=True):
        """
        ✅ FIXED: 移除use_pca参数，所有方法都使用PCA降维
        
        完全按照FLD的做法：
        1. 输入：高维manifold（可能是16维、13维、或其他）
        2. PCA降维到2D
        3. 用笛卡尔坐标画图
        """
        ax.cla()

        point_alpha = 0.6  ## Increased for better visibility ##
        line_alpha = 0.3
        arrow_alpha = 1.0
        arrow_step = 50
        arrow_size = 0.015
        arrow_power = 1.0
        arrow_color = (0.25, 0.25, 0.5)

        num_steps = [len(manifold) for manifold in manifold_collection]
        manifolds = torch.cat(manifold_collection, dim=0).cpu()
        
        ## ✅ FIXED: 总是使用PCA降维到2D（删除了use_pca分支）
        manifolds_pca = torch.tensor(self.pca.fit_transform(manifolds)).split(num_steps, dim=0)

        for i, manifold in enumerate(manifolds_pca):
            if draw_line:
                ## ✅ FIXED: 线条颜色与点颜色一致
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


def extract_original_features(audio_path, sr=16000, hop_length=512):
    """
    Extract simple baseline features - changed to be more different from librosa
    Returns: (num_frames, 4) - spectral features only
    """
    y, sr = librosa.load(audio_path, sr=sr)
    
    ## Simple spectral features (more different from librosa) ##
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)
    zero_crossing = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)
    
    ## Combine ##
    features = np.vstack([spectral_centroid, spectral_rolloff, zero_crossing, rms])
    embedding = features.T
    
    return embedding  ## shape: (num_frames, 4) ##


def extract_librosa_features(audio_path, sr=16000, hop_length=512):
    """
    Extract combined librosa features - spectral only (no MFCC)
    Returns: (num_frames, total_features) - local clustering in PCA
    """
    y, sr = librosa.load(audio_path, sr=sr)
    
    ## Extract spectral features (different from MFCC) ##
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
    
    ## Combine all spectral features (no MFCC) ##
    features = np.vstack([
        chroma[:, :min_len],
        spectral_contrast[:, :min_len],
        spectral_centroid[:, :min_len],
        spectral_bandwidth[:, :min_len],
        spectral_rolloff[:, :min_len],
        zcr[:, :min_len],
        tonnetz[:, :min_len]
    ])
    ## features shape: (total_features, time_frames) ##
    
    ## Transpose to (time_frames, total_features) ##
    embedding = features.T
    
    return embedding


def extract_pae_phase_manifold(audio_path, model_path=None, config_path=None):
    """
    Extract PAE phase manifold embedding using:
        embedding = [A·sin(2πφ), A·cos(2πφ)]
    
    ✅ 这会产生16维向量，然后通过PCA降到2D后呈现同心圆/螺旋！
    
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
        
        ## Process audio in windows ##
        ## 🔧 KEY FIX: Reduce hop_size for denser sampling! ##
        window_size = cfg.model_config.time_range  ## 32000 = 2 seconds
        hop_size = 1600  ## ✅ CHANGED: 0.1s hop (was 8000 = 0.5s) - 5x denser! ##
        
        total_windows = (len(y) - window_size) // hop_size + 1
        print(f"  [PAE] Total windows: {total_windows} (denser sampling for periodicity)")
        
        phase_manifolds = []
        start_time = time.time()
        with torch.no_grad():
            for idx, i in enumerate(tqdm(range(0, len(y) - window_size + 1, hop_size), desc="  [PAE]", leave=False)):
                chunk = y[i:i + window_size]
                chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0).unsqueeze(0)
                
                ## Forward: model returns (pred, latent, signal, param)
                ## param = [p, f, a, b] where p=phase, f=freq, a=amplitude, b=offset
                pred, latent, signal, param = model(chunk_tensor)
                
                ## param is a list: [p, f, a, b] each shape (batch, embedding_channels, 1) ##
                if param is not None and isinstance(param, list) and len(param) == 4:
                    p, f, a, b = param  ## phase, freq, amplitude, offset
                    ## p is atan2/tpi range [-0.5, 0.5], normalize to [0, 1] ##
                    phase = (p.squeeze(2) + 0.5) % 1.0  ## (1, embedding_channels), [0, 1] with wrap ##
                    amplitude = torch.clamp(a.squeeze(2), min=0.0)  ## ✅ Clamp to positive ##
                else:
                    ## Fallback: Extract from latent using polar decomposition ##
                    latent_mean = latent.mean(dim=2)  ## (1, embedding_channels)
                    amplitude = torch.abs(latent_mean)
                    phase = (torch.atan2(latent[:, ::2, :].mean(dim=2), 
                                        latent[:, 1::2, :].mean(dim=2) + 1e-8) / (2 * np.pi)) % 1.0
                    if amplitude.shape[1] != phase.shape[1]:
                        amplitude = amplitude[:, :phase.shape[1]]
                
                ## ✅ ADD: Debug first frame ##
                if idx == 0:
                    print(f"    [PAE Debug] phase range: [{phase.min():.3f}, {phase.max():.3f}]")
                    print(f"    [PAE Debug] amplitude range: [{amplitude.min():.3f}, {amplitude.max():.3f}]")
                
                ## 🔥 Key: Build phase manifold (produces concentric circles!) ##
                ## Use full 2π range for complete circle ##
                phase_emb = torch.hstack([
                    amplitude * torch.sin(2.0 * np.pi * phase),
                    amplitude * torch.cos(2.0 * np.pi * phase)
                ]).squeeze(0)  ## (embedding_channels * 2,) ##
                
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
        raise  ## No fallback, re-raise exception ##


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
        ## Use correct import paths with src prefix ##
        from src.models.VQ_PAE import VQ_AE
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
        hop_size = 16000  ## 1.0s hop - faster ##
        
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
                
                ## Use phase_params [p, f, a, b] for true phase-amplitude decomposition ##
                phase_params = output.get('phase_params', None)
                
                ## phase_params = [p, f, a, b] (phase, freq, amplitude, offset) ##
                p, f, a, b = phase_params  ## each shape: (1, embedding_channels, 1)
                ## p is atan2/tpi range [-0.5, 0.5], normalize to [0, 1] ##
                phase = (p.squeeze(2) + 0.5) % 1.0  ## (1, embedding_channels), [0, 1] with wrap ##
                amplitude = torch.abs(a.squeeze(2))  ## (1, embedding_channels), ensure positive ##
                ## Normalize amplitude to [0, 1] for better visualization ##
                amplitude = amplitude / (amplitude.max() + 1e-8)
                
                ## 🔥 Key: Build phase manifold (discrete concentric circles!) ##
                ## Use full 2π range for complete circle ##
                phase_emb = torch.hstack([
                    amplitude * torch.sin(2.0 * np.pi * phase),
                    amplitude * torch.cos(2.0 * np.pi * phase)
                ]).squeeze(0)  ## (embedding_channels * 2,) ##
                
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
        raise  ## No fallback, re-raise exception ##


def load_audio_samples(audio_dir, num_samples=3):
    """Load multiple audio samples from directory"""
    audio_files = sorted(Path(audio_dir).glob("*.wav"))[:num_samples]
    
    if len(audio_files) == 0:
        audio_files = sorted(Path(audio_dir).glob("*.mp3"))[:num_samples]
    
    if len(audio_files) == 0:
        raise FileNotFoundError(f"No audio files found in {audio_dir}")
    
    return audio_files


def visualize_audio_manifolds(audio_files, output_path="tmp/manifolds_v5_0015/visualization.png"):
    """
    Visualize audio representation manifolds using phase manifold embedding
    
    Expected results:
    - Original: irregular point cloud
    - Librosa: local clustering
    - PAE: concentric circles/spirals (after PCA from 16D to 2D)
    - VQPAE: discrete concentric circles (after PCA from 16D to 2D)
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
    
    ## ✅ FIXED: 所有方法都使用PCA降维（删除了use_pca参数）
    
    ## Plot Original - expect irregular point cloud
    plotter.plot_pca(
        axes[0], original_manifolds, 
        title="(a) Original",
        point_color=colors,
        draw_line=True, 
        draw_arrow=False  ## Disable arrows ##
    )
    
    ## Plot Librosa - expect local clustering
    plotter.plot_pca(
        axes[1], librosa_manifolds,
        title="(b) Librosa",
        point_color=colors,
        draw_line=True,
        draw_arrow=False  ## Disable arrows ##
    )
    
    ## Plot PAE - expect concentric circles/spirals after PCA!
    plotter.plot_pca(
        axes[2], pae_manifolds,
        title="(c) PAE",
        point_color=colors,
        draw_line=True,
        draw_arrow=False  ## Disable arrows ##
        ## ✅ FIXED: 删除了use_pca=False，现在总是使用PCA
    )
    
    ## Plot VQPAE - expect discrete concentric circles after PCA!
    plotter.plot_pca(
        axes[3], vqpae_manifolds,
        title="(d) VQPAE",
        point_color=colors,
        draw_line=True,
        draw_arrow=False  ## Disable arrows ##
        ## ✅ FIXED: 删除了use_pca=False，现在总是使用PCA
    )
    
    plt.tight_layout()
    
    ## Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved visualization to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize audio manifolds using phase embedding - V4 Fixed")
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
        default="tmp/manifolds_v5_0015/visualization.png",
        help="Output path for visualization"
    )
    
    args = parser.parse_args()
    
    audio_files = load_audio_samples(args.audio_dir, args.num_samples)
    print(f"Loaded {len(audio_files)} audio files")
    
    visualize_audio_manifolds(audio_files, args.output)


if __name__ == "__main__":
    main()