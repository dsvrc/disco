"""
Visualize audio manifolds V9 - Fixed VQPAE phase extraction

Key fixes:
- PAE: Use single window latent time series (works)
- VQPAE: Use z_recon (reconstructed phase signal) time series
         OR use multiple windows with phase accumulation

conda activate music2dance \
python PAE_training/DeepPhase4Audio-main/plotting/tmps/visualize_5audios_v9.py \
--audio_dir PAE_training/DeepPhase4Audio-main/plotting/wav \
--num_samples 5 \
--output tmp/manifolds_v9/visualization.png
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import librosa
from tqdm import tqdm

## Add PAE path ##
pae_base = Path("PAE_training/DeepPhase4Audio-main")
if str(pae_base) not in sys.path:
    sys.path.insert(0, str(pae_base))

from sklearn.decomposition import PCA
import matplotlib.colors as mcolors


class Plotter:
    """Plotter for PCA visualization"""
    def __init__(self) -> None:
        self.pca = PCA(n_components=2)

    def plot_pca(self, ax, manifold_collection, title=None, point_color=None, draw_line=False, draw_arrow=False):
        """PCA visualization with optional lines"""
        ax.cla()

        point_alpha = 0.6
        line_alpha = 0.3

        num_steps = [len(manifold) for manifold in manifold_collection]
        manifolds = torch.cat(manifold_collection, dim=0).cpu()
        
        ## PCA to 2D ##
        manifolds_pca = torch.tensor(self.pca.fit_transform(manifolds)).split(num_steps, dim=0)

        for i, manifold in enumerate(manifolds_pca):
            if draw_line:
                color = point_color[i] if point_color else (0.0, 0.0, 0.0)
                ax.plot(manifold[:, 0], manifold[:, 1], color=color, alpha=line_alpha, linewidth=0.5)
            if point_color is None:
                ax.scatter(manifold[:, 0], manifold[:, 1], alpha=point_alpha, s=10, label=i)
            else:
                ax.scatter(manifold[:, 0], manifold[:, 1], color=point_color[i], s=10, alpha=point_alpha, label=i)
        
        ax.legend()
        ax.set_axis_off()
        if title is not None:
            ax.set_title(title)


def extract_original_features(audio_path, sr=16000, frame_length=2048, hop_length=512, max_frames=200):
    """Extract raw waveform chunks"""
    y, sr = librosa.load(audio_path, sr=sr)
    
    frames = []
    for i in range(0, min(len(y) - frame_length, max_frames * hop_length), hop_length):
        chunk = y[i:i + frame_length]
        frames.append(chunk)
    
    embedding = np.array(frames)
    return embedding


def extract_librosa_features(audio_path, sr=16000, hop_length=512, max_frames=200):
    """Extract spectral features"""
    y, sr = librosa.load(audio_path, sr=sr)
    
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    
    min_len = min(chroma.shape[1], spectral_contrast.shape[1], spectral_centroid.shape[1],
                   spectral_bandwidth.shape[1], spectral_rolloff.shape[1], zcr.shape[1], tonnetz.shape[1])
    min_len = min(min_len, max_frames)
    
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


def extract_pae_single_window(audio_path, window_start=0, downsample=10):
    """
    PAE: Extract from single window's latent time series - SINGLE DOMINANT CHANNEL
    downsample: keep every Nth sample to reduce density
    """
    from src.models.PAE import PAEInputFlattened
    from src.utils.config import load_config
    from src.utils.helpers import resolve_model_class
    
    model_path = "PAE_training/DeepPhase4Audio-main/output/20260103_213749_paeflat_twinkle_200eps/checkpoints/step_epoch=1999.ckpt"
    config_path = "PAE_training/DeepPhase4Audio-main/configs/paeflat200.yaml"
    
    cfg = load_config(config_path, flatten=False)
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model = resolve_model_class(cfg.model_config.model_name, cfg.model_config)
    state_dict = {}
    for key, val in checkpoint['state_dict'].items():
        if key.startswith('model.'):
            state_dict[key[6:]] = val
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    ## Load audio ##
    y, sr = librosa.load(audio_path, sr=16000)
    
    ## Single window ##
    window_size = cfg.model_config.time_range  ## 32000 = 2s ##
    chunk = y[window_start:window_start + window_size]
    chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        pred, latent, signal, param = model(chunk_tensor)
    
    ## latent shape: (1, channels, time_steps) ##
    latent = latent.squeeze(0)  ## (channels, time_steps) ##
    num_channels = latent.shape[0] // 2
    time_steps = latent.shape[1]
    
    phase_channels = latent[::2, :]
    amp_channels = latent[1::2, :]
    
    phase = torch.atan2(phase_channels, amp_channels + 1e-8) / (2 * np.pi)
    phase = (phase + 0.5) % 1.0
    amplitude = torch.sqrt(phase_channels**2 + amp_channels**2)
    
    ## Find dominant channel ##
    mean_amps = amplitude.mean(dim=1).cpu().numpy()
    dominant_ch = np.argmax(mean_amps)
    print(f"  [PAE] Dominant ch: {dominant_ch}, amp={mean_amps[dominant_ch]:.3f}")
    
    ## Extract dominant channel ##
    phase_dom = phase[dominant_ch, :].cpu().numpy()
    amp_dom = amplitude[dominant_ch, :].cpu().numpy()
    
    ## Downsample to reduce density ##
    phase_dom = phase_dom[::downsample]
    amp_dom = amp_dom[::downsample]
    
    ## Build 2D manifold ##
    x = amp_dom * np.sin(2 * np.pi * phase_dom)
    y_coord = amp_dom * np.cos(2 * np.pi * phase_dom)
    embedding = np.column_stack([x, y_coord])
    
    print(f"  [PAE] Shape: {embedding.shape} (downsampled {downsample}x)")
    return embedding


def extract_vqpae_single_window(audio_path, window_start=0, downsample=800):
    """
    VQPAE: Use z_pae (encoder output) - SAME STRUCTURE AS PAE's latent!
    z_pae shape: (batch, embedding_channels, time_steps)
    
    Window: 6 seconds (96000 samples @ 16kHz)
    Note: VQPAE encoder has stride=1, so z_pae has 96000 time steps (not downsampled)
    We need higher downsample rate (100x instead of 10x)
    """
    from src.models.VQ_PAE import VQ_AE
    from src.utils.config import load_config
    
    model_path = "PAE_training/DeepPhase4Audio-main/output/20260111_023242_VQ_PAE-aistpp_1000_eps-8emb/checkpoints/step_epoch=199.ckpt"
    config_path = "PAE_training/DeepPhase4Audio-main/configs/vqpae_aistpp_1000_eps_8emb.yaml"
    
    cfg = load_config(config_path, flatten=False)
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model = VQ_AE(cfg.model_config)
    state_dict = {}
    for key, val in checkpoint['state_dict'].items():
        if key.startswith('model.'):
            state_dict[key[6:]] = val
        else:
            state_dict[key] = val
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    ## Load audio ##
    y, sr = librosa.load(audio_path, sr=16000)
    
    ## Single 6-second window ##
    window_size = cfg.model_config.time_range  ## 96000 = 6s ##
    print(f"  [VQPAE] Window size: {window_size} samples = {window_size/16000:.1f}s")
    
    ## Ensure enough audio ##
    if len(y) < window_size:
        y = np.pad(y, (0, window_size - len(y)), mode='constant')
    
    chunk = y[window_start:window_start + window_size]
    chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0)
    
    with torch.no_grad():
        output = model(chunk_tensor)
    
    ## Use z_pae (encoder output) - SAME as PAE's latent! ##
    z_pae = output['z_pae']  ## (1, embedding_channels, time_steps) ##
    z_pae = z_pae.squeeze(0)  ## (channels, time_steps) ##
    
    num_channels = z_pae.shape[0] // 2  ## Should be 4 (8 channels / 2) ##
    time_steps = z_pae.shape[1]
    print(f"  [VQPAE] z_pae shape: {z_pae.shape}, num_channel_pairs: {num_channels}")
    
    ## Extract phase and amplitude - SAME LOGIC AS PAE ##
    phase_channels = z_pae[::2, :]  ## (4, time_steps) ##
    amp_channels = z_pae[1::2, :]  ## (4, time_steps) ##
    
    phase = torch.atan2(phase_channels, amp_channels + 1e-8) / (2 * np.pi)
    phase = (phase + 0.5) % 1.0  ## [0, 1] ##
    amplitude = torch.sqrt(phase_channels**2 + amp_channels**2)
    
    ## Calculate mean amps for logging ##
    mean_amps = amplitude.mean(dim=1).cpu().numpy()
    
    ## Use channel 0 (not dominant) for consistency ##
    selected_ch = 0
    print(f"  [VQPAE] Using channel: {selected_ch} (all amps: {mean_amps})")
    
    ## Extract selected channel ##
    phase_dom = phase[selected_ch, :].cpu().numpy()
    amp_dom = amplitude[selected_ch, :].cpu().numpy()
    
    ## Downsample to reduce density ##
    phase_dom = phase_dom[::downsample]
    amp_dom = amp_dom[::downsample]
    
    ## Build 2D manifold - NO normalization, NO centering ##
    x = amp_dom * np.sin(2 * np.pi * phase_dom)
    y_coord = amp_dom * np.cos(2 * np.pi * phase_dom)
    
    print(f"  [VQPAE] Raw range: x=[{x.min():.2f}, {x.max():.2f}], y=[{y_coord.min():.2f}, {y_coord.max():.2f}]")
    
    embedding = np.column_stack([x, y_coord])
    
    print(f"  [VQPAE] Shape: {embedding.shape} (downsampled {downsample}x)")
    return embedding


def load_audio_samples(audio_dir, num_samples=3):
    """Load audio samples"""
    audio_files = sorted(Path(audio_dir).glob("*.wav"))[:num_samples]
    if len(audio_files) == 0:
        audio_files = sorted(Path(audio_dir).glob("*.mp3"))[:num_samples]
    if len(audio_files) == 0:
        raise FileNotFoundError(f"No audio files in {audio_dir}")
    return audio_files


def visualize_audio_manifolds(audio_files, output_path="tmp/manifolds_v9/visualization.png"):
    """Visualize using single window time series"""
    
    plotter = Plotter()
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    print("="*60)
    print("Extracting features (V9 - fixed VQPAE)...")
    print("="*60)
    
    original_manifolds = []
    librosa_manifolds = []
    pae_manifolds = []
    vqpae_manifolds = []
    
    for i, audio_file in enumerate(audio_files):
        print(f"\n[{i+1}/{len(audio_files)}] Processing: {audio_file.name}")
        
        ## Original ##
        original_feat = extract_original_features(str(audio_file))
        original_manifolds.append(torch.from_numpy(original_feat).float())
        
        ## Librosa ##
        librosa_feat = extract_librosa_features(str(audio_file))
        librosa_manifolds.append(torch.from_numpy(librosa_feat).float())
        
        ## PAE ##
        pae_feat = extract_pae_single_window(str(audio_file))
        pae_manifolds.append(pae_feat)
        
        ## VQPAE ##
        vqpae_feat = extract_vqpae_single_window(str(audio_file))
        vqpae_manifolds.append(vqpae_feat)
    
    ## Colors ##
    colors = list(mcolors.TABLEAU_COLORS.values())[:len(audio_files)]
    
    print("\n" + "="*60)
    print("Creating visualization...")
    print("="*60)
    
    ## Original & Librosa: use PCA with lines ##
    plotter.plot_pca(axes[0], original_manifolds, title="(a) Original", point_color=colors, draw_line=True)
    plotter.plot_pca(axes[1], librosa_manifolds, title="(b) Librosa", point_color=colors, draw_line=True)
    
    ## PAE: already 2D, direct plot ##
    for i, (pae_feat, color) in enumerate(zip(pae_manifolds, colors)):
        print(f"  [PAE {i}] x range: [{pae_feat[:, 0].min():.3f}, {pae_feat[:, 0].max():.3f}], y range: [{pae_feat[:, 1].min():.3f}, {pae_feat[:, 1].max():.3f}]")
        axes[2].plot(pae_feat[:, 0], pae_feat[:, 1], color=color, alpha=0.3, linewidth=0.5)
        axes[2].scatter(pae_feat[:, 0], pae_feat[:, 1], color=color, s=10, alpha=0.6, label=i)
    axes[2].set_title("(c) PAE")
    axes[2].legend()
    axes[2].axis('equal')
    axes[2].set_axis_off()
    
    ## VQPAE: already 2D, direct plot - remove outliers ##
    for i, (vqpae_feat, color) in enumerate(zip(vqpae_manifolds, colors)):
        ## Remove outliers using IQR method ##
        x_data = vqpae_feat[:, 0]
        y_data = vqpae_feat[:, 1]
        
        ## Calculate IQR for x and y ##
        q1_x, q3_x = np.percentile(x_data, [25, 75])
        q1_y, q3_y = np.percentile(y_data, [25, 75])
        iqr_x = q3_x - q1_x
        iqr_y = q3_y - q1_y
        
        ## Filter out outliers (1.5 * IQR) ##
        mask = ((x_data >= q1_x - 1.5 * iqr_x) & (x_data <= q3_x + 1.5 * iqr_x) &
                (y_data >= q1_y - 1.5 * iqr_y) & (y_data <= q3_y + 1.5 * iqr_y))
        
        vqpae_filtered = vqpae_feat[mask]
        removed = len(vqpae_feat) - len(vqpae_filtered)
        
        print(f"  [VQPAE {i}] x range: [{vqpae_filtered[:, 0].min():.3f}, {vqpae_filtered[:, 0].max():.3f}], y range: [{vqpae_filtered[:, 1].min():.3f}, {vqpae_filtered[:, 1].max():.3f}] (removed {removed} outliers)")
        axes[3].plot(vqpae_filtered[:, 0], vqpae_filtered[:, 1], color=color, alpha=0.3, linewidth=0.5)
        axes[3].scatter(vqpae_filtered[:, 0], vqpae_filtered[:, 1], color=color, s=10, alpha=0.6, label=i)
    axes[3].set_title("(d) VQPAE")
    axes[3].legend()
    axes[3].axis('equal')
    axes[3].set_axis_off()
    
    plt.tight_layout()
    
    ## Save ##
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize audio manifolds V9")
    parser.add_argument("--audio_dir", type=str, default="raw_data/aistpp/wavs")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--output", type=str, default="tmp/manifolds_v9/visualization.png")
    
    args = parser.parse_args()
    
    audio_files = load_audio_samples(args.audio_dir, args.num_samples)
    print(f"Loaded {len(audio_files)} audio files")
    
    visualize_audio_manifolds(audio_files, args.output)


if __name__ == "__main__":
    main()
