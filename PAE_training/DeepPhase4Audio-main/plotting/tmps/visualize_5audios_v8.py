"""
Visualize audio manifolds V8 - Single window time series

Key changes from V7:
- Use single window's internal time dimension (no sliding windows)
- Phase is continuous within the window
- Should produce clean spiral/circular trajectories

conda activate music2dance \
&& python PAE_training/DeepPhase4Audio-main/plotting/tmps/visualize_5audios_v8.py \
--audio_dir PAE_training/DeepPhase4Audio-main/plotting/wav \
--num_samples 5 \
--output tmp/manifolds_v8/visualization.png
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


def extract_pae_single_window(audio_path, window_start=0):
    """
    Extract PAE from single window's time series - SINGLE DOMINANT CHANNEL
    Returns: 2D phase manifold from dominant channel only
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
    
    ## Extract phase and amplitude from latent time series ##
    latent = latent.squeeze(0)  ## (channels, time_steps) ##
    num_channels = latent.shape[0] // 2  ## Should be 8 ##
    time_steps = latent.shape[1]
    
    phase_channels = latent[::2, :]  ## (8, time_steps) ##
    amp_channels = latent[1::2, :]  ## (8, time_steps) ##
    
    phase = torch.atan2(phase_channels, amp_channels + 1e-8) / (2 * np.pi)
    phase = (phase + 0.5) % 1.0  ## [0, 1] ##
    amplitude = torch.sqrt(phase_channels**2 + amp_channels**2)
    
    ## Find dominant channel ##
    mean_amps = amplitude.mean(dim=1).cpu().numpy()
    dominant_ch = np.argmax(mean_amps)
    print(f"  [PAE] Dominant channel: {dominant_ch}, mean_amp={mean_amps[dominant_ch]:.3f}")
    
    ## Extract only dominant channel ##
    phase_dom = phase[dominant_ch, :].cpu().numpy()  ## (time_steps,) ##
    amp_dom = amplitude[dominant_ch, :].cpu().numpy()  ## (time_steps,) ##
    
    ## Build 2D phase manifold for dominant channel only ##
    x = amp_dom * np.sin(2 * np.pi * phase_dom)
    y = amp_dom * np.cos(2 * np.pi * phase_dom)
    embedding = np.column_stack([x, y])  ## (time_steps, 2) ##
    
    print(f"  [PAE] Single channel embedding shape: {embedding.shape}")
    return embedding


def extract_vqpae_single_window(audio_path, window_start=0):
    """
    Extract VQPAE from single window's time series - SINGLE DOMINANT CHANNEL
    Returns: 2D phase manifold from dominant channel only
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
    
    ## Single window ##
    window_size = cfg.model_config.time_range  ## 96000 = 6s ##
    chunk = y[window_start:window_start + window_size]
    chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0)
    
    with torch.no_grad():
        output = model(chunk_tensor)
    
    ## Get z_vq (quantized latent) time series ##
    z_vq = output['z_vq']  ## (1, channels, time_steps) ##
    z_vq = z_vq.squeeze(0)  ## (channels, time_steps) ##
    
    num_channels = z_vq.shape[0] // 2  ## Should be 8 ##
    time_steps = z_vq.shape[1]
    
    ## Extract phase and amplitude ##
    phase_channels = z_vq[::2, :]
    amp_channels = z_vq[1::2, :]
    
    phase = torch.atan2(phase_channels, amp_channels + 1e-8) / (2 * np.pi)
    phase = (phase + 0.5) % 1.0
    amplitude = torch.sqrt(phase_channels**2 + amp_channels**2)
    
    ## Find dominant channel ##
    mean_amps = amplitude.mean(dim=1).cpu().numpy()
    dominant_ch = np.argmax(mean_amps)
    print(f"  [VQPAE] Dominant channel: {dominant_ch}, mean_amp={mean_amps[dominant_ch]:.3f}")
    
    ## Extract only dominant channel ##
    phase_dom = phase[dominant_ch, :].cpu().numpy()
    amp_dom = amplitude[dominant_ch, :].cpu().numpy()
    
    ## Build 2D phase manifold ##
    x = amp_dom * np.sin(2 * np.pi * phase_dom)
    y = amp_dom * np.cos(2 * np.pi * phase_dom)
    embedding = np.column_stack([x, y])  ## (time_steps, 2) ##
    
    print(f"  [VQPAE] Single channel embedding shape: {embedding.shape}")
    return embedding


def load_audio_samples(audio_dir, num_samples=3):
    """Load audio samples"""
    audio_files = sorted(Path(audio_dir).glob("*.wav"))[:num_samples]
    if len(audio_files) == 0:
        audio_files = sorted(Path(audio_dir).glob("*.mp3"))[:num_samples]
    if len(audio_files) == 0:
        raise FileNotFoundError(f"No audio files in {audio_dir}")
    return audio_files


def visualize_audio_manifolds(audio_files, output_path="tmp/manifolds_v8/visualization.png"):
    """Visualize using single window time series"""
    
    plotter = Plotter()
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    print("="*60)
    print("Extracting features (V8 - single window)...")
    print("="*60)
    
    original_manifolds = []
    librosa_manifolds = []
    pae_manifolds = []
    vqpae_manifolds = []
    
    for i, audio_file in enumerate(audio_files):
        print(f"\n[{i+1}/{len(audio_files)}] Processing: {audio_file.name}")
        
        ## Original ##
        print("  [Original] Extracting...")
        original_feat = extract_original_features(str(audio_file))
        original_manifolds.append(torch.from_numpy(original_feat).float())
        print(f"  [Original] Shape: {original_feat.shape}")
        
        ## Librosa ##
        print("  [Librosa] Extracting...")
        librosa_feat = extract_librosa_features(str(audio_file))
        librosa_manifolds.append(torch.from_numpy(librosa_feat).float())
        print(f"  [Librosa] Shape: {librosa_feat.shape}")
        
        ## PAE single window ##
        print("  [PAE] Extracting single window...")
        pae_feat = extract_pae_single_window(str(audio_file))
        pae_manifolds.append(torch.from_numpy(pae_feat).float())
        
        ## VQPAE single window ##
        print("  [VQPAE] Extracting single window...")
        vqpae_feat = extract_vqpae_single_window(str(audio_file))
        vqpae_manifolds.append(torch.from_numpy(vqpae_feat).float())
    
    ## Colors ##
    colors = list(mcolors.TABLEAU_COLORS.values())[:len(audio_files)]
    
    print("\n" + "="*60)
    print("Creating PCA visualization...")
    print("="*60)
    
    ## Plot all - PAE/VQPAE now 2D, no PCA needed ##
    plotter.plot_pca(axes[0], original_manifolds, title="(a) Original", point_color=colors, draw_line=False)
    plotter.plot_pca(axes[1], librosa_manifolds, title="(b) Librosa", point_color=colors, draw_line=False)
    
    ## PAE/VQPAE: already 2D, just scatter plot directly ##
    for i, (pae_feat, color) in enumerate(zip(pae_manifolds, colors)):
        axes[2].plot(pae_feat[:, 0], pae_feat[:, 1], color=color, alpha=0.3, linewidth=0.5)
        axes[2].scatter(pae_feat[:, 0], pae_feat[:, 1], color=color, s=10, alpha=0.6, label=i)
    axes[2].set_title("(c) PAE - Single Channel")
    axes[2].legend()
    axes[2].axis('equal')
    axes[2].set_axis_off()
    
    for i, (vqpae_feat, color) in enumerate(zip(vqpae_manifolds, colors)):
        axes[3].plot(vqpae_feat[:, 0], vqpae_feat[:, 1], color=color, alpha=0.3, linewidth=0.5)
        axes[3].scatter(vqpae_feat[:, 0], vqpae_feat[:, 1], color=color, s=10, alpha=0.6, label=i)
    axes[3].set_title("(d) VQPAE - Single Channel")
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
    
    parser = argparse.ArgumentParser(description="Visualize audio manifolds V8")
    parser.add_argument("--audio_dir", type=str, default="raw_data/aistpp/wavs")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--output", type=str, default="tmp/manifolds_v8/visualization.png")
    
    args = parser.parse_args()
    
    audio_files = load_audio_samples(args.audio_dir, args.num_samples)
    print(f"Loaded {len(audio_files)} audio files")
    
    visualize_audio_manifolds(audio_files, args.output)


if __name__ == "__main__":
    main()
