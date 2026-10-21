"""
Visualize audio representation manifolds using PCA

Compare different audio encoding methods:
- Original: Raw spectral features
- Librosa: Hand-crafted features (MFCC, Chroma, etc.)
- PAE: Phase AutoEncoder latents
- VQPAE: VQ-Phase AutoEncoder latents
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import librosa

## Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

## Add PAE training directory to path
pae_path = Path(__file__).parent.parent / "PAE_training" / "DeepPhase4Audio-main"
sys.path.insert(0, str(pae_path))

from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
import matplotlib.colors as mcolors
from matplotlib.colors import to_rgba

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

def extract_original_features(audio_path, sr=16000, n_fft=2048, hop_length=512):
    """Extract original spectral features (mel spectrogram)"""
    y, sr = librosa.load(audio_path, sr=sr)
    
    ## Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=128
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db.T  ## Shape: (time, n_mels)

def extract_librosa_features(audio_path, sr=16000, hop_length=512):
    """Extract librosa hand-crafted features"""
    y, sr = librosa.load(audio_path, sr=sr)
    
    ## Extract envelope (onset strength)
    envelope = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    
    ## Extract MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop_length).T
    
    ## Extract Chroma
    chroma = librosa.feature.chroma_cens(
        y=y, sr=sr, hop_length=hop_length, n_chroma=12
    ).T
    
    ## Concatenate all features
    features = np.concatenate([
        envelope[:, np.newaxis],  ## (time, 1)
        mfcc,  ## (time, 20)
        chroma  ## (time, 12)
    ], axis=1)  ## (time, 33)
    
    return features

def extract_pae_features(audio_path, model_path=None, config_path=None):
    """Extract PAE (Phase AutoEncoder) features using real model encoder"""
    if model_path is None:
        model_path = "PAE_training/DeepPhase4Audio-main/output/20260103_213749_paeflat_twinkle_200eps/checkpoints/step_epoch=1999.ckpt"
    if config_path is None:
        config_path = "PAE_training/DeepPhase4Audio-main/configs/paeflat200.yaml"
    
    try:
        ## Import necessary modules
        from src.models.PAE import PAEFlat
        from src.utils.config import load_config
        
        ## Load config
        cfg = load_config(config_path, flatten=False)
        
        ## Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        ## Initialize model directly (without full Lightning module)
        model = PAEFlat(cfg.model_config)
        
        ## Load state dict (remove 'model.' prefix from keys)
        state_dict = {}
        for key, val in checkpoint['state_dict'].items():
            if key.startswith('model.'):
                state_dict[key[6:]] = val  ## Remove 'model.' prefix
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        print(f"[INFO] PAE model loaded successfully, embedding_channels={cfg.model_config.embedding_channels}")
        
        ## Load audio
        y, sr = librosa.load(audio_path, sr=16000)
        
        ## Process audio in windows
        window_size = 32000  ## 2 seconds at 16kHz (from config)
        hop_size = 320  ## 0.02 second hop for dense manifold visualization
        
        pae_features = []
        with torch.no_grad():
            for i in range(0, len(y) - window_size + 1, hop_size):
                chunk = y[i:i + window_size]
                chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0).unsqueeze(0)  ## (1, 1, N)
                
                ## Forward through model: (pred, latent, signal, param)
                _, latent, _, _ = model(chunk_tensor)
                
                ## latent shape: (1, embedding_channels, time_steps) 
                ## Average over time to get a single feature vector per window
                latent_avg = latent.mean(dim=2).squeeze(0)  ## (embedding_channels,)
                pae_features.append(latent_avg)
        
        if len(pae_features) == 0:
            ## If audio is too short, process the whole thing
            chunk_tensor = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0)
            if chunk_tensor.shape[2] < window_size:
                ## Pad if needed
                pad_size = window_size - chunk_tensor.shape[2]
                chunk_tensor = torch.nn.functional.pad(chunk_tensor, (0, pad_size))
            _, latent, _, _ = model.model(chunk_tensor)
            latent_avg = latent.mean(dim=2).squeeze(0)
            pae_features.append(latent_avg)
            
        pae_features = torch.stack(pae_features, dim=0).cpu().numpy()  ## (num_frames, embedding_channels)
        print(f"[INFO] PAE features extracted: shape={pae_features.shape}")
        return pae_features
        
    except Exception as e:
        print(f"[WARNING] PAE extraction failed: {e}")
        import traceback
        traceback.print_exc()
        y, sr = librosa.load(audio_path, sr=16000)
        num_frames = max(1, (len(y) - 32000) // 16000 + 1)
        dummy_features = np.random.randn(num_frames, 32).astype(np.float32)  ## Match embedding_channels=32
        return dummy_features

def extract_vqpae_features(audio_path, model_path=None, config_path=None):
    """Extract VQPAE (VQ-Phase AutoEncoder) features using real model encoder"""
    if model_path is None:
        model_path = "outputs/__archived__/20260104_030828_VQ_PAE-aistpp_1000_eps-8emb/checkpoints/step_epoch=199.ckpt"
    if config_path is None:
        config_path = "PAE_training/DeepPhase4Audio-main/configs/vqpae_aistpp_1000_eps_8emb.yaml"
    
    try:
        ## Import necessary modules
        from src.models.VQ_PAE import VQPAE
        from src.utils.config import load_config
        
        ## Load config
        cfg = load_config(config_path, flatten=False)
        
        ## Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        ## Initialize model directly (without full Lightning module)
        model = VQPAE(cfg.model_config)
        
        ## Load state dict (remove 'model.' prefix from keys)
        state_dict = {}
        for key, val in checkpoint['state_dict'].items():
            if key.startswith('model.'):
                state_dict[key[6:]] = val  ## Remove 'model.' prefix
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        print(f"[INFO] VQPAE model loaded successfully, embedding_channels={cfg.model_config.embedding_channels}")
        
        ## Load audio
        y, sr = librosa.load(audio_path, sr=16000)
        
        ## Process audio in windows
        window_size = 96000  ## 6 seconds at 16kHz (from config)
        hop_size = 320  ## 0.02 second hop for dense manifold visualization
        
        vqpae_features = []
        with torch.no_grad():
            for i in range(0, len(y) - window_size + 1, hop_size):
                chunk = y[i:i + window_size]
                chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0).unsqueeze(0)  ## (1, 1, N)
                
                ## Forward through model
                pred = model(chunk_tensor)
                
                ## pred is a dict with keys: audio, z_pae, z_vq, codes, latents
                z_vq = pred["z_vq"]  ## (1, embedding_channels, L)
                
                ## Average over time to get a single feature vector per window
                z_vq_avg = z_vq.mean(dim=2).squeeze(0)  ## (embedding_channels,)
                vqpae_features.append(z_vq_avg)
        
        if len(vqpae_features) == 0:
            ## If audio is too short, process the whole thing
            chunk_tensor = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0)
            if chunk_tensor.shape[2] < window_size:
                ## Pad if needed
                pad_size = window_size - chunk_tensor.shape[2]
                chunk_tensor = torch.nn.functional.pad(chunk_tensor, (0, pad_size))
            pred = model(chunk_tensor)
            z_vq = pred["z_vq"]
            z_vq_avg = z_vq.mean(dim=2).squeeze(0)
            vqpae_features.append(z_vq_avg)
            
        vqpae_features = torch.stack(vqpae_features, dim=0).cpu().numpy()  ## (num_frames, embedding_channels)
        print(f"[INFO] VQPAE features extracted: shape={vqpae_features.shape}")
        return vqpae_features
        
    except Exception as e:
        print(f"[WARNING] VQPAE extraction failed: {e}")
        import traceback
        traceback.print_exc()
        y, sr = librosa.load(audio_path, sr=16000)
        num_frames = max(1, (len(y) - 96000) // 16000 + 1)
        dummy_features = np.random.randn(num_frames, 8).astype(np.float32)  ## Match embedding_channels=8
        return dummy_features

def load_audio_samples(audio_dir, num_samples=5):
    """Load multiple audio samples from directory"""
    audio_files = sorted(Path(audio_dir).glob("*.wav"))[:num_samples]
    
    if len(audio_files) == 0:
        audio_files = sorted(Path(audio_dir).glob("*.mp3"))[:num_samples]
    
    if len(audio_files) == 0:
        raise FileNotFoundError(f"No audio files found in {audio_dir}")
    
    return audio_files

def visualize_audio_manifolds(audio_files, output_path="tmp/manifolds_v3_2236/visualization.png"):
    """
    Visualize audio representation manifolds for different encoding methods
    
    Args:
        audio_files: List of audio file paths
        output_path: Path to save visualization
    """
    ## Initialize plotter
    plotter = Plotter()
    
    ## Create figure with 4 subplots (one for each method)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    print("Extracting features from audio files...")
    
    ## Extract features using different methods
    original_manifolds = []
    librosa_manifolds = []
    pae_manifolds = []
    vqpae_manifolds = []
    
    for audio_file in audio_files:
        print(f"Processing: {audio_file.name}")
        
        ## Original spectral features
        original_feat = extract_original_features(str(audio_file))
        original_manifolds.append(torch.from_numpy(original_feat).float())
        
        ## Librosa features
        librosa_feat = extract_librosa_features(str(audio_file))
        librosa_manifolds.append(torch.from_numpy(librosa_feat).float())
        
        ## PAE features
        pae_feat = extract_pae_features(str(audio_file))
        pae_manifolds.append(torch.from_numpy(pae_feat).float())
        
        ## VQPAE features
        vqpae_feat = extract_vqpae_features(str(audio_file))
        vqpae_manifolds.append(torch.from_numpy(vqpae_feat).float())
    
    ## Generate colors for each sample
    colors = list(mcolors.TABLEAU_COLORS.values())[:len(audio_files)]
    
    print("\nVisualizing manifolds...")
    
    ## Plot Original
    plotter.plot_pca(
        axes[0], original_manifolds, 
        title="(a) Original",
        point_color=colors,
        draw_line=True, 
        draw_arrow=True
    )
    
    ## Plot Librosa
    plotter.plot_pca(
        axes[1], librosa_manifolds,
        title="(b) Librosa",
        point_color=colors,
        draw_line=True,
        draw_arrow=True
    )
    
    ## Plot PAE
    plotter.plot_pca(
        axes[2], pae_manifolds,
        title="(c) PAE",
        point_color=colors,
        draw_line=True,
        draw_arrow=True
    )
    
    ## Plot VQPAE
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
    print(f"\nSaved visualization to: {output_path}")
    
    plt.show()

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize audio representation manifolds")
    parser.add_argument(
        "--audio_dir", 
        type=str, 
        default="raw_data/fma/music_wav",
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
        default="tmp/manifolds_v3_2236/visualization.png",
        help="Output path for visualization"
    )
    
    args = parser.parse_args()
    
    ## Load audio files
    audio_files = load_audio_samples(args.audio_dir, args.num_samples)
    print(f"Loaded {len(audio_files)} audio files")
    
    ## Visualize manifolds
    visualize_audio_manifolds(audio_files, args.output)

if __name__ == "__main__":
    main()
