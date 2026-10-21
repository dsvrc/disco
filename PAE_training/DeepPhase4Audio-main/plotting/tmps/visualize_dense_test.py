"""
Test dense sampling for spiral visualization - Single audio

conda activate music2dance \
&& python PAE_training/DeepPhase4Audio-main/plotting/tmps/visualize_dense_test.py \
--audio_file PAE_training/DeepPhase4Audio-main/plotting/wav/0.wav \
--output tmp/manifolds_dense_test/visualization.png
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import librosa
import time
from tqdm import tqdm

## Add PAE training directory to path ##
pae_base = Path("PAE_training/DeepPhase4Audio-main")
if str(pae_base) not in sys.path:
    sys.path.insert(0, str(pae_base))

from sklearn.decomposition import PCA


def smooth_amplitude(amplitude, window=5):
    """Apply moving average smoothing to amplitude"""
    if len(amplitude) < window:
        return amplitude
    kernel = np.ones(window) / window
    smoothed = np.convolve(amplitude, kernel, mode='same')
    return smoothed


def unwrap_phase(phase):
    """
    Unwrap phase to make it continuous (avoid 0.99->0.01 jumps)
    phase: (N,) array in [0, 1] range
    returns: (N,) array with continuous values
    """
    phase = np.array(phase)
    unwrapped = np.unwrap(phase * 2 * np.pi) / (2 * np.pi)  ## Convert to radians, unwrap, convert back ##
    return unwrapped


def extract_pae_dense(audio_path, max_duration=10.0):
    """
    Extract PAE with DENSE sampling (hop_size=320)
    max_duration: only process first N seconds to speed up testing
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
    
    print(f"[PAE] Model loaded, embedding_channels={cfg.model_config.embedding_channels}")
    
    ## Load audio ##
    y, sr = librosa.load(audio_path, sr=16000)
    
    ## Limit duration for testing ##
    max_samples = int(max_duration * sr)
    y = y[:max_samples]
    
    ## DENSE sampling ##
    window_size = cfg.model_config.time_range  ## 32000 = 2s ##
    hop_size = 320  ## 0.02s hop - very dense! ##
    
    total_windows = (len(y) - window_size) // hop_size + 1
    print(f"  [PAE] Dense sampling: hop={hop_size} ({hop_size/sr:.3f}s), total_windows={total_windows}")
    
    all_phases = []
    all_amplitudes = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(y) - window_size + 1, hop_size), desc="  [PAE]", leave=False):
            chunk = y[i:i + window_size]
            chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0).unsqueeze(0)
            
            pred, latent, signal, param = model(chunk_tensor)
            
            if param is not None and isinstance(param, list) and len(param) == 4:
                p, f, a, b = param
                phase = (p.squeeze(2) + 0.5) % 1.0
                amplitude = torch.clamp(a.squeeze(2), min=0.0)
            else:
                latent_mean = latent.mean(dim=2)
                amplitude = torch.abs(latent_mean)
                phase = (torch.atan2(latent[:, ::2, :].mean(dim=2), 
                                    latent[:, 1::2, :].mean(dim=2) + 1e-8) / (2 * np.pi)) % 1.0
                if amplitude.shape[1] != phase.shape[1]:
                    amplitude = amplitude[:, :phase.shape[1]]
            
            all_phases.append(phase.cpu().numpy())
            all_amplitudes.append(amplitude.cpu().numpy())
    
    all_phases = np.vstack(all_phases)  ## (N, 8) ##
    all_amplitudes = np.vstack(all_amplitudes)  ## (N, 8) ##
    
    ## Unwrap phase for each channel to avoid wrapping jumps ##
    for ch in range(all_phases.shape[1]):
        all_phases[:, ch] = unwrap_phase(all_phases[:, ch])
    
    ## Optional: smooth amplitude to reduce jitter ##
    for ch in range(all_amplitudes.shape[1]):
        all_amplitudes[:, ch] = smooth_amplitude(all_amplitudes[:, ch], window=5)
    
    ## Build 16D phase manifold ##
    embedding = []
    for ch in range(8):
        embedding.append(all_amplitudes[:, ch] * np.sin(2 * np.pi * all_phases[:, ch]))
        embedding.append(all_amplitudes[:, ch] * np.cos(2 * np.pi * all_phases[:, ch]))
    embedding = np.column_stack(embedding)
    
    print(f"[PAE] Dense embedding shape: {embedding.shape}")
    return embedding


def extract_vqpae_dense(audio_path, max_duration=10.0):
    """
    Extract VQPAE with DENSE sampling (hop_size=3200)
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
    
    print(f"[VQPAE] Model loaded, embedding_channels={cfg.model_config.embedding_channels}")
    
    ## Load audio ##
    y, sr = librosa.load(audio_path, sr=16000)
    
    ## Limit duration ##
    max_samples = int(max_duration * sr)
    y = y[:max_samples]
    
    ## DENSE sampling ##
    window_size = cfg.model_config.time_range  ## 96000 = 6s ##
    hop_size = 3200  ## 0.2s hop - denser ##
    
    total_windows = (len(y) - window_size) // hop_size + 1
    print(f"  [VQPAE] Dense sampling: hop={hop_size} ({hop_size/sr:.3f}s), total_windows={total_windows}")
    
    all_phases = []
    all_amplitudes = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(y) - window_size + 1, hop_size), desc="  [VQPAE]", leave=False):
            chunk = y[i:i + window_size]
            chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0)
            
            output = model(chunk_tensor)
            phase_params = output.get('phase_params', None)
            p, f, a, b = phase_params
            
            phase = (p.squeeze(2) + 0.5) % 1.0
            amplitude = torch.clamp(a.squeeze(2), min=0.0)
            
            all_phases.append(phase.cpu().numpy())
            all_amplitudes.append(amplitude.cpu().numpy())
    
    all_phases = np.vstack(all_phases)
    all_amplitudes = np.vstack(all_amplitudes)
    
    ## Unwrap phase for each channel ##
    for ch in range(all_phases.shape[1]):
        all_phases[:, ch] = unwrap_phase(all_phases[:, ch])
    
    ## Smooth amplitude ##
    for ch in range(all_amplitudes.shape[1]):
        all_amplitudes[:, ch] = smooth_amplitude(all_amplitudes[:, ch], window=5)
    
    ## Build 16D phase manifold ##
    embedding = []
    for ch in range(8):
        embedding.append(all_amplitudes[:, ch] * np.sin(2 * np.pi * all_phases[:, ch]))
        embedding.append(all_amplitudes[:, ch] * np.cos(2 * np.pi * all_phases[:, ch]))
    embedding = np.column_stack(embedding)
    
    print(f"[VQPAE] Dense embedding shape: {embedding.shape}")
    return embedding


def visualize_comparison(audio_file, output_path):
    """Compare PAE and VQPAE with dense sampling"""
    
    print("\n" + "="*60)
    print(f"Processing: {Path(audio_file).name}")
    print("="*60)
    
    ## Extract features ##
    pae_feat = extract_pae_dense(audio_file, max_duration=10.0)
    vqpae_feat = extract_vqpae_dense(audio_file, max_duration=10.0)
    
    ## PCA to 2D ##
    pca_pae = PCA(n_components=2)
    pca_vqpae = PCA(n_components=2)
    
    pae_2d = pca_pae.fit_transform(pae_feat)
    vqpae_2d = pca_vqpae.fit_transform(vqpae_feat)
    
    ## Plot ##
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ## PAE ##
    colors_pae = plt.cm.viridis(np.linspace(0, 1, len(pae_2d)))
    axes[0].plot(pae_2d[:, 0], pae_2d[:, 1], 'gray', alpha=0.3, linewidth=0.5)
    axes[0].scatter(pae_2d[:, 0], pae_2d[:, 1], c=colors_pae, s=10, alpha=0.6)
    axes[0].set_title(f"PAE Dense Sampling\n{len(pae_2d)} frames", fontsize=12)
    axes[0].axis('equal')
    axes[0].grid(True, alpha=0.3)
    
    ## VQPAE ##
    colors_vqpae = plt.cm.plasma(np.linspace(0, 1, len(vqpae_2d)))
    axes[1].plot(vqpae_2d[:, 0], vqpae_2d[:, 1], 'gray', alpha=0.3, linewidth=0.5)
    axes[1].scatter(vqpae_2d[:, 0], vqpae_2d[:, 1], c=colors_vqpae, s=10, alpha=0.6)
    axes[1].set_title(f"VQPAE Dense Sampling\n{len(vqpae_2d)} frames", fontsize=12)
    axes[1].axis('equal')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    ## Save ##
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\n✅ Saved to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test dense sampling visualization")
    parser.add_argument("--audio_file", type=str, required=True, help="Audio file path")
    parser.add_argument("--output", type=str, default="tmp/manifolds_dense_test/visualization.png")
    
    args = parser.parse_args()
    
    visualize_comparison(args.audio_file, args.output)


if __name__ == "__main__":
    main()
