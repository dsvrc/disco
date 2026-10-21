"""
Visualize single dominant channel to verify circular/spiral structure

conda activate music2dance \
&& python PAE_training/DeepPhase4Audio-main/plotting/tmps/visualize_single_channel.py \
--audio_file PAE_training/DeepPhase4Audio-main/plotting/wav/0.wav \
--output tmp/manifolds_single_channel/visualization.png
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


def unwrap_phase(phase):
    """Unwrap phase to continuous values"""
    phase = np.array(phase)
    unwrapped = np.unwrap(phase * 2 * np.pi) / (2 * np.pi)
    return unwrapped


def smooth_amplitude(amplitude, window=5):
    """Smooth amplitude with moving average"""
    if len(amplitude) < window:
        return amplitude
    kernel = np.ones(window) / window
    smoothed = np.convolve(amplitude, kernel, mode='same')
    return smoothed


def extract_pae_single_channel(audio_path, max_duration=10.0):
    """Extract PAE and return single dominant channel"""
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
    y = y[:int(max_duration * sr)]
    
    ## Dense sampling ##
    window_size = cfg.model_config.time_range
    hop_size = 320  ## 0.02s ##
    
    print(f"[PAE] Dense sampling: hop={hop_size/sr:.3f}s")
    
    all_phases = []
    all_amplitudes = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(y) - window_size + 1, hop_size), desc="Extracting", leave=False):
            chunk = y[i:i + window_size]
            chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0).unsqueeze(0)
            
            pred, latent, signal, param = model(chunk_tensor)
            
            if param is not None and isinstance(param, list) and len(param) == 4:
                p, f, a, b = param
                phase = (p.squeeze(2) + 0.5) % 1.0  ## (1, 8) ##
                amplitude = torch.clamp(a.squeeze(2), min=0.0)  ## (1, 8) ##
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
    
    ## Find dominant channel (highest mean amplitude) ##
    mean_amps = all_amplitudes.mean(axis=0)
    dominant_ch = np.argmax(mean_amps)
    print(f"\n[Dominant Channel] ch{dominant_ch}, mean_amplitude={mean_amps[dominant_ch]:.3f}")
    print(f"All channel amplitudes: {mean_amps}")
    
    return all_phases, all_amplitudes, dominant_ch


def visualize_all_channels(all_phases, all_amplitudes, output_path):
    """Visualize all 8 channels separately"""
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for ch in range(8):
        ## Unwrap phase ##
        phase_unwrapped = unwrap_phase(all_phases[:, ch])
        amplitude = smooth_amplitude(all_amplitudes[:, ch], window=5)
        
        ## Build 2D phase manifold for this channel ##
        x = amplitude * np.sin(2 * np.pi * phase_unwrapped)
        y = amplitude * np.cos(2 * np.pi * phase_unwrapped)
        
        ## Plot with time color ##
        colors = plt.cm.viridis(np.linspace(0, 1, len(x)))
        axes[ch].plot(x, y, 'gray', alpha=0.3, linewidth=0.5)
        axes[ch].scatter(x, y, c=colors, s=5, alpha=0.7)
        axes[ch].set_title(f"Channel {ch}\nmean_amp={all_amplitudes[:, ch].mean():.3f}", fontsize=10)
        axes[ch].axis('equal')
        axes[ch].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\n✅ Saved to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize single channel phase manifold")
    parser.add_argument("--audio_file", type=str, required=True)
    parser.add_argument("--output", type=str, default="tmp/manifolds_single_channel/all_channels.png")
    
    args = parser.parse_args()
    
    all_phases, all_amplitudes, dominant_ch = extract_pae_single_channel(args.audio_file)
    visualize_all_channels(all_phases, all_amplitudes, args.output)


if __name__ == "__main__":
    main()
