import torch
import numpy as np
import soundfile as sf
import scipy.signal as signal
from tqdm import tqdm
import os

from src.experiments.vq_pae import VQ_PAEModel
from src.utils.config import load_config
from src.utils.helpers import get_device_accelerator

# Configuration
cfg_path = "configs/vqpaeAistpp_50eps_64emb.yaml"
model_ckpt_path = "output/20260104_022638_VQ_PAE-aistpp_5000_eps-64emb/checkpoints/step_epoch=199.ckpt"
audio_file = "../../raw_data/aistpp/wavs/gBR_sBM_cAll_d04_mBR0_ch07.wav"
output_dir = "output/vqpae_full_reconstruction-_VQ_PAE-aistpp_5000_eps-64emb"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load config and model
print("Loading model...")
cfg = load_config(cfg_path, flatten=False)
model = VQ_PAEModel.load_from_checkpoint(model_ckpt_path, cfg=cfg, strict=False)
model.to(device)
model.eval()
print(f"Model loaded on {device}")

# Audio processing function (from notebook)
def process_audio(file_path, steps=None, lowpass_cutoff=8000, original_rate=48000, target_rate=16000, window_length=2):
    """Process audio with sliding window reconstruction"""
    
    audio, sr = sf.read(file_path)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)  # Convert to mono
    
    assert sr == original_rate, f"Expected {original_rate} Hz, but got {sr} Hz"
    
    # Low-pass filter
    nyquist = sr / 2
    norm_cutoff = lowpass_cutoff / nyquist
    b, a = signal.butter(5, norm_cutoff, btype='low')
    filtered_audio = signal.filtfilt(b, a, audio)
    
    # Downsample to 16 KHz
    downsampled_audio = signal.resample_poly(filtered_audio, target_rate, sr)
    
    # Normalize to -1 to 1
    downsampled_audio = downsampled_audio / np.max(np.abs(downsampled_audio))
    
    # Make 2-second windows
    window_size = target_rate * window_length
    
    if steps is not None:
        assert steps <= len(downsampled_audio) // window_size
    else:
        steps = len(downsampled_audio) // window_size
    
    print(f"Processing {steps} windows of {window_length}s each...")
    
    out_samples = []
    with torch.no_grad():
        for i in tqdm(range(steps), desc="Reconstructing audio"):
            s = torch.tensor(
                downsampled_audio[i * window_size: (i+1) * window_size], 
                device=device
            ).float().unsqueeze(0)  # Add batch dimension
            
            pred = model.model(s)
            x_recon = pred["audio"].squeeze(0).cpu()  # Get reconstructed audio
            out_samples.append(x_recon)
    
    return downsampled_audio, torch.cat(out_samples, dim=0)

# Process full audio
print(f"\nProcessing audio file: {audio_file}")
original_audio, reconstructed_audio = process_audio(audio_file)

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Save results
print("\nSaving results...")
sf.write(os.path.join(output_dir, 'original_16khz.wav'), original_audio, 16000)
sf.write(os.path.join(output_dir, 'reconstructed_16khz.wav'), reconstructed_audio.numpy().flatten(), 16000)

print(f"Original audio length: {len(original_audio)/16000:.2f}s")
print(f"Reconstructed audio length: {len(reconstructed_audio)/16000:.2f}s")
print(f"Results saved to {output_dir}/")
print("Done!")
