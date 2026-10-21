# Music PAE Training and Integration

This directory contains code for training a Periodic Autoencoder (PAE) on music features and integrating it with the Diffusion Forcing model.

## Overview

The Music PAE extracts phase-based features (PFAB) from music:
- **P (Phase)**: Normalized phase in [-0.5, 0.5]
- **F (Frequency)**: Dominant frequency
- **A (Amplitude)**: Signal amplitude
- **B (Offset)**: DC offset

These features capture the periodic and rhythmic structure of music, making them ideal for music-to-motion generation.

## Training Music PAE

### 1. Prepare Music Features

Extract librosa features (35 dimensions) from your music files:

```python
import librosa
import numpy as np

def extract_music_features(audio_path, sr=22050, hop_length=512):
    """
    Extract 35-dim music features:
    - MFCC: 13 dims
    - Chroma: 12 dims
    - Tempogram: 8 dims
    - RMS: 1 dim
    - Spectral Centroid: 1 dim
    """
    y, sr = librosa.load(audio_path, sr=sr)
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    tempogram = librosa.feature.tempogram(y=y, sr=sr, hop_length=hop_length)[:8]
    rms = librosa.feature.rms(y=y, hop_length=hop_length)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
    
    # Stack all features
    features = np.vstack([mfcc, chroma, tempogram, rms, spectral_centroid])
    return features.T  # (time, 35)

# Extract and save
features = extract_music_features('music.wav')
np.save('music_features/music_001.npy', features)
```

### 2. Train PAE

```bash
python PAE_training/train_music_pae.py \
    --data_dir raw_data/fma/debug/concatenated \
    --save_dir ./music_pae_checkpoints \
    --input_channels 35 \
    --window_size 150 \
    --embedding_channels 5 \
    --batch_size 64 \
    --epochs 100 \
    --lr 1e-3

python PAE_training/train_music_pae.py \
    --data_dir raw_data/fma/processed/concatenated \
    --input_channels 35 \
    --window_size 150 \
    --embedding_channels 5
```

**Parameters:**
- `--data_dir`: Directory with .npy files (shape: T x 35)
- `--window_size`: Time window size (default: 150 frames = 5 seconds @ 30fps)
- `--embedding_channels`: Number of phase channels (default: 5)
- `--window_duration`: Time duration in seconds (default: 5.0s)
- `--stride`: Sliding window stride (default: 1 for frame-by-frame, matching real-time inference)

**Note:** stride=1 ensures training matches real-time streaming inference where we process frame-by-frame.

### 3. Monitor Training

Training will save:
- Checkpoints: `music_pae_checkpoints/music_pae_epoch_*.pt`
- Best model: `music_pae_checkpoints/music_pae_best.pt`
- Visualizations: `music_pae_checkpoints/reconstruction_epoch_*.png`

## Using PAE in Diffusion Model

### Model Architecture

The `DiffForcingWanPAEModel` processes music as follows:

```
Raw Music (B*T, 150, 35)  ← 150 frames = 5 seconds @ 30fps
    ↓ [Pretrained PAE - Frozen]
PFAB Features (B*T, 20)
    ↓ [Periodic Phase Encoder]
    ├─ P: sin/cos encoding with 8 frequencies → 80 dims
    └─ FAB: direct concat → 15 dims
    ↓ [Linear Projection]
Encoded Music (B*T, 1024)
    ↓ [Cross-Attention in WanModel]
Conditioned Motion Generation
```

### Key Components

1. **PeriodicPhaseEncoder**: Preserves phase periodicity using multi-frequency sin/cos encoding
2. **Music PAE**: Extracts PFAB features (frozen during diffusion training)
3. **Cross-Attention**: Integrates music features into motion generation

### Real-time Streaming Design

The model is designed for **frame-by-frame streaming inference**:
- Training uses stride=1 (frame-by-frame sliding window)
- At inference time t, extract window [t-149:t] (150 frames = past 5 seconds)
- PAE processes window → PFAB → generate motion for frame t
- Move to t+1 and repeat (causal, real-time compatible)

### Configuration Example

```yaml
model:
    target: models.diffusion_forcing_wan_pae.DiffForcingWanPAEModel
    params:
        pae_checkpoint_path: "./music_pae_checkpoints/music_pae_best.pt"
        input_dim: 4  # Motion latent dimension after VAE
        hidden_dim: 1024
        ffn_dim: 2048
        music_window_size: 150  # Must match PAE training (5 seconds @ 30fps)
        phase_channels: 5  # Must match PAE training
        use_music_cond: true
        cfg_scale: 5.0  # Classifier-free guidance
        drop_out: 0.1  # CFG training dropout
```

## Phase Encoding Details

### Why Periodic Encoding?

Phase values P ∈ [-0.5, 0.5] are **periodic** - values near -0.5 and +0.5 are close in the cycle.
Direct linear projection breaks this periodicity!

### Solution: Multi-Frequency Encoding

For each phase channel, we compute:

```python
for freq in [1, 2, 4, 8, 16, 32, 64, 128]:
    encoded.append(sin(2π * freq * P))
    encoded.append(cos(2π * freq * P))
```

**Benefits:**
- ✅ P=-0.5 and P=0.5 map to similar encodings
- ✅ Multi-scale: captures both fine-grained and coarse phase patterns
- ✅ Proven effective (NeRF, Transformer positional encoding)

### Encoding Dimensions

- Phase (P): 5 channels × 8 freqs × 2 (sin/cos) = **80 dims**
- Frequency (F): 5 channels = **5 dims**
- Amplitude (A): 5 channels = **5 dims**
- Offset (B): 5 channels = **5 dims**
- **Total**: 95 dims → Project to 1024 dims

## Training Tips

### Music PAE Training

1. **Data**: Use diverse music with clear rhythmic patterns
2. **Window Size**: 150 frames @ 30fps = 5 seconds context (captures longer-term rhythmic patterns)
3. **Stride**: Use stride=1 for frame-by-frame training (matches real-time inference)
4. **Epochs**: 50-100 epochs usually sufficient
5. **Learning Rate**: Start with 1e-3, decay with cosine schedule

### Real-time Inference Considerations

1. **Causal Window**: Use past 5 seconds [t-149:t] for frame t generation
2. **Latency**: 5-second context = 150 frames lookback
3. **Edge Cases**: Pad with zeros for first 150 frames of a song

### Diffusion Model Training

1. **Freeze PAE**: PAE parameters should be frozen during diffusion training
2. **CFG Training**: Use `drop_out=0.1` to enable classifier-free guidance
3. **CFG Scale**: Test range [3.0, 7.0] during inference
4. **Warmup**: Consider warming up with lower CFG scale initially

## Troubleshooting

### PAE produces poor reconstructions
- Increase `embedding_channels` (try 8-10)
- Train longer (100+ epochs)
- Check if music features have good periodic structure

### Diffusion model ignores music
- Verify PAE checkpoint path is correct
- Check CFG scale (try higher values like 5.0-7.0)
- Ensure music features are properly aligned with motion

### Memory issues
- Reduce batch size in `encode_music_with_pae` (default: 128)
- Use smaller `hidden_dim`
- Process music offline and cache encodings

## File Structure

```
PAE_training/
├── music_pae.py              # PAE model definition
├── train_music_pae.py         # Training script
└── README.md                  # This file

models/
└── diffusion_forcing_wan_pae.py  # Diffusion model with PAE integration
```

## Citation

This implementation is based on:
- **DeepPhase** (SIGGRAPH 2022): Periodic Autoencoder for motion
- **DiscoForcing**: Music-to-dance generation

Adapted for music feature encoding with periodic phase representation.
