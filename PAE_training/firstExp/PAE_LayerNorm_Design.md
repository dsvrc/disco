# PAE LayerNorm Design Memo

## Overview
PAE (Periodic Autoencoder) uses a special LayerNorm design (`LN_v2`) that normalizes along the **time dimension** instead of the traditional channel dimension.

## Implementation Details

### Shape Convention
- Input: `(B, C, T)` where B=batch, C=channels, T=time_steps
- Learnable parameters: `alpha` and `beta` with shape `[1, 1, time_range]`

### Code (from SIGGRAPH 2022)
```python
class LN_v2(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones([1, 1, dim]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)  # Normalize along time (T)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y
```

## Why Normalize Along Time Dimension?

### 1. **Signal Processing Perspective**
Removing DC component and standardizing amplitude helps reveal periodic patterns:
```
Original:  [5, 7, 5, 3, 5, 7, 5, 3]  # Has DC offset and scale
De-meaned: [0, 2, 0, -2, 0, 2, 0, -2]  # Periodic pattern clearer
Normalized: [0, 1, 0, -1, 0, 1, 0, -1]  # Amplitude standardized
```

### 2. **PAE Goal: Extract Periodic Parameters**
PAE learns 4 parameters per latent channel:
- **P** (Phase): Where the cycle starts
- **F** (Frequency): How fast it oscillates  
- **A** (Amplitude): How big the oscillation is
- **B** (Offset): The DC component

Time-dimension normalization removes DC offset and standardizes amplitude, making it easier to learn these parameters via FFT.

### 3. **Learnable Alpha/Beta as Position Encoding**
- Shape `[1, 1, time_range]` means **per-timestep** learnable weights
- Not just global scale—allows network to learn importance of each time position
- Similar to positional encoding in Transformers

### 4. **Why NOT Channel-Dimension Normalization?**
- Music features have different physical meanings (MFCC, Chroma, Spectral Centroid, etc.)
- Normalizing across channels would mix these features and destroy their individual meanings
- PAE needs each feature channel to independently process its temporal pattern

## Key Insight
This is **not** traditional LayerNorm for feature normalization. It's a **signal processing technique** to:
1. Remove temporal bias (DC component)
2. Standardize temporal variance
3. Prepare signals for frequency domain analysis (FFT)

## Reference
- Original implementation: `/root/_projects/SIGGRAPH_2022/PyTorch/Library/Utility.py`
- Paper: SIGGRAPH 2022 - Periodic Autoencoder
- Applied to: Motion capture data (original) → Music features (this project)
