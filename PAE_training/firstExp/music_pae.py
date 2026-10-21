"""
Music PAE Model
Adapted from SIGGRAPH 2022 PAE for music feature encoding
Input: librosa music features (35 dimensions)
Output: Phase, Frequency, Amplitude, Offset (PFAB) parameters
"""

import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """
    Layer Normalization matching SIGGRAPH 2022 PAE LN_v2
    Normalizes along time dimension for each channel independently
    Simplified to work with dynamic time dimensions
    """
    def __init__(self, epsilon=1e-5):
        super(LayerNorm, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, x):
        """
        x: (B, C, T)
        Normalizes along time dimension T for each channel independently
        """
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        return y


class MusicPAE(nn.Module):
    """
    Periodic Autoencoder for Music Features with Dynamic Window Size
    
    Args:
        input_channels: Number of input feature channels (35 for librosa features)
        embedding_channels: Number of phase channels to learn (default: 5)
        time_range: Default number of time steps in the window (default: 150)
        window: Time duration in seconds (default: 5.0s for 150 frames @ 30fps)
    
    Note:
        Input format: (B, W, 35) where W can be different from time_range
        Maintains a cache for different W values to avoid recomputing parameters
    """
    def __init__(self, input_channels=35, embedding_channels=5, time_range=150, window=5.0):
        super(MusicPAE, self).__init__()
        self.input_channels = input_channels
        self.embedding_channels = embedding_channels
        self.time_range = time_range
        self.window = window
        
        # Cache for dynamic window sizes
        self._window_cache = {}

        # Constants
        self.tpi = Parameter(
            torch.from_numpy(np.array([2.0*np.pi], dtype=np.float32)), 
            requires_grad=False
        )
        # Default args and freqs for time_range
        self.register_buffer('args', 
            torch.from_numpy(
                np.linspace(-self.window/2, self.window/2, self.time_range, dtype=np.float32)
            )
        )
        # Frequencies for FFT (remove DC component)
        self.register_buffer('freqs',
            torch.fft.rfftfreq(time_range)[1:] * time_range / self.window
        )

        # Encoder
        intermediate_channels = max(int(input_channels / 2), embedding_channels * 2)
        
        self.conv1 = nn.Conv1d(
            input_channels, intermediate_channels, time_range, 
            stride=1, padding=int((time_range - 1) / 2), 
            dilation=1, groups=1, bias=True, padding_mode='zeros'
        )
        self.norm1 = LayerNorm()
        
        self.conv2 = nn.Conv1d(
            intermediate_channels, embedding_channels, time_range, 
            stride=1, padding=int((time_range - 1) / 2), 
            dilation=1, groups=1, bias=True, padding_mode='zeros'
        )

        # Phase prediction using global pooling + linear
        self.phase_fc = nn.Linear(embedding_channels, embedding_channels * 2)

        # Decoder
        self.deconv1 = nn.Conv1d(
            embedding_channels, intermediate_channels, time_range, 
            stride=1, padding=int((time_range - 1) / 2), 
            dilation=1, groups=1, bias=True, padding_mode='zeros'
        )
        self.denorm1 = LayerNorm()
        
        self.deconv2 = nn.Conv1d(
            intermediate_channels, input_channels, time_range, 
            stride=1, padding=int((time_range - 1) / 2), 
            dilation=1, groups=1, bias=True, padding_mode='zeros'
        )
    
    def _get_window_params(self, W, device):
        """
        Get or create parameters for a specific window size W
        
        Args:
            W: Window size
            device: Device to create tensors on
        
        Returns:
            args: Time arguments for signal reconstruction
            freqs: Frequencies for FFT
        """
        if W not in self._window_cache:
            # Create new parameters for this window size
            args = torch.from_numpy(
                np.linspace(-self.window/2, self.window/2, W, dtype=np.float32)
            ).to(device)
            freqs = (torch.fft.rfftfreq(W)[1:] * W / self.window).to(device)
            self._window_cache[W] = (args, freqs)
        
        return self._window_cache[W]

    def FFT(self, function, dim, W, freqs):
        """
        Extract frequency, amplitude, and offset from signal using FFT
        
        Args:
            function: Input signal (B, C, W)
            dim: Dimension to perform FFT on
            W: Window size (time dimension)
            freqs: Frequency bins for this window size
        
        Returns:
            freq: Dominant frequency (B, C)
            amp: Amplitude (B, C)
            offset: DC offset (B, C)
        """
        rfft = torch.fft.rfft(function, dim=dim)
        magnitudes = rfft.abs()
        spectrum = magnitudes[:,:,1:]  # Remove DC component
        power = spectrum**2

        # Frequency: weighted average of spectrum
        # Ensure freqs matches spectrum length
        freqs_trimmed = freqs[:spectrum.shape[2]]
        freq = torch.sum(freqs_trimmed * power, dim=dim) / torch.sum(power, dim=dim)

        # Amplitude: 2 * RMS of power
        amp = 2 * torch.sqrt(torch.sum(power, dim=dim)) / W

        # Offset: DC component
        offset = rfft.real[:,:,0] / W

        return freq, amp, offset

    def forward(self, x):
        """
        Forward pass - input must be exactly time_range length
        
        Args:
            x: Input music features (B, time_range, 35)
        
        Returns:
            y: Reconstructed features (B, time_range, 35)
            latent: Latent embedding (B, embedding_channels, time_range)
            signal: Parametrized signal (B, embedding_channels, time_range)
            params: [phase, freq, amp, offset] each (B, embedding_channels, 1)
        """
        B, W, C = x.shape
        assert W == self.time_range, f"Input window size {W} must equal time_range {self.time_range}"
        assert C == self.input_channels, f"Expected {self.input_channels} channels, got {C}"
        
        # Transpose to (B, C, time_range) for Conv1d
        y = x.transpose(1, 2)  # (B, 35, time_range)

        # Encoder
        y = self.conv1(y)
        y = self.norm1(y)
        y = F.elu(y)

        y = self.conv2(y)
        
        # Ensure output is time_range length (Conv1d may reduce by 1-2 due to padding)
        if y.shape[2] != self.time_range:
            y = F.pad(y, (0, self.time_range - y.shape[2]))

        latent = y  # Save latent (B, embedding_channels, time_range)

        # Extract F, A, B from FFT
        f, a, b = self.FFT(y, dim=2, W=self.time_range, freqs=self.freqs)

        # Predict Phase: pool over time then predict
        y_pooled = y.mean(dim=-1)  # (B, embedding_channels)
        phase_raw = self.phase_fc(y_pooled)  # (B, embedding_channels*2)
        phase_raw = phase_raw.view(B, self.embedding_channels, 2)  # (B, embedding_channels, 2)
        p = torch.atan2(phase_raw[:,:,1], phase_raw[:,:,0]) / self.tpi  # (B, embedding_channels)

        # Reshape parameters
        p = p.unsqueeze(2)  # (B, embedding_channels, 1)
        f = f.unsqueeze(2)
        a = a.unsqueeze(2)
        b = b.unsqueeze(2)
        params = [p, f, a, b]

        # Reconstruct signal from parameters
        y = a * torch.sin(self.tpi * (f * self.args + p)) + b

        signal = y  # Save signal (B, embedding_channels, time_range)

        # Decoder
        y = self.deconv1(y)
        y = self.denorm1(y)
        y = F.elu(y)

        y = self.deconv2(y)  # (B, C, ~time_range)
        
        # Ensure output is time_range length
        if y.shape[2] != self.time_range:
            if y.shape[2] < self.time_range:
                y = F.pad(y, (0, self.time_range - y.shape[2]))
            else:
                y = y[:, :, :self.time_range]

        # Transpose back to (B, time_range, C)
        y = y.transpose(1, 2)  # (B, time_range, 35)

        return y, latent, signal, params

    def encode(self, x):
        """
        Encode input to PFAB parameters only (for inference)
        
        Args:
            x: Input music features (B, W, 35)
        
        Returns:
            params: [phase, freq, amp, offset] each (B, embedding_channels, 1)
        """
        with torch.no_grad():
            _, _, _, params = self.forward(x)
        return params

    def get_pfab_features(self, x):
        """
        Get PFAB features as a single tensor
        
        Args:
            x: Input music features (B, W, 35)
        
        Returns:
            pfab: (B, embedding_channels*4) concatenated PFAB features
        """
        params = self.encode(x)
        p, f, a, b = params
        
        # Squeeze and concatenate
        p = p.squeeze(-1)  # (B, embedding_channels)
        f = f.squeeze(-1)
        a = a.squeeze(-1)
        b = b.squeeze(-1)
        
        pfab = torch.cat([p, f, a, b], dim=1)  # (B, embedding_channels*4)
        return pfab
