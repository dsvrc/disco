# Model Manager & Audio Processor

import sys
import os
import threading
import queue
import time
from collections import deque

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import librosa
from torch_ema import ExponentialMovingAverage
from utils.initialize import instantiate, load_config, compare_statedict_and_parameters
from utils.motion_process import StreamMotionRecovery272


# ============ Audio Processing Constants ============
AUDIO_FPS = 30
AUDIO_HOP_LENGTH = 512
AUDIO_SAMPLE_RATE = AUDIO_FPS * AUDIO_HOP_LENGTH  # 15360 Hz

# VQ-PAE constants (from utils/VQPAE_preprocess.py)
VQPAE_SAMPLE_RATE = 16000
VQPAE_HOP_SAMPLES = VQPAE_SAMPLE_RATE // AUDIO_FPS  # 533 samples per frame @ 30fps
VQPAE_TIME_RANGE = 96000  # 6 seconds @ 16kHz
VQPAE_WINDOW_SIZE = 180   # 6s * 30fps
VQPAE_EMBEDDING_CHANNELS = 8


class StreamAudioProcessor:
    """
    Real-time audio feature extractor for music-to-motion generation.
    Processes audio at 30Hz and extracts features compatible with the model.
    Output shape: (window_size, 35) per frame
    Features: [envelope(1), mfcc(20), chroma(12), peak(1), beat(1)]
    """
    
    def __init__(self, window_size=32, sample_rate=44100):
        self.window_size = window_size
        self.input_sample_rate = sample_rate
        self.target_sample_rate = AUDIO_SAMPLE_RATE
        self.hop_length = AUDIO_HOP_LENGTH
        
        # Audio buffer for accumulating samples
        self.audio_buffer = np.array([], dtype=np.float32)
        
        # Feature buffer: store extracted features for each frame
        # Shape: list of (window_size, 35) arrays
        self.feature_buffer = deque(maxlen=1000)
        
        # Frame counter
        self.frame_count = 0
        
        # Beat tracking state
        self.start_bpm = 120.0
        
        # Stats for debugging
        self.stats = {
            'resample_time': 0,
            'feature_extract_time': 0,
            'total_frames_processed': 0
        }
        
        self.lock = threading.Lock()
        
        print(f"StreamAudioProcessor initialized: window={window_size}, "
              f"input_sr={sample_rate}, target_sr={self.target_sample_rate}")
    
    def reset(self):
        """Reset processor state"""
        with self.lock:
            self.audio_buffer = np.array([], dtype=np.float32)
            self.feature_buffer.clear()
            self.frame_count = 0
    
    def add_audio_samples(self, samples, sample_rate=None):
        """
        Add raw audio samples to buffer and extract features.
        Every hop_length samples = 1 audio frame at 30fps.
        Feature is extracted immediately with left-padding if needed.
        """
        if sample_rate is None:
            sample_rate = self.input_sample_rate
        
        with self.lock:
            # Resample if necessary
            if sample_rate != self.target_sample_rate:
                t0 = time.time()
                samples = librosa.resample(
                    samples.astype(np.float32), 
                    orig_sr=sample_rate, 
                    target_sr=self.target_sample_rate
                )
                self.stats['resample_time'] = time.time() - t0
            
            # Append to buffer
            self.audio_buffer = np.concatenate([self.audio_buffer, samples])
            
            # Process frames - extract feature for each new audio frame
            self._process_pending_frames()
    
    def _process_pending_frames(self):
        """
        Process pending audio frames.
        Reference: librosa_preprocess.py
        
        For each hop_length samples received, we extract one (window_size, 35) feature.
        The audio window for frame t covers samples for frames [t - window_size + 1, t].
        Use LEFT PADDING (zeros) when t < window_size.
        """
        samples_per_frame = self.hop_length  # 512 samples = 1 frame at 30fps
        full_window_samples = self.window_size * self.hop_length  # 32 * 512 = 16384
        
        # How many complete frames can we extract?
        # We need at least 1 frame (hop_length samples) to extract features
        num_frames_available = len(self.audio_buffer) // samples_per_frame
        
        # Process each new frame
        while self.frame_count < num_frames_available:
            current_frame = self.frame_count  # 0-indexed
            
            # Calculate which samples this frame needs
            # Frame t needs samples from frame (t - window_size + 1) to frame t
            # In samples: from (t - window_size + 1) * hop_length to (t + 1) * hop_length
            end_sample = (current_frame + 1) * samples_per_frame
            start_sample = end_sample - full_window_samples  # Can be negative
            
            # Extract audio window with left-padding if needed
            if start_sample < 0:
                # Need left padding
                pad_samples = -start_sample
                audio_segment = self.audio_buffer[:end_sample]
                audio_window = np.pad(audio_segment, (pad_samples, 0), mode='constant')
            else:
                audio_window = self.audio_buffer[start_sample:end_sample]
            
            # Ensure correct length
            assert len(audio_window) == full_window_samples, f"Window size mismatch: {len(audio_window)} vs {full_window_samples}"
            
            # Extract features for this frame: (window_size, 35)
            t0 = time.time()
            features = self._extract_frame_features(audio_window)
            self.stats['feature_extract_time'] = time.time() - t0
            self.stats['total_frames_processed'] = self.frame_count + 1
            
            self.feature_buffer.append(features)
            self.frame_count += 1
        
        # Trim buffer to prevent memory growth (keep last window_size frames)
        max_samples_to_keep = full_window_samples + samples_per_frame * 10  # Keep some extra
        if len(self.audio_buffer) > max_samples_to_keep:
            self.audio_buffer = self.audio_buffer[-max_samples_to_keep:]
            # Adjust frame_count calculation won't work after trimming, so we track separately
    
    def _extract_frame_features(self, audio_window):
        """
        Extract features from audio window.
        Reference: librosa_preprocess.py
        
        Args:
            audio_window: np.ndarray of shape (window_size * hop_length,)
        Returns:
            features: np.ndarray of shape (window_size, 35)
        """
        y = audio_window.astype(np.float32)
        sr = self.target_sample_rate
        hop = self.hop_length
        W = self.window_size
        
        # Envelope: onset strength -> (W+1,) -> take first W
        envelope = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)[:W]
        if len(envelope) < W:
            envelope = np.pad(envelope, (0, W - len(envelope)), mode='constant')
        
        # MFCC: (20, W+1) -> take first W columns -> transpose to (W, 20)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop)[:, :W].T
        if mfcc.shape[0] < W:
            mfcc = np.pad(mfcc, ((0, W - mfcc.shape[0]), (0, 0)), mode='constant')
        
        # Chroma: (12, W+1) -> take first W columns -> transpose to (W, 12)
        chroma = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop, n_chroma=12)[:, :W].T
        if chroma.shape[0] < W:
            chroma = np.pad(chroma, ((0, W - chroma.shape[0]), (0, 0)), mode='constant')
        
        # Peak detection on envelope
        peak_onehot = np.zeros(W, dtype=np.float32)
        try:
            peaks = librosa.onset.onset_detect(onset_envelope=envelope, sr=sr, hop_length=hop)
            peaks = peaks[peaks < W]
            if len(peaks) > 0:
                peak_onehot[peaks] = 1.0
        except:
            pass
        
        # Beat detection
        beat_onehot = np.zeros(W, dtype=np.float32)
        try:
            _, beats = librosa.beat.beat_track(
                onset_envelope=envelope,
                sr=sr,
                hop_length=hop,
                start_bpm=self.start_bpm,
                tightness=100
            )
            beats = beats[beats < W]
            if len(beats) > 0:
                beat_onehot[beats] = 1.0
        except:
            pass
        
        # Concatenate: (W, 35)
        features = np.concatenate([
            envelope[:, None],      # (W, 1)
            mfcc,                   # (W, 20)
            chroma,                 # (W, 12)
            peak_onehot[:, None],   # (W, 1)
            beat_onehot[:, None]    # (W, 1)
        ], axis=-1)
        
        return features.astype(np.float32)
    
    def get_latest_features(self):
        """
        Get the latest feature frame for model input.
        Returns:
            features: np.ndarray of shape (window_size, 35) or None if no features available
        """
        with self.lock:
            if len(self.feature_buffer) > 0:
                return self.feature_buffer[-1].copy()
            return None
    
    def get_features_at_index(self, idx):
        """
        Get feature at specific index.
        Returns:
            features: np.ndarray of shape (window_size, 35) or None
        """
        with self.lock:
            if 0 <= idx < len(self.feature_buffer):
                return self.feature_buffer[idx].copy()
            elif len(self.feature_buffer) > 0:
                # Return last available if index out of range
                return self.feature_buffer[-1].copy()
            return None
    
    def has_features(self):
        """Check if any features are available"""
        with self.lock:
            return len(self.feature_buffer) > 0
    
    def get_feature_count(self):
        """Get number of extracted feature frames"""
        with self.lock:
            return len(self.feature_buffer)
    
    def get_status(self):
        """Get processor status"""
        with self.lock:
            return {
                "audio_buffer_samples": len(self.audio_buffer),
                "audio_buffer_seconds": len(self.audio_buffer) / self.target_sample_rate if len(self.audio_buffer) > 0 else 0,
                "feature_frames": len(self.feature_buffer),
                "frame_count": self.frame_count,
                "stats": self.stats.copy()
            }


class StreamVQPAEProcessor:
    """
    Real-time VQ-PAE audio feature extractor for music-to-motion generation.
    Processes audio at 30Hz through VQ-PAE neural network.
    Output shape: (180, 16) per frame — concat of z_vq(8) + z_pae(8).
    """

    def __init__(self, vqpae_model, device='cuda', window_size=VQPAE_WINDOW_SIZE, sample_rate=44100):
        self.window_size = window_size  # 180
        self.input_sample_rate = sample_rate
        self.target_sample_rate = VQPAE_SAMPLE_RATE  # 16000
        self.hop_samples = VQPAE_HOP_SAMPLES  # 533

        self.vqpae_model = vqpae_model
        self.device = device

        # Audio buffer (resampled to 16kHz)
        self.audio_buffer = np.array([], dtype=np.float32)

        # Feature buffer: each entry is (180, 16)
        self.feature_buffer = deque(maxlen=1000)

        self.frame_count = 0

        self.stats = {
            'resample_time': 0,
            'feature_extract_time': 0,
            'total_frames_processed': 0
        }

        self.lock = threading.Lock()

        print(f"StreamVQPAEProcessor initialized: window={window_size}, "
              f"input_sr={sample_rate}, target_sr={self.target_sample_rate}, device={device}")

    def reset(self):
        """Reset processor state"""
        with self.lock:
            self.audio_buffer = np.array([], dtype=np.float32)
            self.feature_buffer.clear()
            self.frame_count = 0

    def add_audio_samples(self, samples, sample_rate=None):
        """Add raw audio samples to buffer and extract features via VQ-PAE."""
        if sample_rate is None:
            sample_rate = self.input_sample_rate

        with self.lock:
            # Resample to 16kHz
            if sample_rate != self.target_sample_rate:
                t0 = time.time()
                samples = librosa.resample(
                    samples.astype(np.float32),
                    orig_sr=sample_rate,
                    target_sr=self.target_sample_rate
                )
                self.stats['resample_time'] = time.time() - t0

            self.audio_buffer = np.concatenate([self.audio_buffer, samples])
            self._process_pending_frames()

    def _process_pending_frames(self):
        """Process pending audio frames through VQ-PAE."""
        num_frames_available = len(self.audio_buffer) // self.hop_samples

        while self.frame_count < num_frames_available:
            t0 = time.time()

            # Current frame ends at this sample position
            end_sample = (self.frame_count + 1) * self.hop_samples

            # We need a 6-second window (TIME_RANGE samples) ending at end_sample
            start_sample = end_sample - VQPAE_TIME_RANGE

            # Extract window with zero-padding for early frames
            if start_sample < 0:
                pad_len = -start_sample
                audio_segment = self.audio_buffer[:end_sample]
                audio_window = np.pad(audio_segment, (pad_len, 0), mode='constant')
            else:
                audio_window = self.audio_buffer[start_sample:end_sample]

            # Ensure exactly TIME_RANGE samples
            if len(audio_window) < VQPAE_TIME_RANGE:
                audio_window = np.pad(audio_window, (0, VQPAE_TIME_RANGE - len(audio_window)), mode='constant')
            elif len(audio_window) > VQPAE_TIME_RANGE:
                audio_window = audio_window[:VQPAE_TIME_RANGE]

            # VQ-PAE forward pass
            features = self._vqpae_forward(audio_window)
            self.stats['feature_extract_time'] = time.time() - t0
            self.stats['total_frames_processed'] = self.frame_count + 1

            self.feature_buffer.append(features)
            self.frame_count += 1

        # Trim buffer to prevent memory growth
        max_keep = VQPAE_TIME_RANGE + self.hop_samples * 10
        if len(self.audio_buffer) > max_keep:
            self.audio_buffer = self.audio_buffer[-max_keep:]

    @torch.no_grad()
    def _vqpae_forward(self, audio_window):
        """
        Run VQ-PAE forward and return (180, 16) feature array.
        Mirrors utils/VQPAE_preprocess.py extraction logic.
        """
        audio_tensor = torch.from_numpy(audio_window).float().unsqueeze(0).to(self.device)
        output = self.vqpae_model(audio_tensor)

        z_pae = output['z_pae']   # [1, 8, 96000]
        z_vq = output['z_vq']     # [1, 8, 96000]

        z_pae_down = torch.nn.functional.adaptive_avg_pool1d(z_pae, output_size=self.window_size)  # [1, 8, 180]
        z_vq_down = torch.nn.functional.adaptive_avg_pool1d(z_vq, output_size=self.window_size)    # [1, 8, 180]

        z_concat = torch.cat([z_vq_down, z_pae_down], dim=1)  # [1, 16, 180]
        features = z_concat.squeeze(0).permute(1, 0).cpu().numpy()  # [180, 16]

        return features.astype(np.float32)

    def get_latest_features(self):
        """Get the latest feature frame. Returns (180, 16) or None."""
        with self.lock:
            if len(self.feature_buffer) > 0:
                return self.feature_buffer[-1].copy()
            return None

    def get_features_at_index(self, idx):
        """Get feature at specific index. Returns (180, 16) or None."""
        with self.lock:
            if 0 <= idx < len(self.feature_buffer):
                return self.feature_buffer[idx].copy()
            elif len(self.feature_buffer) > 0:
                return self.feature_buffer[-1].copy()
            return None

    def has_features(self):
        """Check if any features are available"""
        with self.lock:
            return len(self.feature_buffer) > 0

    def get_feature_count(self):
        """Get number of extracted feature frames"""
        with self.lock:
            return len(self.feature_buffer)

    def get_status(self):
        """Get processor status"""
        with self.lock:
            return {
                "audio_buffer_samples": len(self.audio_buffer),
                "audio_buffer_seconds": len(self.audio_buffer) / self.target_sample_rate if len(self.audio_buffer) > 0 else 0,
                "feature_frames": len(self.feature_buffer),
                "frame_count": self.frame_count,
                "stats": self.stats.copy()
            }


class FrameBuffer:
    """
    Thread-safe frame buffer that maintains a queue of generated frames
    """
    def __init__(self, target_buffer_size=4):
        self.buffer = deque(maxlen=100)  # Max 100 frames in buffer
        self.target_size = target_buffer_size
        self.lock = threading.Lock()
        
    def add_frame(self, joints):
        """Add a frame to the buffer"""
        with self.lock:
            self.buffer.append(joints)
    
    def get_frame(self):
        """Get the next frame from buffer"""
        with self.lock:
            if len(self.buffer) > 0:
                return self.buffer.popleft()
            return None
    
    def get_frames(self, count):
        """Get multiple frames from buffer at once"""
        with self.lock:
            frames = []
            for _ in range(min(count, len(self.buffer))):
                frames.append(self.buffer.popleft())
            return frames
    
    def size(self):
        """Get current buffer size"""
        with self.lock:
            return len(self.buffer)
    
    def clear(self):
        """Clear the buffer"""
        with self.lock:
            self.buffer.clear()
    
    def needs_generation(self):
        """Check if buffer needs more frames"""
        return self.size() < self.target_size


class ModelManager:
    """
    Manages model loading and real-time frame generation
    Supports both text and audio (music) conditioning modes
    """
    def __init__(self, config_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load models
        self.vae, self.model, cfg = self._load_models(config_path)

        # Determine music_dim from config to choose audio processor type
        self.music_dim = getattr(cfg.model.params, 'music_dim', 35)
        self.use_vqpae = (self.music_dim == 16)

        # Load VQ-PAE model if needed
        self.vqpae_model = None
        if self.use_vqpae:
            vqpae_ckpt = getattr(cfg, 'vqpae_ckpt', None)
            if vqpae_ckpt is None:
                raise ValueError("music_dim=16 requires vqpae_ckpt in config")
            # Resolve relative path from project root (parent of web_demo/)
            project_root = os.path.dirname(os.path.dirname(__file__))
            if not os.path.isabs(vqpae_ckpt):
                vqpae_ckpt = os.path.join(project_root, vqpae_ckpt)
            print(f"Loading VQ-PAE model from {vqpae_ckpt}...")
            from utils.VQPAE_preprocess import load_vqpae_model
            self.vqpae_model = load_vqpae_model(vqpae_ckpt)
            self.vqpae_model.to(self.device)
            self.vqpae_model.eval()
            print(f"VQ-PAE model loaded (music_dim={self.music_dim})")
        else:
            print(f"Using librosa audio processor (music_dim={self.music_dim})")

        # Frame buffer
        self.frame_buffer = FrameBuffer(target_buffer_size=16)
        
        # Stream joint recovery with smoothing
        self.smoothing_alpha = 0.5  # Default: medium smoothing
        self.stream_recovery = StreamMotionRecovery272(joints_num=22, smoothing_alpha=self.smoothing_alpha, enable_smpl=False)
        
        # Generation state
        self.current_text = ""
        self.is_generating = False
        self.generation_thread = None
        self.should_stop = False
        
        # Model generation state
        self.first_chunk = True
        self.history_length = 30  # Default history window length
        self.denoise_steps = 10  # Default denoising steps
        
        # ============ Audio Mode Support ============
        self.generation_mode = "text"  # "text" or "audio"
        self.audio_processor = None  # StreamAudioProcessor instance
        self.audio_window_size = 32  # Default audio window size
        self.current_motion_idx = 0  # Track motion frame index for audio alignment
        self.vae_scale_factor = 4  # VAE upsampling factor
        
        # ============ Performance Stats ============
        self.perf_lock = threading.Lock()
        self.perf_stats = {
            'prepare_time': 0,
            'model_inference_time': 0,
            'vae_decode_time': 0,
            'frame_process_time': 0,
            'post_time': 0,
            'audio_process_time': 0,
            'total_steps': 0
        }
        
        print("ModelManager initialized successfully")
    
    def _load_models(self, config_path):
        """Load VAE and diffusion models"""
        torch.set_float32_matmul_precision("high")
        
        # Change to parent directory to load config properly
        original_dir = os.getcwd()
        parent_dir = os.path.dirname(os.path.dirname(__file__))
        os.chdir(parent_dir)
        
        try:
            # Load config (same as generate_ldf.py)
            cfg = load_config(config_path=config_path)
            
            # Load VAE
            print("Loading VAE...")
            vae = instantiate(
                target=cfg.test_vae.target,
                cfg=None,
                hfstyle=False,
                **cfg.test_vae.params,
            )
            vae_ckpt = torch.load(cfg.test_vae_ckpt, map_location="cpu", weights_only=False)
            
            if "ema_state" in vae_ckpt:
                vae.load_state_dict(vae_ckpt["state_dict"], strict=True)
                vae_ema = ExponentialMovingAverage(
                    vae.parameters(), decay=cfg.test_vae.ema_decay
                )
                vae_ema.load_state_dict(vae_ckpt["ema_state"])
                vae_ema.copy_to(vae.parameters())
                print(f"Loaded VAE with EMA")
            else:
                vae.load_state_dict(vae_ckpt["state_dict"], strict=True)
                print(f"Loaded VAE without EMA")
            
            vae.to(self.device)
            vae.eval()
            
            # Load diffusion model
            print("Loading diffusion model...")
            model = instantiate(
                target=cfg.model.target, cfg=None, hfstyle=False, **cfg.model.params
            )
            checkpoint = torch.load(cfg.test_ckpt, map_location="cpu", weights_only=False)

            state_dict = checkpoint["state_dict"]
            model.load_state_dict(state_dict, strict=True)

            if "ema_state" in checkpoint:
                ema = ExponentialMovingAverage(model.parameters(), decay=cfg.model.ema_decay)
                ema.load_state_dict(checkpoint["ema_state"])
                ema.copy_to(model.parameters())
                print(f"Loaded model with EMA")
            else:
                print(f"Loaded model without EMA")
            
            model.to(self.device)
            model.eval()

            return vae, model, cfg

        finally:
            # Restore original directory
            os.chdir(original_dir)
    
    def start_generation(self, text, history_length=None):
        """Start or update generation with new text (TEXT MODE)"""
        self.generation_mode = "text"
        self.current_text = text
        
        if history_length is not None:
            self.history_length = history_length
        
        if not self.is_generating:
            # Reset state before starting (only once at the beginning)
            self.frame_buffer.clear()
            self.stream_recovery.reset()
            self.vae.clear_cache()
            self.first_chunk = True
            self.current_motion_idx = 0
            self.model.init_generated(self.history_length, batch_size=1, num_denoise_steps=self.denoise_steps)
            print(f"[TEXT MODE] Model initialized with history length: {self.history_length}, denoise steps: {self.denoise_steps}")
            
            # Start generation thread
            self.should_stop = False
            self.generation_thread = threading.Thread(target=self._generation_loop)
            self.generation_thread.daemon = True
            self.generation_thread.start()
            self.is_generating = True
    
    # ============ Audio Mode Methods ============
    def start_audio_generation(self, history_length=None, audio_window_size=32, input_sample_rate=44100):
        """Start generation with audio (music) conditioning (AUDIO MODE)"""
        self.generation_mode = "audio"
        self.audio_window_size = audio_window_size
        
        if history_length is not None:
            self.history_length = history_length
        
        if not self.is_generating:
            # Reset state before starting
            self.frame_buffer.clear()
            self.stream_recovery.reset()
            self.vae.clear_cache()
            self.first_chunk = True
            self.current_motion_idx = 0
            
            # Initialize audio processor based on music_dim
            if self.use_vqpae:
                self.audio_window_size = VQPAE_WINDOW_SIZE  # 180
                self.audio_processor = StreamVQPAEProcessor(
                    vqpae_model=self.vqpae_model,
                    device=self.device,
                    window_size=self.audio_window_size,
                    sample_rate=input_sample_rate
                )
                silence_sr = VQPAE_SAMPLE_RATE
            else:
                self.audio_processor = StreamAudioProcessor(
                    window_size=self.audio_window_size,
                    sample_rate=input_sample_rate
                )
                silence_sr = AUDIO_SAMPLE_RATE

            # NOTE: Pre-fill with silence to enable immediate generation start
            # This creates initial low-quality features but eliminates startup delay
            silence_duration = 1.0  # 1 second of silence
            silence_samples = np.zeros(int(silence_sr * silence_duration), dtype=np.float32)
            self.audio_processor.add_audio_samples(silence_samples, silence_sr)
            print(f"[AUDIO MODE] Pre-filled audio buffer with {silence_duration}s silence for immediate start")
            
            self.model.init_generated(self.history_length, batch_size=1, num_denoise_steps=self.denoise_steps)
            print(f"[AUDIO MODE] Model initialized with history length: {self.history_length}, "
                  f"denoise steps: {self.denoise_steps}, audio window: {audio_window_size}")
            
            # Start generation thread
            self.should_stop = False
            self.generation_thread = threading.Thread(target=self._generation_loop)
            self.generation_thread.daemon = True
            self.generation_thread.start()
            self.is_generating = True
    
    def add_audio_data(self, audio_samples, sample_rate=None):
        """Add audio data to the processor (AUDIO MODE)"""
        if self.audio_processor is not None:
            self.audio_processor.add_audio_samples(audio_samples, sample_rate)
    
    def get_audio_status(self):
        """Get audio processor status"""
        if self.audio_processor is not None:
            return self.audio_processor.get_status()
        return {"audio_buffer_samples": 0, "audio_buffer_seconds": 0, "feature_frames": 0, "frame_count": 0}
    
    def update_text(self, text):
        """Update text without resetting state (continuous generation with new text)"""
        if text != self.current_text:
            old_text = self.current_text
            self.current_text = text
            # Don't reset first_chunk, stream_recovery, or vae cache
            # This allows continuous generation with text changes
            print(f"Text updated: '{old_text}' -> '{text}' (continuous generation)")
    
    def pause_generation(self):
        """Pause generation (keeps all state)"""
        self.should_stop = True
        if self.generation_thread:
            self.generation_thread.join(timeout=2.0)
        self.is_generating = False
        print("Generation paused (state preserved)")
    
    def resume_generation(self):
        """Resume generation from paused state"""
        if self.is_generating:
            print("Already generating, ignoring resume")
            return
        
        # Restart generation thread with existing state
        self.should_stop = False
        self.generation_thread = threading.Thread(target=self._generation_loop)
        self.generation_thread.daemon = True
        self.generation_thread.start()
        self.is_generating = True
        print("Generation resumed")
    
    def reset(self, history_length=None, smoothing_alpha=None, denoise_steps=None):
        """Reset generation state completely
        
        Args:
            history_length: History window length for the model
            smoothing_alpha: EMA smoothing factor (0.0 to 1.0)
                - 1.0 = no smoothing (default)
                - 0.0 = infinite smoothing
                - Recommended: 0.3-0.7 for visible smoothing
            denoise_steps: Number of denoising steps (1-50, default 10)
        """
        # Stop if running
        if self.is_generating:
            self.pause_generation()
        
        # Clear everything
        self.frame_buffer.clear()
        self.vae.clear_cache()
        self.first_chunk = True
        self.current_motion_idx = 0
        
        # Reset audio processor if exists
        if self.audio_processor is not None:
            self.audio_processor.reset()
        
        if history_length is not None:
            self.history_length = history_length
        
        if denoise_steps is not None:
            # Ensure denoise_steps is multiple of chunk_size (5)
            chunk_size = 5
            denoise_steps = np.clip(denoise_steps, chunk_size, 50)
            # Round to nearest multiple of chunk_size
            self.denoise_steps = int(np.round(denoise_steps / chunk_size) * chunk_size)
            print(f"Denoising steps updated to: {self.denoise_steps} (must be multiple of {chunk_size})")
        
        # Update smoothing alpha if provided and recreate stream recovery
        if smoothing_alpha is not None:
            self.smoothing_alpha = np.clip(smoothing_alpha, 0.0, 1.0)
            print(f"Smoothing alpha updated to: {self.smoothing_alpha}")
        
        # Recreate stream recovery with new smoothing alpha
        self.stream_recovery = StreamMotionRecovery272(
            joints_num=22,
            smoothing_alpha=self.smoothing_alpha,
            enable_smpl=False
        )
        
        # Initialize model with denoise steps
        self.model.init_generated(self.history_length, batch_size=1, num_denoise_steps=self.denoise_steps)
        print(f"Model reset - history: {self.history_length}, smoothing: {self.smoothing_alpha}, steps: {self.denoise_steps}")
    
    def _generation_loop(self):
        """Main generation loop that runs in background thread
        Supports both TEXT and AUDIO modes
        """
        import sys
        print(f"\n🚀 [DEBUG] Generation loop started! Mode: {self.generation_mode}", file=sys.stderr, flush=True)
        
        import time
        step_count = 0
        first_feature_logged = False  # Track when first audio feature arrives
        
        with torch.no_grad():
            while not self.should_stop:
                # Check if buffer needs more frames
                if self.frame_buffer.needs_generation():
                    try:
                        loop_start = time.time()
                        
                        # 1. Prepare condition
                        t0 = time.time()
                        if self.generation_mode == "audio":
                            # AUDIO MODE: Get music features
                            if self.audio_processor is None:
                                print(f"[AUDIO MODE] Audio processor not initialized, waiting...", file=sys.stderr, flush=True)
                                time.sleep(0.02)
                                continue
                            
                            if not self.audio_processor.has_features():
                                # No audio data received from frontend yet
                                time.sleep(0.02)
                                continue
                            
                            # Log first feature arrival
                            if not first_feature_logged:
                                feature_count = self.audio_processor.get_feature_count()
                                print(f"✅ [AUDIO MODE] First audio features available! Count: {feature_count}", file=sys.stderr, flush=True)
                                first_feature_logged = True
                            
                            # Get latest available feature (real-time mode)
                            audio_feat = self.audio_processor.get_latest_features()
                            
                            if audio_feat is None:
                                time.sleep(0.01)
                                print(f"No audio features yet, waiting...", file=sys.stderr, flush=True)
                                continue
                            
                            x = {"music": [[audio_feat]]}
                            
                        else:
                            # TEXT MODE
                            x = {"text": [self.current_text]}
                        t_prepare = time.time() - t0
                        
                        # ============ Generate from model (1 token) ============
                        t_model_start = time.time()
                        output = self.model.stream_generate_step(
                            x, first_chunk=self.first_chunk
                        )
                        generated = output["generated"]
                        t_model = time.time() - t_model_start
                        
                        # ============ Decode with VAE (1 token -> 4 frames) ============
                        t_vae_start = time.time()
                        decoded = self.vae.stream_decode(
                            generated[0][None, :], first_chunk=self.first_chunk
                        )[0]
                        t_vae = time.time() - t_vae_start
                        
                        # ============ Convert each frame to joints ============
                        t_frame_start = time.time()
                        for i in range(decoded.shape[0]):
                            frame_data = decoded[i].cpu().numpy()
                            joints = self.stream_recovery.process_frame(frame_data)
                            self.frame_buffer.add_frame(joints)
                        frame_time = time.time() - t_frame_start
                        
                        # Post-processing
                        t_post_start = time.time()
                        self.first_chunk = False
                        self.current_motion_idx += self.vae_scale_factor
                        t_post = time.time() - t_post_start
                        
                        # Update stats
                        with self.perf_lock:
                            self.perf_stats['prepare_time'] = t_prepare
                            self.perf_stats['model_inference_time'] = t_model
                            self.perf_stats['vae_decode_time'] = t_vae
                            self.perf_stats['frame_process_time'] = frame_time
                            self.perf_stats['post_time'] = t_post
                            self.perf_stats['total_steps'] += 1
                        
                        loop_time = time.time() - loop_start
                        step_count += 1
                        
                        # Print performance stats EVERY loop
                        fps = (decoded.shape[0] / loop_time) if loop_time > 0 else 0
                        mode_info = f"mode={self.generation_mode}"
                        if self.generation_mode == "audio":
                            mode_info += f" audio_idx={self.current_motion_idx}"
                        
                        msg = (
                            f"[PERF] Step:{step_count:03d} {mode_info} | "
                            f"Total:{loop_time*1000:6.1f}ms (P:{t_prepare*1000:4.1f} M:{t_model*1000:5.1f} V:{t_vae*1000:4.1f} Post:{t_post*1000:4.1f}) | "
                            f"FPS:{fps:4.1f} Buf:{self.frame_buffer.size()}"
                        )
                        print(msg, file=sys.stderr, flush=True)
                        
                    except Exception as e:
                        print(f"❌ [DEBUG] Error in generation loop: {e}", file=sys.stderr, flush=True)
                        import traceback
                        traceback.print_exc()
                        time.sleep(0.1)
                else:
                    # Buffer is full, wait a bit
                    time.sleep(0.01)
        
        print("Generation loop stopped", file=sys.stderr, flush=True)
        
    
    def get_next_frame(self):
        """Get the next frame from buffer"""
        return self.frame_buffer.get_frame()
    
    def get_buffer_status(self):
        """Get buffer status"""
        # Get performance stats
        with self.perf_lock:
            perf_stats = self.perf_stats.copy()
        
        # Add audio processor stats if in audio mode
        if self.audio_processor:
            audio_status = self.audio_processor.get_status()
            audio_stats = audio_status.get('stats', {})
            perf_stats.update({
                'audio_resample_time': audio_stats.get('resample_time', 0.0),
                'audio_feature_time': audio_stats.get('feature_extract_time', 0.0)
            })
        
        status = {
            "buffer_size": self.frame_buffer.size(),
            "target_size": self.frame_buffer.target_size,
            "is_generating": self.is_generating,
            "current_text": self.current_text,
            "smoothing_alpha": self.smoothing_alpha,
            "denoise_steps": self.denoise_steps,
            "generation_mode": self.generation_mode,
            "perf_stats": perf_stats,
        }
        
        # Add audio status if in audio mode
        if self.generation_mode == "audio" and self.audio_processor is not None:
            audio_status = self.audio_processor.get_status()
            status.update({
                "audio_feature_frames": audio_status["feature_frames"],
                "audio_buffer_seconds": audio_status["audio_buffer_seconds"],
                "current_motion_idx": self.current_motion_idx,
            })
        
        return status


# Global model manager instance
_model_manager = None


def get_model_manager(config_path=None):
    """Get or create the global model manager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager(config_path)
    return _model_manager

