"""
Real-time audio capture and VQPAE feature extraction
Captures system audio and processes it with VQPAE in real-time at 30 FPS

FILE STRUCTURE OVERVIEW:
========================

1. Module Import Strategy (Lines 5-18)
   - Explains the critical module caching strategy to avoid conflicts
   - PAE paths are NOT added at module level to prevent interference

2. Constants (Lines 37-43)
   - FPS: 30 (target motion frame rate)
   - SAMPLE_RATE: 16kHz (VQ-PAE requirement)
   - WINDOW_LENGTH: 6 seconds (VQPAE context window)
   - OUTPUT_WINDOW_SIZE: 180 frames (6s × 30fps)

3. load_vqpae_model() (Lines 46-125)
   - Loads VQ-PAE model with module cache management
   - Key technique: backs up and restores sys.modules['models'] to avoid conflicts
   - Creates model config and loads checkpoint weights

4. process_vqpae_window() (Lines 128-145)
   - Processes single 6-second audio window through VQPAE
   - Input: (96000,) audio samples
   - Output: (180, 16) features (180 frames, 16 channels: 8 VQ + 8 PAE)

5. RealtimeVQPAE Class (Lines 148-end)
   - Main class for real-time audio capture and processing
   
   Components:
   a) __init__: Initializes model and buffers
   b) audio_callback: Captures audio chunks (runs in audio thread)
   c) processing_thread: Processes audio → VQPAE features (runs in separate thread)
      - Implements zero-padding strategy for cold start
      - Maintains 6-second sliding window
      - Outputs features at 30 FPS
   d) record: Main entry point for recording
   e) stop: Gracefully stops recording and saves results

THREADING MODEL:
================
- Main thread: Controls execution
- Audio capture thread: sounddevice callback, captures audio chunks
- Processing thread: Reads from queue, runs VQPAE inference

DATA FLOW:
==========
System Audio → audio_callback → audio_queue → processing_thread → VQPAE model → features_list

ZERO-PADDING STRATEGY :
================================================
- First 6 seconds: Gradually fills buffer with real audio, pads zeros at beginning
- After 6 seconds: Uses full 6-second sliding window of real audio
- This enables immediate output with no delay (but quality improves over first 6s)

Usage:
    python utils/realtime_vqpae.py --checkpoint <path_to_vqpae_ckpt> --duration 30 --output output.npy
"""

import os
import sys
import time
import queue
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
from tqdm import tqdm

## DO NOT add PAE paths here - they will be added inside load_vqpae_model() to avoid conflicts

# Constants
FPS = 30  ## Motion frame rate
SAMPLE_RATE = 16000  ## VQ-PAE target sample rate
WINDOW_LENGTH = 6.0  ## seconds - VQPAE training window
TIME_RANGE = 96000  ## 6 seconds @ 16kHz
EMBEDDING_CHANNELS = 8  ## VQ-PAE embedding channels
OUTPUT_WINDOW_SIZE = 180  ## frames per window (6s * 30fps)
FRAME_STEP_SAMPLES = SAMPLE_RATE // FPS  ## 533 samples per frame @ 30fps


def load_vqpae_model(checkpoint_path, device='cuda'):
    """Load pretrained VQ-PAE model using same strategy as VQPAE_preprocess.py"""
    import sys
    
    # Add BOTH PAE paths (parent for 'src' imports, src for 'models' imports)
    pae_parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../PAE_training/DeepPhase4Audio-main'))
    pae_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../PAE_training/DeepPhase4Audio-main/src'))
    
    if pae_parent_path not in sys.path:
        sys.path.insert(0, pae_parent_path)
    if pae_src_path not in sys.path:
        sys.path.insert(0, pae_src_path)
    
    # Backup and clear 'models' module cache to avoid conflict with project's models/ directory
    models_backup = sys.modules.pop('models', None)
    
    try:
        # Import VQ_PAE from PAE's models package
        from models.VQ_PAE import VQ_AE
    finally:
        # Restore original project's models module
        if models_backup is not None:
            sys.modules['models'] = models_backup
    
    class Config:
        input_channels = 1
        embedding_channels = EMBEDDING_CHANNELS
        time_range = TIME_RANGE
        window = WINDOW_LENGTH
        use_fft_mlp = False
        vq_only = False
        intermediate_channels = 64
        enc_dilation_rates = [9, 3, 1]
        dec_dilation_rates = [1, 3, 9]
        enc_kernel_sizes = [7, 3, 3]
        dec_kernel_sizes = [3, 3, 7]
        activation = 'elu'
        combiner = 'concat'
        
        class vq:
            @staticmethod
            def to_dict():
                return {
                    'input_dim': EMBEDDING_CHANNELS,
                    'n_codebooks': 9,
                    'codebook_size': 1024,
                    'codebook_dim': 8,
                    'quantizer_dropout': 0.5,
                }
    
    cfg = Config()
    model = VQ_AE(cfg)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    model = model.to(device)
    return model


def process_vqpae_window(audio_window, model, device='cuda'):
    """Process a single 6-second audio window through VQPAE
    
    Args:
        audio_window: numpy array of shape (96000,)
        model: VQ-PAE model
        device: device to run on
    
    Returns:
        features: numpy array of shape (180, 16)
    """
    with torch.no_grad():
        audio_tensor = torch.from_numpy(audio_window).float().unsqueeze(0).to(device)
        output = model(audio_tensor)
        
        z_pae = output['z_pae']  ## [1, 8, 96000]
        z_downsampled = torch.nn.functional.adaptive_avg_pool1d(z_pae, output_size=OUTPUT_WINDOW_SIZE)  ## [1, 8, 180]
        
        z_vq_full = output['z_vq']  ## [1, 8, 96000]
        z_vq = torch.nn.functional.adaptive_avg_pool1d(z_vq_full, output_size=OUTPUT_WINDOW_SIZE)  ## [1, 8, 180]
        
        z_concat = torch.cat([z_vq, z_downsampled], dim=1)  ## [1, 16, 180]
        z_concat = z_concat.squeeze(0).permute(1, 0).cpu().numpy()  ## [180, 16]
        
    return z_concat


class RealtimeVQPAE:
    """Real-time audio capture and VQPAE processing"""
    
    def __init__(self, checkpoint_path, device='cuda', verbose=True):
        self.device = device
        self.verbose = verbose
        
        # Load model
        if self.verbose:
            print(f"Loading VQ-PAE model from {checkpoint_path}")
        self.model = load_vqpae_model(checkpoint_path, device=device)
        if self.verbose:
            print(f"Model loaded successfully on {device}")
        
        # Audio buffers
        self.audio_queue = queue.Queue()  ## Thread-safe queue for audio chunks
        self.audio_history = []  ## Accumulated audio samples
        self.features_list = []  ## Collected features
        
        # State variables
        self.recording = False
        self.processing = False
        self.frame_counter = 0
        self.start_time = None
        
        # Performance tracking
        self.process_times = []
        
    def audio_callback(self, indata, frames, time_info, status):
        """Callback for sounddevice audio stream (runs in separate thread)"""
        if status and self.verbose:
            print(f"Audio callback status: {status}")
        
        # Convert to mono and add to queue
        audio_chunk = indata[:, 0].copy()  ## Take first channel
        self.audio_queue.put(audio_chunk)
    
    def processing_thread(self):
        """Processing thread: consumes audio from queue and runs VQPAE"""
        samples_since_last_frame = 0
        
        if self.verbose:
            print(f"Processing thread started")
            print(f"Target: 30 FPS (frame every {1000/FPS:.1f}ms, {FRAME_STEP_SAMPLES} samples)")
        
        while self.processing or not self.audio_queue.empty():
            try:
                # Get audio chunk from queue (timeout to allow checking self.processing)
                audio_chunk = self.audio_queue.get(timeout=0.1)
                self.audio_history.extend(audio_chunk)
                samples_since_last_frame += len(audio_chunk)
                
                # Check if we should process a frame
                if samples_since_last_frame >= FRAME_STEP_SAMPLES:
                    frame_start_time = time.time()
                    
                    # Construct 6-second window with zero-padding if needed
                    current_samples = len(self.audio_history)
                    
                    if current_samples < TIME_RANGE:
                        # Zero-pad at the beginning (cold start phase)
                        pad_size = TIME_RANGE - current_samples
                        audio_window = np.concatenate([
                            np.zeros(pad_size, dtype=np.float32),
                            np.array(self.audio_history, dtype=np.float32)
                        ])
                    else:
                        # Use last 6 seconds
                        audio_window = np.array(self.audio_history[-TIME_RANGE:], dtype=np.float32)
                    
                    # Process through VQPAE
                    features = process_vqpae_window(audio_window, self.model, self.device)
                    self.features_list.append(features)
                    
                    # Update counters
                    self.frame_counter += 1
                    samples_since_last_frame -= FRAME_STEP_SAMPLES
                    
                    # Track performance
                    process_time = (time.time() - frame_start_time) * 1000  ## ms
                    self.process_times.append(process_time)
                    
                    if self.verbose and self.frame_counter % 30 == 0:  ## Print every second
                        elapsed = time.time() - self.start_time
                        avg_process_time = np.mean(self.process_times[-30:])
                        print(f"Frame {self.frame_counter:4d} | "
                              f"Elapsed: {elapsed:.1f}s | "
                              f"Process time: {avg_process_time:.1f}ms | "
                              f"Buffer: {current_samples} samples | "
                              f"Features shape: {features.shape}")
                
            except queue.Empty:
                continue
        
        if self.verbose:
            print(f"Processing thread finished. Total frames: {self.frame_counter}")
    
    def record(self, duration=None, output_path=None):
        """
        Start real-time recording and processing
        
        Args:
            duration: Recording duration in seconds (None for infinite)
            output_path: Path to save features (None for no save)
        
        Returns:
            features: numpy array of shape (n_frames, 180, 16)
        """
        # Reset state
        self.audio_history = []
        self.features_list = []
        self.frame_counter = 0
        self.process_times = []
        
        # List available audio devices
        if self.verbose:
            print("\n=== Available Audio Devices ===")
            print(sd.query_devices())
            print("=" * 50)
            print(f"\nUsing default input device")
        
        # Start processing thread
        self.processing = True
        process_thread = threading.Thread(target=self.processing_thread, daemon=True)
        process_thread.start()
        
        # Start recording
        self.recording = True
        self.start_time = time.time()
        
        if self.verbose:
            print(f"\n=== Starting Recording ===")
            print(f"Duration: {duration}s" if duration else "Duration: Infinite (Ctrl+C to stop)")
            print(f"Sample rate: {SAMPLE_RATE} Hz")
            print(f"Target FPS: {FPS}")
            print("=" * 50 + "\n")
        
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                callback=self.audio_callback,
                blocksize=int(SAMPLE_RATE * 0.1)  ## 100ms blocks
            ):
                if duration:
                    time.sleep(duration)
                else:
                    # Infinite recording until Ctrl+C
                    while True:
                        time.sleep(1)
        
        except KeyboardInterrupt:
            if self.verbose:
                print("\n\nRecording interrupted by user (Ctrl+C)")
        
        finally:
            # Stop recording and processing
            self.recording = False
            if self.verbose:
                print("Stopping recording...")
            
            # Wait for processing thread to finish
            self.processing = False
            process_thread.join(timeout=5)
            
            if self.verbose:
                print("Processing thread stopped")
        
        # Collect results
        features = np.stack(self.features_list, axis=0) if self.features_list else np.array([])
        
        # Print statistics
        if self.verbose and len(self.process_times) > 0:
            print("\n=== Performance Statistics ===")
            print(f"Total frames: {self.frame_counter}")
            print(f"Total duration: {time.time() - self.start_time:.2f}s")
            print(f"Actual FPS: {self.frame_counter / (time.time() - self.start_time):.1f}")
            print(f"Avg process time: {np.mean(self.process_times):.2f}ms")
            print(f"Max process time: {np.max(self.process_times):.2f}ms")
            print(f"Features shape: {features.shape}")
            print("=" * 50)
        
        # Save if requested
        if output_path and len(features) > 0:
            np.save(output_path, features)
            if self.verbose:
                print(f"\nFeatures saved to: {output_path}")
        
        return features


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time audio capture and VQPAE processing")
    parser.add_argument("--checkpoint", required=True, help="Path to VQ-PAE checkpoint")
    parser.add_argument("--duration", type=float, default=None, 
                        help="Recording duration in seconds (default: infinite, stop with Ctrl+C)")
    parser.add_argument("--output", default="output_realtime.npy", 
                        help="Output file path for features")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], 
                        help="Device to run model on")
    parser.add_argument("--list-devices", action="store_true", 
                        help="List available audio devices and exit")
    
    args = parser.parse_args()
    
    # List devices and exit if requested
    if args.list_devices:
        print("=== Available Audio Devices ===")
        print(sd.query_devices())
        return
    
    # Create processor and start recording
    processor = RealtimeVQPAE(args.checkpoint, device=args.device, verbose=True)
    features = processor.record(duration=args.duration, output_path=args.output)
    
    print(f"\nDone! Processed {len(features)} frames")


if __name__ == "__main__":
    main()
