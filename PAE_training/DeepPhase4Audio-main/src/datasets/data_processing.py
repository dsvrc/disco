import os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import scipy.signal as signal
import soundfile as sf
try:
    import librosa
except ImportError:
    librosa = None


def preprocess_audio(file_path, lowpass_cutoff=8000, original_rate=44100, target_rate=16000, window_length=2):
    # Load audio based on file format - mp3 files MUST use librosa
    if file_path.endswith('.mp3'):
        if librosa is None:
            raise ImportError("librosa is required to load mp3 files. Install with: pip install librosa")
        audio, sr = librosa.load(file_path, sr=original_rate, mono=True)
    else:
        # For wav files, try soundfile first
        try:
            audio, sr = sf.read(file_path)
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)  # Convert to mono if stereo
        except Exception as e:
            # Fall back to librosa if soundfile fails
            if librosa is None:
                raise
            print(f"Warning: soundfile failed for {file_path}, using librosa instead: {e}")
            audio, sr = librosa.load(file_path, sr=original_rate, mono=True)
    
    if sr != original_rate:
        raise ValueError(f"Expected {original_rate} Hz, but got {sr} Hz. File: {file_path}")
    
    # low-pass filter
    nyquist = sr / 2
    norm_cutoff = lowpass_cutoff / nyquist
    b, a = signal.butter(5, norm_cutoff, btype='low')
    filtered_audio = signal.filtfilt(b, a, audio)
    
    # Downsample the audio to target rate
    downsampled_audio = signal.resample_poly(filtered_audio, target_rate, sr)
    
    # normalize to -1 to 1
    max_val = np.max(np.abs(downsampled_audio))
    if max_val > 0:
        downsampled_audio = downsampled_audio / max_val
    
    # make windows
    window_size = int(target_rate * window_length)
    step_size = window_size  ## no overlap for faster training

    windows = []
    for i in range(0, len(downsampled_audio) - window_size + 1, step_size):
        s = downsampled_audio[i:i + window_size]
        windows.append(s)
    
    return windows

class AudioDataset(Dataset):
    def __init__(self, dataset_root, split, lowpass_cutoff=8000, original_rate=44100, target_rate=16000, window_length=2):
        self.audio_dir = dataset_root
        self.split = split
        self.lowpass_cutoff = lowpass_cutoff
        self.original_rate = original_rate
        self.target_rate = target_rate
        self.window_length = window_length
        
        # load and preprocess audio files
        self.data = []
        # Recursively find all .wav and .mp3 files
        for root, dirs, files in os.walk(dataset_root):
            for file_name in sorted(files):
                if file_name.endswith((".wav", ".mp3")):
                    file_path = os.path.join(root, file_name)
                    try:
                        self.data.extend(preprocess_audio(file_path, self.lowpass_cutoff, self.original_rate, self.target_rate, self.window_length))
                    except Exception as e:
                        print(f"Warning: Failed to load {file_path}: {e}")
                        continue
        
        # Split into train, test, and validation
        if len(self.data) == 0:
            print(f"Warning: No audio data found in {dataset_root}")
        num_train = int(0.8 * len(self.data))
        num_valid = int(0.1 * len(self.data))
        num_test = len(self.data) - num_train - num_valid
        train_data, valid_data, test_data = random_split(self.data, [num_train, num_valid, num_test])
        self.data = {"train": train_data, "valid": valid_data, "test": test_data}[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)


if __name__ == "__main__":
    dataset_root = os.path.join(os.getcwd(), "data")
    train_dataset = AudioDataset(dataset_root, split='train')
    valid_dataset = AudioDataset(dataset_root, split='valid')
    test_dataset = AudioDataset(dataset_root, split='test')

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(valid_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Access a sample
    sample = train_dataset[0]
    print(f"Sample shape: {sample.shape}") # B, 2, 32000 if using pos
    print(sample)
    
    # print(sample.squeeze(0).norm)
