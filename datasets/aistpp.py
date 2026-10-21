import os
import random
from typing import Dict, List

import numpy as np
import torch
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from tqdm import tqdm

class LengthMismatchError(Exception):
    pass


class AistPPDataset(Dataset):
    def __init__(self, cfg, split="train"):
        self.cfg = cfg
        self.split = split
        self.stream_mode = cfg.data.get("stream_mode", False)
        if self.split == "train":
            self.file_list = cfg.data.train_meta_paths
            self.min_length = cfg.data.min_length
            self.max_length = cfg.data.max_length
            self.window_length = cfg.data.get("window_length", cfg.data.max_length)
            self.random_length = cfg.data.get("random_length", 0)
        elif self.split == "val":
            self.file_list = cfg.data.val_meta_paths
            self.min_length = cfg.data.min_length
            self.max_length = cfg.data.max_length
            self.window_length = cfg.data.max_length
            self.random_length = 0
        elif self.split == "test":
            self.file_list = cfg.data.test_meta_paths
            self.min_length = 0
            self.max_length = cfg.data.max_length
            self.window_length = cfg.data.max_length
            self.random_length = 0
        self.feature_path = cfg.data.get("feature_path", None)
        self.token_path = cfg.data.get("token_path", None)
        self.music_path = cfg.data.get("music_path", None)
        self.is_vae = cfg.get("is_vae", False) 

        self.dataset = self._load_file_list()


        rank_zero_info(f"Loaded {len(self.dataset)} samples from {split} dataset.")

    def _load_file_list(self) -> List[str]:
        """Load a list of npy file paths from a text file."""
        dataset = []
        ignored_cnt = 0
        for path in self.file_list:
            if os.path.exists(path):
                data_path = os.path.dirname(path)
                dataset_name = os.path.basename(data_path)
                rank_zero_info(f"Loading {path} ...")
                with open(path, "r") as f:
                    for name in f:
                        name = name.strip()
                        if name:
                            data = {}
                            try:
                                data["name"] = name
                                data["dataset"] = dataset_name
                                ##############################
                                # feature
                                ##############################
                                if self.feature_path is not None:
                                    # If feature_path is absolute, use it directly
                                    if os.path.isabs(self.feature_path):
                                        feature_dir = self.feature_path
                                    else:
                                        # Otherwise treat it relative to the dataset list file location
                                        feature_dir = os.path.join(data_path, self.feature_path)
                                    feature_path = os.path.join(feature_dir, name + ".npy")
                                    
                                    if not os.path.exists(feature_path):
                                        # rank_zero_info(f"Feature file not found: {feature_path}")
                                        continue
                                        
                                    feature = self.load_feature(feature_path)
                                    data["feature"] = feature
                                    data["feature_length"] = feature.shape[0]

                                    
                                ##############################
                                # token
                                ##############################
                                if self.token_path is not None and not self.is_vae:  ##### add this line to exclude music&token loading when training vae, 
                                    if os.path.isabs(self.token_path):               ##### otherwise it raises bugs
                                        token_dir = self.token_path
                                    else:
                                        token_dir = os.path.join(data_path, self.token_path)
                                    
                                    token_path = os.path.join(token_dir, name + ".npy")
                                    
                                    if not os.path.exists(token_path):
                                        continue
                                        
                                    token = self.load_token(token_path)
                                    data["token"] = token
                                    data["token_length"] = token.shape[0]
                                    

                                    
                                ##############################
                                # music
                                ##############################
                                if self.music_path is not None and not self.is_vae: ##### add this line to exclude music&token loading when training vae, 
                                                                                    ##### otherwise it raises bugs
                                    if os.path.isabs(self.music_path):
                                        music_dir = self.music_path
                                    else:
                                        music_dir = os.path.join(data_path, self.music_path)
                                    
                                    music_path = os.path.join(music_dir, name + ".npy")

                                    if not os.path.exists(music_path):
                                        continue

                                    music = self.load_music(music_path)
                                    data["music"] = music
                                    data["music_length"] = music.shape[0]
                                    '''some music and motion are within +- 1 frame differenct in length '''

                                
                                dataset.append(data)
                            except LengthMismatchError:
                                ignored_cnt += 1
                                pass
                            except Exception as e:
                                rank_zero_info(f"Error loading data for {name}: {e}")
                                pass
                            if self.cfg.debug and len(dataset) >= 100:
                                rank_zero_info(f"debug mode, break at {len(dataset)}")
                                break
        if ignored_cnt > 0:
            rank_zero_info(f"Ignored {ignored_cnt} samples due to length mismatch.")
        if len(dataset) == 0:
            rank_zero_info(
                f"No data found in {self.file_list}. Please check the file paths and ensure they are correct."
            )
        else:
            for i in range(3):
                tmp = random.choice(dataset)
                rank_zero_info(f"Random data {tmp['name']}: {tmp['feature'].shape}")
        return dataset

    def load_feature(self, feature_path: str) -> np.ndarray:
        feature = np.load(feature_path).astype(np.float32)
        if np.isnan(feature).any():
            raise ValueError("NaN values found in feature, skip it.")
        if feature.shape[0] < self.min_length or feature.shape[0] > self.max_length:
            raise LengthMismatchError("Feature length out of bounds, skip it.")
        return feature
    
    def load_token(self, token_path: str) -> np.ndarray:
        token = np.load(token_path)
        if np.isnan(token).any():
            raise ValueError("NaN values found in token, skip it.")
        return token
    
    def load_music(self, music_path: str) -> np.ndarray:
        music = np.load(music_path).astype(np.float32)
        if np.isnan(music).any():
            raise ValueError("NaN values found in music feature, skip it.")
        if music.shape[0] < self.min_length or music.shape[0] > self.max_length:
            raise LengthMismatchError("Music length out of bounds, skip it.")
        return music

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        return self._process(data)

    def _process(self, data):
        output = {}
        output["dataset"] = data["dataset"]
        output["name"] = data["name"]
        ##############################
        # feature(motion)
        ##############################
        if "feature" in data:
            feature, feature_length = self.process_feature(data["feature"])
            output["feature"] = feature
            output["feature_length"] = feature_length
        ##############################
        # token
        ##############################
        if "token" in data:
            token, token_length = self.process_token(data["token"])
            output["token"] = token
            output["token_length"] = token_length
        ##############################
        # music
        ##############################
        if "music" in data:
            music, music_length = self.process_music(data["music"])
            if self.stream_mode:
                # Downsample music to align with VAE tokens
                # VAE encoding logic: the first frame is used alone, subsequent frames are grouped every 4 frames
                if "token" in output:
                    # Get the number of tokens:
                    num_tokens = output["token_length"]

                    # Downsample music accordingly
                    music_tokens = []
                    # The first frame maps to the first token
                    if music_length > 0:
                        music_tokens.append(music[0])

                    # For subsequent tokens, take the first frame of every 4-frame group as representative
                    for i in range(1, num_tokens):
                        frame_idx = 1 + (i - 1) * 4  # 1, 5, 9, 13, ...
                        if frame_idx < music_length:
                            music_tokens.append(music[frame_idx])

                    # Stack to (T, W, D) or (T, D) once per-sample to avoid Python work in collate_fn
                    if len(music_tokens) > 0:
                        output["music"] = np.stack(music_tokens, axis=0).astype(
                            np.float32, copy=False
                        )
                        output["music_length"] = output["music"].shape[0]
                    else:
                        output["music"] = np.zeros((0,), dtype=np.float32)
                        output["music_length"] = 0
                else:
                    # Keep all frames, stack to ndarray for faster collate
                    if music_length > 0:
                        output["music"] = np.asarray(music, dtype=np.float32)
                        output["music_length"] = output["music"].shape[0]
                    else:
                        output["music"] = np.zeros((0,), dtype=np.float32)
                        output["music_length"] = 0
            else:
                output["music"] = music
                output["music_length"] = music_length

        # Optional (data.train_max_tokens): train on random contiguous token windows with the aligned music,
        # like the 300-frame FineDance slices; long AIST++ sequences otherwise pad every batch to ~360 tokens.
        max_tokens = self.cfg.data.get("train_max_tokens", None)
        if self.split == "train" and max_tokens and output.get("token_length", 0) > max_tokens:
            start = random.randint(0, output["token_length"] - max_tokens)
            output["token"] = output["token"][start : start + max_tokens]
            output["token_length"] = max_tokens
            if self.stream_mode and isinstance(output.get("music"), np.ndarray) and output["music"].ndim >= 2:
                output["music"] = output["music"][start : start + max_tokens]
                output["music_length"] = output["music"].shape[0]

        return output

    def process_feature(self, feature):
        feature_len = feature.shape[0]
        # if the motion is longer than window_length, randomly crop a window_length clip
        if feature_len > self.window_length:
            start = random.randint(0, feature_len - self.window_length)
            feature = feature[start : start + self.window_length]
            feature_len = self.window_length
        return feature, feature_len
    
    def process_music(self, music: np.ndarray, target_len: int = None):
        music_len = music.shape[0]
        
        # # Only crop if longer than window_length (for training)
        # if target_len is None:
        #     target_len = self.window_length
        
        # if music_len > target_len:
        #     start = random.randint(0, music_len - target_len)
        #     music = music[start : start + target_len]
        #     music_len = target_len
        # # Remove padding logic - keep original length
    
        return music, music_len

    def process_token(self, token):
        token_length = len(token)
        if token_length > self.random_length:
            new_token_length = token_length - random.randint(0, self.random_length)
            start = random.randint(0, token_length - new_token_length)
            token = token[start : start + new_token_length]
        token_length = len(token)
        return token, token_length
    
    
    



def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    output = {}
    keys = batch[0].keys()

    for key in keys:
        if key in ["feature", "token"] and isinstance(batch[0][key], np.ndarray):
            # Pad sequences (T, D) -> (B, T_max, D)
            items = [torch.from_numpy(b[key]) for b in batch]
            output[key] = torch.nn.utils.rnn.pad_sequence(
                items, batch_first=True, padding_value=0
            )
        elif key == "music":
            # Two possible representations:
            # - non-stream: np.ndarray (T, D) -> pad to (B, T_max, D)
            # - stream: list of np.ndarray, each (W, D) (or (D,)) -> pad to (B, T_max, W, D)
            m0 = batch[0].get("music", None)
            if isinstance(m0, np.ndarray):
                items = [torch.from_numpy(b["music"]) for b in batch]
                output["music"] = torch.nn.utils.rnn.pad_sequence(
                    items, batch_first=True, padding_value=0
                )
            elif isinstance(m0, list):
                # Determine max sequence length in this batch.
                lengths = [len(b["music"]) for b in batch]
                t_max = max(lengths) if len(lengths) > 0 else 0

                # Find a sample element to infer (W, D).
                sample = None
                for b in batch:
                    if len(b["music"]) > 0:
                        sample = b["music"][0]
                        break
                if sample is None:
                    # All empty: fall back to empty tensor.
                    output["music"] = torch.zeros((len(batch), 0, 0, 0), dtype=torch.float32)
                else:
                    sample = np.asarray(sample)
                    if sample.ndim == 1:
                        w, d = 1, sample.shape[0]
                    else:
                        w, d = sample.shape[0], sample.shape[1]

                    music_btwd = torch.zeros(
                        (len(batch), t_max, w, d), dtype=torch.float32
                    )
                    for i, b in enumerate(batch):
                        seq = b["music"]
                        for t, m in enumerate(seq[:t_max]):
                            m = np.asarray(m)
                            if m.ndim == 1:
                                m = m[None, :]
                            music_btwd[i, t, : m.shape[0], : m.shape[1]] = torch.from_numpy(
                                m.astype(np.float32, copy=False)
                            )
                    output["music"] = music_btwd
            else:
                # Unknown type: keep as list
                output["music"] = [b["music"] for b in batch]
        elif key in ["feature_length", "token_length","music_length"]:
            # Stack scalars
            output[key] = torch.tensor([b[key] for b in batch])
        else:
            # Default to list
            output[key] = [b[key] for b in batch]
    return output