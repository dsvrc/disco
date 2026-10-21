import bisect
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import OmegaConf
from utils.initialize import instantiate

class _EmptyDataset(Dataset):
    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise IndexError("Empty dataset")

class MultiDataset(Dataset):
    def __init__(self, cfg, split="train"):
        self.datasets = []
        self.cfg = cfg
        self.split = split

        if not hasattr(cfg.data, "datasets"):
             rank_zero_info("MultiDataset: cfg.data.datasets not found. No datasets initialized.")
             return

        rank_zero_info(f"Initializing MultiDataset for split {split} with {len(cfg.data.datasets)} sub-datasets...")

        for ds_conf in cfg.data.datasets:
            # Optional per-dataset split enabling flags
            # Default: enabled for all splits
            if split == "train" and ds_conf.get("enable_train", True) is False:
                continue
            if split == "val" and ds_conf.get("enable_val", True) is False:
                continue
            if split == "test" and ds_conf.get("enable_test", True) is False:
                continue

            # ds_conf contains the specific configuration for this dataset (e.g., target, paths)
            # Use OmegaConf.merge to create a clean, independent config for each sub-dataset.
            # We start with a copy of the full config and then overwrite the 'data' section 
            # with the specific sub-dataset parameters.
            ds_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            ds_cfg.data = ds_conf
            
            target = ds_cfg.data.get("target", None)
            
            if target is None:
                raise ValueError(f"No target specified for dataset in MultiDataset: {ds_conf}")
                
            rank_zero_info(f"  - Initializing sub-dataset: {target}")
            
            # Instantiate the dataset with the isolated config
            dataset = instantiate(target, cfg=ds_cfg, split=split)
            self.datasets.append(dataset)
            
        # Use ConcatDataset to handle indexing and length automatically
        # (ConcatDataset requires a non-empty list)
        if len(self.datasets) == 0:
            rank_zero_info(
                f"MultiDataset: no sub-datasets enabled for split '{split}'. Using empty dataset."
            )
            self.concat_dataset = _EmptyDataset()
        else:
            self.concat_dataset = ConcatDataset(self.datasets)
        rank_zero_info(f"MultiDataset loaded. Total samples: {len(self.concat_dataset)}")

    def __len__(self):
        return len(self.concat_dataset)

    def __getitem__(self, idx):
        return self.concat_dataset[idx]

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
