import os

import numpy as np
import torch
from lightning import seed_everything
from lightning.pytorch.utilities import rank_zero_info
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

from utils.initialize import instantiate, load_config

RAW_DATA_PATH = "raw_data"
PATH_FILES = [
    [
        f"{RAW_DATA_PATH}/finedance/train.txt",
        f"{RAW_DATA_PATH}/finedance/val.txt",
        f"{RAW_DATA_PATH}/finedance/test.txt",
    ],
]
FEATURE_PATH = [
    "motion_272_30fps"
]
TOKEN_PATH = [
    "TOKENS_vae_wan_1d_multi_20260121_180702_ckpt_130000",
]
RECOVERED_PATH = [
    "recovered_tokens_vae_wan_1d_multi_20260121_180702_ckpt_130000",
]


def main():
    # init
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config()
    seed_everything(cfg.seed)
    # vae
    vae_model = instantiate(
        cfg.model.target,
        cfg=None,
        hfstyle=False,
        **cfg.model.params,
    )
    vae_ckpt = torch.load(cfg.test_ckpt, map_location="cpu", weights_only=False)
    if "ema_state" in vae_ckpt:
        vae_model.load_state_dict(vae_ckpt["state_dict"], strict=True)
        vae_ema = ExponentialMovingAverage(
            vae_model.parameters(), decay=cfg.model.ema_decay
        )
        vae_ema.load_state_dict(vae_ckpt["ema_state"])
        vae_ema.copy_to(vae_model.parameters())
        rank_zero_info(f"Loaded VAE model from {cfg.test_ckpt} with EMA")
    else:
        vae_model.load_state_dict(vae_ckpt["state_dict"], strict=True)
        rank_zero_info(f"Loaded VAE model from {cfg.test_ckpt} w/o EMA")
    vae_model = vae_model.to(device)

    # an optional `pretokenize` section in the config (path_files / feature_path / token_path /
    # recovered_path) replaces the hard-coded lists above
    pt = cfg.get("pretokenize", None)
    jobs = zip(PATH_FILES, FEATURE_PATH, TOKEN_PATH, RECOVERED_PATH) if pt is None else [
        (list(pt.path_files), pt.feature_path, pt.token_path, pt.get("recovered_path", None))
    ]
    for path_files, feature_path, token_path, recovered_path in jobs:
        tokenize_motion(
            vae_model, path_files, device, cfg, feature_path, token_path, recovered_path
        )


@torch.no_grad()
def tokenize_motion(
    vae_model, path_files, device, cfg, feature_path_root, token_path_root, recovered_path_root=None
):
    for path_file in path_files:
        if os.path.exists(path_file):
            data_path = os.path.dirname(path_file)
            # Use local variables to avoid modifying parameters in a loop
            curr_feature_path = os.path.join(data_path, feature_path_root)
            curr_token_path = os.path.join(data_path, token_path_root)
            
            if not os.path.exists(curr_token_path):
                os.makedirs(curr_token_path, exist_ok=True)
            
            curr_recovered_path = None
            if recovered_path_root is not None:
                curr_recovered_path = os.path.join(data_path, recovered_path_root)
                if not os.path.exists(curr_recovered_path):
                    os.makedirs(curr_recovered_path, exist_ok=True)
            
            print(f"Processing {path_file}\n  Source: {curr_feature_path}\n  Target: {curr_token_path}")
            
            with open(path_file, "r") as f:
                names = [line.strip() for line in f if line.strip()]
                for name in tqdm(
                    names, desc=f"Processing {os.path.basename(path_file)}"
                ):
                    single_feature_path = os.path.join(curr_feature_path, f"{name}.npy")
                    single_token_path = os.path.join(curr_token_path, f"{name}.npy")
                    
                    if curr_recovered_path is not None:
                        single_recovered_path = os.path.join(curr_recovered_path, f"{name}.npy")
                    
                    if os.path.exists(single_feature_path):
                        try:
                            feature = np.load(single_feature_path, allow_pickle=True)
                            # Explicitly convert to float32 to match model weights
                            feature = torch.from_numpy(feature).to(device).float()[None, :, :]
                            token = vae_model.encode(feature)
                            recovered = vae_model.decode(token)
                            np.save(single_token_path, token.detach().cpu().numpy()[0])
                            if curr_recovered_path is not None:
                                np.save(
                                    single_recovered_path,
                                    recovered.detach().cpu().numpy()[0],
                                )
                        except Exception as e:
                            rank_zero_info(f"Error processing {name}: {e}")
                    else:
                        rank_zero_info(f"Feature file not found, skipping: {single_feature_path}")


if __name__ == "__main__":
    main()

# python pretokenize_vae.py --config configs/vae_wan_1d_multi_272_stand.yaml