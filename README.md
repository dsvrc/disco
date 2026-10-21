# DiscoForcing

**A Unified Framework for Real-Time Audio-Driven Character Control with Diffusion Forcing.**

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://discoforcing.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2605.28491-b31b1b)](https://arxiv.org/pdf/2605.28491)

DiscoForcing is a streaming audio-driven diffusion framework for real-time music-to-dance motion generation. It couples a causal music encoder (VQ-PAE) that captures rhythmic structure and phase dynamics with a diffusion-forcing sequence model trained under heterogeneous noise levels across the temporal horizon, enabling strictly causal, bounded-latency streaming generation of coherent full-body dance motion at interactive frame rates.

### Highlights

- **Real-time streaming** – strictly causal, bounded-latency generation suitable for interactive applications.
- **Causal music conditioning** – decomposed music encoding via VQ-PAE that extracts discrete rhythmic tokens and continuous phase-alignment features from a sliding audio window.
- **Temporal guided sampling** – hybrid temporal schedule that balances responsiveness to abrupt music changes against long-horizon consistency.
- **End-to-end interactive system** – supports both online avatar playback and physics-based humanoid deployment via ROS2.
- **Multiple datasets** – trained and evaluated on AIST++ and FineDance.

---

## Installation

### Environment Setup

```bash
# Create conda environment
conda create -n music2dance python=3.10
conda activate music2dance

# Install dependencies
pip install -r requirements.txt

# Install Flash Attention
# visit https://blog.csdn.net/dongbidsaxue/article/details/146322800 to download wheels for flash attention, then run
pip install your-flash-attention-wheel-version.whl

# for rendering
sudo apt update && sudo apt install -y ffmpeg
```

## Dataset Preparation

### Download datasets

```bash
bash scripts/download_aistpp.sh

bash scripts/download_finedance.sh
```

### Dataset preprocessing
> **NOTE**: download SMPL and put it under `deps/smpl/SMPL_NEUTRAL.npz` first.

For detailed explanation of different motion representations, see `utils/dataset_process.py` (annotations in function `mean_variance`).

```bash
# AIST++, 272-dim, 30fps
python utils/dataset_process.py --dim 272 --dataset aistpp --make_animation --tgt_fps 30

# FineDance, 272-dim, 30fps
python utils/dataset_process.py --dim 272 --dataset finedance --make_animation --tgt_fps 30
```

### Music feature extraction

```bash
# Windowed librosa features (T, window_size, 35)
python utils/librosa_preprocess.py --src raw_data/aistpp/wavs --dest raw_data/aistpp/baseline_feats_window_64 --window_size 64

python utils/librosa_preprocess.py --src raw_data/finedance/music_wav --dest raw_data/finedance/baseline_feats_window_64 --window_size 64

# VQ-PAE features
python utils/VQPAE_preprocess.py \
  --checkpoint <your_vqpae_checkpoint_path> \
  --stride 90 --src raw_data/aistpp/wavs --dest raw_data/aistpp/vqpae_feats

python utils/VQPAE_preprocess.py \
  --checkpoint <your_vqpae_checkpoint_path> \
  --stride 90 --src raw_data/finedance/music_wav --dest raw_data/finedance/vqpae_feats
```

---

## Configuration

Create `configs/paths.yaml` from the example:

```bash
cp configs/paths_default.yaml configs/paths.yaml
# Edit paths.yaml to point to your data directories
```

---

## Training

### 1. Train VAE (Motion Encoder)

```bash
# Train VAE
python train_vae.py --config <your_vae_config>.yaml --override train=True

# Test VAE
python train_vae.py --config <your_vae_config>.yaml
```

### 2. Pretokenize Dataset

Precompute VAE tokens for diffusion training.

In `pretokenize_vae.py`, configure the following variables according to your setup:

- **`PATH_FILES`** – lists of train/val/test split files.
- **`FEATURE_PATH`** – subdirectory name for raw motion features (e.g. `"motions_272_30fps"`).
- **`TOKEN_PATH`** – subdirectory name for output VAE tokens.
- **`RECOVERED_PATH`** – subdirectory name for reconstructed motions (optional, for quality checks).

Then run:

```bash
python pretokenize_vae.py --config <your_vae_config>.yaml
```

### 3. Train Latent Diffusion Forcing

```bash
# Train
python train_ldf.py --config <your_ldf_config>.yaml --override train=True

# Test / Evaluate
python train_ldf.py --config <your_ldf_config>.yaml
```

---

## Generation

```bash
# Generate from all .wav files under a directory
python generate_ldf.py --config <your_stream_config>.yaml --audio_root <your_wav_dir>

# Generate from specific .wav files
python generate_ldf.py --config <your_stream_config>.yaml --audio_paths a.wav b.wav

# Concatenate multiple audio files for one continuous output
python generate_ldf.py --config <your_stream_config>.yaml --audio_paths a.wav b.wav --concat

# Specify VQ-PAE checkpoint and output directory
python generate_ldf.py --config <your_stream_config>.yaml \
  --audio_root <your_wav_dir> \
  --vqpae_ckpt <your_vqpae_checkpoint_path> \
  --output_dir <your_output_dir>
```

---



## Web Real-time Demo

For real-time interactive demo with streaming generation, see [`web_demo/README.md`](web_demo/README.md).

## ROS2 Demo

A ROS2-based demo is available in [`ros2_demo/`](ros2_demo/). See [`ros2_demo/README.md`](ros2_demo/README.md) for setup instructions.

---

## Model Architecture

### VAE (Variational Autoencoder)

- **Input**: T × D motion features (D = 263 or 272)
- **Latent**: (T/4) × z_dim tokens
- **Architecture**: Causal 1D encoder-decoder based on WAN 2.2

### Latent Diffusion Forcing

- **Backbone**: DiT-style transformer based on WAN 2.2
- **Music Conditioning**: Decomposed causal music encoding (VQ-PAE discrete rhythmic tokens + continuous phase features) via cross-attention
- **Diffusion Schedule**: Triangular noise schedule with heterogeneous noise levels
- **Streaming**: Autoregressive latent generation with sliding window and history-guided sampling

---

## Project Structure

```
DiscoForcing/
├── configs/                        # Configuration files (YAML)
├── datasets/                       # Dataset loaders
├── models/                         # Model implementations
│   ├── diffusion_forcing_wan.py    # LDF with librosa music conditioning
│   ├── vae_wan_1d.py               # 1D causal VAE encoder-decoder
│   └── tools/                      # WAN model, attention, VAE internals
├── metrics/                        # Evaluation metrics
├── utils/                          # Utilities
│   ├── initialize.py               # Config & model loading
│   ├── motion_process.py           # Motion data processing
│   ├── visualize.py                # Rendering utilities
│   ├── dataset_process.py          # Dataset preprocessing
│   ├── VQPAE_preprocess.py         # VQ-PAE feature extraction
│   └── ...
├── PAE_training/                   # VQ-PAE training code
├── web_demo/                       # Real-time web demo
├── ros2_demo/                      # ROS2 integration demo
├── scripts/                        # Helper scripts (download, eval, train launchers, etc.)
├── train_vae.py                    # VAE training script
├── train_ldf.py                    # LDF training script
├── pretokenize_vae.py              # Dataset pretokenization
├── generate_ldf.py                 # Streaming generation
└── visualize_motion.py             # Motion visualization
```

**External data directories** (not included in repo):

```
deps/
└── smpl/
    └── SMPL_NEUTRAL.npz            # SMPL body model (required)

raw_data/
├── aistpp/
│   ├── wavs/                       # Raw audio files
│   ├── motions_272_30fps/          # Processed motion features
│   ├── baseline_feats_window_64/   # Windowed librosa features
│   ├── vqpae_feats/                # VQ-PAE features
│   └── ...
└── finedance/
    ├── music_wav/                   # Raw audio files
    ├── motions_272_30fps/           # Processed motion features
    ├── baseline_feats_window_64/    # Windowed librosa features
    ├── vqpae_feats/                 # VQ-PAE features
    └── ...

outputs/                             # Pretrained model checkpoints
```

---

## Citation

If you find this code useful in your research, please cite:

```bibtex
@inproceedings{ji2026discoforcing,
  title={DiscoForcing: A Unified Framework for Real-Time Audio-Driven Character Control with Diffusion Forcing},
  author={Ji, Kaiyang and Qian, Bingsheng and Wu, Binghuan and Chen, Kangyi and Shi, Ye and Wang, Jingya},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026}
}
```

## Acknowledgments

- [PyTorch Lightning](https://lightning.ai/) – Training framework
- [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) – Quaternion operations code
- [Alibaba Wan Team](https://github.com/Wan-Video/Wan2.2) – WAN model architecture and components
- [DeepPhase](https://github.com/sebastianstarke/AI4Animation) – Periodic Autoencoder for phase-based motion features
- [AIST++](https://google.github.io/aistplusplus_dataset/) – Music-dance dataset
- [FineDance](https://github.com/li-ronghui/FineDance) – Music-dance dataset
- [FloodDiffusion](https://github.com/ShandaAI/FloodDiffusion) – Streaming motion generation
