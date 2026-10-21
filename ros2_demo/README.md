# ros2_demo

This directory contains ROS2 demo packages for DiscoForcing.

- `textop_ctrl`: Custom message definitions (rosidl interface package)
- `music_manager`: Node that captures live audio and publishes VQPAE features
- `motion_manager`: Node that subscribes to VQPAE features and generates motion in real-time, published as `Gmr` messages

---

## 1) Prerequisites

- ROS2 installed (with `colcon`)
- Python environment with: `torch`, `numpy`, `sounddevice`
- A VQ-PAE checkpoint ready (e.g. from `PAE_training/DeepPhase4Audio-main/.../*.ckpt`)

> Note: `music_manager` reuses `utils/realtime_vqpae.py` from this repo to load the VQ-PAE model.

### 1.1 Python Dependencies

Install the required dependencies in your ROS2 Python environment (at minimum `sounddevice`; inference also needs `torch/lightning/torch_ema`, etc.):

```zsh
pip install -r requirements.txt
```

If you only want to run audio capture (without motion generation), the minimum is:

```zsh
pip install sounddevice numpy
```

---

## 2) Build (colcon build)

In a ROS2 workspace, place the packages under `src/` (or symlink the packages from `ros2_demo` into your workspace `src/`).

Recommended workspace structure (using `~/ros2_ws` as an example):

```zsh
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
ln -s <REPO_ROOT>/ros2_demo/textop_ctrl .
ln -s <REPO_ROOT>/ros2_demo/music_manager .
ln -s <REPO_ROOT>/ros2_demo/motion_manager .
```

You need at least these three packages:

- `textop_ctrl`
- `music_manager`
- `motion_manager`

After building, remember to source the setup file:

```zsh
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

---

## 3) Using music_manager to Capture Audio and Publish VQPAE Features

### 3.1 Node Overview

`music_manager` provides the node: `vqpae_audio_publisher`

What it does:

1. Captures audio in real-time from the input device (default microphone)
2. Maintains a 6-second sliding window (zero-padded if less than 6 seconds)
3. Runs VQ-PAE encoding on the current window each frame (30 FPS)
4. Publishes features to a ROS topic

Default parameters:

- Sample rate: 16 kHz
- FPS: 30
- Window length: 6 seconds
- Output feature shape: `(180, 16)`

### 3.2 Published Topic / Message

- Topic: `/music/vqpae` (configurable via the `topic` parameter)
- Message: `textop_ctrl/Vqpae`
  - `timestamp`
  - `features`: `std_msgs/Float32MultiArray`

When `publish_mode=latest` (default):
- `features.layout` describes dimensions `(180, 16)`

When `publish_mode=batch`:
- `features.layout` describes dimensions `(N, 180, 16)`

### 3.3 Selecting an Input Device (Microphone / Speaker Loopback)

#### A) Microphone
The default `input_device=default` uses the system default input device.

#### B) Speaker Loopback (System Audio)
On Linux, you typically need to select a PulseAudio/PipeWire **monitor source** (e.g. `alsa_output.*.monitor`).

You can have the node print the device list, then set the corresponding device index via `input_device`:

- `list_devices:=true`
- `input_device:="<index>"`

> Note: The monitor device name varies across machines/sound cards — this is expected.

### 3.4 Launch Commands (ros2 run)

1) (Optional) List audio devices:

```zsh
source ~/ros2_ws/install/setup.bash
ros2 run music_manager vqpae_audio_publisher --ros-args \
  -p list_devices:=true \
  -p checkpoint:="<VQPAE_CKPT>"
```

2) Normal launch (publish to `/music/vqpae`):

```zsh
source ~/ros2_ws/install/setup.bash
ros2 run music_manager vqpae_audio_publisher --ros-args \
  -p checkpoint:="<VQPAE_CKPT>" \
  -p device:=cuda \
  -p input_device:="default" \
  -p topic:="/music/vqpae"
```

---

## 4) Verifying the Topic (Optional)

You can use `ros2 topic echo` to quickly confirm data is being published:

```zsh
source ~/ros2_ws/install/setup.bash
ros2 topic echo /music/vqpae
```

Expected feature array lengths:
- Single window: approximately `180 * 16 = 2880`
- Batch: approximately `N * 180 * 16`

---

## 5) Troubleshooting

### Q1: No audio / no data
- Confirm `sounddevice` is available in the current ROS Python environment
- Confirm the correct input device is selected (system audio requires a monitor source)

### Q2: Checkpoint loading fails
- Confirm the checkpoint path is correct
- Confirm the checkpoint matches the VQ-PAE architecture in `utils/realtime_vqpae.py`

---

## 6) Using motion_manager to Subscribe to VQPAE Features and Publish Motion (Gmr)

### 6.1 Node Overview

`motion_manager` provides the node: `motion_generator`

What it does:

1. Subscribes to VQPAE features published by `music_manager` (`textop_ctrl/Vqpae`)
2. Calls the diffusion model in streaming mode: `stream_generate_step()`
3. Decodes with VAE `stream_decode()` to obtain **272-dim** motion representation
4. Recovers SMPL-ish parameters via `utils.motion_process.StreamMotionRecovery272(enable_smpl=True)`
5. Publishes `textop_ctrl/Gmr` (`pose_body`, `trans`, `root_orient`)

### 6.2 Subscribed / Published Topics

- Subscribe: `/music/vqpae` (configurable via `vqpae_topic`)
  - Message: `textop_ctrl/Vqpae`
- Publish: `/motion/gmr` (configurable via `gmr_topic`)
  - Message: `textop_ctrl/Gmr`
    - `pose_body`: `std_msgs/Float64MultiArray` (flattened axis-angle)
    - `trans`: `std_msgs/Float64MultiArray` (root translation, length 3)
    - `root_orient`: `std_msgs/Float64MultiArray` (root axis-angle, length 3)

### 6.3 Typical Usage (Start Audio First, Then Motion Generation)

1) Launch `music_manager` (audio capture + publish VQPAE features)
- Required parameter: `checkpoint:=<VQPAE_CKPT>`

```zsh
source ~/ros2_ws/install/setup.bash
ros2 run music_manager vqpae_audio_publisher --ros-args \
  -p checkpoint:="<VQPAE_CKPT>" \
  -p device:=cuda \
  -p topic:="/music/vqpae"
```

2) Launch `motion_manager` (subscribe VQPAE + generate motion)
- Key parameters:
  - `config:=configs/stream.yaml` (contains diffusion/VAE checkpoint paths)
  - `device:=auto|cuda|cpu`

> Both nodes use `/music/vqpae` by default, so they connect automatically without changing parameters.

```zsh
source ~/ros2_ws/install/setup.bash
ros2 run motion_manager motion_generator --ros-args \
  -p config:="configs/stream.yaml" \
  -p device:=auto \
  -p vqpae_topic:="/music/vqpae" \
  -p gmr_topic:="/motion/gmr"
```

### 6.4 Verification (Optional)

Check if `/motion/gmr` is producing output:

```zsh
source ~/ros2_ws/install/setup.bash
ros2 topic echo /motion/gmr
```

### 6.5 Troubleshooting

- If `motion_manager` starts but produces no output:
  - First confirm `/music/vqpae` has data (`ros2 topic echo /music/vqpae`)
  - Check that the `config` points to valid diffusion/VAE checkpoint paths
  - Confirm the model outputs 272-dim (this node expects 272-dim representation)
