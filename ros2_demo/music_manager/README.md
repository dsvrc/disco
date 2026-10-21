# music_manager (ROS2)

This package publishes **VQ-PAE encoded features** from live audio (mic or loopback) as a ROS2 topic.

For an end-to-end ROS2 workspace build + run guide, see `ros2_demo/README.md`.

## What it publishes

Topic (default): `/music/vqpae`

Message type: `textop_ctrl/Vqpae`

- `publish_mode=latest` (default): one window each publish, features shape `(180, 16)`
- `publish_mode=batch`: packs `batch_size` windows, features shape `(N, 180, 16)`

The feature tensor is in `msg.features` (a `std_msgs/Float32MultiArray`).
Layout is described via `msg.features.layout.dim`.

## Parameters

- `checkpoint` (string, required): path to VQ-PAE checkpoint
- `device` (string): `cuda` or `cpu`
- `input_device` (string): `default`, an integer device index (e.g. `"8"`), or a device name
- `topic` (string): output topic name
- `publish_mode` (string): `latest` or `batch`
- `batch_size` (int): used when `publish_mode=batch`
- `list_devices` (bool): print audio devices at startup

## Notes on "speaker" loopback on Linux

For system audio (speaker) capture you usually need a **monitor source**:

- PulseAudio: `alsa_output.*.monitor`
- PipeWire: similar monitor sources

Use `input_device` to select the correct device/index.

## Dependencies

Runtime:
- ROS2 `rclpy`, `std_msgs`
- Python: `numpy`, `torch`
- Audio capture: `sounddevice`

This node reuses the repo VQ-PAE code in `utils/realtime_vqpae.py`.
