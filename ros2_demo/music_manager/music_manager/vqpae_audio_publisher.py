"""ROS2 node: capture audio (microphone or loopback) and publish VQ-PAE features.

Publishes `std_msgs/Float32MultiArray` with layout:
- dim[0] (label="frames", size=180, stride=180*16)
- dim[1] (label="channels", size=16, stride=16)
- or batched: if `publish_mode=batch`, packs N windows as (N, 180, 16)

Notes
-----
- This node reuses the project VQ-PAE loader in `utils.realtime_vqpae.load_vqpae_model`.
- For *speaker/system audio* loopback, PortAudio host support varies by OS/device.
  On Linux, typical solutions are PulseAudio/PIPEWIRE monitor sources.
"""

from __future__ import annotations

import os
import time
import queue
import threading
from dataclasses import dataclass
from typing import Optional, List

import numpy as np

try:
    import sounddevice as _sd  # type: ignore
except Exception:  # pragma: no cover
    _sd = None

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from textop_ctrl.msg import Vqpae


# We import sounddevice lazily so the node can start even if audio isn't available.


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    fps: int = 30
    window_length_s: float = 6.0
    channels: int = 1
    block_seconds: float = 0.1

    @property
    def time_range_samples(self) -> int:
        return int(self.sample_rate * self.window_length_s)

    @property
    def frame_step_samples(self) -> int:
        return self.sample_rate // self.fps

    @property
    def output_window_size(self) -> int:
        return int(self.window_length_s * self.fps)


class VQPAEAudioPublisher(Node):
    def __init__(self) -> None:
        super().__init__("vqpae_audio_publisher")

        # Parameters
        self.declare_parameter("checkpoint", "")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("input_device", "default")
        self.declare_parameter("list_devices", False)
        self.declare_parameter("topic", "/music/vqpae")
        self.declare_parameter("publish_mode", "latest")  # latest|batch
        self.declare_parameter("batch_size", 1)
        self.declare_parameter("verbose", True)

        self.cfg = AudioConfig()

        self.topic = self.get_parameter("topic").get_parameter_value().string_value
        self.publisher_ = self.create_publisher(Vqpae, self.topic, 10)

        self.verbose = bool(self.get_parameter("verbose").value)

        ckpt = self.get_parameter("checkpoint").get_parameter_value().string_value
        if not ckpt:
            raise RuntimeError("Parameter 'checkpoint' is required and must be a valid VQ-PAE checkpoint path")

        self.device = self.get_parameter("device").get_parameter_value().string_value
        self.input_device = self.get_parameter("input_device").get_parameter_value().string_value

        if bool(self.get_parameter("list_devices").value):
            if _sd is None:
                self.get_logger().warn("sounddevice is not available; cannot list audio devices")
            else:
                self.get_logger().info("Available audio devices:\n%s" % _sd.query_devices())

        # Import project utility for VQPAE
        # Ensure repo root is on sys.path when running as ROS node.
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        if repo_root not in os.sys.path:
            os.sys.path.insert(0, repo_root)

        from utils.realtime_vqpae import load_vqpae_model, process_vqpae_window

        self._process_vqpae_window = process_vqpae_window

        self.get_logger().info(f"Loading VQ-PAE model from: {ckpt}")
        self.model = load_vqpae_model(ckpt, device=self.device)
        self.get_logger().info(f"VQ-PAE ready on {self.device}")

        # Runtime state
        self.audio_queue = queue.Queue()
        self.audio_history = []
        self._processing = True
        self._samples_since_last_frame = 0

        self.publish_mode = self.get_parameter("publish_mode").get_parameter_value().string_value
        self.batch_size = int(self.get_parameter("batch_size").value)
        if self.publish_mode not in {"latest", "batch"}:
            raise RuntimeError("publish_mode must be 'latest' or 'batch'")
        if self.batch_size < 1:
            raise RuntimeError("batch_size must be >= 1")

        # Keep last N outputs for batch mode
        self._pending_windows = []

        self._proc_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._proc_thread.start()

        self._audio_stream = None
        self._start_audio_stream()

        self.get_logger().info(
            f"Publishing to {self.topic} | mode={self.publish_mode} | "
            f"sr={self.cfg.sample_rate} fps={self.cfg.fps} window={self.cfg.window_length_s}s"
        )

    def destroy_node(self):  # type: ignore[override]
        self._stop_audio_stream()
        self._processing = False
        if self._proc_thread.is_alive():
            self._proc_thread.join(timeout=2)
        return super().destroy_node()

    def _resolve_device(self):
        """Return sounddevice device argument: None for default, or int index."""
        if self.input_device in ("default", ""):
            return None
        try:
            return int(self.input_device)
        except ValueError:
            # Allow passing a device name
            return self.input_device

    def _audio_callback(self, indata, frames, time_info, status):
        if status and self.verbose:
            self.get_logger().warn(f"Audio status: {status}")
        # Mono
        self.audio_queue.put(indata[:, 0].copy())

    def _start_audio_stream(self) -> None:
        if _sd is None:
            raise RuntimeError(
                "sounddevice is required for audio capture. Install it in your ROS env: pip install sounddevice"
            )

        device = self._resolve_device()
        self.get_logger().info(f"Starting audio input stream (device={device})")
        self._audio_stream = _sd.InputStream(
            samplerate=self.cfg.sample_rate,
            channels=self.cfg.channels,
            callback=self._audio_callback,
            blocksize=int(self.cfg.sample_rate * self.cfg.block_seconds),
            device=device,
        )
        self._audio_stream.start()

    def _stop_audio_stream(self) -> None:
        if self._audio_stream is None:
            return
        try:
            self._audio_stream.stop()
            self._audio_stream.close()
        finally:
            self._audio_stream = None

    def _make_msg(self, data: np.ndarray) -> Vqpae:
        out = Vqpae()
        out.timestamp = self.get_clock().now().to_msg()

        feats = Float32MultiArray()
        feats.layout.dim = []

        if data.ndim == 2:
            # (180, 16)
            frames, ch = data.shape
            feats.layout.dim.append(MultiArrayDimension(label="frames", size=frames, stride=frames * ch))
            feats.layout.dim.append(MultiArrayDimension(label="channels", size=ch, stride=ch))
            feats.data = data.astype(np.float32).reshape(-1).tolist()
        elif data.ndim == 3:
            # (N, 180, 16) -> flatten but keep metadata as single-window shape
            n, frames, ch = data.shape
            feats.layout.dim.append(MultiArrayDimension(label="batch", size=n, stride=n * frames * ch))
            feats.layout.dim.append(MultiArrayDimension(label="frames", size=frames, stride=frames * ch))
            feats.layout.dim.append(MultiArrayDimension(label="channels", size=ch, stride=ch))
            feats.data = data.astype(np.float32).reshape(-1).tolist()
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")

        out.features = feats
        return out

    def _processing_loop(self) -> None:
        time_range = self.cfg.time_range_samples
        frame_step = self.cfg.frame_step_samples

        last_log = time.time()
        n_frames = 0

        while self._processing:
            try:
                audio_chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            self.audio_history.extend(audio_chunk.tolist())
            self._samples_since_last_frame += len(audio_chunk)

            if self._samples_since_last_frame < frame_step:
                continue

            current_samples = len(self.audio_history)
            if current_samples < time_range:
                pad = time_range - current_samples
                audio_window = np.concatenate(
                    [np.zeros(pad, dtype=np.float32), np.array(self.audio_history, dtype=np.float32)]
                )
            else:
                audio_window = np.array(self.audio_history[-time_range:], dtype=np.float32)

            try:
                feats = self._process_vqpae_window(audio_window, self.model, device=self.device)  # (180,16)
            except Exception as e:
                self.get_logger().error(f"VQPAE processing failed: {e}")
                continue

            if self.publish_mode == "latest":
                msg = self._make_msg(feats)
                self.publisher_.publish(msg)
            else:
                self._pending_windows.append(feats)
                if len(self._pending_windows) >= self.batch_size:
                    batch = np.stack(self._pending_windows[: self.batch_size], axis=0)
                    self._pending_windows = self._pending_windows[self.batch_size :]
                    msg = self._make_msg(batch)
                    self.publisher_.publish(msg)

            self._samples_since_last_frame -= frame_step
            n_frames += 1

            if self.verbose and (time.time() - last_log) > 5.0:
                self.get_logger().info(f"Published {n_frames} feature windows")
                last_log = time.time()


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = VQPAEAudioPublisher()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
