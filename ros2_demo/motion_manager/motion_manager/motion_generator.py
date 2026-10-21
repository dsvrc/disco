"""ROS2 node: subscribe VQPAE features and generate motion, publishing as Gmr.

Pipeline
--------
- Subscribe: `textop_ctrl/Vqpae` on `/music/vqpae`
- Condition model with VQPAE feature window (180,16)
- Stream-generate latent tokens -> VAE stream decode -> 272-dim motion representation
- Recover joints with `utils.motion_process.StreamMotionRecovery272`
- Publish: `textop_ctrl/Gmr` (pose_body, trans, root_orient)

Assumptions / Notes
-------------------
- The diffusion+VAE models are loaded similarly to `generate_ldf.py --audio_type vqpae`.
- We assume VAE decode returns a 272-dim representation per frame (or per chunk).
- `Gmr` message fields are minimal in this repo; we fill them as Float64MultiArray.
"""

from __future__ import annotations

import os
import math
import time
from typing import Optional

import numpy as np

try:
    import torch as _torch  # type: ignore
except Exception:  # pragma: no cover
    _torch = None

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, MultiArrayDimension

from textop_ctrl.msg import Vqpae, Gmr


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


def _float64_multiarray(data: np.ndarray, label: str = "") -> Float64MultiArray:
    msg = Float64MultiArray()
    msg.layout.dim = []

    if data.ndim == 1:
        msg.layout.dim.append(MultiArrayDimension(label=label or "dim0", size=data.shape[0], stride=data.shape[0]))
    elif data.ndim == 2:
        d0, d1 = data.shape
        msg.layout.dim.append(MultiArrayDimension(label=label or "dim0", size=d0, stride=d0 * d1))
        msg.layout.dim.append(MultiArrayDimension(label="dim1", size=d1, stride=d1))
    else:
        flat = data.reshape(-1)
        msg.layout.dim.append(MultiArrayDimension(label=label or "flat", size=flat.shape[0], stride=flat.shape[0]))
        data = flat

    msg.data = data.astype(np.float64).reshape(-1).tolist()
    return msg


class MotionGenerator(Node):
    def __init__(self) -> None:
        super().__init__("motion_generator")

        # Parameters
        self.declare_parameter("config", "configs/stream.yaml")
        self.declare_parameter("vae_scale_factor", 4)
        self.declare_parameter("history_length", 60)
        self.declare_parameter("num_denoise_steps", 10)
        self.declare_parameter("device", "auto")
        self.declare_parameter("vqpae_topic", "/music/vqpae")
        self.declare_parameter("gmr_topic", "/motion/gmr")
        self.declare_parameter("smoothing_alpha", 1.0)

        # Load models (diffusion + VAE)
        repo_root = _repo_root()
        if repo_root not in os.sys.path:
            os.sys.path.insert(0, repo_root)

        if _torch is None:
            raise RuntimeError("PyTorch is required to run motion_manager. Please install torch in your ROS env.")

        torch = _torch  # type: ignore
        from lightning import seed_everything  # type: ignore
        from torch_ema import ExponentialMovingAverage  # type: ignore
        from utils.initialize import instantiate, load_config, compare_statedict_and_parameters
        from utils.motion_process import StreamMotionRecovery272

        cfg_path = self.get_parameter("config").get_parameter_value().string_value
        self.cfg = load_config(config_path=cfg_path)
        seed_everything(self.cfg.seed)

        device_param = self.get_parameter("device").get_parameter_value().string_value
        if device_param == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device_param

        torch.set_float32_matmul_precision("high")

        # VAE
        self.vae = instantiate(
            target=self.cfg.test_vae.target,
            cfg=None,
            hfstyle=False,
            **self.cfg.test_vae.params,
        )
        vae_ckpt = torch.load(self.cfg.test_vae_ckpt, map_location="cpu", weights_only=False)
        if "ema_state" in vae_ckpt:
            self.vae.load_state_dict(vae_ckpt["state_dict"], strict=True)
            vae_ema = ExponentialMovingAverage(self.vae.parameters(), decay=self.cfg.test_vae.ema_decay)
            vae_ema.load_state_dict(vae_ckpt["ema_state"])
            vae_ema.copy_to(self.vae.parameters())
            self.get_logger().info("Loaded VAE with EMA")
        else:
            self.vae.load_state_dict(vae_ckpt["state_dict"], strict=True)
            self.get_logger().info("Loaded VAE w/o EMA")

        compare_statedict_and_parameters(
            state_dict=self.vae.state_dict(),
            named_parameters=self.vae.named_parameters(),
            named_buffers=self.vae.named_buffers(),
        )
        self.vae.to(self.device)
        self.vae.eval()

        # Diffusion model
        self.model = instantiate(target=self.cfg.model.target, cfg=None, hfstyle=False, **self.cfg.model.params)
        ckpt = torch.load(self.cfg.test_ckpt, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["state_dict"], strict=True)
        if "ema_state" in ckpt:
            ema = ExponentialMovingAverage(self.model.parameters(), decay=self.cfg.model.ema_decay)
            ema.load_state_dict(ckpt["ema_state"])
            ema.copy_to(self.model.parameters())
            self.get_logger().info("Loaded diffusion model with EMA")
        else:
            self.get_logger().info("Loaded diffusion model w/o EMA")

        compare_statedict_and_parameters(
            state_dict=self.model.state_dict(),
            named_parameters=self.model.named_parameters(),
            named_buffers=self.model.named_buffers(),
        )
        self.model.to(self.device)
        self.model.eval()

        # Stream state
        self.first_chunk = True
        self.vae_scale_factor = int(self.get_parameter("vae_scale_factor").value)
        self.history_length = int(self.get_parameter("history_length").value)
        self.num_denoise_steps = int(self.get_parameter("num_denoise_steps").value)

        self.model.init_generated(self.history_length, batch_size=1, num_denoise_steps=self.num_denoise_steps)
        try:
            self.vae.clear_cache()
        except Exception:
            pass

        self.recovery = StreamMotionRecovery272(
            joints_num=22,
            smoothing_alpha=float(self.get_parameter("smoothing_alpha").value),
            device=self.device,
            enable_smpl=True,
        )

        # Pub/sub
        vqpae_topic = self.get_parameter("vqpae_topic").get_parameter_value().string_value
        gmr_topic = self.get_parameter("gmr_topic").get_parameter_value().string_value
        self.sub_ = self.create_subscription(Vqpae, vqpae_topic, self._on_vqpae, 10)
        self.pub_ = self.create_publisher(Gmr, gmr_topic, 10)

        self.get_logger().info(f"Subscribed: {vqpae_topic} | Publishing: {gmr_topic} | device={self.device}")

    def _on_vqpae(self, msg: Vqpae) -> None:
        """Consume latest VQPAE window and generate the next motion chunk."""
        # Parse features
        data = np.array(msg.features.data, dtype=np.float32)
        # Determine expected shape from layout if provided.
        if len(msg.features.layout.dim) >= 2:
            d0 = msg.features.layout.dim[0].size
            d1 = msg.features.layout.dim[1].size
            if d0 * d1 == data.size:
                feats = data.reshape(d0, d1)
            else:
                feats = data.reshape(-1, 16)
        else:
            feats = data.reshape(-1, 16)

        # Model expects (1, 180, 16) in a list, consistent with generate_ldf.py
        audio_ctx = feats[:180, :16][None, ...]

        x = {"music": [audio_ctx]}

        if _torch is None:
            self.get_logger().error("torch is not available; cannot run inference")
            return

        torch = _torch  # type: ignore
        with torch.no_grad():
            out = self.model.stream_generate_step(x, first_chunk=self.first_chunk)
            latent = out["generated"]  # list/tuple

            decoded = self.vae.stream_decode(latent[0][None, :], first_chunk=self.first_chunk)[0]

        self.first_chunk = False

        # decoded could be (T, D) or (D,)
        decoded_np = decoded.detach().cpu().numpy()
        if decoded_np.ndim == 1:
            decoded_np = decoded_np[None, :]

        # Convert each frame to SMPL-ish outputs using streaming 272 recovery
        # Then publish just the last frame (real-time)
        last = None
        for i in range(decoded_np.shape[0]):
            last = self.recovery.process_frame(decoded_np[i].astype(np.float32))

        if last is None:
            return

        # last is dict when enable_smpl=True
        pose_aa = last["pose_aa"].astype(np.float64)
        root_trans = last["root_trans"].astype(np.float64)
        root_rot = last["root_rot"].astype(np.float64)

        out_msg = Gmr()
        out_msg.pose_body = _float64_multiarray(pose_aa, label="pose_body")
        out_msg.trans = _float64_multiarray(root_trans, label="trans")
        out_msg.root_orient = _float64_multiarray(root_rot, label="root_orient")

        self.pub_.publish(out_msg)


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = MotionGenerator()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
