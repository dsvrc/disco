# motion_manager (ROS2)

Subscribe live **VQPAE features** and generate a motion stream in real-time.

## I/O

- Subscribe: `textop_ctrl/Vqpae` (default topic: `/music/vqpae`)
- Publish: `textop_ctrl/Gmr` (default topic: `/motion/gmr`)

## What it does

For each incoming VQPAE feature window (usually `(180,16)`):

1. Run diffusion `stream_generate_step()` using `x={"music": [audio_ctx]}`
2. VAE `stream_decode()` produces motion representation frames
3. Recover joints with `utils.motion_process.StreamMotionRecovery272` (streaming 272-recovery)
4. Publish SMPL-ish params as `Gmr`:
   - `pose_body` (axis-angle flattened)
   - `trans` (root translation)
   - `root_orient` (root axis-angle)

## Parameters

- `config` (string): config yaml for model/ckpts (default `configs/stream.yaml`)
- `vqpae_topic` (string): input topic (default `/music/vqpae`)
- `gmr_topic` (string): output topic (default `/motion/gmr`)
- `device` (string): `auto` / `cuda` / `cpu`
- `history_length` (int): stream history window for model init
- `num_denoise_steps` (int): denoise steps
- `vae_scale_factor` (int): upsampling factor per latent token (usually 4)
- `smoothing_alpha` (float): EMA smoothing for joint recovery (1.0 = no smoothing)

## Notes

- This node assumes the current model+VAE pipeline outputs a **272-dim** representation suitable for `StreamMotionRecovery272(enable_smpl=True)`.
- If your VAE outputs 263-dim instead, we need to switch to `StreamJointRecovery263` and adjust the published message.
