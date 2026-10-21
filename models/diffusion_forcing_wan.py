import os
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
from .tools.wan_model import WanModel


class DiffForcingWanModel(nn.Module):
    def __init__(
        self,
        input_dim=256,
        hidden_dim=1024,
        ffn_dim=2048,
        freq_dim=256,
        num_heads=8,
        num_layers=8,
        time_embedding_scale=1.0,
        chunk_size=5,
        noise_steps=10,
        use_music_cond=True,
        music_dim=35,
        music_len=512,
        drop_out=0.1,  # external condition (music) dropout prob during training
        history_drop_out: Optional[float] = None,  # history dropout prob during training (None -> same as drop_out)
        tie_training_dropouts: bool = False,  # if True: history_dropout == music_dropout
        history_drop_soft_noise: bool = False,  # if True: when history is dropped, apply a soft noise schedule on history (gradient on last chunk OR random k~U(0,1)); else: full mask (k=1)
        history_loss_weight: float = 1.0,  # auxiliary loss weight for noisy history region (k_t > eps)
        cfg_scale=5.0,
        prediction_type="vel",  # "vel", "x0", "noise"
        causal=False,
        max_seq_len=1024,
        # History Guidance parameters
        use_history_guidance=False,
        history_guidance_scale=2.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.time_embedding_scale = time_embedding_scale
        self.chunk_size = chunk_size
        self.noise_steps = noise_steps
        self.use_music_cond = use_music_cond
        self.music_dim = music_dim
        self.drop_out = drop_out
        self.history_drop_out = history_drop_out
        self.tie_training_dropouts = tie_training_dropouts
        self.history_drop_soft_noise = history_drop_soft_noise
        self.history_loss_weight = history_loss_weight
        self.cfg_scale = cfg_scale
        self.prediction_type = prediction_type
        self.causal = causal
        self.max_seq_len = max_seq_len

        # History guidance settings
        self.use_history_guidance = use_history_guidance
        self.history_guidance_scale = history_guidance_scale

        self.music_len = music_len

        self.music_embedding = nn.Sequential(
            nn.Linear(self.music_dim, self.hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.hidden_dim, self.ffn_dim),
        )

        self.model = WanModel(
            patch_size=(1, 1, 1),
            music_len=self.music_len,
            in_dim=self.input_dim,
            dim=self.hidden_dim,
            ffn_dim=self.ffn_dim,
            freq_dim=self.freq_dim,
            music_dim=self.music_dim,
            out_dim=self.input_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            window_size=(-1, -1),
            qk_norm=True,
            cross_attn_norm=True,
            eps=1e-6,
            causal=self.causal,
            max_seq_len=self.max_seq_len,
        )
        self.param_dtype = torch.float32

    def preprocess(self, x):
        # (bs, T, C) -> (bs, C, T, 1, 1)
        x = x.permute(0, 2, 1)[:, :, :, None, None]
        return x

    def postprocess(self, x):
        # (bs, C, T, 1, 1) ->  (bs, T, C)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(x.size(0), x.size(2), -1)
        return x

    def _get_noise_levels(self, device, seq_len, time_steps):
        """Get noise levels"""
        noise_level = torch.clamp(
            1
            + torch.arange(seq_len, device=device) / self.chunk_size
            - time_steps.unsqueeze(1), 
            min=0.0,
            max=1.0,
        )
        return noise_level

    def add_noise(self, x, noise_level, noise: Optional[torch.Tensor] = None):
        """Add noise
        Args:
            x: (B, T, D)
            noise_level: (B, T)
        """
        noise = torch.randn_like(x) if noise is None else noise
        noise_level = noise_level.unsqueeze(-1)
        noisy_x = x * (1 - noise_level) + noise_level * noise
        return noisy_x, noise

    def forward(self, x):
        feature = x["feature"]  # (B, T, C)
        feature_length = x["feature_length"]  # (B,)
        batch_size, seq_len, _ = feature.shape
        device = feature.device

        # Randomly use a time step
        time_steps = []
        for i in range(batch_size):
            valid_len = feature_length[i].item()
            max_time = valid_len / self.chunk_size
            time_steps.append(torch.FloatTensor(1).uniform_(0, max_time).item())
        time_steps = torch.tensor(time_steps, device=device)  # (B,)
        noise_level = self._get_noise_levels(device, seq_len, time_steps)  # (B, T)

        # Encode music condition (CFG training: optionally drop external conditioning)
        music_dropped = [False for _ in range(batch_size)]
        if self.use_music_cond and "music" in x:
            music_list = x["music"]
            all_music_context = []
            # Fast-path: collated stream-mode music tensor, shape (B, T, W, D).
            # This avoids Python loops + per-frame H2D copies from numpy.
            if isinstance(music_list, torch.Tensor) and music_list.dim() == 4:
                # Ensure time length matches seq_len (feature is already padded to seq_len).
                if music_list.size(1) < seq_len:
                    pad_t = seq_len - music_list.size(1)
                    music_list = torch.cat(
                        [
                            music_list,
                            torch.zeros(
                                (batch_size, pad_t, music_list.size(2), music_list.size(3)),
                                device=music_list.device,
                                dtype=music_list.dtype,
                            ),
                        ],
                        dim=1,
                    )
                elif music_list.size(1) > seq_len:
                    music_list = music_list[:, :seq_len]

                # Per-sample dropout (drop whole external condition for a sample).
                if self.training and self.drop_out > 0:
                    drop_mask = (
                        torch.rand((batch_size,), device=music_list.device) < self.drop_out
                    )
                    for i in range(batch_size):
                        music_dropped[i] = bool(drop_mask[i].item())
                    music_list = torch.where(
                        drop_mask.view(batch_size, 1, 1, 1),
                        torch.zeros_like(music_list),
                        music_list,
                    )

                # WanModel cross-attn stream mode expects context with batch dimension B*T.
                # Shape: (B*T, W, D)
                music_context = (
                    music_list.to(dtype=self.param_dtype).reshape(
                        batch_size * seq_len, music_list.size(2), music_list.size(3)
                    )
                )
            elif isinstance(music_list[0], list):
                sample_m = music_list[0][0] if len(music_list) > 0 and len(music_list[0]) > 0 else None
                W = sample_m.shape[0] if sample_m is not None else self.music_len
                D = sample_m.shape[1] if sample_m is not None else self.music_dim
                self.model.music_len = W

                for i in range(batch_size):
                    single_music_list = music_list[i]
                    
                    if self.training and np.random.rand() < self.drop_out:
                        music_dropped[i] = True
                        single_music_context = [
                            torch.zeros((W, D), device=device, dtype=self.param_dtype)
                            for _ in range(seq_len)
                        ]
                    else:
                        single_music_context = [
                            torch.from_numpy(m).to(device).to(self.param_dtype)
                            for m in single_music_list
                        ]
                        
                        if len(single_music_context) < seq_len:
                            padding = [
                                torch.zeros((W, D), device=device, dtype=self.param_dtype)
                                for _ in range(seq_len - len(single_music_context))
                            ]
                            single_music_context.extend(padding)
                        else:
                            single_music_context = single_music_context[:seq_len]
                    
                    all_music_context.extend(single_music_context)
                music_context = all_music_context
            else:
                # Assume music_list is a tensor-like external condition (B, T, D) or similar.
                music_context = music_list
                if self.training and self.drop_out > 0:
                    # Per-sample dropout on batch dimension when possible
                    if isinstance(music_context, torch.Tensor) and music_context.dim() >= 1:
                        drop_mask = (
                            torch.rand((batch_size,), device=music_context.device)
                            < self.drop_out
                        )
                        for i in range(batch_size):
                            music_dropped[i] = bool(drop_mask[i].item())
                        # Broadcast drop_mask over remaining dims
                        view_shape = (batch_size,) + (1,) * (music_context.dim() - 1)
                        music_context = torch.where(
                            drop_mask.view(view_shape),
                            torch.zeros_like(music_context),
                            music_context,
                        )
        else:
            music_context = [
                torch.zeros((self.music_len, self.music_dim), device=device, dtype=self.param_dtype)
                for _ in range(batch_size * seq_len)
            ]

        # Training: independently drop history vs external condition by default.
        # - If tie_training_dropouts=True, history_drop == music_drop.
        # - Else, history_drop is sampled with prob=history_drop_out (defaults to drop_out).
        history_dropped = [False for _ in range(batch_size)]
        if self.training:
            if self.tie_training_dropouts:
                history_dropped = list(music_dropped)
            else:
                p_hist = self.drop_out if self.history_drop_out is None else self.history_drop_out
                if p_hist > 0:
                    for i in range(batch_size):
                        history_dropped[i] = (np.random.rand() < p_hist)

        # If history is dropped for a sample, mask (set to full noise) the "history" part
        # of the input (all tokens except the last chunk).
        if self.training and any(history_dropped):
             for i in range(batch_size):
                if not history_dropped[i]:
                    continue
                
                # Calculate history length
                t = time_steps[i].item()
                end_index = int(self.chunk_size * t) + 1
                valid_len = feature_length[i].item()
                end_index = min(valid_len, end_index)
                hist_len_i = max(0, end_index - self.chunk_size)
                
                if hist_len_i <= 0:
                    continue

                if self.history_drop_soft_noise:
                    # Apply a soft history-noise schedule:
                    # - Gradient noise on the most recent history chunk (0.8 -> 0.0)
                    # - OR random noise levels k~U(0,1) per history frame
                    #
                    # Assuming total history_drop_out is ~0.5.
                    # - 0.25/0.5 = 0.5 prob: Gradient Noise
                    # - 0.25/0.5 = 0.5 prob: Random k~U(0,1)
                    
                    if np.random.rand() < 0.5:
                        # Gradient Noise:
                        # 1. Full mask (1.0) for older history
                        noise_level[i, :hist_len_i] = 1.0
                        
                        # 2. Linear decay (0.8 -> 0.0) for the most recent chunk
                        grad_len = min(hist_len_i, self.chunk_size)
                        if grad_len > 0:
                            # linspace(0.8, 0.0, chunk_size) -> [0.8, 0.6, 0.4, 0.2, 0.0]
                            # We take the last `grad_len` values to align with the end of history
                            vals = torch.linspace(0.8, 0.0, self.chunk_size, device=device)
                            # Add randomness to noise levels: + U(0, 0.2)
                            vals = vals + torch.rand_like(vals) * 0.2
                            noise_level[i, hist_len_i - grad_len : hist_len_i] = vals[-grad_len:]
                            
                        # Update noise_ref for the gradient part (full mask part is handled by add_noise implicitly 
                        # but we need to update noise_ref here if we want manual control, 
                        # HOWEVER, add_noise() uses the modified noise_level to generate noise.
                        # Since we modified noise_level in-place, the subsequent add_noise call 
                        # will generate the correct noisy_feature and return the effective noise.
                        # So we don't need to manually update noise_ref here anymore (as we removed that code block).
                        # The update happens automatically via `noise_ref.append(noise[...])` later.
                    else:
                        # Random Levels: continuous uniform in [0, 1)
                        nl = torch.rand(
                            (hist_len_i,),
                            device=device,
                            dtype=torch.float32,
                        )
                        noise_level[i, :hist_len_i] = nl
                else:
                    # Full masking: set noise level to 1.0
                    noise_level[i, :hist_len_i] = 1.0

        # Add noise to entire sequence
        noisy_feature, noise = self.add_noise(feature, noise_level)  # (B, T, D)

        feature = self.preprocess(feature)  # (B, C, T, 1, 1)
        noisy_feature = self.preprocess(noisy_feature)  # (B, C, T, 1, 1)
        noise = self.preprocess(noise)  # (B, C, T, 1, 1)

        feature_ref = []
        noise_ref = []
        noisy_feature_input = []
        for i in range(batch_size):
            t = time_steps[i].item()
            end_index = int(self.chunk_size * t) + 1
            valid_len = feature_length[i].item()
            end_index = min(valid_len, end_index)
            feature_ref.append(feature[i, :, :end_index, ...])
            noise_ref.append(noise[i, :, :end_index, ...])
            noisy_feature_input.append(noisy_feature[i, :, :end_index, ...])

        # Through WanModel
        predicted_result = self.model(
            noisy_feature_input,
            noise_level * self.time_embedding_scale,
            music_context,
            seq_len,
            y=None,
        )

        loss = 0.0
        
        for b in range(batch_size):
            # Weighted multi-task loss:
            # - Active window (last chunk) is the main objective.
            # - Noisy history (k_t > eps) is an auxiliary denoising objective.
            if self.prediction_type == "vel":
                target = feature_ref[b] - noise_ref[b]
            elif self.prediction_type == "x0":
                target = feature_ref[b]
            elif self.prediction_type == "noise":
                target = noise_ref[b]
            else:
                raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

            # Per-step MSE over non-time dims -> (T,)
            diff = predicted_result[b] - target
            mse_per_step = (diff ** 2).mean(dim=(0, 2, 3))

            # Active window mask (last chunk within this sample's current window)
            T = mse_per_step.shape[0]
            active_len = min(self.chunk_size, T)
            mask_active = torch.zeros_like(mse_per_step)
            mask_active[T - active_len : T] = 1.0

            # History mask: only noisy history (k_t > eps)
            mask_history = torch.zeros_like(mse_per_step)
            hist_end = T - active_len
            if hist_end > 0:
                current_noise_levels = noise_level[b, :T]
                mask_history[:hist_end] = (current_noise_levels[:hist_end] > 1e-5).float()

            loss_active = (mse_per_step * mask_active).sum() / (mask_active.sum() + 1e-8)
            if mask_history.sum() > 0:
                loss_hist = (mse_per_step * mask_history).sum() / (mask_history.sum() + 1e-8)
            else:
                loss_hist = 0.0

            sample_loss = loss_active + self.history_loss_weight * loss_hist
            loss += sample_loss
        loss = loss / batch_size

        loss_dict = {"total": loss, "mse": loss}
        return loss_dict

    def _denoise_step_with_history_guidance(
        self,
        noisy_input: List[torch.Tensor],
        noise_level: torch.Tensor,
        music_context: List,
        music_null_context: Optional[List],
        seq_len: int,
        start_index: int,
        end_index: int,
        dt: float,
        history_eps_window: Optional[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor]:
        """
        Perform one denoising step with history guidance.
        
        Args:
            noisy_input: List of noisy inputs per batch
            noise_level: Noise levels (B, T)
            music_context: Music conditioning
            music_null_context: Null music conditioning for CFG
            seq_len: Sequence length
            start_index: Start index for update
            end_index: End index for denoising
            dt: Time step size
            
        Returns:
            Updated noisy inputs
        """
        batch_size = len(noisy_input)
        device = noisy_input[0].device
        
        # Infer history length from the current window.
        # In streaming setup, `noisy_input[i]` is shaped (C, T, 1, 1) where:
        # - the first `hist_len` tokens are treated as (already denoised) history,
        # - the last `chunk_size` tokens correspond to the current chunk being denoised.
        T = noisy_input[0].shape[1]
        hist_len = max(0, T - self.chunk_size)

        if not self.use_history_guidance:
            # Standard denoising without history guidance (optionally with CFG via cfg_scale)
            predicted_result = self.model(
                noisy_input,
                noise_level * self.time_embedding_scale,
                music_context,
                seq_len,
                y=None,
            )
            
            if self.cfg_scale != 1.0 and music_null_context is not None:
                predicted_result_null = self.model(
                    noisy_input,
                    noise_level * self.time_embedding_scale,
                    music_null_context,
                    seq_len,
                    y=None,
                )
                predicted_result = [
                    self.cfg_scale * pv - (self.cfg_scale - 1) * pvn
                    for pv, pvn in zip(predicted_result, predicted_result_null)
                ]
            
            return predicted_result
        
        # History guidance enabled:
        # We do NOT apply external CFG (cfg_scale) on top; guidance is implemented solely via:
        #   v(noise, null) + w * (v(clean, music) - v(noise, null))
        #
        # Ensure we have a null conditioning for the "noise, null" branch even when cfg_scale==1.
        if music_null_context is None:
            if isinstance(music_context, torch.Tensor):
                music_null_context = torch.zeros_like(music_context)
            else:
                music_null_context = [torch.zeros_like(m) for m in music_context]

        # Use provided per-window eps (recommended for streaming); otherwise sample per-call.
        if history_eps_window is not None:
            assert len(history_eps_window) == batch_size
            hist_eps_per_batch = [
                history_eps_window[i][:, :hist_len, ...] for i in range(batch_size)
            ]
        else:
            hist_eps_per_batch = [
                torch.randn_like(noisy_input[i][:, :hist_len, ...]) for i in range(batch_size)
            ]

        # History guidance (no history_weights_dict): v(noise, null) + w * (v(clean, music) - v(noise, null))
        # - "noise": deterministic soft re-noising on history:
        #   * older history: nl=1.0 (fully noised)
        #   * most recent history chunk: nl decays [0.8, 0.6, 0.4, 0.2, 0.0] (for chunk_size==5)
        # - "clean": keep history tokens clean (nl=0)
        noise_level_noise = noise_level.clone()
        nl_hist = torch.ones((hist_len,), device=device, dtype=torch.float32)
        # Always apply the fixed 5-frame schedule to the most recent 5 history frames
        # (independent of chunk_size).
        grad_len = min(hist_len, 5)
        if grad_len > 0:
            vals = torch.tensor(
                [0.8, 0.6, 0.4, 0.2, 0.0],
                device=device,
                dtype=torch.float32,
            )
            nl_hist[hist_len - grad_len : hist_len] = vals[-grad_len:]
        noise_level_noise[:, :hist_len] = nl_hist.unsqueeze(0).expand(batch_size, -1)
        noise_level_clean = noise_level.clone()
        noise_level_clean[:, :hist_len] = 0.0

        noisy_inputs_noise: List[torch.Tensor] = []
        for i in range(batch_size):
            u = noisy_input[i].clone()
            # x_noisy = x_clean * (1-nl) + eps * nl
            nl_ct = nl_hist.view(1, hist_len, 1, 1)  # broadcast over C
            u[:, :hist_len, ...] = (
                u[:, :hist_len, ...] * (1.0 - nl_ct) + hist_eps_per_batch[i] * nl_ct
            )
            noisy_inputs_noise.append(u)

        pred_clean_null = self.model(
            noisy_input,
            noise_level_clean * self.time_embedding_scale,
            music_null_context,
            seq_len,
            y=None,
        )

        pred_noise_music = self.model(
            noisy_inputs_noise,
            noise_level_noise * self.time_embedding_scale,
            music_context,
            seq_len,
            y=None,
        )

        w = float(self.history_guidance_scale)
        return [
            pv0 + w * (pv1 - pv0)
            for pv0, pv1 in zip(pred_clean_null, pred_noise_music)
        ]

    @torch.no_grad()
    def generate(self, x, num_denoise_steps=None):
        """
        Generation with history guidance support.
        """
        feature_length = x["feature_length"]
        batch_size = len(feature_length)
        seq_len = max(feature_length).item()

        if num_denoise_steps is None:
            num_denoise_steps = self.noise_steps
        assert num_denoise_steps % self.chunk_size == 0

        device = next(self.parameters()).device

        # Initialize with noise
        generated = torch.randn(
            batch_size, seq_len + self.chunk_size, self.input_dim, device=device
        )
        generated = self.preprocess(generated)

        max_t = 1 + (seq_len - 1) / self.chunk_size
        dt = 1 / num_denoise_steps
        total_steps = int(max_t / dt)

        # Prepare music condition
        generated_length = feature_length
        if self.use_music_cond and "music" in x:
            music_list = x["music"]
            all_music_context = []
            # Fast-path: collated stream-mode music tensor, shape (B, T, W, D).
            if isinstance(music_list, torch.Tensor) and music_list.dim() == 4:
                target_len = seq_len + self.chunk_size
                if music_list.size(1) < target_len:
                    pad_t = target_len - music_list.size(1)
                    music_list = torch.cat(
                        [
                            music_list,
                            torch.zeros(
                                (batch_size, pad_t, music_list.size(2), music_list.size(3)),
                                device=music_list.device,
                                dtype=music_list.dtype,
                            ),
                        ],
                        dim=1,
                    )
                elif music_list.size(1) > target_len:
                    music_list = music_list[:, :target_len]

                music_context = (
                    music_list.to(dtype=self.param_dtype).reshape(
                        batch_size * target_len, music_list.size(2), music_list.size(3)
                    )
                )
            elif isinstance(music_list[0], list):
                sample_m = music_list[0][0] if len(music_list) > 0 and len(music_list[0]) > 0 else None
                W = sample_m.shape[0] if sample_m is not None else self.music_len
                D = sample_m.shape[1] if sample_m is not None else self.music_dim
                self.model.music_len = W
                
                for i in range(batch_size):
                    single_music_list = music_list[i]
                    single_music_context = [
                        torch.from_numpy(m).to(device).to(self.param_dtype)
                        for m in single_music_list
                    ]

                    target_len = seq_len + self.chunk_size
                    if len(single_music_context) < target_len:
                        padding = [
                            torch.zeros((W, D), device=device, dtype=self.param_dtype)
                            for _ in range(target_len - len(single_music_context))
                        ]
                        single_music_context.extend(padding)
                    else:
                        single_music_context = single_music_context[:target_len]

                    all_music_context.extend(single_music_context)
                music_context = all_music_context
            else:
                music_context = music_list # (B, T, 35)
        else:
            music_context = [
                torch.zeros((self.music_len, self.music_dim), device=device, dtype=self.param_dtype)
                for _ in range(batch_size * (seq_len + self.chunk_size))
            ]

        # Null condition for CFG (must match the type/shape passed into WanModel)
        if self.cfg_scale != 1.0:
            if isinstance(music_context, torch.Tensor):
                music_null_context = torch.zeros_like(music_context)
            else:
                music_null_context = [torch.zeros_like(m) for m in music_context]
        else:
            music_null_context = None
        
        # Fixed epsilon per (latent) frame: reuse across denoise steps where `end_index` is unchanged.
        prev_end_index: Optional[int] = None
        eps_window_per_batch: Optional[List[torch.Tensor]] = None
        
        # Progressive generation
        for step in range(total_steps):
            t = step * dt
            start_index = max(0, int(self.chunk_size * (t - 1)) + 1)
            end_index = int(self.chunk_size * t) + 1
            time_steps = torch.full((batch_size,), t, device=device)

            noise_level = self._get_noise_levels(
                device, seq_len + self.chunk_size, time_steps
            )

            noisy_input = []
            for i in range(batch_size):
                noisy_input.append(generated[i, :, :end_index, ...])

            hist_len = max(0, end_index - self.chunk_size)

            # Resample epsilon only when a new latent frame becomes active (end_index changes).
            if prev_end_index != end_index:
                prev_end_index = end_index
                if hist_len > 0:
                    eps_window_per_batch = [
                        torch.randn(
                            self.input_dim,
                            hist_len,
                            1,
                            1,
                            device=device,
                            dtype=generated.dtype,
                        )
                        for _ in range(batch_size)
                    ]
                else:
                    eps_window_per_batch = None

            # Denoise with history guidance
            predicted_result = self._denoise_step_with_history_guidance(
                noisy_input=noisy_input,
                noise_level=noise_level,
                music_context=music_context,
                music_null_context=music_null_context,
                seq_len=seq_len + self.chunk_size,
                start_index=start_index,
                end_index=end_index,
                dt=dt,
                history_eps_window=eps_window_per_batch,
            )

            # Update generated sequence
            for i in range(batch_size):
                predicted_result_i = predicted_result[i]
                if self.prediction_type == "vel":
                    predicted_vel = predicted_result_i[:, start_index:end_index, ...]
                    generated[i, :, start_index:end_index, ...] += predicted_vel * dt
                elif self.prediction_type == "x0":
                    predicted_vel = (
                        predicted_result_i[:, start_index:end_index, ...]
                        - generated[i, :, start_index:end_index, ...]
                    ) / (
                        noise_level[i, start_index:end_index]
                        .unsqueeze(0)
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                    )
                    generated[i, :, start_index:end_index, ...] += predicted_vel * dt
                elif self.prediction_type == "noise":
                    predicted_vel = (
                        generated[i, :, start_index:end_index, ...]
                        - predicted_result_i[:, start_index:end_index, ...]
                    ) / (
                        1
                        + dt
                        - noise_level[i, start_index:end_index]
                        .unsqueeze(0)
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                    )
                    generated[i, :, start_index:end_index, ...] += predicted_vel * dt

        generated = self.postprocess(generated)
        y_hat_out = []
        for i in range(batch_size):
            single_generated = generated[i, : generated_length[i], :]
            y_hat_out.append(single_generated)
        out = {}
        out["generated"] = y_hat_out

        return out

    @torch.no_grad()
    def stream_generate(self, x, num_denoise_steps=None):
        """
        Streaming generation with history guidance support.
        """
        feature_length = x["feature_length"]
        batch_size = len(feature_length)
        seq_len = max(feature_length).item()

        if num_denoise_steps is None:
            num_denoise_steps = self.noise_steps
        assert num_denoise_steps % self.chunk_size == 0

        device = next(self.parameters()).device

        generated = torch.randn(
            batch_size, seq_len + self.chunk_size, self.input_dim, device=device
        )
        generated = self.preprocess(generated)

        max_t = 1 + (seq_len - 1) / self.chunk_size
        dt = 1 / num_denoise_steps
        total_steps = int(max_t / dt)

        generated_length = feature_length

        if self.use_music_cond and "music" in x:
            music_list = x["music"]
            all_music_context = []
            # Fast-path: collated stream-mode music tensor, shape (B, T, W, D).
            if isinstance(music_list, torch.Tensor) and music_list.dim() == 4:
                target_len = seq_len + self.chunk_size
                if music_list.size(1) < target_len:
                    pad_t = target_len - music_list.size(1)
                    music_list = torch.cat(
                        [
                            music_list,
                            torch.zeros(
                                (batch_size, pad_t, music_list.size(2), music_list.size(3)),
                                device=music_list.device,
                                dtype=music_list.dtype,
                            ),
                        ],
                        dim=1,
                    )
                elif music_list.size(1) > target_len:
                    music_list = music_list[:, :target_len]

                all_music_context = (
                    music_list.to(dtype=self.param_dtype).reshape(
                        batch_size * target_len, music_list.size(2), music_list.size(3)
                    )
                )
            elif isinstance(music_list[0], list):
                sample_m = music_list[0][0] if len(music_list) > 0 and len(music_list[0]) > 0 else None
                W = sample_m.shape[0] if sample_m is not None else self.music_len
                D = sample_m.shape[1] if sample_m is not None else self.music_dim
                self.model.music_len = W

                for i in range(batch_size):
                    single_music_list = music_list[i]
                    single_music_context = [
                        torch.from_numpy(m).to(device).to(self.param_dtype)
                        for m in single_music_list
                    ]

                    target_len = seq_len + self.chunk_size
                    if len(single_music_context) < target_len:
                        padding = [
                            torch.zeros((W, D), device=device, dtype=self.param_dtype)
                            for _ in range(target_len - len(single_music_context))
                        ]
                        single_music_context.extend(padding)
                    else:
                        single_music_context = single_music_context[:target_len]

                    all_music_context.extend(single_music_context)
                all_music_context = all_music_context
            else:
                all_music_context = music_list
        else:
            all_music_context = [
                torch.zeros((self.music_len, self.music_dim), device=device, dtype=self.param_dtype)
                for _ in range(batch_size * (seq_len + self.chunk_size))
            ]

        # Null condition for CFG (must match the type/shape passed into WanModel)
        if self.cfg_scale != 1.0:
            if isinstance(all_music_context, torch.Tensor):
                music_null_context = torch.zeros_like(all_music_context)
            else:
                music_null_context = [torch.zeros_like(m) for m in all_music_context]
        else:
            music_null_context = None

        # Fixed epsilon per (latent) frame: reuse across denoise steps where `end_index` is unchanged.
        prev_end_index: Optional[int] = None
        eps_window_per_batch: Optional[List[torch.Tensor]] = None

        commit_index = 0
        for step in range(total_steps):
            t = step * dt
            start_index = max(0, int(self.chunk_size * (t - 1)) + 1)
            end_index = int(self.chunk_size * t) + 1
            time_steps = torch.full((batch_size,), t, device=device)

            noise_level = self._get_noise_levels(
                device, seq_len + self.chunk_size, time_steps
            )

            noisy_input = []
            for i in range(batch_size):
                noisy_input.append(generated[i, :, :end_index, ...])

            hist_len = max(0, end_index - self.chunk_size)

            # Resample epsilon only when a new latent frame becomes active (end_index changes).
            if prev_end_index != end_index:
                prev_end_index = end_index
                if hist_len > 0:
                    eps_window_per_batch = [
                        torch.randn(
                            self.input_dim,
                            hist_len,
                            1,
                            1,
                            device=device,
                            dtype=generated.dtype,
                        )
                        for _ in range(batch_size)
                    ]
                else:
                    eps_window_per_batch = None

            # Denoise with history guidance
            predicted_result = self._denoise_step_with_history_guidance(
                noisy_input=noisy_input,
                noise_level=noise_level,
                music_context=all_music_context,
                music_null_context=music_null_context,
                seq_len=seq_len + self.chunk_size,
                start_index=start_index,
                end_index=end_index,
                dt=dt,
                history_eps_window=eps_window_per_batch,
            )

            for i in range(batch_size):
                predicted_result_i = predicted_result[i]
                if self.prediction_type == "vel":
                    predicted_vel = predicted_result_i[:, start_index:end_index, ...]
                    generated[i, :, start_index:end_index, ...] += predicted_vel * dt
                elif self.prediction_type == "x0":
                    predicted_vel = (
                        predicted_result_i[:, start_index:end_index, ...]
                        - generated[i, :, start_index:end_index, ...]
                    ) / (
                        noise_level[i, start_index:end_index]
                        .unsqueeze(0)
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                    )
                    generated[i, :, start_index:end_index, ...] += predicted_vel * dt
                elif self.prediction_type == "noise":
                    predicted_vel = (
                        generated[i, :, start_index:end_index, ...]
                        - predicted_result_i[:, start_index:end_index, ...]
                    ) / (
                        1
                        + dt
                        - noise_level[i, start_index:end_index]
                        .unsqueeze(0)
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                    )
                    generated[i, :, start_index:end_index, ...] += predicted_vel * dt

            if commit_index < start_index:
                output = generated[:, :, commit_index:start_index, ...]
                output = self.postprocess(output)
                y_hat_out = []
                for i in range(batch_size):
                    if commit_index < generated_length[i]:
                        y_hat_out.append(
                            output[i, : generated_length[i] - commit_index, ...]
                        )
                    else:
                        y_hat_out.append(None)

                out = {}
                out["generated"] = y_hat_out
                yield out
                commit_index = start_index

        output = generated[:, :, commit_index:, ...]
        output = self.postprocess(output)
        y_hat_out = []
        for i in range(batch_size):
            if commit_index < generated_length[i]:
                y_hat_out.append(output[i, : generated_length[i] - commit_index, ...])
            else:
                y_hat_out.append(None)
        out = {}
        out["generated"] = y_hat_out
        yield out

    def init_generated(self, seq_len, batch_size=1, num_denoise_steps=None):
        self.seq_len = seq_len
        self.batch_size = batch_size
        if num_denoise_steps is None:
            self.num_denoise_steps = self.noise_steps
        else:
            self.num_denoise_steps = num_denoise_steps
        assert self.num_denoise_steps % self.chunk_size == 0
        self.dt = 1 / self.num_denoise_steps
        self.current_step = 0
        self.music_condition_list = [[] for _ in range(self.batch_size)]
        self.generated = torch.randn(
            self.batch_size, self.seq_len * 2 + self.chunk_size, self.input_dim
        )
        self.generated = self.preprocess(self.generated)
        self.commit_index = 0

    @torch.no_grad()
    def stream_generate_step(self, x, first_chunk=True):
        """
        Streaming generation step with history guidance support.
        """
        device = next(self.parameters()).device
        
        if first_chunk:
            self.generated = self.generated.to(device)

        # Encode music condition
        if self.use_music_cond and "music" in x:
            music_list = x["music"]  # List[List[np.ndarray]] or Tensor (B, T, W, D)
            new_music_context = []
            for i in range(self.batch_size):
                # Fast-path: collated stream-mode tensor
                if isinstance(music_list, torch.Tensor) and music_list.dim() == 4:
                    m = music_list[i, -1]
                    W = m.shape[0] if m is not None else self.music_len
                    D = m.shape[1] if m is not None else self.music_dim
                    self.model.music_len = W  # update music_len for WanModel
                    new_music_context.append(m.to(device).to(self.param_dtype))
                    continue

                # Fallback: list-of-numpy per sample
                m = music_list[i][-1]
                W = m.shape[0] if m is not None else self.music_len
                D = m.shape[1] if m is not None else self.music_dim
                self.model.music_len = W  # update music_len for WanModel
                new_music_context.append(torch.from_numpy(m).to(device).to(self.param_dtype))
        else:
            new_music_context = [
                torch.zeros((self.music_len, self.music_dim), device=device, dtype=self.param_dtype)
                for _ in range(self.batch_size)
            ]

        # NOTE: for CFG, null context must match the *actual* context list passed into WanModel.
        # We'll build `music_null_context` after we assemble `music_condition` for this step.

        for i in range(self.batch_size):
            if first_chunk:
                self.music_condition_list[i].extend(
                    [new_music_context[i]] * self.chunk_size
                )
            else:
                self.music_condition_list[i].extend([new_music_context[i]])

        # (Streaming) We want a fixed epsilon per output latent frame, not per denoise step.
        # Sample once per call; reused across the inner `while self.current_step < end_step` loop.
        # No need to sample `seq_len` long; we only need up to the maximum possible `hist_len`
        # during this call.
        end_index_max = min(self.seq_len, self.commit_index + self.chunk_size)
        hist_len_max = max(0, end_index_max - self.chunk_size)
        eps_window_per_batch: Optional[List[torch.Tensor]]
        if hist_len_max > 0:
            eps_window_per_batch = [
                torch.randn(
                    self.input_dim,
                    hist_len_max,
                    1,
                    1,
                    device=device,
                    dtype=self.generated.dtype,
                )
                for _ in range(self.batch_size)
            ]
        else:
            eps_window_per_batch = None

        end_step = (
            (self.commit_index + self.chunk_size)
            * self.num_denoise_steps
            / self.chunk_size
        )
        
        while self.current_step < end_step:
            current_time = self.current_step * self.dt
            start_index = max(0, int(self.chunk_size * (current_time - 1)) + 1)
            end_index = int(self.chunk_size * current_time) + 1
            time_steps = torch.full((self.batch_size,), current_time, device=device)

            noise_level = self._get_noise_levels(device, end_index, time_steps)[
                :, -self.seq_len :
            ]

            noisy_input = []
            for i in range(self.batch_size):
                noisy_input.append(
                    self.generated[i, :, :end_index, ...][:, -self.seq_len :]
                )

            music_condition = []
            for i in range(self.batch_size):
                music_condition.extend(
                    self.music_condition_list[i][:end_index][-self.seq_len :]
                )
            
            if self.cfg_scale != 1.0:
                music_null_context = [torch.zeros_like(m) for m in music_condition]
            else:
                music_null_context = None

            hist_len = max(0, min(end_index, self.seq_len) - self.chunk_size)
            # Denoise with history guidance
            predicted_result = self._denoise_step_with_history_guidance(
                noisy_input=noisy_input,
                noise_level=noise_level,
                music_context=music_condition,
                music_null_context=music_null_context,
                seq_len=min(end_index, self.seq_len),
                start_index=start_index,
                end_index=end_index,
                dt=self.dt,
                history_eps_window=eps_window_per_batch,
            )

            for i in range(self.batch_size):
                predicted_result_i = predicted_result[i]
                if end_index > self.seq_len:
                    predicted_result_i = torch.cat(
                        [
                            torch.zeros(
                                predicted_result_i.shape[0],
                                end_index - self.seq_len,
                                predicted_result_i.shape[2],
                                predicted_result_i.shape[3],
                                device=device,
                            ),
                            predicted_result_i,
                        ],
                        dim=1,
                    )
                if self.prediction_type == "vel":
                    predicted_vel = predicted_result_i[:, start_index:end_index, ...]
                    self.generated[i, :, start_index:end_index, ...] += (
                        predicted_vel * self.dt
                    )
                elif self.prediction_type == "x0":
                    predicted_vel = (
                        predicted_result_i[:, start_index:end_index, ...]
                        - self.generated[i, :, start_index:end_index, ...]
                    ) / (
                        noise_level[i, start_index:end_index]
                        .unsqueeze(0)
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                    )
                    self.generated[i, :, start_index:end_index, ...] += (
                        predicted_vel * self.dt
                    )
                elif self.prediction_type == "noise":
                    predicted_vel = (
                        self.generated[i, :, start_index:end_index, ...]
                        - predicted_result_i[:, start_index:end_index, ...]
                    ) / (
                        1
                        + self.dt
                        - noise_level[i, start_index:end_index]
                        .unsqueeze(0)
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                    )
                    self.generated[i, :, start_index:end_index, ...] += (
                        predicted_vel * self.dt
                    )
            self.current_step += 1
            
        output = self.generated[:, :, self.commit_index : self.commit_index + 1, ...]
        output = self.postprocess(output)
        out = {}
        out["generated"] = output
        self.commit_index += 1

        if self.commit_index == self.seq_len * 2:
            self.generated = torch.cat(
                [
                    self.generated[:, :, self.seq_len :, ...],
                    torch.randn(
                        self.batch_size,
                        self.input_dim,
                        self.seq_len,
                        1,
                        1,
                        device=device,
                    ),
                ],
                dim=2,
            )
            self.current_step -= self.seq_len * self.num_denoise_steps / self.chunk_size
            self.commit_index -= self.seq_len
            for i in range(self.batch_size):
                self.music_condition_list[i] = self.music_condition_list[i][
                    self.seq_len :
                ]
        return out
