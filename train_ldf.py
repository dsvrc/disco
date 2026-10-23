import os
from datetime import timedelta
from typing import Any

import numpy as np
import torch
import torch.serialization
import wandb
from lightning import Trainer, seed_everything
from omegaconf import DictConfig, ListConfig
from omegaconf.base import ContainerMetadata
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage

from metrics.t2m import T2MMetrics
from metrics.m2d import M2DMetrics

from utils.cl_replay import STAR, ReplayStream
from utils.initialize import (
    compare_statedict_and_parameters,
    get_function,
    get_shared_run_time,
    instantiate,
    load_config,
    save_config_and_codes,
)
from utils.lightning_module import BasicLightningModule
from utils.motion_process import convert_motion_to_joints
from utils.visualize import (  # evaluate_video
    make_composite_compare_videos,
    render_video,
)

# Set tokenizers parallelism to false to avoid warnings in multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Register OmegaConf types as safe globals for PyTorch 2.6+ checkpoint loading.
# PyTorch 2.6 defaults torch.load(weights_only=True) and will refuse non-allowlisted globals.
# Lightning uses torch.load under the hood when resuming from .ckpt, so we allowlist the
# OmegaConf types commonly found inside the checkpoint dict.
torch.serialization.add_safe_globals([DictConfig, ListConfig, ContainerMetadata, Any])


class CustomLightningModule(BasicLightningModule):
    def initialize_metrics(self):

        # vae
        self.vae = instantiate(
            target=self.cfg.test_vae.target,
            cfg=None,
            hfstyle=False,
            **self.cfg.test_vae.params,
        )
        vae_ckpt = torch.load(
            self.cfg.test_vae_ckpt, map_location="cpu", weights_only=False
        )
        if "ema_state" in vae_ckpt:
            self.vae.load_state_dict(vae_ckpt["state_dict"], strict=True)
            self.vae_ema = ExponentialMovingAverage(
                self.vae.parameters(), decay=self.cfg.test_vae.ema_decay
            )
            self.vae_ema.load_state_dict(vae_ckpt["ema_state"])
            self.vae_ema.copy_to(self.vae.parameters())
            rank_zero_info(f"Loaded VAE model from {self.cfg.test_vae_ckpt} with EMA")
        else:
            self.vae.load_state_dict(vae_ckpt["state_dict"], strict=True)
            rank_zero_info(f"Loaded VAE model from {self.cfg.test_vae_ckpt} w/o EMA")

        compare_statedict_and_parameters(
            state_dict=self.vae.state_dict(),
            named_parameters=self.vae.named_parameters(),
            named_buffers=self.vae.named_buffers(),
        )
        
        # metric models
        self.recover_dim = self.cfg.metrics.dim
        self.features2joints = convert_motion_to_joints
        if self.cfg.metrics.t2m.enable:
            self.t2m_metrics = T2MMetrics(self.cfg.metrics.t2m)
        if self.cfg.metrics.m2d.enable:
            self.m2d_metrics = M2DMetrics(self.cfg.metrics.m2d)
        return


    def _step(self, batch, is_training=True):
        # Create a copy and replace motion fields with token fields
        model_batch = batch.copy()
        model_batch["feature"] = batch["token"]
        model_batch["feature_length"] = batch["token_length"]
        if "token_text_end" in batch:
            model_batch["feature_text_end"] = batch["token_text_end"]
        out = self.model(model_batch)
        return out

    def on_train_start(self):
        # continual learning: replay the buffers of sessions 0..cl_session-1 (utils/cl_replay.py, config `replay`)
        self.replay, self.star = None, None
        rcfg = self.cfg.get("replay", None)
        if rcfg is None or not rcfg.get("buffer_dir"):
            return
        self.replay = ReplayStream(
            self.cfg, rcfg.buffer_dir, self.cfg.cl_session,
            batch_size=rcfg.get("batch_size") or self.cfg.data.train_bs, num_workers=self.cfg.data.num_workers,
        )
        rank_zero_info(f"replay: {self.replay.summary}, loss weight {rcfg.get('weight', 1.0)}")
        scfg = rcfg.get("star", None)
        if scfg is not None and scfg.get("enable", False):
            self.star = STAR(self.model, lam=scfg.lam, gamma=scfg.gamma, eps=scfg.eps, steps=scfg.get("steps", 1))
            rank_zero_info(f"STAR: lam={scfg.lam} gamma={scfg.gamma} eps={scfg.eps}, {len(self.star.names)} weight tensors")

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        if getattr(self, "replay", None) is None:
            return loss
        rcfg = self.cfg.replay
        replay_batch = self.replay.next(self.device)
        per_sample = self.model(replay_batch, per_sample=True)["per_sample"]
        if rcfg.get("use_sample_weights", True):
            w = torch.tensor(replay_batch["weight"], device=per_sample.device, dtype=per_sample.dtype)
        else:
            w = torch.ones_like(per_sample)
        replay_loss = (w * per_sample).mean()
        total = loss + rcfg.get("weight", 1.0) * replay_loss
        bs = len(replay_batch["name"])
        self.log("train_loss/replay", replay_loss.item(), on_step=True, on_epoch=True, prog_bar=True, batch_size=bs)
        if self.star is not None:
            star_loss = self.star.loss(self.model, replay_batch)
            total = total + star_loss
            self.log("train_loss/star", star_loss.item(), on_step=True, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log("train_loss/cl_total", total.item(), on_step=True, on_epoch=False, batch_size=bs)
        return total

    def update_metrics(self, batch):
        with self.ema.average_parameters(self.model.parameters()):
            model_batch = batch.copy()
            model_batch["feature"] = batch["token"]
            model_batch["feature_length"] = batch["token_length"]
            if "token_text_end" in batch:
                model_batch["feature_text_end"] = batch["token_text_end"]
            output = self.model.generate(model_batch)
        generated = output["generated"] # latent (B, T / 4, 4)
        ground_truth_token = batch["token"] # latent (B, T / 4, 4)
        gt_token_length = batch["token_length"]
        ground_truth_feature = batch["feature"] # feat (B, T, 263)
        gt_feature_length = batch["feature_length"]
        

        for i in range(len(generated)):
            # Decode motion
            single_generated = generated[i]
            decoded_single_generated = self.vae.decode(
                single_generated[None, :].to(self.device)
            )[0]
            decoded_single_generated = decoded_single_generated.float().to(self.device)
            # Decode ground truth
            single_gt_r = ground_truth_token[i][: gt_token_length[i]]
            decoded_single_gt_r = self.vae.decode(single_gt_r[None, :].to(self.device))[
                0
            ]
            decoded_single_gt_r = decoded_single_gt_r.float().to(self.device)
            # Original ground truth
            single_gt_o = ground_truth_feature[i]
            decoded_single_gt_o = single_gt_o[: gt_feature_length[i], :].to(self.device)
            decoded_single_gt_o = decoded_single_gt_o.float().to(self.device)

            # Update T2M metrics
            if self.cfg.metrics.t2m.enable:
                # Some datasets (e.g., Babel after switching to music conditioning) may not provide text tokens.
                if "text_tokens" not in batch:
                    continue
                text_tokens_single = batch["text_tokens"][i]
                if self.cfg.metrics.t2m.fid_target == "vae":
                    self.t2m_metrics.update(
                        feats_rst=decoded_single_generated[None, ...],
                        feats_ref=decoded_single_gt_r[None, ...],
                        lengths_rst=[int(decoded_single_generated.shape[0])],
                        lengths_ref=[int(decoded_single_gt_r.shape[0])],
                        text_tokens=[text_tokens_single],
                    )
                else:
                    self.t2m_metrics.update(
                        feats_rst=decoded_single_generated[None, ...],
                        feats_ref=decoded_single_gt_o[None, ...],
                        lengths_rst=[int(decoded_single_generated.shape[0])],
                        lengths_ref=[int(decoded_single_gt_o.shape[0])],
                        text_tokens=[text_tokens_single],
                    )
            
            # Update M2D metrics
            if self.cfg.metrics.m2d.enable:
                joints_generated = self.features2joints(
                    decoded_single_generated.float().cpu().numpy(), 
                    dim=self.recover_dim
                ) # (T, 22, 3)
                joints_gt_o = self.features2joints(
                    decoded_single_gt_o.float().cpu().numpy(), 
                    dim=self.recover_dim
                ) # (T, 22, 3)
                joints_gt_r = self.features2joints(
                    decoded_single_gt_r.float().cpu().numpy(), 
                    dim=self.recover_dim
                ) # (T, 22, 3)
                # audio_features_single = batch.get("music")[i].to(self.device) # (T, 35) 
                # beat_features_single = audio_features_single[:, -1].astype(bool) # (T, ) one hot beat feature

                # Resolve music wav path for BAS (beat alignment score).
                # New configs store it per-dataset at `test_setting.<dataset>.music_dir`,
                # while older configs used a flat `test_setting.music_dir`.
                dataset_id = None
                if isinstance(batch, dict) and "dataset" in batch and len(batch["dataset"]) > i:
                    dataset_id = batch["dataset"][i]

                music_dir = None
                try:
                    from omegaconf import OmegaConf
                    if dataset_id is not None:
                        music_dir = OmegaConf.select(
                            self.cfg, f"test_setting.{dataset_id}.music_dir", default=None
                        )
                    if music_dir is None:
                        music_dir = OmegaConf.select(self.cfg, "test_setting.music_dir", default=None)
                except Exception:
                    # Best-effort fallback for non-OmegaConf configs
                    if dataset_id is not None and hasattr(self.cfg.test_setting, str(dataset_id)):
                        ds_ts = getattr(self.cfg.test_setting, str(dataset_id))
                        music_dir = getattr(ds_ts, "music_dir", None)
                    if music_dir is None:
                        music_dir = getattr(self.cfg.test_setting, "music_dir", None)

                music_path = None
                if getattr(self.m2d_metrics, "evaluate_music", False):
                    if music_dir is None:
                        raise ValueError(
                            "M2D metrics require wav paths, but no music_dir is configured. "
                            "Set either `test_setting.music_dir` (legacy) or "
                            "`test_setting.<dataset>.music_dir` (recommended, e.g. `test_setting.aistpp.music_dir`)."
                        )
                    music_path = os.path.join(music_dir, batch["name"][i] + ".wav")
                    if not os.path.exists(music_path):
                        raise FileNotFoundError(
                            f"Expected wav not found for M2D metrics: {music_path} "
                            f"(dataset={dataset_id}, name={batch['name'][i]})."
                        )

                if self.cfg.metrics.m2d.fid_target == "vae":
                    self.m2d_metrics.update(
                        joints_ref=torch.tensor(joints_gt_r,)[None, ...],
                        joints_rst=torch.tensor(joints_generated)[None, ...],
                        lengths_ref=[int(joints_gt_r.shape[0])],
                        lengths_rst=[int(joints_generated.shape[0])],
                        music_paths=[music_path] if music_path is not None else None,
                    )
                else:
                    self.m2d_metrics.update(
                        joints_ref=torch.tensor(joints_gt_o)[None, ...],
                        joints_rst=torch.tensor(joints_generated)[None, ...],
                        lengths_ref=[int(joints_gt_o.shape[0])],
                        lengths_rst=[int(joints_generated.shape[0])],
                        music_paths=[music_path] if music_path is not None else None,
                    )

        return

    def compute_metrics(self):
        # In DDP, metrics that keep Python list / CPU state (e.g. M2DMetrics)
        # cannot be synchronized via NCCL. We compute/log only on rank 0.
        if getattr(self.trainer, "world_size", 1) > 1 and getattr(self.trainer, "global_rank", 0) != 0:
            return

        if self.cfg.metrics.t2m.enable:
            t2m_output = self.t2m_metrics.compute(sanity_flag=self.trainer.sanity_checking)
            for key, value in t2m_output.items():
                self.log(f"metrics/t2m_metrics/{key}", value, sync_dist=False)
                try:
                    rank_zero_info(f"[metrics][t2m] {key}={float(value):.6f}")
                except Exception:
                    rank_zero_info(f"[metrics][t2m] {key}={value}")
        if self.cfg.metrics.m2d.enable:
            m2d_output = self.m2d_metrics.compute(sanity_flag=self.trainer.sanity_checking)
            for key, value in m2d_output.items():
                self.log(f"metrics/m2d_metrics/{key}", value, sync_dist=False)
                try:
                    rank_zero_info(f"[metrics][m2d] {key}={float(value):.6f}")
                except Exception:
                    rank_zero_info(f"[metrics][m2d] {key}={value}")
        return

    def update_test(self, batch):
        with self.ema.average_parameters(self.model.parameters()):
            model_batch = batch.copy()
            model_batch["feature"] = batch["token"]
            model_batch["feature_length"] = batch["token_length"]
            if "token_text_end" in batch:
                model_batch["feature_text_end"] = batch["token_text_end"]
            output = self.model.generate(model_batch)
        generated = output["generated"]
        # Save motion
        generated_id = batch["name"]  # [batch_size]
        dataset_id = batch["dataset"]  # [batch_size]

        # Decode motion to latent space
        # NOTE: inside the to() function, if will check current_tensor.device == target_device, then run this in each loop is fine.
        for i in range(len(generated)):
            single_dataset_id = dataset_id[i]
            single_generated_id = generated_id[i]
            
            # Limit to first 20 samples per dataset
            if single_dataset_id not in self.test_sample_counters:
                self.test_sample_counters[single_dataset_id] = 0
            
            if self.test_sample_counters[single_dataset_id] >= self.max_test_samples:
                continue
            
            self.test_sample_counters[single_dataset_id] += 1
            single_generated = generated[i]
            if "feature_text_end" in batch:
                single_feature_text_end = batch["feature_text_end"][i]
                frames = np.array(single_feature_text_end)
            else:
                frames = None
            try:
                decoded_single_generated = self.vae.decode(
                    single_generated[None, :].to(self.device)
                )[0]

                # Save music_name for rendering
                os.makedirs(
                    f"{self.cfg.save_dir}/{single_dataset_id}/music_name", exist_ok=True
                )
                with open(
                    f"{self.cfg.save_dir}/{single_dataset_id}/music_name/{single_generated_id}.txt",
                    "w",
                ) as f:
                    f.write(single_generated_id)
                
                os.makedirs(
                    f"{self.cfg.save_dir}/{single_dataset_id}/token",
                    exist_ok=True,
                )
                np.save(
                    f"{self.cfg.save_dir}/{single_dataset_id}/token/{single_generated_id}.npy",
                    single_generated.float().cpu().numpy(),
                )
                os.makedirs(
                    f"{self.cfg.save_dir}/{single_dataset_id}/feature",
                    exist_ok=True,
                )
                np.save(
                    f"{self.cfg.save_dir}/{single_dataset_id}/feature/{single_generated_id}.npy",
                    decoded_single_generated.float().cpu().numpy(),
                )
                # Save text_end if available
                if frames is not None:
                    os.makedirs(
                        f"{self.cfg.save_dir}/{single_dataset_id}/frames", exist_ok=True
                    )
                    np.save(
                        f"{self.cfg.save_dir}/{single_dataset_id}/frames/{single_generated_id}.npy",
                        frames,
                    )
            except Exception as e:
                rank_zero_info(
                    f"Error in saving motion {single_generated_id} of dataset {single_dataset_id}: {e}"
                )

        return {"output": output}

    def process_test_results(self):
        for dataset_id in os.listdir(self.cfg.save_dir):
            feature_dir = f"{self.cfg.save_dir}/{dataset_id}/feature"
            if not os.path.exists(feature_dir):
                continue
            # render video and save
            if self.cfg.test_setting.render and not self.cfg.debug:
                dataset_config = self.cfg.test_setting.get(dataset_id, {}) or {}
                music_dir = dataset_config.get(
                    "music_dir", self.cfg.test_setting.get("music_dir", None)
                )
                render_video(
                    motion_dir=feature_dir,
                    save_dir=f"{self.cfg.save_dir}/{dataset_id}/video",
                    render_setting=self.cfg.test_setting,
                    frames_dir=f"{self.cfg.save_dir}/{dataset_id}/frames",
                    music_dir=music_dir,
                )

                # Create composite videos
                # Get compare folders and names for this dataset
                compare_folders = dataset_config.get("compare_folders", None)
                compare_names = dataset_config.get("compare_names", None)
                
                rank_zero_info(f"Processing dataset: {dataset_id}")
                rank_zero_info(f"Compare folders: {compare_folders}")
                rank_zero_info(f"Compare names: {compare_names}")
                rank_zero_info(f"Music dir: {music_dir}")
                
                make_composite_compare_videos(
                    result_folder=f"{self.cfg.save_dir}/{dataset_id}/video",
                    compare_folders=compare_folders,
                    compare_names=compare_names,
                    music_name_folder=f"{self.cfg.save_dir}/{dataset_id}/music_name", 
                    music_dir=music_dir,
                    save_dir=f"{self.cfg.save_dir}/{dataset_id}/composite",
                )

                # wandb log video
                if (
                    not self.cfg.debug
                    and self.logger is not None
                    and isinstance(self.logger, WandbLogger)
                ):
                    video_to_log = []
                    for video_path in sorted(
                        os.listdir(f"{self.cfg.save_dir}/{dataset_id}/composite")
                    ):
                        video_to_log.append(
                            wandb.Video(
                                f"{self.cfg.save_dir}/{dataset_id}/composite/{video_path}",
                                format="gif",
                            )
                        )
                    wandb.log(
                        {f"{dataset_id}_video": video_to_log},
                        step=self.global_step,
                    )


def main():
    # init
    torch.set_float32_matmul_precision("high")
    cfg = load_config()
    seed_everything(cfg.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    run_time = get_shared_run_time(cfg.save_dir)
    save_dir = os.path.join(cfg.save_dir, f"{run_time}_{cfg.exp_name}")
    os.makedirs(save_dir, exist_ok=True)
    OmegaConf.update(cfg.config, "save_dir", save_dir)
    rank_zero_info(
        f"Save dir: {save_dir}, current working dir: {os.getcwd()}, exp_name: {cfg.exp_name}"
    )
    save_config_and_codes(cfg, cfg.save_dir)

    logger = None
    if not cfg.debug:
        wandb_key = cfg.logger.wandb.wandb_key
        if wandb_key and wandb_key.strip():
            os.environ["WANDB_API_KEY"] = wandb_key
            logger = WandbLogger(
                project=cfg.logger.wandb.project,
                name=f"{cfg.exp_name}_{run_time}",
                entity=cfg.logger.wandb.entity,
                config=OmegaConf.to_container(cfg.config, resolve=True),
                save_dir=cfg.save_dir,
            )
            rank_zero_info("WandB logging enabled")
        else:
            rank_zero_info("WandB API key not provided, skipping WandB logging")

    # dataloader
    collate_fn = (
        get_function(cfg.data.collate_fn) if cfg.data.get("collate_fn", None) else None
    )

    train_dataset = (
        instantiate(cfg.data.target, cfg=cfg.config, split="train")
        if cfg.train
        else None
    )
    val_dataset = instantiate(
        cfg.data.get("val_target", cfg.data.target), cfg=cfg.config, split="val"
    )
    test_dataset = instantiate(
        cfg.data.get("test_target", cfg.data.target), cfg=cfg.config, split="test"
    )

    rank_zero_info(
        f"Train dataset: {len(train_dataset) if train_dataset is not None else 0}, Val dataset: {len(val_dataset) if val_dataset is not None else 0}, Test dataset: {len(test_dataset)}"
    )

    train_dataloader = (
        DataLoader(
            train_dataset,
            batch_size=cfg.data.train_bs,
            shuffle=True,
            drop_last=False,
            num_workers=cfg.data.num_workers,
            persistent_workers=cfg.data.get("persistent_workers", True),
            prefetch_factor=cfg.data.get("prefetch_factor", 8),
            collate_fn=collate_fn,
            pin_memory=cfg.data.get("pin_memory", True),
        )
        if cfg.train
        else None
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.data.val_bs,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.data.num_workers,
        persistent_workers=cfg.data.get("persistent_workers", False),
        prefetch_factor=cfg.data.get("prefetch_factor", 8),
        collate_fn=collate_fn,
        pin_memory=cfg.data.get("pin_memory", True),
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.data.test_bs,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.data.num_workers,
        persistent_workers=cfg.data.get("persistent_workers", False),
        prefetch_factor=cfg.data.get("prefetch_factor", 8),
        collate_fn=collate_fn,
        pin_memory=cfg.data.get("pin_memory", True),
    )

    # lightning module, model is inside the lightning module
    model = CustomLightningModule(cfg=cfg.config)
    # continual learning: session k starts from the weights after session k-1 (init_ckpt), with a fresh
    # optimizer / LR schedule / step counter; resume_ckpt instead continues an interrupted run of the same session
    if cfg.train and cfg.get("init_ckpt") and not cfg.resume_ckpt:
        model.init_from_checkpoint(cfg.init_ckpt, use_ema=cfg.get("init_from_ema", True))

    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.save_dir,
        filename="step_{step}",
        every_n_train_steps=cfg.validation.save_every_n_steps,
        save_top_k=cfg.validation.save_top_k,
        monitor="step",
        mode="max",
        save_last=True,
        save_on_train_epoch_end=False,
    )
    if cfg.train:
        callbacks.append(checkpoint_callback)

    # Handle devices as either int or list
    num_devices = (
        cfg.trainer.devices
        if isinstance(cfg.trainer.devices, int)
        else len(cfg.trainer.devices)
    )

    trainer = Trainer(
        **cfg.trainer,
        logger=logger,
        strategy=DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=10800))
        if num_devices > 1
        else "auto",
        callbacks=callbacks,
        default_root_dir=cfg.save_dir,
        val_check_interval=cfg.validation.validation_steps,
        check_val_every_n_epoch=None,
    )

    if cfg.train:
        # PyTorch 2.6+ defaults torch.load(weights_only=True) which can reject common objects
        # saved in Lightning checkpoints (e.g., OmegaConf metadata, typing.Any, lists).
        # Our checkpoints are produced locally, so we force weights_only=False for resume.
        if cfg.resume_ckpt:
            from lightning.pytorch.trainer.connectors.checkpoint_connector import (
                _CheckpointConnector,
            )

            original_load = _CheckpointConnector.resume_start

            def patched_resume_start(self, checkpoint_path, weights_only=None):
                return original_load(self, checkpoint_path, weights_only=False)

            _CheckpointConnector.resume_start = patched_resume_start
        trainer.fit(
            model,
            train_dataloader,
            val_dataloaders=[val_dataloader, test_dataloader],
            ckpt_path=cfg.resume_ckpt,
        )
    else:
        for i in range(cfg.config.val_repeat):
            # Set different seed for each validation run to get diverse results
            # But keep it deterministic: same i -> same seed -> same result
            seed_everything(cfg.seed + i)
            # Use weights_only=False for checkpoint loading (trusted source)
            from lightning.pytorch.trainer.connectors.checkpoint_connector import _CheckpointConnector
            original_load = _CheckpointConnector.resume_start
            def patched_resume_start(self, checkpoint_path, weights_only=None):
                return original_load(self, checkpoint_path, weights_only=False)
            _CheckpointConnector.resume_start = patched_resume_start
            
            trainer.validate(
                model,
                dataloaders=[val_dataloader, test_dataloader],
                ckpt_path=cfg.test_ckpt,
            )
            model.cfg.test_setting.render = False  # only render once

    if not cfg.debug and logger is not None:
        wandb.finish()


if __name__ == "__main__":
    # train
    # train.py --config configs/ldf.yaml
    main()
