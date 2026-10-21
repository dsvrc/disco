import os

import numpy as np
import torch
import wandb
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from metrics.mr import MRMetrics
from metrics.t2m import T2MMetrics
from metrics.m2d import M2DMetrics
from utils.initialize import (
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


class CustomLightningModule(BasicLightningModule):
    def initialize_metrics(self):
        self.recover_dim = self.cfg.metrics.dim
        self.features2joints = convert_motion_to_joints
        self.mr_metrics = MRMetrics()
        self.t2m_metrics = None
        self.m2d_metrics = None
        if self.cfg.metrics.t2m.get("enable", False):
            self.t2m_metrics = T2MMetrics(self.cfg.metrics.t2m)
        if self.cfg.metrics.m2d.get("enable", False):
            self.m2d_metrics = M2DMetrics(self.cfg.metrics.m2d)

    def update_metrics(self, batch):
        with self.ema.average_parameters(self.model.parameters()):
            output = self.model.generate(batch)
        motion = output["generated"]
        ground_truth = batch["feature"]
        # recover to joint positions
        for i in range(len(motion)):
            single_motion = motion[i]
            single_gt = ground_truth[i]
            length = min(single_motion.shape[0], single_gt.shape[0])
            single_motion = single_motion[:length]
            single_gt = single_gt[:length]
            joints = self.features2joints(
                single_motion.float().cpu().numpy(),
                self.recover_dim,
            )
            gt_joints = self.features2joints(
                single_gt.float().cpu().numpy(),
                self.recover_dim,
            )
            # float32
            single_motion = single_motion.float().to(self.device)
            single_gt = single_gt.float().to(self.device)
            self.mr_metrics.update(
                joints_rst=torch.tensor(joints)[None, ...],
                joints_ref=torch.tensor(gt_joints)[None, ...],
                lengths=[length],
            )
            if self.t2m_metrics is not None:
                self.t2m_metrics.update(
                    feats_rst=single_motion[None, ...],
                    feats_ref=single_gt[None, ...],
                    lengths_rst=[length],
                    lengths_ref=[length],
            )
            if self.m2d_metrics is not None:
                self.m2d_metrics.update(
                    joints_ref=torch.tensor(gt_joints)[None, ...],
                    joints_rst=torch.tensor(joints)[None, ...],
                    lengths_ref=[length],
                    lengths_rst=[length],
            )

    def compute_metrics(self):
        mr_output = self.mr_metrics.compute(sanity_flag=self.trainer.sanity_checking)
        for key, value in mr_output.items():
            self.log(f"metrics/mr_metrics/{key}", value, sync_dist=False)
        if self.t2m_metrics is not None:
            t2m_output = self.t2m_metrics.compute(sanity_flag=self.trainer.sanity_checking)
            for key, value in t2m_output.items():
                self.log(f"metrics/t2m_metrics/{key}", value, sync_dist=False)
        if self.m2d_metrics is not None:
            m2d_output = self.m2d_metrics.compute(sanity_flag=self.trainer.sanity_checking)
            for key, value in m2d_output.items():
                self.log(f"metrics/m2d_metrics/{key}", value, sync_dist=False)

    def update_test(self, batch):
        
        # Only rank 0 should save test results to avoid duplication in multi-GPU
        if self.trainer.global_rank != 0:
            return
            
        with self.ema.average_parameters(self.model.parameters()):
            output = self.model.generate(batch)
        motion = output["generated"]
        # Save motion
        motion_id = batch["name"]  # [batch_size]
        dataset_id = batch["dataset"]  # [batch_size]
        text = batch.get("text", [""] * len(motion_id)) # Default to empty string if no text
        
        for single_motion, single_motion_id, single_dataset_id, single_text in zip(
            motion, motion_id, dataset_id, text, strict=False
        ):
            # Limit to first 20 samples per dataset
            if single_dataset_id not in self.test_sample_counters:
                self.test_sample_counters[single_dataset_id] = 0
            
            if self.test_sample_counters[single_dataset_id] >= self.max_test_samples:
                continue
            
            self.test_sample_counters[single_dataset_id] += 1
            
            os.makedirs(
                f"{self.cfg.save_dir}/{single_dataset_id}/motion", exist_ok=True
            )
            np.save(
                f"{self.cfg.save_dir}/{single_dataset_id}/motion/{single_motion_id}.npy",
                single_motion.float().cpu().numpy(),
            )
            
            # Save music name for rendering
            os.makedirs(
                f"{self.cfg.save_dir}/{single_dataset_id}/music_name", exist_ok=True
            )
            with open(
                f"{self.cfg.save_dir}/{single_dataset_id}/music_name/{single_motion_id}.txt",
                "w",
            ) as f:
                f.write(single_motion_id)
            
            # Save text description
            os.makedirs(f"{self.cfg.save_dir}/{single_dataset_id}/text", exist_ok=True)
            with open(
                f"{self.cfg.save_dir}/{single_dataset_id}/text/{single_motion_id}.txt",
                "w",
            ) as f:
                f.write(single_text)
        return

    def process_test_results(self):
        for dataset_id in os.listdir(self.cfg.save_dir):
            motion_dir = f"{self.cfg.save_dir}/{dataset_id}/motion"
            if not os.path.exists(motion_dir):
                continue
            # render video and save
            step_suffix = (
                f"_{self.global_step}"
                if self.cfg.test_setting.get("save_separate", False)
                else ""
            )
            if self.cfg.test_setting.render:
                render_video(
                    motion_dir=motion_dir,
                    save_dir=f"{self.cfg.save_dir}/{dataset_id}/video{step_suffix}",
                    render_setting=self.cfg.test_setting,
                )
                # Create composite videos if enabled
                if self.cfg.test_setting.get("make_composite", True):
                    dataset_config = self.cfg.test_setting.get(dataset_id, {}) or {}
                    music_dir = dataset_config.get(
                        "music_dir", self.cfg.test_setting.get("music_dir", None)
                    )
                    make_composite_compare_videos(
                        result_folder=f"{self.cfg.save_dir}/{dataset_id}/video{step_suffix}",
                        compare_folders=self.cfg.test_setting.get(dataset_id, {}).get(
                            "compare_folders", None
                        ),
                        compare_names=self.cfg.test_setting.get(dataset_id, {}).get(
                            "compare_names", None
                        ),
                        music_name_folder=f"{self.cfg.save_dir}/{dataset_id}/music_name",
                        music_dir=music_dir,
                        save_dir=f"{self.cfg.save_dir}/{dataset_id}/composite{step_suffix}",
                    )

                # wandb log video
                if (
                    not self.cfg.debug
                    and self.logger is not None
                    and isinstance(self.logger, WandbLogger)
                ):
                    video_to_log = []
                    composite_dir = f"{self.cfg.save_dir}/{dataset_id}/composite{step_suffix}"
                    if os.path.exists(composite_dir):
                        for video_path in sorted(os.listdir(composite_dir)):
                            video_to_log.append(
                                wandb.Video(
                                    os.path.join(composite_dir, video_path),
                                    format="gif",
                                )
                            )
                        wandb.log(
                            {f"{dataset_id}_video": video_to_log}, step=self.global_step
                        )


def initialize_config():
    cfg = load_config()
    seed_everything(cfg.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    run_time = get_shared_run_time(cfg.save_dir)
    save_dir = os.path.join(cfg.save_dir, f"{run_time}_{cfg.exp_name}")
    os.makedirs(save_dir, exist_ok=True)
    OmegaConf.update(cfg.config, "save_dir", save_dir)
    OmegaConf.update(cfg.config, "run_time", run_time)
    rank_zero_info(
        f"Save dir: {save_dir}, current working dir: {os.getcwd()}, exp_name: {cfg.exp_name}"
    )
    save_config_and_codes(cfg, cfg.save_dir)
    return cfg


def main():
    # init
    torch.set_float32_matmul_precision("high")
    cfg = load_config() # load from cli, here we load configs/vae_wan_1d.yaml
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
            persistent_workers=True,
            prefetch_factor=8,
            collate_fn=collate_fn,
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
        persistent_workers=False,
        prefetch_factor=8,
        collate_fn=collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.data.test_bs,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.data.num_workers,
        persistent_workers=False,
        prefetch_factor=8,
        collate_fn=collate_fn,
    )

    # lightning module, model is inside the lightning module
    model = CustomLightningModule(cfg=cfg.config)

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
        strategy=DDPStrategy(find_unused_parameters=True)
        if num_devices > 1
        else "auto",
        callbacks=callbacks,
        default_root_dir=cfg.save_dir,
        val_check_interval=cfg.validation.validation_steps,
        check_val_every_n_epoch=None,
    )

    if cfg.train:
        if not cfg.debug:
            trainer.validate(model, dataloaders=[val_dataloader, test_dataloader], weights_only=False)
        trainer.fit(
            model,
            train_dataloader,
            val_dataloaders=[val_dataloader, test_dataloader],
            ckpt_path=cfg.resume_ckpt,
            weights_only=False,
        )
    else:
        for i in range(cfg.config.val_repeat):
            # Set different seed for each validation run to get diverse results
            # But keep it deterministic: same i -> same seed -> same result
            seed_everything(cfg.seed + i)
            trainer.validate(
                model,
                dataloaders=[val_dataloader, test_dataloader],
                ckpt_path=cfg.test_ckpt,
                weights_only=False,
            )
            model.cfg.test_setting.render = False  # only render once

    if not cfg.debug and logger is not None:
        wandb.finish()


if __name__ == "__main__":
    # train
    # train.py --config configs/default_vae.yaml
    main()
