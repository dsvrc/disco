# The implementation of M2DMetrics is adapted from LODGE:
# https://github.com/li-ronghui/LODGE/blob/main/metric/metrics_finedance.py
# @inproceedings{li2024lodge,
#   title={Lodge: A coarse to fine diffusion network for long dance generation guided by the characteristic dance primitives},
#   author={Li, Ronghui and Zhang, YuXiang and Zhang, Yachao and Zhang, Hongwen and Guo, Jie and Zhang, Yan and Liu, Yebin and Li, Xiu},
#   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#   pages={1524--1534},
#   year={2024}
# }

from typing import List, Optional
import torch
from torch import Tensor
from torchmetrics import Metric
import numpy as np
from .m2d_features.kinetic import extract_kinetic_features
from .m2d_features.manual_new import extract_manual_features
from .m2d_features.beat_align import calc_db, BA, get_music_beat_fromwav

from .tools.utils import (
	calculate_activation_statistics_np,
	calculate_diversity_np,
	calculate_frechet_distance_np,
    calculate_avg_distance,
    calculate_foot_skating_ratio
)

def normalize(gt_feat, pred_feat):
    mean = gt_feat.mean(axis=0)
    std = gt_feat.std(axis=0)
    return (gt_feat - mean) / (std + 1e-10), (pred_feat - mean) / (std + 1e-10)


class M2DMetrics(Metric):
    """
    MRMetrics-style Music-to-Dance Metrics

    Metrics:
      - FID_k / FID_g
      - DIV_k / DIV_g
      - BAS (optional, cfg.evaluate_music=True)
    """

    def __init__(
        self,
        cfg,
        diversity_times: int = 300,
        dist_sync_on_step: bool = True,
    ):
        # NOTE:
        # This metric caches per-sample features in Python lists / CPU tensors.
        # In DDP (NCCL), torchmetrics' default sync-on-compute will try to all_gather
        # these states across ranks, which fails for CPU objects:
        #   RuntimeError: No backend type associated with device type cpu
        # We disable sync_on_compute and instead compute/log on rank 0.
        super().__init__(dist_sync_on_step=dist_sync_on_step, sync_on_compute=False)

        self.cfg = cfg
        self.evaluate_music = getattr(cfg, "evaluate_music", False)
        self.evaluate_physics = getattr(cfg, "evaluate_physics", False)
        self.diversity_times = diversity_times

        self.name = "Music-to-Dance"

        # -------- counters --------
        self.add_state("count_seq", default=torch.tensor(0), dist_reduce_fx="sum")

        # -------- feature caches --------
        self.add_state("gen_k_feats", default=[], dist_reduce_fx=None)
        self.add_state("gt_k_feats", default=[], dist_reduce_fx=None)

        self.add_state("gen_g_feats", default=[], dist_reduce_fx=None)
        self.add_state("gt_g_feats", default=[], dist_reduce_fx=None)
        
        self.metrics = ["FID_k", "FID_g", "DIV_k", "DIV_g", "DIV_k_gt", "DIV_g_gt"]
        
        # -------- BAS --------
        if self.evaluate_music:
            self.add_state("BAS_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.metrics.append("BAS")

        # -------- physics --------
        if self.evaluate_physics:
            self.add_state("FSR_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.metrics.append("FSR")  
            

    # -----------------------------------------------------
    # Update: cache a single sequence
    # -----------------------------------------------------
    @torch.no_grad()
    def update(
        self,
        joints_ref: Tensor,              # (bs, T_ref, J, 3)
        joints_rst: Tensor,              # (bs, T_rst, J, 3)
        lengths_ref: List[int],
        lengths_rst: List[int],
        music_paths: Optional[List[str]] = None,
    ):
        """
        Args:
            joints_ref: GT joint positions
            joints_rst: Generated joint positions
            lengths_ref: valid length per GT sequence
            lengths_rst: valid length per generated sequence
            music_paths: wav paths (required if evaluate_music=True)
        """
        assert joints_ref.dim() == 4
        assert joints_rst.dim() == 4
        assert joints_ref.shape[0] == joints_rst.shape[0]

        bs = joints_ref.shape[0]
        self.count_seq += bs

        ref = joints_ref.detach().cpu().numpy()
        rst = joints_rst.detach().cpu().numpy()

        for i in range(bs):
            T_ref = lengths_ref[i]
            T_rst = lengths_rst[i]

            joints_gt = ref[i, :T_ref]
            joints_gen = rst[i, :T_rst]

            # ---------- FSR ----------
            if self.evaluate_physics:
                fsr = calculate_foot_skating_ratio(joints_gen)
                self.FSR_sum += torch.tensor(fsr)

            # Root-relative joints
            joints_gt[:, 1:, :] -= joints_gt[:, 0:1, :]
            joints_gen[:, 1:, :] -= joints_gen[:, 0:1, :]

            # ---------- kinetic features ----------
            gt_k = extract_kinetic_features(joints_gt)
            gen_k = extract_kinetic_features(joints_gen)

            self.gt_k_feats.append(torch.from_numpy(gt_k))
            self.gen_k_feats.append(torch.from_numpy(gen_k))

            # ---------- geometric / manual features ----------
            gt_g = extract_manual_features(joints_gt)
            gen_g = extract_manual_features(joints_gen)

            self.gt_g_feats.append(torch.from_numpy(gt_g))
            self.gen_g_feats.append(torch.from_numpy(gen_g))

            # ---------- BAS ----------
            if self.evaluate_music:
                assert music_paths is not None
                dance_beats, _ = calc_db(joints_gen)
                music_beats = get_music_beat_fromwav(
                    music_paths[i], joints_gen.shape[0]
                )
                bas = BA(music_beats, dance_beats)
                self.BAS_sum += torch.tensor(bas)
                

    # -----------------------------------------------------
    # Compute: global metrics
    # -----------------------------------------------------
    @torch.no_grad()
    def compute(self, sanity_flag: bool = False):
        metrics = {}

        if sanity_flag or len(self.gen_k_feats) == 0:
            for k in self.metrics:
                metrics[k] = torch.tensor(0.0, device=self.device)
            self.reset()
            return metrics

        gen_k = torch.stack(self.gen_k_feats).numpy()
        gt_k = torch.stack(self.gt_k_feats).numpy()
        gen_g = torch.stack(self.gen_g_feats).numpy()
        gt_g = torch.stack(self.gt_g_feats).numpy()

        # Normalize features
        gt_k, gen_k = normalize(gt_k, gen_k)
        gt_g, gen_g = normalize(gt_g, gen_g)

        # ---------- FID ----------
        mu, cov = calculate_activation_statistics_np(gen_k)
        mu_gt, cov_gt = calculate_activation_statistics_np(gt_k)
        metrics["FID_k"] = torch.tensor(
            calculate_frechet_distance_np(mu_gt, cov_gt, mu, cov),
            device=self.device,
        )

        mu, cov = calculate_activation_statistics_np(gen_g)
        mu_gt, cov_gt = calculate_activation_statistics_np(gt_g)
        metrics["FID_g"] = torch.tensor(
            calculate_frechet_distance_np(mu_gt, cov_gt, mu, cov),
            device=self.device,
        )

        # ---------- Diversity ----------
        if gen_k.shape[0] > self.diversity_times:
            metrics["DIV_k"] = torch.tensor(
                calculate_diversity_np(gen_k, self.diversity_times),
                device=self.device,
            )
            metrics["DIV_g"] = torch.tensor(
                calculate_diversity_np(gen_g, self.diversity_times),
                device=self.device,
            )
            metrics["DIV_k_gt"] = torch.tensor(
                calculate_diversity_np(gt_k, self.diversity_times),
                device=self.device,
            )
            metrics["DIV_g_gt"] = torch.tensor(
                calculate_diversity_np(gt_g, self.diversity_times),
                device=self.device,
            )
        else:
            metrics["DIV_k"] = torch.tensor(
                calculate_avg_distance(gen_k), 
                device=self.device
            )
            metrics["DIV_g"] = torch.tensor(
                calculate_avg_distance(gen_g), 
                device=self.device
            )
            metrics["DIV_k_gt"] = torch.tensor(
                calculate_avg_distance(gt_k), 
                device=self.device
            )
            metrics["DIV_g_gt"] = torch.tensor(
                calculate_avg_distance(gt_g),
                device=self.device,
            )

        # ---------- BAS ----------
        if self.evaluate_music:
            metrics["BAS"] = torch.tensor(
                self.BAS_sum / self.count_seq, 
                device=self.device
            )
        
        # ---------- FSR ----------
        if self.evaluate_physics:
            metrics["FSR"] = torch.tensor(
                self.FSR_sum / self.count_seq, 
                device=self.device
            )

        self.reset()
        return metrics
    