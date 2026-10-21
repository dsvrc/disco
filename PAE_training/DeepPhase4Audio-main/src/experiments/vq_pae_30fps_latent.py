import PIL
import PIL.Image
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import os

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from audiotools import AudioSignal
import soundfile as sf
import wandb

from src.datasets.definitions import SPLIT_TRAIN, SPLIT_TEST, SPLIT_VALID
from src.utils.metrics import MAE
from src.utils.helpers import resolve_dataset_class, resolve_lr_scheduler, resolve_model_class, resolve_optimizer
from src.losses.stft_loss import STFTLoss, MultiResolutionSTFTLoss
from src.losses.loss import L1Loss, SISDRLoss, MelSpectrogramLoss

class VQ_PAE30FPSModel(pl.LightningModule):
    """VQ-PAE with temporal downsampling to 30fps for motion alignment"""

    def __init__(self, cfg, **kwargs) -> None:
        super(VQ_PAE30FPSModel, self).__init__()
        self.cfg = cfg
        self.dataset_cfg = cfg.dataset_config
        self.training_config = cfg.training_config
        self.model_config = cfg.model_config
        self.worker_config = cfg.worker_config
        self.loss_config = cfg.loss_config
        self.saved_once = False

        self.D, self.N, self.K = self.model_config.input_channels, self.model_config.time_range, self.model_config.embedding_channels
        dataset_class = resolve_dataset_class(self.dataset_cfg.dataset)
        
        # Prepare dataset kwargs with optional audio processing parameters
        dataset_kwargs = {'dataset_root': self.dataset_cfg.dataset_root, 'split': None}
        if hasattr(self.dataset_cfg, 'original_rate'):
            dataset_kwargs['original_rate'] = self.dataset_cfg.original_rate
        if hasattr(self.dataset_cfg, 'target_rate'):
            dataset_kwargs['target_rate'] = self.dataset_cfg.target_rate
        if hasattr(self.dataset_cfg, 'lowpass_cutoff'):
            dataset_kwargs['lowpass_cutoff'] = self.dataset_cfg.lowpass_cutoff
        if hasattr(self.dataset_cfg, 'window_length'):
            dataset_kwargs['window_length'] = self.dataset_cfg.window_length
        
        self.datasets = {
            split: dataset_class(**{**dataset_kwargs, 'split': split})
            for split in (SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST)
        }

        for split in (SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST):
            print(f'Number of samples in {split} split: {len(self.datasets[split])}')

        self._instantiate_model()
        self.feat_loss = L1Loss()
        self.mel_loss = MelSpectrogramLoss()
        self.sisdr_loss = SISDRLoss()
        self.recon_loss = nn.MSELoss()
        self.stft_loss = STFTLoss()
        
        self.save_hyperparameters()

    def _instantiate_model(self):
        model_class = resolve_model_class(self.model_config.model_name)
        self.model = model_class(self.model_config)

    def training_step(self, batch, batch_idx):
        pred = self.model(batch)
        loss_items = {}

        x_recon = pred["audio"]
        codes = pred["codes"]
        z_vq = pred["z_vq"]
        z_recon = pred["z_recon"]
        phase_params = pred["phase_params"]

        if self.D == 1:
            batch = batch.reshape(batch.shape[0], self.N)
            x_recon = x_recon.reshape(x_recon.shape[0], self.N)
        else:
            x_recon = x_recon.reshape(x_recon.shape[0], self.D, self.N)

        loss_items['recon_loss'] = self.recon_loss(x_recon, batch)
        loss_items['stft_loss_spectral_convergence'], loss_items['stft_loss_magnitude'] = self.stft_loss(batch, x_recon)
        loss_items['vq/vodebook_loss'] = pred["vq/codebook_loss"].mean()
        loss_items['vq/commitment_loss'] = pred["vq/commitment_loss"].mean()
        loss_items['feature_matching_loss'] = self.feat_loss(x_recon, batch)
        loss_items['mel_spectrogram_loss'] = self.mel_loss(x_recon, batch)

        total_loss = sum([loss_items[key] * self.loss_config.loss_weights[key] for key in loss_items.keys()])

        self.log_dict(
            {f"loss_train/{k}": v for k, v in loss_items.items()},
            on_step=True, on_epoch=True, prog_bar=True
        )
        self.log("loss_train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        pred = self.model(batch)
        loss_items = {}

        x_recon = pred["audio"]
        codes = pred["codes"]
        z_vq = pred["z_vq"]
        z_recon = pred["z_recon"]
        phase_params = pred["phase_params"]

        if self.D == 1:
            batch = batch.reshape(batch.shape[0], self.N)
            x_recon = x_recon.reshape(x_recon.shape[0], self.N)
        else:
            x_recon = x_recon.reshape(x_recon.shape[0], self.D, self.N)

        loss_items['recon_loss'] = self.recon_loss(x_recon, batch)
        loss_items['stft_loss_spectral_convergence'], loss_items['stft_loss_magnitude'] = self.stft_loss(batch, x_recon)
        loss_items['vq/vodebook_loss'] = pred["vq/codebook_loss"].mean()
        loss_items['vq/commitment_loss'] = pred["vq/commitment_loss"].mean()
        loss_items['feature_matching_loss'] = self.feat_loss(x_recon, batch)
        loss_items['mel_spectrogram_loss'] = self.mel_loss(x_recon, batch)

        total_loss = sum([loss_items[key] * self.loss_config.loss_weights[key] for key in loss_items.keys()])

        self.log_dict(
            {f"loss_val/{k}": v for k, v in loss_items.items()},
            on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("loss_val/total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # log figures, randomly select one validation signal
        k = torch.randint(0, batch.shape[0], (1,)).item()
        x = batch[k, ...]
        y = x_recon[k, ...]
        selected_latent_signal = z_recon[k, ...] if z_recon is not None else None
        
        # save figures to step folder
        step_dir = os.path.join(self.cfg.output_dir, f'val_step_{self.global_step}')
        os.makedirs(step_dir, exist_ok=True)
        
        title_prefix = 'images_val'
        fig_reconstruction = self._visualize_reconstruction(x, y, selected_latent_signal)
        figure_log_spectogram = self._visualize_log_spectogram(x, y)
        
        fig_reconstruction.save(os.path.join(step_dir, 'reconstruction.png'))
        figure_log_spectogram.save(os.path.join(step_dir, 'log_spectogram.png'))
        
        self._log_figure(title=f"{title_prefix}/Reconstruction", caption="Input vs Reconstructed Signal", 
                         img=fig_reconstruction)
        self._log_figure(title=f"{title_prefix}/Log Spectogram", caption="Input and Output Log Spectograms", 
                         img=figure_log_spectogram)
        
        if not self.model_config.vq_only:
            selected_params = [p[k, ...] for p in phase_params]
            fig_latent_val = self._visualize_latent_values(selected_params)
            fig_latent_val.save(os.path.join(step_dir, 'latent_values.png'))
            self._log_figure(title=f"{title_prefix}/Latent Values", caption="Value Histogram of Learned Latent Parameters", 
                            img=fig_latent_val)
    
    def test_step(self, batch, batch_idx):
        if torch.cuda.is_available():
            batch = batch.to(self.device)
    
        pred = self.model(batch)
        loss_items = {}

        x_recon = pred["audio"]
        codes = pred["codes"]
        z_vq = pred["z_vq"]
        z_recon = pred["z_recon"]
        phase_params = pred["phase_params"]

        if self.D == 1:
            batch = batch.reshape(batch.shape[0], self.N)
            x_recon = x_recon.reshape(x_recon.shape[0], self.N)
        else:
            x_recon = x_recon.reshape(x_recon.shape[0], self.D, self.N)

        loss_items['recon_loss'] = self.recon_loss(x_recon, batch)
        loss_items['stft_loss_spectral_convergence'], loss_items['stft_loss_magnitude'] = self.stft_loss(batch, x_recon)
        loss_items['vq/vodebook_loss'] = pred["vq/codebook_loss"].mean()
        loss_items['vq/commitment_loss'] = pred["vq/commitment_loss"].mean()
        loss_items['feature_matching_loss'] = self.feat_loss(x_recon, batch)
        loss_items['mel_spectrogram_loss'] = self.mel_loss(x_recon, batch)

        self.log_dict(
            {f"loss_test/{k}": v for k, v in loss_items.items()},
            on_step=False, on_epoch=True, prog_bar=True
        )

        k = torch.randint(0, batch.shape[0], (1,)).item()
        x = batch[k, ...]
        y = x_recon[k, ...]
        selected_latent_signal = z_recon[k, ...] if z_recon is not None else None
        
        # Save test audio
        if not self.saved_once:
            if not self.device == 'cpu':
                act = x.cpu().numpy().flatten()
                pre = y.cpu().numpy().flatten()
            else:
                act = x.numpy().flatten()
                pre = y.numpy().flatten()
            
            if len(act) != len(pre):
                min_len = min(len(act), len(pre))
                act = act[:min_len]
                pre = pre[:min_len]
            
            test_dir = os.path.join(self.cfg.output_dir, 'test_audio')
            os.makedirs(test_dir, exist_ok=True)
            sf.write(os.path.join(test_dir, f'gt_epoch_{self.training_config.num_epochs}.wav'), act, 16000)
            sf.write(os.path.join(test_dir, f'pred_epoch_{self.training_config.num_epochs}.wav'), pre, 16000)
            self.saved_once = True
        
        """
        title_prefix = 'images_test'
        fig_reconstruction = self._visualize_reconstruction(x, y, selected_latent_signal)
        figure_log_spectogram = self._visualize_log_spectogram(x, y)

        self._log_figure(title=f"{title_prefix}/Reconstruction", caption="Input vs Reconstructed Signal", img=fig_reconstruction)
        self._log_figure(title=f"{title_prefix}/Log Spectogram", caption="Input and Output Log Spectograms", img=figure_log_spectogram)
    
        if not self.model_config.vq_only:
            selected_params = [p[k, ...] for p in phase_params]
            fig_latent_val = self._visualize_latent_values(selected_params)
            self._log_figure(title=f"{title_prefix}/Latent Values", caption="Value Histogram of Learned Latent Parameters", img=fig_latent_val)
        """

    def train_dataloader(self):
        return DataLoader(
            self.datasets[SPLIT_TRAIN],
            self.training_config.batch_size,
            shuffle=True,
            num_workers=self.worker_config.workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return self._create_val_test_dataloader(SPLIT_VALID)


    def test_dataloader(self):
        return self._create_val_test_dataloader(SPLIT_TEST)
    

    def configure_optimizers(self):
        optimizer = resolve_optimizer(self.training_config, self.model.parameters())
        lr_scheduler = resolve_lr_scheduler(self.training_config, optimizer)
        return [optimizer], [lr_scheduler]
    

    def _create_train_dataloader(self):
        return DataLoader(
            self.datasets[SPLIT_TRAIN],
            self.training_config.batch_size,
            shuffle=True,
            num_workers=self.worker_config.workers,
            pin_memory=True,
            drop_last=True,
        )


    def _create_val_test_dataloader(self, split):
        return DataLoader(
            self.datasets[split],
            self.training_config.batch_size_validation,
            shuffle=False,
            num_workers=self.worker_config.workers_validation,
            pin_memory=True,
            drop_last=False,
        )


    @staticmethod
    def _fig_to_PIL_img(fig) -> PIL.Image:
        canvas = FigureCanvas(fig)
        canvas.draw()
        width, height = canvas.get_width_height()
        img = PIL.Image.frombytes(
            'RGBA',
            (width, height),
            canvas.buffer_rgba().tobytes()
        ).convert('RGB')
        
        return img


    def _visualize_reconstruction(self, x, y, z):
        fig, axs = plt.subplots(3, 1, figsize=(20, 10))
        
        axs[0].plot(x.cpu().detach().numpy())
        axs[0].set_title('Ground Truth')
        axs[0].set_xlabel('Time (samples)')
        axs[0].set_ylabel('Amplitude')
        
        axs[1].plot(y.cpu().detach().numpy())
        axs[1].set_title('Reconstruction')
        axs[1].set_xlabel('Time (samples)')
        axs[1].set_ylabel('Amplitude')
        
        if z is not None:
            z_np = z.cpu().detach().numpy()
            for i in range(min(z_np.shape[0], 8)):
                axs[2].plot(z_np[i, :], label=f'Channel {i}', alpha=0.7)
            axs[2].set_title('Latent Signal (first 8 channels)')
            axs[2].set_xlabel('Time')
            axs[2].set_ylabel('Value')
            axs[2].legend(loc='upper right', fontsize='small', ncol=4)
        
        plt.tight_layout()
        img = self._fig_to_PIL_img(fig)
        plt.close(fig)
        
        return img


    def _visualize_log_spectogram(self, x, y):
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        
        # Convert to AudioSignal
        x_signal = AudioSignal(x.cpu().detach().numpy().reshape(1, 1, -1), sample_rate=16000)
        y_signal = AudioSignal(y.cpu().detach().numpy().reshape(1, 1, -1), sample_rate=16000)
        
        x_stft = x_signal.stft()
        y_stft = y_signal.stft()
        
        x_mag = 20 * torch.log10(x_stft.abs() + 1e-7)
        y_mag = 20 * torch.log10(y_stft.abs() + 1e-7)
        
        im0 = axs[0].imshow(x_mag.squeeze().cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
        axs[0].set_title('Ground Truth Log Spectrogram')
        axs[0].set_xlabel('Time Frames')
        axs[0].set_ylabel('Frequency Bins')
        plt.colorbar(im0, ax=axs[0], format='%+2.0f dB')
        
        im1 = axs[1].imshow(y_mag.squeeze().cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
        axs[1].set_title('Reconstructed Log Spectrogram')
        axs[1].set_xlabel('Time Frames')
        axs[1].set_ylabel('Frequency Bins')
        plt.colorbar(im1, ax=axs[1], format='%+2.0f dB')
        
        plt.tight_layout()
        img = self._fig_to_PIL_img(fig)
        plt.close(fig)
        
        return img


    def _visualize_latent_values(self, phase_params):
        p, f, a, b = [param.cpu().detach().numpy() for param in phase_params]
        
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        axs[0, 0].hist(p.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        axs[0, 0].set_title('Phase (p) Distribution')
        axs[0, 0].set_xlabel('Phase Value')
        axs[0, 0].set_ylabel('Frequency')
        axs[0, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
        
        axs[0, 1].hist(f.flatten(), bins=50, alpha=0.7, color='green', edgecolor='black')
        axs[0, 1].set_title('Frequency (f) Distribution')
        axs[0, 1].set_xlabel('Frequency Value')
        axs[0, 1].set_ylabel('Frequency')
        axs[0, 1].xaxis.set_major_locator(MaxNLocator(integer=True))
        
        axs[1, 0].hist(a.flatten(), bins=50, alpha=0.7, color='red', edgecolor='black')
        axs[1, 0].set_title('Amplitude (a) Distribution')
        axs[1, 0].set_xlabel('Amplitude Value')
        axs[1, 0].set_ylabel('Frequency')
        axs[1, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
        
        axs[1, 1].hist(b.flatten(), bins=50, alpha=0.7, color='purple', edgecolor='black')
        axs[1, 1].set_title('Bias (b) Distribution')
        axs[1, 1].set_xlabel('Bias Value')
        axs[1, 1].set_ylabel('Frequency')
        axs[1, 1].xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.tight_layout()
        img = self._fig_to_PIL_img(fig)
        plt.close(fig)
        
        return img


    def _log_figure(self, title, caption, img):
        if isinstance(self.logger, WandbLogger):
            self.logger.log_image(
                key=title,
                images=[img],
                caption=[caption]
            )
