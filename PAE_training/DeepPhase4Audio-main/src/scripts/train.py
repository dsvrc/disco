import os
# Force use GPU 7 BEFORE any CUDA/PyTorch imports
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import uuid
import torch

from datetime import datetime

from sklearn.model_selection import ParameterGrid
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, Callback  ## added Callback
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class PeriodicTestCallback(Callback):
    """Run test every N epochs during training"""
    def __init__(self, test_every_n_epochs=500):
        super().__init__()
        self.test_every_n_epochs = test_every_n_epochs
    
    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.test_every_n_epochs == 0:
            trainer.test(pl_module)


from src.utils.config import command_line_parser, yaml_config_parser
from src.experiments.pae_deep import PAEDeepModel # don't remove this
from src.experiments.paella import PAEllaModel
from src.experiments.pae_wave import PAEWaveModel
from src.experiments.pae_flattened import PAEInputFlattenedModel
from src.experiments.ae import AEModel
from src.experiments.vq_pae import VQ_PAEModel
from src.utils.helpers import get_device_accelerator 


def main():
    # Print GPU information
    print("=" * 80)
    print("GPU Configuration Check:")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
        print(f"torch.cuda.current_device(): {torch.cuda.current_device()}")
        print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
        # Show actual physical GPU being used
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used', '--format=csv,noheader'], 
                              capture_output=True, text=True)
        print(f"All GPUs:\n{result.stdout}")
    print("=" * 80)
    
    cfg = yaml_config_parser() # note that the namespaces are accessed differently
    seed_everything(42)
    # Resolve name task
    model_name = cfg.model_config.experiment_name
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config_name = cfg.metadata.run_name
    output_dir = os.path.join('output', f'{timestamp}_{config_name}')  ## timestamp + config name
    os.makedirs(output_dir, exist_ok=True)
    cfg.output_dir = output_dir
    
    model = eval(model_name)(cfg)
    run_name = f'T{timestamp}_{cfg.metadata.run_name}_{str(uuid.uuid4())[:5]}'

    """
    Define callbacks
    """
    if cfg.metadata.logging:
        csv_logger = CSVLogger(
            save_dir='logs',
            name=cfg.model_config.experiment_name,
        )
        wandb_logger = WandbLogger(
            name=run_name,
            project='Deep Learning',
            config=cfg
        )

    checkpoint_local_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, 'checkpoints'),  ## save to timestamped folder
        filename='step_{epoch}',
        save_last=False,
        save_top_k=3,
        monitor='loss_val/total_loss',
        mode='min',
    )

    # if this doesn't work (e.g. rich install fails) use TQDM progressbar instead (below)
    rich_progress_bar = RichProgressBar( 
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
            metrics_text_delimiter="\n",
            metrics_format=".3e",
        )
    )
    early_stop_callback = EarlyStopping(monitor="loss_train/stft_loss_spectral_convergence", min_delta=0.00, patience=50, verbose=False, mode="max")
    periodic_test_callback = PeriodicTestCallback(test_every_n_epochs=getattr(cfg, 'test_every_n_epochs', 500))  ## test every 500 epochs

    trainer = Trainer(
        logger=[wandb_logger, csv_logger] if cfg.metadata.logging else False,
        callbacks=[checkpoint_local_callback, rich_progress_bar, periodic_test_callback],  ## added periodic_test_callback
        accelerator=get_device_accelerator(preferred_accelerator='cuda'),
        devices=1,  ## Use the only visible GPU (mapped by CUDA_VISIBLE_DEVICES)
        default_root_dir=cfg.training_config.ckpt_save_dir, 
        max_epochs=cfg.training_config.num_epochs,
        num_sanity_val_steps=1,
        precision=16 if cfg.training_config.optimizer_float_16 else 32,
        log_every_n_steps=10,
        check_val_every_n_epoch=getattr(cfg, 'check_val_every_n_epoch', 100),  ## default to 100 if not specified
        profiler='simple',
        enable_checkpointing=True,
    )

    # train and test the model
    trainer.fit(model, ckpt_path=cfg.dataset_config.resume)
    print("saving model...")
    trainer.save_checkpoint(f'logs/models/{model_name}_epoch_{cfg.training_config.num_epochs}.ckpt',weights_only=False)  ## use training_config
    print("...done")
    trainer.test(model)


if __name__ == '__main__':
    main()