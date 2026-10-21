import torch
import torch.nn as nn
from src.models.PAE import *
from datetime import datetime  ## added for timestamp

from src.utils.config import command_line_parser, yaml_config_parser
import src.utils as utils
from src.experiments.pae_flattened import PAEInputFlattenedModel
from pytorch_lightning import Trainer
from src.utils.helpers import *
from src.datasets.definitions import SPLIT_TRAIN, SPLIT_TEST, SPLIT_VALID
from torch.utils.data import DataLoader
from src.losses.stft_loss import STFTLoss, MultiResolutionSTFTLoss
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np


"""change the config file to the one you want to use"""

cfg = utils.config.load_config("configs/paeflat200.yaml", flatten=False)  ## add flatten parameter

#model = PAEInputFlattenedModel(cfg)

#print(torch.load('logs/models/PAEInputFlattenedModel_epoch_50.ckpt', map_location=torch.device('cpu')))
model = PAEInputFlattenedModel.load_from_checkpoint(f"saved_models/PAEInputFlattenedModel_epoch_{cfg.training_config.num_epochs}.ckpt",cfg=cfg)  ## use training_config.num_epochs
trainer=Trainer(accelerator=get_device_accelerator(preferred_accelerator='cpu'),
        devices=1,
        default_root_dir=cfg.training_config.ckpt_save_dir,  ## use training_config
        max_epochs=cfg.training_config.num_epochs,  ## use training_config
        num_sanity_val_steps=1,
        precision=16 if cfg.training_config.optimizer_float_16 else 32,  ## use training_config
        profiler='simple')

model.eval()

#now we need to load the data and test the model
dataset_class = resolve_dataset_class(cfg.dataset_config.dataset)  ## use dataset_config
model.datasets = {split: dataset_class(dataset_root=cfg.dataset_config.dataset_root,split=split) 
            for split in (SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST)}

for split in (SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST):
    print(f'Number of samples in {split} split: {len(model.datasets[split])}')

# Create DataLoader for the test dataset
test_loader = DataLoader(model.datasets[SPLIT_TEST], batch_size=cfg.training_config.batch_size, shuffle=False, num_workers=cfg.worker_config.workers)  ## use training_config and worker_config

# Define loss functions
mse_loss = torch.nn.MSELoss()
stft_loss = STFTLoss()


# Initialize metrics
total_mse_loss = 0.0
total_stft_sc_loss = 0.0
total_stft_mag_loss = 0.0
num_batches = 0
D, N, K = cfg.model_config.input_channels, cfg.model_config.time_range, cfg.model_config.embedding_channels  ## use model_config
"""
# Evaluate the model on the test dataset
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        pred = model.model(batch)[0]
        if D == 1:
            batch = batch.reshape(batch.shape[0], N)
            pred = pred.reshape(pred.shape[0], N)
        else:
            pred = pred.reshape(pred.shape[0], D, N)
        mse = mse_loss(pred, batch)
        stft_l = stft_loss(batch, pred)
        stft_sc, stft_mag = stft_l[0], stft_l[1]

        total_mse_loss += mse.item()
        total_stft_sc_loss += stft_sc.item()
        total_stft_mag_loss += stft_mag.item()
        num_batches += 1

# Compute average losses
avg_mse_loss = total_mse_loss / num_batches
avg_stft_sc_loss = total_stft_sc_loss / num_batches
avg_stft_mag_loss = total_stft_mag_loss / num_batches

# Print results
print(f"Average MSE Loss: {avg_mse_loss}")
print(f"Average STFT Spectral Convergence Loss: {avg_stft_sc_loss}")
print(f"Average STFT Magnitude Loss: {avg_stft_mag_loss}")

# Write results to a report
with open("test_report.txt", "w") as report_file:
    report_file.write(f"Average MSE Loss: {avg_mse_loss}\n")
    report_file.write(f"Average STFT Spectral Convergence Loss: {avg_stft_sc_loss}\n")
    report_file.write(f"Average STFT Magnitude Loss: {avg_stft_mag_loss}\n")
"""
with torch.no_grad():
    batch = next(iter(test_loader))


    # define your mask, where 1 means keep and 0 means remove that signal
    fc_mask1 = np.ones(cfg.model_config.embedding_channels)  ## use model_config
    latent_mask1 = np.zeros(cfg.model_config.embedding_channels)
    fm1 = 0
    lm1 =16
    #fc_mask1[:fm1]=1
    #latent_mask1[:fm1]=1
    


    fc_mask2 = np.zeros(cfg.model_config.embedding_channels)
    latent_mask2 = np.zeros(cfg.model_config.embedding_channels)

   
    print("latent_mask:",latent_mask1)

    pred1, latent1, signal1, params1 = model.model.masked_forward(batch,fc_mask1,latent_mask1)  # Adjust mask as needed
    #pred2, latent2, signal2, params2 = model.model.masked_forward(batch,fc_mask2,latent_mask2)  # Adjust mask as needed

    if D == 1:
            batch = batch.reshape(batch.shape[0], N)
            pred1 = pred1.reshape(pred1.shape[0], N)
            #pred2 = pred2.reshape(pred2.shape[0], N)
    else:
         pred1 = pred1.reshape(pred1.shape[0], D, N)
         #pred2 = pred2.reshape(pred2.shape[0], D, N)
    
   
    k = 0 
    x = batch[k, ...]
    y1 = pred1[k, ...]
    #y2 =pred2[k, ...]
    selected_latent_signal1 = signal1[k, ...]
    #selected_latent_signal2 = signal2[k, ...] 
    selected_params = [p[k, ...] for p in params1]
    img1=model._visualize_reconstruction(x, y1,selected_latent_signal1)
    #img2=model._visualize_reconstruction(x, y2,selected_latent_signal2)
 
    img1.save(f"reconstruction_plot_{cfg.training_config.num_epochs}_{fm1}_fcmask_{lm1}_latentmask.png")  ## use training_config
    #img2.save(f"reconstruction_plot_{cfg.training_config.num_epochs}_nomask.png")
    
    # Save audio files for comparison
    import soundfile as sf
    import os
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f'save_{timestamp}'  ## save directory with timestamp
    os.makedirs(save_dir, exist_ok=True)
    act = x.cpu().numpy().flatten() if x.is_cuda else x.numpy().flatten()
    pred = y1.cpu().numpy().flatten() if y1.is_cuda else y1.numpy().flatten()
    sf.write(f'{save_dir}/gt_epoch_{cfg.training_config.num_epochs}.wav', act, 16000)
    sf.write(f'{save_dir}/pred_epoch_{cfg.training_config.num_epochs}.wav', pred, 16000)
    print(f"Saved audio files to {save_dir}/ directory")  ## audio comparison saved







