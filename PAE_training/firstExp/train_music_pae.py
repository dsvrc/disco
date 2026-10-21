"""
Training script for Music PAE
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
from datetime import datetime

from music_pae import MusicPAE


class MusicFeatureDataset(Dataset):
    """
    Dataset for music features
    
    Expected data format:
        - feature_dir: Directory containing .npy files with music features
        - Each .npy file shape: (T, 35) where T is number of frames
    
    Returns complete sequences, windowing is done in collate_fn
    """
    def __init__(self, feature_dir, window_size=32):
        self.feature_dir = feature_dir
        self.window_size = window_size
        
        self.feature_files = []
        for fname in sorted(os.listdir(feature_dir)):
            if fname.endswith('.npy'):
                fpath = os.path.join(feature_dir, fname)
                features = np.load(fpath)
                if len(features) >= window_size:
                    self.feature_files.append(fpath)
        
        print(f"Found {len(self.feature_files)} valid feature files (>= {window_size} frames)")
    
    def __len__(self):
        return len(self.feature_files)
    
    def __getitem__(self, idx):
        features = np.load(self.feature_files[idx])
        return torch.from_numpy(features).float()


def collate_fn(batch, window_size):
    """
    Collate function to extract random windows from sequences
    
    Args:
        batch: List of tensors with shape (T, 35)
        window_size: Size of window to extract
    
    Returns:
        Tensor of shape (B, window_size, 35)
    """
    windows = []
    for seq in batch:
        T = seq.shape[0]
        if T > window_size:
            start = np.random.randint(0, T - window_size + 1)
            window = seq[start:start + window_size]
        else:
            window = seq
        windows.append(window)
    
    return torch.stack(windows)


def plot_reconstruction(model, dataloader, device, save_path, num_samples=4):
    """Plot original vs reconstructed signals"""
    model.eval()
    
    # Get a batch
    batch = next(iter(dataloader))
    batch = batch[:num_samples].to(device)
    
    with torch.no_grad():
        recon, latent, signal, params = model(batch)
    
    # Reshape for plotting
    input_channels = model.input_channels
    time_range = batch.shape[1]  # Dynamic window size
    
    # Transpose for plotting: (B, W, 35) -> (B, 35, W)
    original = batch.transpose(1, 2).cpu().numpy()
    reconstructed = recon.transpose(1, 2).cpu().numpy()
    latent_np = latent.cpu().numpy()
    signal_np = signal.cpu().numpy()
    
    p, f, a, b = [x.squeeze(-1).cpu().numpy() for x in params]
    
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4*num_samples))
    
    for i in range(num_samples):
        # Original
        axes[i, 0].imshow(original[i], aspect='auto', cmap='viridis')
        axes[i, 0].set_title(f'Sample {i+1}: Original')
        axes[i, 0].set_ylabel('Features')
        
        # Reconstructed
        axes[i, 1].imshow(reconstructed[i], aspect='auto', cmap='viridis')
        axes[i, 1].set_title('Reconstructed')
        
        # Latent
        axes[i, 2].imshow(latent_np[i], aspect='auto', cmap='viridis')
        axes[i, 2].set_title('Latent Embedding')
        
        # Signal
        axes[i, 3].imshow(signal_np[i], aspect='auto', cmap='viridis')
        axes[i, 3].set_title('Parametrized Signal')
        
        # PFAB parameters
        axes[i, 4].bar(range(len(p[i])), p[i], alpha=0.5, label='Phase')
        axes[i, 4].bar(range(len(f[i])), f[i], alpha=0.5, label='Freq')
        axes[i, 4].bar(range(len(a[i])), a[i], alpha=0.5, label='Amp')
        axes[i, 4].bar(range(len(b[i])), b[i], alpha=0.5, label='Offset')
        axes[i, 4].set_title('PFAB Parameters')
        axes[i, 4].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {save_path}")


def train(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create timestamped save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"pae_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving to: {save_dir}")
    
    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project="music-pae",
            config=vars(args),
            name=f"pae_w{args.window_size}_e{args.embedding_channels}"
        )
    
    # Dataset
    print("Loading dataset...")
    dataset = MusicFeatureDataset(
        args.data_dir, 
        window_size=args.window_size
    )
    
    from functools import partial
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=partial(collate_fn, window_size=args.window_size)
    )
    
    # Model
    print("Initializing model...")
    model = MusicPAE(
        input_channels=args.input_channels,
        embedding_channels=args.embedding_channels,
        time_range=args.window_size,
        window=args.window_duration
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs, 
        eta_min=1e-5  # Don't decay to zero
    )
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Training loop
    print("Starting training...")
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(device)
            
            # Forward
            recon, latent, signal, params = model(batch)
            
            # Loss (both are now (B, W, 35))
            loss = criterion(recon, batch)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        # Epoch statistics
        avg_loss = epoch_loss / len(dataloader)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{args.epochs} - Avg Loss: {avg_loss:.6f} - LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Log to wandb
        if not args.no_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': avg_loss,
                'train/lr': scheduler.get_last_lr()[0]
            })
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_path = os.path.join(save_dir, f"music_pae_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }, save_path)
            print(f"Saved checkpoint to {save_path}")
        
        # Visualize
        if (epoch + 1) % args.plot_every == 0:
            plot_path = os.path.join(save_dir, f"reconstruction_epoch_{epoch+1}.png")
            plot_reconstruction(model, dataloader, device, plot_path)
            if not args.no_wandb:
                wandb.log({f'reconstruction_epoch_{epoch+1}': wandb.Image(plot_path)})
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(save_dir, "music_pae_best.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }, best_path)
            print(f"Saved best model (loss: {best_loss:.6f})")
    
    print("Training complete!")
    
    if not args.no_wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Train Music PAE')
    
    # Data
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing music feature .npy files')
    parser.add_argument('--input_channels', type=int, default=35,
                        help='Number of input feature channels')
    parser.add_argument('--window_size', type=int, default=150,
                        help='Time window size (number of frames)')
    parser.add_argument('--window_duration', type=float, default=5.0,
                        help='Time window duration in seconds')
    
    # Model
    parser.add_argument('--embedding_channels', type=int, default=5,
                        help='Number of phase channels')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=1000000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Saving
    parser.add_argument('--save_dir', type=str, default='PAE_training/music_pae_checkpoints')
    parser.add_argument('--save_every', type=int, default=5000)
    parser.add_argument('--plot_every', type=int, default=5000)
    
    # Logging
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging')
    
    args = parser.parse_args()
    
    train(args)


if __name__ == '__main__':
    main()
