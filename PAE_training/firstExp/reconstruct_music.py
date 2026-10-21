"""
Reconstruct music features from trained PAE model
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from music_pae import MusicPAE


def load_model(checkpoint_path, device='cuda'):
    """Load trained PAE model"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = MusicPAE(
        input_channels=35,
        embedding_channels=5,
        time_range=150,
        window=5.0
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.2f}")
    return model


def reconstruct_features(model, features, device='cuda'):
    """
    Reconstruct music features using PAE
    
    Args:
        model: Trained MusicPAE model
        features: numpy array (T, 35)
        device: torch device
    
    Returns:
        recon: Reconstructed features (T, 35)
        pfab_params: Dict with Phase, Freq, Amp, Offset
    """
    model.eval()
    T = features.shape[0]
    time_range = model.time_range
    
    reconstructed = []
    all_pfab = {'phase': [], 'freq': [], 'amp': [], 'offset': []}
    
    with torch.no_grad():
        for start in range(0, T - time_range + 1, time_range):
            window = features[start:start + time_range]
            if window.shape[0] < time_range:
                break
            
            window_tensor = torch.from_numpy(window).unsqueeze(0).float().to(device)
            recon, latent, signal, params = model(window_tensor)
            
            p, f, a, b = params
            all_pfab['phase'].append(p.cpu().numpy())
            all_pfab['freq'].append(f.cpu().numpy())
            all_pfab['amp'].append(a.cpu().numpy())
            all_pfab['offset'].append(b.cpu().numpy())
            
            reconstructed.append(recon[0].cpu().numpy())
    
    reconstructed = np.concatenate(reconstructed, axis=0)
    
    for key in all_pfab:
        all_pfab[key] = np.concatenate(all_pfab[key], axis=0)
    
    return reconstructed, all_pfab


def visualize_reconstruction(original, reconstructed, pfab_params, save_path):
    """Visualize original vs reconstructed features and PFAB parameters"""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    # Original features
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax1.set_title('Original Music Features (35 dims)')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Feature Dimension')
    ax1.set_yticks(range(0, 35, 5))
    
    # Reconstructed features
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(reconstructed.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax2.set_title('Reconstructed Music Features')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Feature Dimension')
    ax2.set_yticks(range(0, 35, 5))
    
    # Error heatmap
    ax3 = fig.add_subplot(gs[1, :])
    error = np.abs(original - reconstructed)
    im = ax3.imshow(error.T, aspect='auto', cmap='hot', interpolation='nearest')
    ax3.set_title('Reconstruction Error (Absolute Difference)')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Feature Dimension')
    plt.colorbar(im, ax=ax3, label='Error')
    
    time_indices = np.arange(len(pfab_params['phase']))
    
    # Phase
    ax4 = fig.add_subplot(gs[2, 0])
    for i in range(pfab_params['phase'].shape[1]):
        ax4.plot(time_indices, pfab_params['phase'][:, i, 0], label=f'Ch{i+1}', alpha=0.7)
    ax4.set_title('Phase Parameters')
    ax4.set_xlabel('Window Index')
    ax4.set_ylabel('Phase')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # Frequency
    ax5 = fig.add_subplot(gs[2, 1])
    for i in range(pfab_params['freq'].shape[1]):
        ax5.plot(time_indices, pfab_params['freq'][:, i, 0], label=f'Ch{i+1}', alpha=0.7)
    ax5.set_title('Frequency Parameters')
    ax5.set_xlabel('Window Index')
    ax5.set_ylabel('Frequency')
    ax5.legend(loc='upper right', fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # Amplitude
    ax6 = fig.add_subplot(gs[3, 0])
    for i in range(pfab_params['amp'].shape[1]):
        ax6.plot(time_indices, pfab_params['amp'][:, i, 0], label=f'Ch{i+1}', alpha=0.7)
    ax6.set_title('Amplitude Parameters')
    ax6.set_xlabel('Window Index')
    ax6.set_ylabel('Amplitude')
    ax6.legend(loc='upper right', fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    # Offset
    ax7 = fig.add_subplot(gs[3, 1])
    for i in range(pfab_params['offset'].shape[1]):
        ax7.plot(time_indices, pfab_params['offset'][:, i, 0], label=f'Ch{i+1}', alpha=0.7)
    ax7.set_title('Offset Parameters')
    ax7.set_xlabel('Window Index')
    ax7.set_ylabel('Offset')
    ax7.legend(loc='upper right', fontsize=8)
    ax7.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Reconstruct music features using trained PAE')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./pae_reconstructions')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = load_model(args.checkpoint, device)
    
    print(f"Loading features from {args.input}")
    features = np.load(args.input)
    print(f"Input shape: {features.shape}")
    
    print("Reconstructing...")
    reconstructed, pfab_params = reconstruct_features(model, features, device)
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Metrics
    mse = np.mean((features[:reconstructed.shape[0]] - reconstructed) ** 2)
    rmse = np.sqrt(mse)
    relative_error = rmse / (np.abs(features[:reconstructed.shape[0]]).mean() + 1e-8)
    
    print(f"\nReconstruction Metrics:")
    print(f"  MSE:  {mse:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  Relative Error: {relative_error*100:.2f}%")
    
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    
    # Save reconstructed features
    recon_path = os.path.join(args.output_dir, f"{base_name}_reconstructed.npy")
    np.save(recon_path, reconstructed)
    print(f"\nSaved reconstructed features to {recon_path}")
    
    # Save PFAB parameters
    pfab_path = os.path.join(args.output_dir, f"{base_name}_pfab.npz")
    np.savez(pfab_path, **pfab_params)
    print(f"Saved PFAB parameters to {pfab_path}")
    
    # Visualize
    vis_path = os.path.join(args.output_dir, f"{base_name}_visualization.png")
    visualize_reconstruction(
        features[:reconstructed.shape[0]], 
        reconstructed, 
        pfab_params, 
        vis_path
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()
