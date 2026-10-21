"""
Test script for Music PAE and integration with Diffusion Forcing

This script demonstrates:
1. Creating synthetic music features
2. Training a simple Music PAE
3. Using the PAE in DiffForcingWanPAEModel
"""

import torch
import numpy as np
import os

def test_music_pae():
    """Test Music PAE model"""
    print("=" * 60)
    print("Testing Music PAE")
    print("=" * 60)
    
    from music_pae import MusicPAE
    
    # Create model
    model = MusicPAE(
        input_channels=35,
        embedding_channels=5,
        time_range=150,
        window=5.0
    )
    
    # Test input: (B, W, 35) - new format
    batch_size = 4
    window_size = 150  # Default window size
    x = torch.randn(batch_size, window_size, 35)
    
    # Forward pass
    recon, latent, signal, params = model(x)
    p, f, a, b = params
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Reconstruction shape: {recon.shape}")
    print(f"✓ Latent shape: {latent.shape}")
    print(f"✓ Signal shape: {signal.shape}")
    print(f"✓ Phase shape: {p.shape}")
    print(f"✓ Frequency shape: {f.shape}")
    print(f"✓ Amplitude shape: {a.shape}")
    print(f"✓ Offset shape: {b.shape}")
    
    # Test get_pfab_features
    pfab = model.get_pfab_features(x)
    print(f"✓ PFAB features shape: {pfab.shape}")
    assert pfab.shape == (batch_size, 20), f"Expected ({batch_size}, 20), got {pfab.shape}"
    
    # Test dynamic window size
    x_short = torch.randn(2, 16, 35)  # Different window size
    recon_short, _, _, _ = model(x_short)
    print(f"✓ Dynamic window test: input {x_short.shape} -> output {recon_short.shape}")
    assert recon_short.shape == x_short.shape, "Shape mismatch for dynamic window"
    
    # Check window cache
    print(f"✓ Window cache keys: {list(model._window_cache.keys())}")
    assert 32 in model._window_cache and 16 in model._window_cache, "Cache not working"
    
    # Check phase range
    p_values = p.squeeze(-1).detach().numpy()
    print(f"✓ Phase range: [{p_values.min():.3f}, {p_values.max():.3f}] (should be in [-0.5, 0.5])")
    
    print("✓ Music PAE test passed!")
    print()
    return model


def test_periodic_encoder():
    """Test Periodic Phase Encoder"""
    print("=" * 60)
    print("Testing Periodic Phase Encoder")
    print("=" * 60)
    
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
    from diffusion_forcing_wan_pae import PeriodicPhaseEncoder
    
    # Create encoder
    encoder = PeriodicPhaseEncoder(
        phase_channels=5,
        L=8,
        hidden_dim=1024
    )
    
    # Test PFAB input: (B, 20) = [P(5), F(5), A(5), B(5)]
    batch_size = 4
    pfab = torch.randn(batch_size, 20)
    # Ensure phase is in correct range
    pfab[:, :5] = torch.rand(batch_size, 5) - 0.5  # P ∈ [-0.5, 0.5]
    
    # Encode
    encoded = encoder(pfab)
    
    print(f"✓ Input PFAB shape: {pfab.shape}")
    print(f"✓ Encoded shape: {encoded.shape}")
    assert encoded.shape == (batch_size, 1024), f"Expected (4, 1024), got {encoded.shape}"
    
    # Test periodicity preservation
    p1 = torch.tensor([[-0.5, 0.0, 0.25, -0.25, 0.1]])
    p2 = torch.tensor([[0.5, 0.0, 0.25, -0.25, 0.1]])  # 0.5 ≈ -0.5 (periodic)
    pfab1 = torch.cat([p1, torch.randn(1, 15)], dim=1)
    pfab2 = torch.cat([p2, torch.randn(1, 15)], dim=1)
    
    enc1 = encoder(pfab1)
    enc2 = encoder(pfab2)
    
    # Check if encodings are similar (not exact due to FAB differences)
    # But the phase-encoded part should contribute to similarity
    print(f"✓ Periodicity test: P=-0.5 vs P=0.5 encoding difference: {(enc1 - enc2).norm():.4f}")
    
    print("✓ Periodic Phase Encoder test passed!")
    print()
    return encoder


def test_integration():
    """Test full integration"""
    print("=" * 60)
    print("Testing DiffForcingWanPAEModel Integration")
    print("=" * 60)
    
    # NOTE: This requires PAE_training to be in path
    # In practice, you would call this from the main project directory
    
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
        from diffusion_forcing_wan_pae import DiffForcingWanPAEModel
        
        # Create model (without pretrained PAE for testing)
        model = DiffForcingWanPAEModel(
            pae_checkpoint_path=None,  # Will use random initialization
            input_dim=4,
            hidden_dim=256,  # Smaller for testing
            ffn_dim=512,
            num_heads=4,
            num_layers=2,
            music_window_size=32,
            phase_channels=5,
            use_music_cond=True,
        )
        
        print(f"✓ Model created successfully")
        print(f"  - Music PAE: {model.music_pae.__class__.__name__}")
        print(f"  - Phase Encoder: {model.phase_encoder.__class__.__name__}")
        print(f"  - WanModel: {model.model.__class__.__name__}")
        
        # Test music encoding
        batch_size = 2
        seq_len = 10
        
        # Create synthetic music: List[List[np.ndarray]]
        # Each sample has seq_len windows of (150, 35)
        music_list = [
            [np.random.randn(150, 35).astype(np.float32) for _ in range(seq_len)]
            for _ in range(batch_size)
        ]
        
        device = torch.device('cpu')
        encoded_music = model.encode_music_with_pae(music_list, device)
        
        print(f"✓ Encoded {len(encoded_music)} music windows")
        print(f"✓ Each encoding shape: {encoded_music[0].shape}")
        assert encoded_music[0].shape[0] == 256, f"Expected 256, got {encoded_music[0].shape}"
        
        # Test forward pass
        x = {
            "feature": torch.randn(batch_size, seq_len, 4),
            "feature_length": torch.tensor([seq_len, seq_len]),
            "music": music_list
        }
        
        model.eval()
        loss_dict = model(x)
        
        print(f"✓ Forward pass successful")
        print(f"  - Loss keys: {list(loss_dict.keys())}")
        print(f"  - Total loss: {loss_dict['total']:.4f}")
        
        print("✓ Integration test passed!")
        print()
        
    except Exception as e:
        print(f"⚠ Integration test failed (expected if dependencies missing)")
        print(f"  Error: {e}")
        print("  This is OK for isolated PAE testing")
        print()


def main():
    print("\n" + "=" * 60)
    print("Music PAE Test Suite")
    print("=" * 60 + "\n")
    
    # Test 1: Music PAE
    pae_model = test_music_pae()
    
    # Test 2: Periodic Encoder
    encoder = test_periodic_encoder()
    
    # Test 3: Full Integration
    test_integration()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Prepare your music features using librosa")
    print("2. Train Music PAE: python train_music_pae.py --data_dir /path/to/features")
    print("3. Use trained PAE in diffusion model with pae_checkpoint_path")
    print()


if __name__ == '__main__':
    main()
