"""
单元测试：验证phase manifold是否产生同心圆

测试步骤：
1. 生成理想的相位-幅度数据，验证是否产生完美同心圆
2. 测试PCA是否破坏同心圆结构
3. 检查PAE实际提取的phase和amplitude值
4. 分析为什么可视化结果不是同心圆
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path


def test_1_ideal_concentric_circles():
    """
    测试1: 理想的相位-幅度嵌入是否产生同心圆
    
    如果这个测试失败，说明embedding公式本身有问题
    """
    print("\n" + "="*60)
    print("测试1: 理想相位-幅度嵌入 → 同心圆")
    print("="*60)
    
    ## 生成理想数据：5条轨迹，不同半径 ##
    num_samples_per_circle = 100
    radii = [0.2, 0.4, 0.6, 0.8, 1.0]  ## 5个不同半径
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    all_embeddings = []
    all_colors = []
    
    for i, radius in enumerate(radii):
        ## 生成完整的圆：phase从0到1 ##
        phases = np.linspace(0, 1, num_samples_per_circle, endpoint=False)
        amplitudes = np.ones(num_samples_per_circle) * radius
        
        ## Phase manifold embedding: [A*sin(2πφ), A*cos(2πφ)] ##
        embedding = np.column_stack([
            amplitudes * np.sin(2 * np.pi * phases),
            amplitudes * np.cos(2 * np.pi * phases)
        ])
        
        all_embeddings.append(embedding)
        all_colors.extend([colors[i]] * num_samples_per_circle)
        
        ## 画原始的2D embedding ##
        axes[0].scatter(embedding[:, 0], embedding[:, 1], 
                       c=colors[i], alpha=0.6, label=f'r={radius}')
    
    axes[0].set_title("(a) 原始2D Embedding\n应该是完美同心圆")
    axes[0].legend()
    axes[0].axis('equal')
    axes[0].grid(True)
    
    ## 现在扩展到高维（模拟8个embedding channels = 16维） ##
    print("\n测试: 扩展到16维...")
    all_embeddings_16d = []
    for i, radius in enumerate(radii):
        phases = np.linspace(0, 1, num_samples_per_circle, endpoint=False)
        
        ## 8个channels，每个都用相同的phase但可能有不同的amplitude ##
        embedding_16d = []
        for ch in range(8):
            amp = radius * (1.0 + 0.1 * np.sin(ch))  ## 每个channel略微不同的幅度
            embedding_16d.append(amp * np.sin(2 * np.pi * phases))
            embedding_16d.append(amp * np.cos(2 * np.pi * phases))
        
        embedding_16d = np.column_stack(embedding_16d)  ## (100, 16)
        all_embeddings_16d.append(embedding_16d)
    
    ## 拼接所有数据 ##
    all_data_16d = np.vstack(all_embeddings_16d)  ## (500, 16)
    print(f"16D数据形状: {all_data_16d.shape}")
    
    ## 用PCA降到2D ##
    pca = PCA(n_components=2)
    all_data_2d = pca.fit_transform(all_data_16d)
    print(f"PCA解释方差比: {pca.explained_variance_ratio_}")
    
    ## 分割回每个圆 ##
    start = 0
    for i, radius in enumerate(radii):
        end = start + num_samples_per_circle
        data_2d = all_data_2d[start:end]
        axes[1].scatter(data_2d[:, 0], data_2d[:, 1], 
                       c=colors[i], alpha=0.6, label=f'r={radius}')
        start = end
    
    axes[1].set_title("(b) PCA降维到2D\n应该保持同心圆结构")
    axes[1].legend()
    axes[1].axis('equal')
    axes[1].grid(True)
    
    ## 测试：没有PCA，直接取前2维 ##
    for i, radius in enumerate(radii):
        embedding = all_embeddings_16d[i][:, :2]  ## 只取前2维
        axes[2].scatter(embedding[:, 0], embedding[:, 1], 
                       c=colors[i], alpha=0.6, label=f'r={radius}')
    
    axes[2].set_title("(c) 直接取前2维\n应该也是完美同心圆")
    axes[2].legend()
    axes[2].axis('equal')
    axes[2].grid(True)
    
    plt.tight_layout()
    output_path = Path("tmp/manifolds_test/test1_ideal_circles.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 测试1完成，保存到: {output_path}")
    print("结论: 如果(a)(b)(c)都是同心圆，说明公式和PCA都没问题")


def test_2_varying_phase_same_amplitude():
    """
    测试2: 相同幅度，不同相位变化速度
    
    模拟实际音频：每帧phase可能变化不均匀
    """
    print("\n" + "="*60)
    print("测试2: 相同幅度，相位非线性变化")
    print("="*60)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    num_frames = 200
    radius = 1.0
    
    ## 情况1: 线性相位变化（完美圆） ##
    phases_linear = np.linspace(0, 3, num_frames)  ## 3圈
    embedding_linear = np.column_stack([
        radius * np.sin(2 * np.pi * phases_linear),
        radius * np.cos(2 * np.pi * phases_linear)
    ])
    
    axes[0].scatter(embedding_linear[:, 0], embedding_linear[:, 1], 
                   c=np.arange(num_frames), cmap='viridis', alpha=0.6)
    axes[0].set_title("线性相位变化（完美圆）")
    axes[0].axis('equal')
    axes[0].grid(True)
    
    ## 情况2: 非线性相位变化（不均匀的圆） ##
    phases_nonlinear = np.cumsum(np.random.uniform(0.8, 1.2, num_frames))
    phases_nonlinear = phases_nonlinear / phases_nonlinear[-1] * 3  ## 归一化到3圈
    
    embedding_nonlinear = np.column_stack([
        radius * np.sin(2 * np.pi * phases_nonlinear),
        radius * np.cos(2 * np.pi * phases_nonlinear)
    ])
    
    axes[1].scatter(embedding_nonlinear[:, 0], embedding_nonlinear[:, 1], 
                   c=np.arange(num_frames), cmap='viridis', alpha=0.6)
    axes[1].set_title("非线性相位变化（仍是圆）")
    axes[1].axis('equal')
    axes[1].grid(True)
    
    plt.tight_layout()
    output_path = Path("tmp/manifolds_test/test2_phase_variation.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 测试2完成，保存到: {output_path}")
    print("结论: 即使相位变化非线性，只要幅度固定，仍然是圆")


def test_3_varying_amplitude_linear_phase():
    """
    测试3: 线性相位，幅度随时间变化
    
    这是关键：如果幅度变化，会产生螺旋而不是圆
    """
    print("\n" + "="*60)
    print("测试3: 线性相位，幅度变化 → 螺旋")
    print("="*60)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    num_frames = 200
    phases = np.linspace(0, 3, num_frames)  ## 3圈
    
    ## 情况1: 幅度线性增长 ##
    amplitudes_linear = np.linspace(0.2, 1.0, num_frames)
    embedding_linear = np.column_stack([
        amplitudes_linear * np.sin(2 * np.pi * phases),
        amplitudes_linear * np.cos(2 * np.pi * phases)
    ])
    
    axes[0].scatter(embedding_linear[:, 0], embedding_linear[:, 1], 
                   c=np.arange(num_frames), cmap='viridis', alpha=0.6)
    axes[0].set_title("幅度线性增长 → 螺旋")
    axes[0].axis('equal')
    axes[0].grid(True)
    
    ## 情况2: 幅度正弦波动 ##
    amplitudes_sine = 0.6 + 0.4 * np.sin(2 * np.pi * phases / 3)
    embedding_sine = np.column_stack([
        amplitudes_sine * np.sin(2 * np.pi * phases),
        amplitudes_sine * np.cos(2 * np.pi * phases)
    ])
    
    axes[1].scatter(embedding_sine[:, 0], embedding_sine[:, 1], 
                   c=np.arange(num_frames), cmap='viridis', alpha=0.6)
    axes[1].set_title("幅度正弦波动 → 花瓣形")
    axes[1].axis('equal')
    axes[1].grid(True)
    
    ## 情况3: 幅度随机波动 ##
    amplitudes_random = 0.5 + 0.5 * np.random.uniform(0, 1, num_frames)
    embedding_random = np.column_stack([
        amplitudes_random * np.sin(2 * np.pi * phases),
        amplitudes_random * np.cos(2 * np.pi * phases)
    ])
    
    axes[2].scatter(embedding_random[:, 0], embedding_random[:, 1], 
                   c=np.arange(num_frames), cmap='viridis', alpha=0.6)
    axes[2].set_title("幅度随机波动 → 模糊环")
    axes[2].axis('equal')
    axes[2].grid(True)
    
    plt.tight_layout()
    output_path = Path("tmp/manifolds_test/test3_amplitude_variation.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 测试3完成，保存到: {output_path}")
    print("结论: 幅度变化会破坏同心圆结构！")


def test_4_check_pae_actual_values():
    """
    测试4: 检查PAE实际提取的phase和amplitude值
    
    从一个真实音频文件提取，看看值的分布
    """
    print("\n" + "="*60)
    print("测试4: 检查PAE实际提取的值")
    print("="*60)
    
    import sys
    from pathlib import Path
    import librosa
    
    ## 添加路径
    pae_base = Path("PAE_training/DeepPhase4Audio-main")
    if str(pae_base) not in sys.path:
        sys.path.insert(0, str(pae_base))
    
    try:
        from src.models.PAE import PAEInputFlattened
        from src.utils.config import load_config
        from src.utils.helpers import resolve_model_class
        
        ## 加载模型
        model_path = "PAE_training/DeepPhase4Audio-main/output/20260103_213749_paeflat_twinkle_200eps/checkpoints/step_epoch=1999.ckpt"
        config_path = "PAE_training/DeepPhase4Audio-main/configs/paeflat200.yaml"
        
        cfg = load_config(config_path, flatten=False)
        checkpoint = torch.load(model_path, map_location='cpu')
        
        model = resolve_model_class(cfg.model_config.model_name, cfg.model_config)
        state_dict = {}
        for key, val in checkpoint['state_dict'].items():
            if key.startswith('model.'):
                state_dict[key[6:]] = val
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        print(f"✅ 模型加载成功，embedding_channels={cfg.model_config.embedding_channels}")
        
        ## 加载音频 - use existing file ##
        audio_path = "PAE_training/DeepPhase4Audio-main/plotting/wav/0.wav"
        y, sr = librosa.load(audio_path, sr=16000)
        
        ## 处理前50个窗口
        window_size = cfg.model_config.time_range
        hop_size = 1600
        num_windows = 50
        
        all_phases = []
        all_amplitudes = []
        
        with torch.no_grad():
            for i in range(0, min(num_windows * hop_size, len(y) - window_size), hop_size):
                chunk = y[i:i + window_size]
                chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0).unsqueeze(0)
                
                pred, latent, signal, param = model(chunk_tensor)
                
                if param is not None and isinstance(param, list) and len(param) == 4:
                    p, f, a, b = param
                    phase = (p.squeeze(2) + 0.5) % 1.0  ## (1, 8)
                    amplitude = torch.clamp(a.squeeze(2), min=0.0)  ## (1, 8)
                    
                    all_phases.append(phase.cpu().numpy())
                    all_amplitudes.append(amplitude.cpu().numpy())
        
        all_phases = np.vstack(all_phases)  ## (num_windows, 8)
        all_amplitudes = np.vstack(all_amplitudes)  ## (num_windows, 8)
        
        print(f"\n提取了 {len(all_phases)} 个窗口")
        print(f"Phase shape: {all_phases.shape}")
        print(f"Amplitude shape: {all_amplitudes.shape}")
        
        ## 分析统计
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        ## (a) Phase分布（每个channel）
        for ch in range(8):
            axes[0, 0].plot(all_phases[:, ch], alpha=0.6, label=f'ch{ch}')
        axes[0, 0].set_title("(a) Phase值随时间变化（每个channel）")
        axes[0, 0].set_xlabel("Frame")
        axes[0, 0].set_ylabel("Phase [0, 1]")
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        ## (b) Amplitude分布（每个channel）
        for ch in range(8):
            axes[0, 1].plot(all_amplitudes[:, ch], alpha=0.6, label=f'ch{ch}')
        axes[0, 1].set_title("(b) Amplitude值随时间变化（每个channel）")
        axes[0, 1].set_xlabel("Frame")
        axes[0, 1].set_ylabel("Amplitude")
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        ## (c) Phase直方图
        axes[1, 0].hist(all_phases.flatten(), bins=50, alpha=0.7)
        axes[1, 0].set_title(f"(c) Phase分布直方图\nmean={all_phases.mean():.3f}, std={all_phases.std():.3f}")
        axes[1, 0].set_xlabel("Phase")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].grid(True)
        
        ## (d) Amplitude直方图
        axes[1, 1].hist(all_amplitudes.flatten(), bins=50, alpha=0.7)
        axes[1, 1].set_title(f"(d) Amplitude分布直方图\nmean={all_amplitudes.mean():.3f}, std={all_amplitudes.std():.3f}")
        axes[1, 1].set_xlabel("Amplitude")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        output_path = Path("tmp/manifolds_test/test4_pae_actual_values.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ 测试4完成，保存到: {output_path}")
        
        ## 关键分析
        print("\n" + "="*60)
        print("关键分析:")
        print("="*60)
        print(f"1. Phase范围: [{all_phases.min():.3f}, {all_phases.max():.3f}]")
        print(f"   - 是否覆盖[0,1]? {all_phases.max() - all_phases.min() > 0.8}")
        print(f"2. Amplitude范围: [{all_amplitudes.min():.3f}, {all_amplitudes.max():.3f}]")
        print(f"   - 是否变化很大? 变化系数={all_amplitudes.std()/all_amplitudes.mean():.3f}")
        print(f"3. 如果amplitude变化系数>0.3，说明幅度波动大，会产生螺旋而非圆")
        
        ## 绘制第一个channel的phase manifold
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        ch = 0  ## 只看第一个channel
        embedding = np.column_stack([
            all_amplitudes[:, ch] * np.sin(2 * np.pi * all_phases[:, ch]),
            all_amplitudes[:, ch] * np.cos(2 * np.pi * all_phases[:, ch])
        ])
        
        axes[0].scatter(embedding[:, 0], embedding[:, 1], 
                       c=np.arange(len(embedding)), cmap='viridis', alpha=0.6)
        axes[0].set_title(f"Channel {ch} Phase Manifold (2D)")
        axes[0].axis('equal')
        axes[0].grid(True)
        
        ## 16维embedding的前2维
        embedding_16d = []
        for ch in range(8):
            embedding_16d.append(all_amplitudes[:, ch] * np.sin(2 * np.pi * all_phases[:, ch]))
            embedding_16d.append(all_amplitudes[:, ch] * np.cos(2 * np.pi * all_phases[:, ch]))
        embedding_16d = np.column_stack(embedding_16d)  ## (num_windows, 16)
        
        axes[1].scatter(embedding_16d[:, 0], embedding_16d[:, 1], 
                       c=np.arange(len(embedding_16d)), cmap='viridis', alpha=0.6)
        axes[1].set_title("16D Embedding前2维")
        axes[1].axis('equal')
        axes[1].grid(True)
        
        plt.tight_layout()
        output_path = Path("tmp/manifolds_test/test4_pae_manifold.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Phase manifold可视化保存到: {output_path}")
        
    except Exception as e:
        print(f"❌ 测试4失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("\n" + "="*80)
    print("单元测试：验证Phase Manifold是否能产生同心圆")
    print("="*80)
    
    test_1_ideal_concentric_circles()
    test_2_varying_phase_same_amplitude()
    test_3_varying_amplitude_linear_phase()
    test_4_check_pae_actual_values()
    
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    print("1. 测试1: 验证理想情况下embedding公式和PCA是否保持同心圆")
    print("2. 测试2: 验证相位非线性变化不会破坏圆形结构")
    print("3. 测试3: 证明幅度变化会产生螺旋/花瓣，破坏同心圆")
    print("4. 测试4: 检查PAE实际提取的phase和amplitude值")
    print("\n关键问题: 如果PAE的amplitude变化很大，就不会是同心圆！")
    print("="*80)


if __name__ == "__main__":
    main()
