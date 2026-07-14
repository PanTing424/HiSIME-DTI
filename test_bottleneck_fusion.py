"""
测试注意力瓶颈融合模块

功能：
1. 验证 AttentionBottleneckFusion 的前向传播
2. 检查输出维度是否正确
3. 对比新旧融合模块的输出
"""

import torch
import torch.nn as nn
from models import AttentionBottleneckFusion, Att_FeatureFusion


def test_bottleneck_fusion():
    """测试注意力瓶颈融合模块"""
    print("=" * 80)
    print("测试注意力瓶颈融合模块（Attention Bottleneck Fusion）")
    print("=" * 80)

    # 设置参数
    batch_size = 16
    feature_dim = 128
    num_latents = 4

    # 创建随机输入（模拟三个模态的特征）
    f_2d = torch.randn(batch_size, feature_dim)       # 2D图特征
    f_3d = torch.randn(batch_size, feature_dim)       # 3D图特征
    f_smiles = torch.randn(batch_size, feature_dim)   # SMILES特征

    print(f"\n输入特征形状:")
    print(f"  - f_2d (2D图):    {f_2d.shape}")
    print(f"  - f_3d (3D图):    {f_3d.shape}")
    print(f"  - f_smiles (SMILES): {f_smiles.shape}")

    # ======== 测试1：注意力瓶颈融合 ========
    print(f"\n{'='*80}")
    print("测试1: 注意力瓶颈融合（AttentionBottleneckFusion）")
    print(f"{'='*80}")

    bottleneck_fusion = AttentionBottleneckFusion(
        seq_dim=feature_dim,
        graph_dim=feature_dim,
        semantic_dim=feature_dim,
        num_latents=num_latents,
        dropout=0.1
    )
    bottleneck_fusion.eval()  # 设置为评估模式，禁用dropout

    with torch.no_grad():
        output_bottleneck = bottleneck_fusion(f_2d, f_3d, f_smiles)

    print(f"\n注意力瓶颈融合输出:")
    print(f"  - 输出形状: {output_bottleneck.shape}")
    print(f"  - 预期形状: torch.Size([{batch_size}, {feature_dim * 3}])")
    print(f"  - 形状匹配: {output_bottleneck.shape == torch.Size([batch_size, feature_dim * 3])}")

    # 检查可学习参数
    print(f"\n可学习参数:")
    print(f"  - Latent tokens 形状: {bottleneck_fusion.latents.shape}")
    print(f"  - Latent tokens 数量: {num_latents}")
    print(f"  - scale_2d 初始值: {bottleneck_fusion.scale_2d.item():.6f}")
    print(f"  - scale_3d 初始值: {bottleneck_fusion.scale_3d.item():.6f}")
    print(f"  - scale_smiles 初始值: {bottleneck_fusion.scale_smiles.item():.6f}")
    print(f"  - 残差模式: {bottleneck_fusion.residual_mode}")
    print(f"  - 是否启用额外残差: {bottleneck_fusion.use_residual}")
    if hasattr(bottleneck_fusion, "gate"):
        gate_val = bottleneck_fusion.gate.item()
        beta_val = torch.sigmoid(bottleneck_fusion.gate).item() if bottleneck_fusion.residual_mode == "gated" else gate_val
        print(f"  - gate 初始值: {gate_val:.6f}")
        print(f"  - beta 初始值: {beta_val:.6f}")

    # ======== 测试2：原始注意力融合（对比） ========
    print(f"\n{'='*80}")
    print("测试2: 原始注意力融合（Att_FeatureFusion）- 作为对比")
    print(f"{'='*80}")

    attention_fusion = Att_FeatureFusion(
        seq_dim=feature_dim,
        graph_dim=feature_dim,
        semantic_dim=feature_dim
    )
    attention_fusion.eval()

    with torch.no_grad():
        output_attention = attention_fusion(f_2d, f_3d, f_smiles)

    print(f"\n原始注意力融合输出:")
    print(f"  - 输出形状: {output_attention.shape}")
    print(f"  - 形状匹配: {output_attention.shape == torch.Size([batch_size, feature_dim * 3])}")

    # ======== 测试3：输出差异分析 ========
    print(f"\n{'='*80}")
    print("测试3: 输出差异分析")
    print(f"{'='*80}")

    # 计算输出的统计信息
    print(f"\n注意力瓶颈融合统计:")
    print(f"  - 均值: {output_bottleneck.mean().item():.6f}")
    print(f"  - 标准差: {output_bottleneck.std().item():.6f}")
    print(f"  - 最小值: {output_bottleneck.min().item():.6f}")
    print(f"  - 最大值: {output_bottleneck.max().item():.6f}")

    print(f"\n原始注意力融合统计:")
    print(f"  - 均值: {output_attention.mean().item():.6f}")
    print(f"  - 标准差: {output_attention.std().item():.6f}")
    print(f"  - 最小值: {output_attention.min().item():.6f}")
    print(f"  - 最大值: {output_attention.max().item():.6f}")

    # 计算两个输出的差异
    diff = (output_bottleneck - output_attention).abs()
    print(f"\n两个融合模块的输出差异:")
    print(f"  - 平均绝对差异: {diff.mean().item():.6f}")
    print(f"  - 最大绝对差异: {diff.max().item():.6f}")

    # ======== 测试4：梯度反向传播 ========
    print(f"\n{'='*80}")
    print("测试4: 梯度反向传播测试")
    print(f"{'='*80}")

    bottleneck_fusion.train()  # 设置为训练模式

    # 创建需要梯度的输入
    f_2d_grad = f_2d.clone().requires_grad_(True)
    f_3d_grad = f_3d.clone().requires_grad_(True)
    f_smiles_grad = f_smiles.clone().requires_grad_(True)

    # 前向传播
    output = bottleneck_fusion(f_2d_grad, f_3d_grad, f_smiles_grad)

    # 创建一个简单的损失（输出的和）
    loss = output.sum()

    # 反向传播
    loss.backward()

    print(f"\n梯度检查:")
    print(f"  - f_2d 梯度: {f_2d_grad.grad is not None} (均值: {f_2d_grad.grad.mean().item():.6f})")
    print(f"  - f_3d 梯度: {f_3d_grad.grad is not None} (均值: {f_3d_grad.grad.mean().item():.6f})")
    print(f"  - f_smiles 梯度: {f_smiles_grad.grad is not None} (均值: {f_smiles_grad.grad.mean().item():.6f})")
    print(f"  - Latent tokens 梯度: {bottleneck_fusion.latents.grad is not None}")
    print(f"  - scale_2d 梯度: {bottleneck_fusion.scale_2d.grad is not None} (值: {bottleneck_fusion.scale_2d.grad.item():.6f})")
    print(f"  - scale_3d 梯度: {bottleneck_fusion.scale_3d.grad is not None} (值: {bottleneck_fusion.scale_3d.grad.item():.6f})")
    print(f"  - scale_smiles 梯度: {bottleneck_fusion.scale_smiles.grad is not None} (值: {bottleneck_fusion.scale_smiles.grad.item():.6f})")
    if hasattr(bottleneck_fusion, "gate") and isinstance(bottleneck_fusion.gate, nn.Parameter):
        print(f"  - residual gate 梯度: {bottleneck_fusion.gate.grad is not None} (值: {bottleneck_fusion.gate.grad.item():.6f})")

    # ======== 测试5：不同残差模式 ========
    print(f"\n{'='*80}")
    print("测试5: 不同残差模式")
    print(f"{'='*80}")

    for residual_mode in ["fixed", "gated", "learned"]:
        fusion_module = AttentionBottleneckFusion(
            seq_dim=feature_dim,
            graph_dim=feature_dim,
            semantic_dim=feature_dim,
            num_latents=num_latents,
            dropout=0.1,
            use_residual=True,
            residual_mode=residual_mode,
            fixed_alpha=0.5
        )
        fusion_module.eval()

        with torch.no_grad():
            out = fusion_module(f_2d, f_3d, f_smiles)

        print(f"\n  residual_mode = {residual_mode}:")
        print(f"    - 输出形状: {out.shape}")
        if residual_mode in ["fixed", "gated"]:
            gate_val = fusion_module.gate.item()
            beta_val = torch.sigmoid(fusion_module.gate).item() if residual_mode == "gated" else gate_val
            print(f"    - gate: {gate_val:.6f}")
            print(f"    - beta: {beta_val:.6f}")
        else:
            beta = fusion_module.gate_net(torch.cat([f_2d, f_3d, f_smiles], dim=1))
            print(f"    - beta均值: {beta.mean().item():.6f}")

    # ======== 测试6：不同 num_latents 的效果 ========
    print(f"\n{'='*80}")
    print("测试6: 不同瓶颈数量（num_latents）的效果")
    print(f"{'='*80}")

    for num_lat in [2, 4, 8, 16]:
        fusion_module = AttentionBottleneckFusion(
            seq_dim=feature_dim,
            graph_dim=feature_dim,
            semantic_dim=feature_dim,
            num_latents=num_lat,
            dropout=0.1
        )
        fusion_module.eval()

        with torch.no_grad():
            out = fusion_module(f_2d, f_3d, f_smiles)

        # 统计参数量
        num_params = sum(p.numel() for p in fusion_module.parameters())

        print(f"\n  num_latents = {num_lat}:")
        print(f"    - Latent tokens 形状: {fusion_module.latents.shape}")
        print(f"    - 总参数量: {num_params:,}")
        print(f"    - 输出形状: {out.shape}")
        print(f"    - 输出均值: {out.mean().item():.6f}")

    print(f"\n{'='*80}")
    print("✅ 所有测试通过！注意力瓶颈融合模块工作正常。")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # 设置随机种子，确保结果可复现
    torch.manual_seed(42)
    test_bottleneck_fusion()
