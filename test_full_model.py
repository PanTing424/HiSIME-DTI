"""
完整模型测试脚本

验证：
1. GraphBAN 模型能否正确初始化
2. 注意力瓶颈融合是否正常工作
3. 两个策略（3D+BAN 和 CrossMamba+3D）能否运行
4. 前向传播和反向传播是否正常
"""

import torch
import numpy as np
from configs import get_cfg_defaults
from models import GraphBAN


def create_dummy_batch(batch_size=4):
    """创建模拟的一个batch数据"""
    import dgl

    # 1. 创建2D分子图 (DGL格式)
    num_nodes_2d = 20
    num_edges = 30

    # 随机生成图结构
    src = np.random.randint(0, num_nodes_2d, num_edges)
    dst = np.random.randint(0, num_nodes_2d, num_edges)

    graphs_2d = []
    for _ in range(batch_size):
        g = dgl.graph((src, dst), num_nodes=num_nodes_2d)
        g = dgl.add_self_loop(g)  # 添加自环，避免0入度节点
        g.ndata['h'] = torch.randn(g.number_of_nodes(), 75)  # 节点特征
        graphs_2d.append(g)

    bg_d = dgl.batch(graphs_2d)

    # 2. 创建3D分子图 (带边特征)
    graphs_3d = []
    for _ in range(batch_size):
        g = dgl.graph((src, dst), num_nodes=num_nodes_2d)
        g.ndata['h'] = torch.randn(num_nodes_2d, 75)  # 节点特征
        # 设置边长数据 (键名必须是 'bond_length')
        g.edata['bond_length'] = torch.rand(g.number_of_edges()) * 3.0  # 边特征（键长，0-3埃）
        # 添加自环，自环边长设为0
        g = dgl.add_self_loop(g)
        # 为自环边添加 bond_length（设为0）
        num_self_loops = num_nodes_2d
        self_loop_lengths = torch.zeros(num_self_loops)
        g.edata['bond_length'] = torch.cat([g.edata['bond_length'][:num_edges], self_loop_lengths], dim=0)
        graphs_3d.append(g)

    bg_d_3d = dgl.batch(graphs_3d)

    # 3. SMILES特征 (ChemBERTa输出是384维)
    bg_smiles = torch.randn(batch_size, 384)

    # 4. 蛋白质特征
    protein_len = 1000
    v_p = torch.randint(0, 26, (batch_size, protein_len))
    v_p_esm = torch.randn(batch_size, 1, 1280)

    # 5. 标签
    labels = torch.randint(0, 2, (batch_size,)).float()

    return bg_d, bg_d_3d, bg_smiles, v_p, v_p_esm, labels


def test_model_initialization():
    """测试1：模型初始化"""
    print("=" * 80)
    print("测试1: 模型初始化（注意力瓶颈融合）")
    print("=" * 80)

    cfg = get_cfg_defaults()
    cfg.defrost()
    cfg.FUSION.TYPE = "bottleneck"
    cfg.FUSION.NUM_LATENTS = 4
    cfg.BCN.TYPE = "ban"
    cfg.freeze()

    print(f"\n融合配置:")
    print(f"  - 类型: {cfg.FUSION.TYPE}")
    print(f"  - 瓶颈数量: {cfg.FUSION.NUM_LATENTS}")
    print(f"  - Dropout: {cfg.FUSION.DROPOUT}")

    model = GraphBAN(**cfg)

    print(f"\n✓ 模型初始化成功")
    print(f"  - 融合模块类型: {type(model.mol_fusion).__name__}")
    print(f"  - 交互层类型: {model.inter_type}")

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    fusion_params = sum(p.numel() for p in model.mol_fusion.parameters())

    print(f"\n参数统计:")
    print(f"  - 总参数量: {total_params:,}")
    print(f"  - 融合模块参数: {fusion_params:,} ({fusion_params/total_params*100:.2f}%)")

    return model, cfg


def test_forward_pass(model, device='cpu'):
    """测试2：前向传播"""
    print("\n" + "=" * 80)
    print("测试2: 前向传播")
    print("=" * 80)

    model = model.to(device)
    model.eval()

    # 创建模拟数据
    bg_d, bg_d_3d, bg_smiles, v_p, v_p_esm, labels = create_dummy_batch(batch_size=4)

    print(f"\n输入数据:")
    print(f"  - 2D图: {bg_d.number_of_nodes()} 节点, {bg_d.number_of_edges()} 边")
    print(f"  - 3D图: {bg_d_3d.number_of_nodes()} 节点, {bg_d_3d.number_of_edges()} 边")
    print(f"  - SMILES: {bg_smiles.shape}")
    print(f"  - 蛋白质: {v_p.shape}")
    print(f"  - 标签: {labels.shape}")

    # 前向传播
    with torch.no_grad():
        v_fusion, v_p_fusion, score, f = model(
            bg_d, bg_d_3d, bg_smiles, v_p, v_p_esm,
            device=device, mode="eval"
        )

    print(f"\n输出:")
    print(f"  - 融合药物特征: {v_fusion.shape}")
    print(f"  - 融合蛋白特征: {v_p_fusion.shape}")
    print(f"  - 预测分数: {score.shape}")
    print(f"  - 交互特征: {f.shape}")

    print(f"\n预测分数统计:")
    print(f"  - 均值: {score.mean().item():.6f}")
    print(f"  - 标准差: {score.std().item():.6f}")
    print(f"  - 最小值: {score.min().item():.6f}")
    print(f"  - 最大值: {score.max().item():.6f}")

    print(f"\n✓ 前向传播成功")

    return v_fusion, v_p_fusion, score, f


def test_backward_pass(model, device='cpu'):
    """测试3：反向传播"""
    print("\n" + "=" * 80)
    print("测试3: 反向传播（梯度检查）")
    print("=" * 80)

    model = model.to(device)
    model.train()

    # 创建模拟数据
    bg_d, bg_d_3d, bg_smiles, v_p, v_p_esm, labels = create_dummy_batch(batch_size=4)

    # 前向传播
    v_fusion, v_p_fusion, f, score = model(
        bg_d, bg_d_3d, bg_smiles, v_p, v_p_esm,
        device=device, mode="train"
    )

    # 计算损失
    from models import binary_cross_entropy, cross_entropy_logits
    if score.shape[1] == 1:
        n, loss = binary_cross_entropy(score, labels)
    else:
        n, loss = cross_entropy_logits(score, labels.long())

    print(f"\n损失:")
    print(f"  - 值: {loss.item():.6f}")

    # 反向传播
    loss.backward()

    # 检查梯度
    print(f"\n梯度检查:")

    # 检查融合模块的梯度
    fusion_module = model.mol_fusion
    print(f"  - Latent tokens 梯度: {fusion_module.latents.grad is not None}")
    print(f"  - scale_2d 梯度: {fusion_module.scale_2d.grad is not None} "
          f"(值: {fusion_module.scale_2d.grad.item():.6f})")
    print(f"  - scale_3d 梯度: {fusion_module.scale_3d.grad is not None} "
          f"(值: {fusion_module.scale_3d.grad.item():.6f})")
    print(f"  - scale_smiles 梯度: {fusion_module.scale_smiles.grad is not None} "
          f"(值: {fusion_module.scale_smiles.grad.item():.6f})")

    # 检查其他模块的梯度
    has_grad_count = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += 1
        if param.grad is not None:
            has_grad_count += 1

    print(f"\n参数梯度统计:")
    print(f"  - 总参数: {total_params}")
    print(f"  - 有梯度的参数: {has_grad_count}")
    print(f"  - 覆盖率: {has_grad_count/total_params*100:.1f}%")

    print(f"\n✓ 反向传播成功")


def test_two_strategies():
    """测试4：两个策略"""
    print("\n" + "=" * 80)
    print("测试4: 双策略（3D+BAN 和 CrossMamba+3D）")
    print("=" * 80)

    cfg = get_cfg_defaults()
    cfg.defrost()
    cfg.FUSION.TYPE = "bottleneck"
    cfg.FUSION.NUM_LATENTS = 4

    strategies = [
        ("3D + BAN", "ban"),
        ("CrossMamba + 3D", "cross_mamba")
    ]

    for strategy_name, bcn_type in strategies:
        print(f"\n{'-'*80}")
        print(f"策略: {strategy_name}")
        print(f"{'-'*80}")

        cfg.BCN.TYPE = bcn_type
        cfg.freeze()

        model = GraphBAN(**cfg)
        model.eval()

        print(f"  - 交互层类型: {model.inter_type}")
        print(f"  - 融合模块: {type(model.mol_fusion).__name__}")

        # 前向传播
        bg_d, bg_d_3d, bg_smiles, v_p, v_p_esm, labels = create_dummy_batch(batch_size=2)

        with torch.no_grad():
            v_fusion, v_p_fusion, score, f = model(
                bg_d, bg_d_3d, bg_smiles, v_p, v_p_esm,
                device='cpu', mode="eval"
            )

        print(f"  - 预测分数: {score.shape}, 均值={score.mean().item():.4f}")
        print(f"  ✓ {strategy_name} 运行成功")

        cfg.defrost()

    print(f"\n✓ 双策略测试成功")


def test_different_num_latents():
    """测试5：不同的瓶颈数量"""
    print("\n" + "=" * 80)
    print("测试5: 不同瓶颈数量（num_latents）")
    print("=" * 80)

    cfg = get_cfg_defaults()
    cfg.defrost()
    cfg.FUSION.TYPE = "bottleneck"
    cfg.BCN.TYPE = "ban"

    latent_counts = [2, 4, 8, 16]

    for num_lat in latent_counts:
        cfg.FUSION.NUM_LATENTS = num_lat
        cfg.freeze()

        model = GraphBAN(**cfg)
        fusion_params = sum(p.numel() for p in model.mol_fusion.parameters())

        # 前向传播
        bg_d, bg_d_3d, bg_smiles, v_p, v_p_esm, _ = create_dummy_batch(batch_size=2)

        model.eval()
        with torch.no_grad():
            v_fusion, v_p_fusion, score, f = model(
                bg_d, bg_d_3d, bg_smiles, v_p, v_p_esm,
                device='cpu', mode="eval"
            )

        print(f"\n  num_latents = {num_lat}:")
        print(f"    - 融合模块参数: {fusion_params:,}")
        print(f"    - 预测分数均值: {score.mean().item():.6f}")
        print(f"    ✓ 运行成功")

        cfg.defrost()

    print(f"\n✓ 不同瓶颈数量测试成功")


def main():
    """主测试函数"""
    print("\n" + "=" * 80)
    print("GraphBAN - 注意力瓶颈融合 - 完整模型测试")
    print("=" * 80)

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 测试1：模型初始化
    model, cfg = test_model_initialization()

    # 测试2：前向传播
    test_forward_pass(model, device='cpu')

    # 测试3：反向传播
    test_backward_pass(model, device='cpu')

    # 测试4：双策略
    test_two_strategies()

    # 测试5：不同瓶颈数量
    test_different_num_latents()

    # 总结
    print("\n" + "=" * 80)
    print("✅ 所有测试通过！模型工作正常，可以开始训练。")
    print("=" * 80)
    print("\n下一步：")
    print("  1. 运行训练: bash run_bottleneck_training.sh BINDINGDB 12")
    print("  2. 或使用: python run_model_adaptive.py --cfg GraphBAN.yaml --data BINDINGDB")
    print()


if __name__ == "__main__":
    main()
