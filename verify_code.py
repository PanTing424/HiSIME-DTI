"""
代码验证脚本

检查自适应训练代码的正确性
"""

import sys
import os

def check_imports():
    """检查所有必要的导入"""
    print("="*80)
    print("检查导入...")
    print("="*80)

    try:
        from configs import get_cfg_defaults
        print("✓ configs.py 导入成功")

        cfg = get_cfg_defaults()
        assert hasattr(cfg, 'ADAPTIVE'), "配置中缺少ADAPTIVE节"
        assert hasattr(cfg.ADAPTIVE, 'ENABLE'), "ADAPTIVE配置缺少ENABLE参数"
        assert hasattr(cfg.ADAPTIVE, 'USE_CROSSMAMBA'), "ADAPTIVE配置缺少USE_CROSSMAMBA参数"
        assert hasattr(cfg.ADAPTIVE, 'SELECTION_METRIC'), "ADAPTIVE配置缺少SELECTION_METRIC参数"
        print("✓ ADAPTIVE配置正确")

    except Exception as e:
        print(f"✗ configs.py 导入失败: {e}")
        return False

    try:
        from adaptive_trainer import AdaptiveModelSelector, run_adaptive_training
        print("✓ adaptive_trainer.py 导入成功")
    except Exception as e:
        print(f"✗ adaptive_trainer.py 导入失败: {e}")
        return False

    try:
        from models import GraphBAN
        print("✓ models.py 导入成功")
    except Exception as e:
        print(f"✗ models.py 导入失败: {e}")
        return False

    print("\n所有导入检查通过！\n")
    return True


def check_config_structure():
    """检查配置结构"""
    print("="*80)
    print("检查配置结构...")
    print("="*80)

    from configs import get_cfg_defaults

    cfg = get_cfg_defaults()

    # 检查ADAPTIVE配置
    print("\nADAPTIVE配置:")
    print(f"  ENABLE: {cfg.ADAPTIVE.ENABLE}")
    print(f"  USE_CROSSMAMBA: {cfg.ADAPTIVE.USE_CROSSMAMBA}")
    print(f"  USE_3D_FEATURES: {cfg.ADAPTIVE.USE_3D_FEATURES}")
    print(f"  EARLY_STOP_EPOCHS: {cfg.ADAPTIVE.EARLY_STOP_EPOCHS}")
    print(f"  SELECTION_METRIC: {cfg.ADAPTIVE.SELECTION_METRIC}")

    # 检查BCN配置
    print("\nBCN配置:")
    print(f"  TYPE: {cfg.BCN.TYPE}")
    print(f"  HEADS: {cfg.BCN.HEADS}")

    print("\n✓ 配置结构正确\n")
    return True


def check_model_initialization():
    """检查模型初始化"""
    print("="*80)
    print("检查模型初始化...")
    print("="*80)

    import torch
    from models import GraphBAN
    from configs import get_cfg_defaults

    device = torch.device('cpu')

    # 测试BAN配置
    print("\n测试配置1: BAN + 3D特征")
    cfg_ban = get_cfg_defaults()
    cfg_ban.defrost()
    cfg_ban.BCN.TYPE = "ban"
    cfg_ban.freeze()

    try:
        model_ban = GraphBAN(**cfg_ban).to(device)
        print(f"✓ BAN模型初始化成功")
        print(f"  交互层类型: {model_ban.inter_type}")
    except Exception as e:
        print(f"✗ BAN模型初始化失败: {e}")
        return False

    # 测试CrossMamba配置
    print("\n测试配置2: CrossMamba + 3D特征")
    cfg_cm = get_cfg_defaults()
    cfg_cm.defrost()
    cfg_cm.BCN.TYPE = "cross_mamba"
    cfg_cm.freeze()

    try:
        model_cm = GraphBAN(**cfg_cm).to(device)
        print(f"✓ CrossMamba模型初始化成功")
        print(f"  交互层类型: {model_cm.inter_type}")
    except Exception as e:
        print(f"✗ CrossMamba模型初始化失败: {e}")
        return False

    print("\n✓ 所有模型配置初始化成功\n")
    return True


def check_adaptive_selector():
    """检查自适应选择器"""
    print("="*80)
    print("检查自适应选择器...")
    print("="*80)

    from adaptive_trainer import AdaptiveModelSelector
    from configs import get_cfg_defaults
    import torch

    cfg = get_cfg_defaults()
    cfg.defrost()
    cfg.ADAPTIVE.ENABLE = True
    cfg.freeze()

    device = torch.device('cpu')

    # 创建虚拟的dataloader（仅用于测试初始化）
    class DummyDataLoader:
        def __len__(self):
            return 10

    train_loader = DummyDataLoader()
    val_loader = DummyDataLoader()
    test_loader = DummyDataLoader()

    try:
        selector = AdaptiveModelSelector(
            device, train_loader, val_loader, test_loader,
            cfg, opt_da=None, discriminator=None, experiment=None
        )
        print("✓ AdaptiveModelSelector初始化成功")

        # 检查配置准备
        configs = selector.prepare_configs()
        print(f"✓ 准备了 {len(configs)} 个配置")

        for i, config_dict in enumerate(configs):
            print(f"\n  配置 {i+1}:")
            print(f"    名称: {config_dict['name']}")
            print(f"    描述: {config_dict['description']}")
            print(f"    BCN类型: {config_dict['config'].BCN.TYPE}")

    except Exception as e:
        print(f"✗ AdaptiveModelSelector初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✓ 自适应选择器检查通过\n")
    return True


def check_3d_features_consistency():
    """检查3D特征代码一致性"""
    print("="*80)
    print("检查3D特征代码一致性...")
    print("="*80)

    from models import MolecularGCN, RBF, EdgeWeightedGCNLayer
    import torch

    # 检查RBF
    try:
        import numpy as np
        centers = np.arange(0, 3, 0.1)
        rbf = RBF(centers, gamma=10.0)
        test_input = torch.tensor([1.5])
        output = rbf(test_input)
        print(f"✓ RBF编码器工作正常，输出维度: {output.shape}")
    except Exception as e:
        print(f"✗ RBF编码器失败: {e}")
        return False

    # 检查EdgeWeightedGCNLayer
    try:
        layer = EdgeWeightedGCNLayer(in_dim=128, out_dim=128, edge_dim=30)
        print(f"✓ EdgeWeightedGCNLayer初始化成功")
    except Exception as e:
        print(f"✗ EdgeWeightedGCNLayer初始化失败: {e}")
        return False

    # 检查MolecularGCN（3D版本）
    try:
        gcn_3d = MolecularGCN(
            in_feats=75,
            dim_embedding=128,
            padding=True,
            hidden_feats=[128, 128, 128],
            max_nodes=290,
            use_edge_features=True
        )
        print(f"✓ MolecularGCN（3D版本）初始化成功")
    except Exception as e:
        print(f"✗ MolecularGCN（3D版本）初始化失败: {e}")
        return False

    print("\n✓ 3D特征代码一致性检查通过\n")
    return True


def main():
    """主函数"""
    print("\n" + "#"*80)
    print("# 自适应训练代码验证")
    print("#"*80 + "\n")

    all_passed = True

    # 检查1: 导入
    if not check_imports():
        all_passed = False
        print("⚠️  导入检查失败，后续检查可能无法进行\n")
        return

    # 检查2: 配置结构
    if not check_config_structure():
        all_passed = False

    # 检查3: 模型初始化
    if not check_model_initialization():
        all_passed = False

    # 检查4: 自适应选择器
    if not check_adaptive_selector():
        all_passed = False

    # 检查5: 3D特征一致性
    if not check_3d_features_consistency():
        all_passed = False

    # 总结
    print("\n" + "#"*80)
    if all_passed:
        print("# ✓ 所有检查通过！代码可以使用。")
    else:
        print("# ✗ 部分检查失败，请修复后再使用。")
    print("#"*80 + "\n")


if __name__ == "__main__":
    main()
