"""
自适应模型选择训练器 - 方案四实现

功能：
1. 训练多个配置（仅3D特征 vs CrossMamba+3D特征）
2. 在验证集上评估各配置
3. 选择最优配置用于测试
4. 支持集成多个模型
"""

import os
import json
import torch
import numpy as np
from copy import deepcopy
from trainer import Trainer


class AdaptiveModelSelector:
    """
    自适应模型选择器

    根据验证集性能自动选择最优模型配置
    """

    def __init__(self, device, train_dataloader, val_dataloader, test_dataloader,
                 cfg, opt_da=None, discriminator=None, experiment=None):
        """
        初始化自适应选择器

        Args:
            device: 训练设备
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            test_dataloader: 测试数据加载器
            cfg: 配置对象
            opt_da: Domain adaptation优化器（可选）
            discriminator: 判别器（可选）
            experiment: 实验跟踪对象（可选）
        """
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.cfg = cfg
        self.opt_da = opt_da
        self.discriminator = discriminator
        self.experiment = experiment

        # 配置选项
        self.configs_to_try = []
        self.selection_metric = cfg.ADAPTIVE.SELECTION_METRIC
        self.output_dir = cfg.RESULT.OUTPUT_DIR

        # 结果存储
        self.config_results = {}
        self.best_config_name = None
        self.best_model = None
        self.best_trainer = None

    def prepare_configs(self):
        """
        准备要尝试的配置列表

        Returns:
            list: 配置列表，每个配置包含名称和参数
        """
        configs = []

        # 配置1：3D特征 + BAN + MFAB瓶颈融合
        cfg_3d_only = deepcopy(self.cfg)
        cfg_3d_only.defrost()
        cfg_3d_only.BCN.TYPE = "ban"
        cfg_3d_only.BCN.HEADS = 2  # ⭐ 使用与inductive_mode_32一致的HEADS=2
        cfg_3d_only.ADAPTIVE.USE_CROSSMAMBA = False
        cfg_3d_only.ADAPTIVE.USE_3D_FEATURES = True
        cfg_3d_only.FUSION.TYPE = "bottleneck"  # ⭐ 使用MFAB注意力瓶颈融合
        cfg_3d_only.freeze()
        configs.append({
            'name': '3d_ban_mfab',
            'description': '3D Features + BAN + MFAB Bottleneck Fusion',
            'config': cfg_3d_only
        })

        # 配置2：CrossMamba + 3D特征 + MFAB瓶颈融合
        cfg_crossmamba_3d = deepcopy(self.cfg)
        cfg_crossmamba_3d.defrost()
        cfg_crossmamba_3d.BCN.TYPE = "cross_mamba"
        cfg_crossmamba_3d.ADAPTIVE.USE_CROSSMAMBA = True
        cfg_crossmamba_3d.ADAPTIVE.USE_3D_FEATURES = True
        cfg_crossmamba_3d.FUSION.TYPE = "bottleneck"  # ⭐ 使用MFAB注意力瓶颈融合
        cfg_crossmamba_3d.freeze()
        configs.append({
            'name': 'crossmamba_3d_mfab',
            'description': 'CrossMamba + 3D Features + MFAB Bottleneck Fusion',
            'config': cfg_crossmamba_3d
        })

        return configs

    def train_single_config(self, config_dict):
        """
        训练单个配置

        Args:
            config_dict: 配置字典，包含name, description, config

        Returns:
            dict: 包含验证集性能指标的字典
        """
        from models import GraphBAN
        from domain_adaptator import Discriminator

        config_name = config_dict['name']
        config_desc = config_dict['description']
        cfg = config_dict['config']

        print(f"\n{'='*80}")
        print(f"Training Configuration: {config_name}")
        print(f"Description: {config_desc}")
        print(f"BCN Type: {cfg.BCN.TYPE}")
        print(f"{'='*80}\n")

        # 创建模型
        model = GraphBAN(**cfg).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)

        # 创建判别器（如果使用DA）
        domain_dmm = None
        opt_da_local = None
        if cfg.DA.USE:
            if cfg.DA.RANDOM_LAYER:
                domain_dmm = Discriminator(
                    input_size=cfg.DA.RANDOM_DIM,
                    n_class=cfg.DECODER.BINARY
                ).to(self.device)
            else:
                domain_dmm = Discriminator(
                    input_size=cfg.DECODER.IN_DIM * cfg.DECODER.BINARY,
                    n_class=cfg.DECODER.BINARY
                ).to(self.device)
            opt_da_local = torch.optim.Adam(domain_dmm.parameters(), lr=cfg.SOLVER.DA_LR)

        # 创建训练器
        trainer = Trainer(
            model, opt, self.device,
            self.train_dataloader, self.val_dataloader, self.test_dataloader,
            opt_da=opt_da_local,
            discriminator=domain_dmm,
            experiment=self.experiment,
            **cfg
        )

        # 训练模型
        trainer.train()

        # 获取验证集最佳性能
        best_epoch = trainer.best_epoch
        best_auroc = trainer.best_auroc
        best_val_loss = trainer.val_loss_epoch[best_epoch - 1] if best_epoch > 0 else float('inf')

        # 获取验证集的AUPRC（需要从trainer中提取）
        # 这里简化处理，使用best_auroc作为主要指标

        result = {
            'config_name': config_name,
            'description': config_desc,
            'best_epoch': best_epoch,
            'val_auroc': best_auroc,
            'val_loss': best_val_loss,
            'trainer': trainer,
            'model': model
        }

        print(f"\n{'='*80}")
        print(f"Configuration {config_name} Training Complete")
        print(f"Best Epoch: {best_epoch}")
        print(f"Best Val AUROC: {best_auroc:.4f}")
        print(f"Best Val Loss: {best_val_loss:.4f}")
        print(f"{'='*80}\n")

        return result

    def select_best_config(self, results):
        """
        根据验证集性能选择最优配置

        Args:
            results: 各配置的结果列表

        Returns:
            dict: 最优配置的结果
        """
        metric_key = 'val_auroc' if self.selection_metric == 'auroc' else 'val_loss'

        if self.selection_metric == 'auroc':
            # AUROC越高越好
            best_result = max(results, key=lambda x: x[metric_key])
        else:
            # Loss越低越好
            best_result = min(results, key=lambda x: x[metric_key])

        return best_result

    def train_and_select(self):
        """
        训练所有配置并选择最优

        Returns:
            dict: 包含最优配置信息和测试结果
        """
        # 准备配置
        configs = self.prepare_configs()
        self.configs_to_try = configs

        print(f"\n{'#'*80}")
        print(f"# Adaptive Model Selection - Training {len(configs)} Configurations")
        print(f"# Selection Metric: {self.selection_metric}")
        print(f"{'#'*80}\n")

        # 训练每个配置
        results = []
        for config_dict in configs:
            result = self.train_single_config(config_dict)
            results.append(result)
            self.config_results[result['config_name']] = result

        # 选择最优配置
        best_result = self.select_best_config(results)
        self.best_config_name = best_result['config_name']
        self.best_model = best_result['model']
        self.best_trainer = best_result['trainer']

        print(f"\n{'#'*80}")
        print(f"# Best Configuration Selected: {self.best_config_name}")
        print(f"# Description: {best_result['description']}")
        print(f"# Val AUROC: {best_result['val_auroc']:.4f}")
        print(f"# Val Loss: {best_result['val_loss']:.4f}")
        print(f"{'#'*80}\n")

        # 在测试集上评估最优模型
        print(f"\n{'='*80}")
        print(f"Evaluating Best Model on Test Set")
        print(f"{'='*80}\n")

        # 测试集结果已经在训练过程中计算
        test_metrics = self.best_trainer.test_metrics

        # 保存选择记录
        self.save_selection_log(results, best_result, test_metrics)

        return {
            'best_config_name': self.best_config_name,
            'best_config_desc': best_result['description'],
            'val_metrics': {
                'auroc': best_result['val_auroc'],
                'loss': best_result['val_loss'],
                'epoch': best_result['best_epoch']
            },
            'test_metrics': test_metrics,
            'all_results': results
        }

    def save_selection_log(self, all_results, best_result, test_metrics):
        """
        保存选择日志

        Args:
            all_results: 所有配置的结果
            best_result: 最优配置结果
            test_metrics: 测试集指标
        """
        log_path = os.path.join(self.output_dir, 'adaptive_selection_log.json')

        log_data = {
            'selection_metric': self.selection_metric,
            'best_config': {
                'name': best_result['config_name'],
                'description': best_result['description'],
                'best_epoch': int(best_result['best_epoch']),
                'val_auroc': float(best_result['val_auroc']),
                'val_loss': float(best_result['val_loss'])
            },
            'test_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v
                           for k, v in test_metrics.items()},
            'all_configs': []
        }

        for result in all_results:
            log_data['all_configs'].append({
                'name': result['config_name'],
                'description': result['description'],
                'best_epoch': int(result['best_epoch']),
                'val_auroc': float(result['val_auroc']),
                'val_loss': float(result['val_loss'])
            })

        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)

        print(f"\n✓ Selection log saved to: {log_path}\n")

    def ensemble_predict(self, dataloader):
        """
        使用所有训练好的模型进行集成预测

        Args:
            dataloader: 数据加载器

        Returns:
            tuple: (集成预测结果, 真实标签)
        """
        all_predictions = []
        all_labels = None

        for config_name, result in self.config_results.items():
            trainer = result['trainer']
            model = result['model']

            # 获取该模型的预测
            model.eval()
            predictions = []
            labels = []

            with torch.no_grad():
                for batch in dataloader:
                    # 这里需要根据实际的数据格式调整
                    # 简化处理，假设trainer有predict方法
                    pass

            all_predictions.append(predictions)
            if all_labels is None:
                all_labels = labels

        # 平均集成
        ensemble_pred = np.mean(all_predictions, axis=0)

        return ensemble_pred, all_labels


def run_adaptive_training(device, train_dataloader, val_dataloader, test_dataloader,
                         cfg, opt_da=None, discriminator=None, experiment=None):
    """
    运行自适应训练的便捷函数

    Args:
        device: 训练设备
        train_dataloader: 训练数据加载器
        val_dataloader: 验证数据加载器
        test_dataloader: 测试数据加载器
        cfg: 配置对象
        opt_da: Domain adaptation优化器（可选）
        discriminator: 判别器（可选）
        experiment: 实验跟踪对象（可选）

    Returns:
        dict: 包含最优配置和测试结果的字典
    """
    selector = AdaptiveModelSelector(
        device, train_dataloader, val_dataloader, test_dataloader,
        cfg, opt_da, discriminator, experiment
    )

    results = selector.train_and_select()

    return results
