# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math
from dgllife.model.gnn import GCN
from ban import BANLayer
from torch.nn.utils.weight_norm import weight_norm
from tqdm import tqdm
from rdkit.Chem import AllChem
from sklearn.preprocessing import StandardScaler
import sys
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
import dgl.function as fn  # ⭐ 新增：DGL消息传递函数

# >>> 新增：Cross-Mamba <<<
from cross_mamba import CrossMambaConfig, CrossMambaHead


# ============================================================
# ⭐ 新增：RBF编码器（用于键长度编码）
# 来源：Innovation 2 models.py 第29-53行
# ============================================================
class RBF(nn.Module):
    """
    Radial Basis Function
    将连续值(如键长度)映射到高维空间
    公式：RBF(x) = exp(-gamma * (x - center)^2)
    """
    def __init__(self, centers, gamma):
        super(RBF, self).__init__()
        self.centers = torch.reshape(torch.tensor(centers), [1, -1])
        self.gamma = gamma

    def forward(self, x):
        """
        Args:
            x(tensor): (-1, 1) 或 (-1,) - 输入的连续值（如键长度）
        Returns:
            y(tensor): (-1, n_centers) - RBF编码后的特征
        """
        x = torch.reshape(x, [-1, 1])
        # 确保centers和x在同一设备上，且数据类型为float32
        centers = self.centers.to(x.device).to(torch.float32)
        x = x.to(torch.float32)
        result = torch.exp(-self.gamma * torch.square(x - centers))
        # 添加数值裁剪防止RBF输出溢出
        return torch.clamp(result, min=1e-6, max=1e6)


# ============================================================
# ⭐ 新增：EdgeWeightedGCNLayer - 支持边特征的GCN层
# 来源：Innovation 2 models.py 第60-96行
# ============================================================
class EdgeWeightedGCNLayer(nn.Module):
    """
    支持边特征的GCN层 - 让边特征参与消息传递

    工作原理：
    1. 节点和边特征分别通过线性变换
    2. 消息传递时：邻居节点特征 × 边特征 = 消息
    3. 聚合消息到目标节点
    4. 残差连接保留原始信息
    """
    def __init__(self, in_dim, out_dim, edge_dim):
        super().__init__()
        self.linear_node = nn.Linear(in_dim, out_dim)
        self.linear_edge = nn.Linear(edge_dim, out_dim)
        self.activation = nn.ReLU()

    def forward(self, g, node_feats, edge_feats):
        with g.local_scope():
            # 1. 节点特征变换
            h = self.linear_node(node_feats)

            # 2. 边特征变换
            e = self.linear_edge(edge_feats)

            # 3. 消息传递：邻居特征 × 边权重
            g.ndata['h'] = h
            g.edata['e'] = e

            # 消息函数：将源节点特征与边特征相乘
            g.update_all(
                message_func=fn.u_mul_e('h', 'e', 'm'),  # msg = h_src * e
                reduce_func=fn.mean('m', 'h_new')         # 聚合
            )

            h_new = g.ndata['h_new']

        return self.activation(h_new + h)  # 残差连接


# ============================================================
# ⭐ 新增：注意力融合模块（融合2D/3D/ChemBERTa三个模态）
# 来源：Innovation 2 models.py 第103-141行
# ============================================================
class AttentionBottleneckFusion(nn.Module):
    """
    基于 Attention Bottleneck 的多模态融合模块
    参考：Attention Bottlenecks for Multimodal Fusion (NeurIPS 2021)

    核心思想：
    1. 使用少量可学习的 latent tokens 作为信息瓶颈
    2. 通过交叉注意力实现：模态 -> latents -> 模态 的信息压缩与分发
    3. 残差连接保留原始特征信息
    """
    def __init__(self, seq_dim, graph_dim, semantic_dim, num_latents=4, dropout=0.1,
                 use_residual=True, residual_mode="gated", fixed_alpha=0.5):
        super(AttentionBottleneckFusion, self).__init__()

        self.seq_dim = seq_dim              # 2D图特征维度: 128
        self.graph_dim = graph_dim          # 3D图特征维度: 128
        self.semantic_dim = semantic_dim    # SMILES特征维度: 128
        self.num_latents = num_latents      # 瓶颈tokens数量: 4-8
        self.use_residual = use_residual
        self.residual_mode = residual_mode

        # 确保所有模态特征维度一致（便于交叉注意力计算）
        assert seq_dim == graph_dim == semantic_dim, \
            f"All modality dimensions must be equal, got {seq_dim}, {graph_dim}, {semantic_dim}"
        self.feature_dim = seq_dim

        # ⭐ 可学习的 latent bottleneck tokens
        # 初始化使用小的正态分布，避免初始化过大导致梯度爆炸
        self.latents = nn.Parameter(torch.empty(1, num_latents, self.feature_dim).normal_(std=0.02))

        # ⭐ 可学习的融合强度参数（初始化为0，表示刚开始不融合）
        self.scale_2d = nn.Parameter(torch.zeros(1))       # 2D图融合强度
        self.scale_3d = nn.Parameter(torch.zeros(1))       # 3D图融合强度
        self.scale_smiles = nn.Parameter(torch.zeros(1))   # SMILES融合强度

        # Dropout层，防止过拟合
        self.dropout = nn.Dropout(dropout)

        if use_residual:
            if residual_mode == "fixed":
                self.register_buffer('gate', torch.tensor(fixed_alpha))
            elif residual_mode == "gated":
                self.gate = nn.Parameter(torch.tensor(0.5))
            elif residual_mode == "learned":
                total_dim = seq_dim + graph_dim + semantic_dim
                self.gate_net = nn.Sequential(
                    nn.Linear(total_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )
            else:
                raise ValueError(f"Unknown residual_mode: {residual_mode}")

        # 输出投影层：将融合后的特征映射到合适的维度
        # 输出维度 = 3 * feature_dim，与原始拼接方式保持一致
        self.output_dim = seq_dim + graph_dim + semantic_dim  # 384

    def scaled_dot_product_attention(self, q, k, v):
        """
        标准的缩放点积注意力
        q: [B, N_q, C]
        k: [B, N_k, C]
        v: [B, N_v, C]
        返回: [B, N_q, C]
        """
        B, N, C = q.shape
        # 计算注意力分数并缩放
        attn = (q @ k.transpose(-2, -1)) * (C ** -0.5)  # [B, N_q, N_k]
        attn = attn.softmax(dim=-1)                      # Softmax归一化
        attn = self.dropout(attn)                        # Dropout
        x = (attn @ v).reshape(B, N, C)                  # 加权求和
        return x

    def bottleneck_fusion(self, f_2d, f_3d, f_smiles):
        """
        注意力瓶颈融合机制

        步骤：
        1. 将三个模态的特征拼接 [B, 3, C]
        2. Latents 从三个模态吸收信息（压缩）: [B, num_latents, C]
        3. 各模态从 latents 获取跨模态信息（分发）: [B, 1, C]
        4. 残差连接保留原始特征

        输入：
            f_2d: [B, C] - 2D分子图特征
            f_3d: [B, C] - 3D分子图特征
            f_smiles: [B, C] - SMILES特征
        输出：
            f_2d_fused, f_3d_fused, f_smiles_fused: 各 [B, C]
        """
        BS = f_2d.shape[0]

        # 扩展维度以便进行注意力计算 [B, 1, C]
        f_2d = f_2d.unsqueeze(1)        # [B, 1, C]
        f_3d = f_3d.unsqueeze(1)        # [B, 1, C]
        f_smiles = f_smiles.unsqueeze(1)  # [B, 1, C]

        # ⭐ 步骤1：拼接所有模态 tokens
        all_modalities = torch.cat([f_2d, f_3d, f_smiles], dim=1)  # [B, 3, C]

        # ⭐ 步骤2：Latents 从所有模态吸收信息（信息压缩）
        # Query: latents，Key & Value: 三个模态
        fused_latents = self.scaled_dot_product_attention(
            q=self.latents.expand(BS, -1, -1),  # [B, num_latents, C]
            k=all_modalities,                    # [B, 3, C]
            v=all_modalities                     # [B, 3, C]
        )  # -> [B, num_latents, C]

        # ⭐ 步骤3：各模态从 latents 获取跨模态信息（信息分发 + 残差连接）
        # 2D图：询问 latents 中有关 3D 和 SMILES 的信息
        f_2d_enhanced = f_2d + self.scale_2d * self.scaled_dot_product_attention(
            q=f_2d,              # [B, 1, C]
            k=fused_latents,     # [B, num_latents, C]
            v=fused_latents      # [B, num_latents, C]
        )  # -> [B, 1, C]

        # 3D图：询问 latents 中有关 2D 和 SMILES 的信息
        f_3d_enhanced = f_3d + self.scale_3d * self.scaled_dot_product_attention(
            q=f_3d,              # [B, 1, C]
            k=fused_latents,     # [B, num_latents, C]
            v=fused_latents      # [B, num_latents, C]
        )  # -> [B, 1, C]

        # SMILES：询问 latents 中有关 2D 和 3D 的信息
        f_smiles_enhanced = f_smiles + self.scale_smiles * self.scaled_dot_product_attention(
            q=f_smiles,          # [B, 1, C]
            k=fused_latents,     # [B, num_latents, C]
            v=fused_latents      # [B, num_latents, C]
        )  # -> [B, 1, C]

        # 去掉多余的维度 [B, 1, C] -> [B, C]
        f_2d_enhanced = f_2d_enhanced.squeeze(1)
        f_3d_enhanced = f_3d_enhanced.squeeze(1)
        f_smiles_enhanced = f_smiles_enhanced.squeeze(1)

        # 在瓶颈增强结果与原始模态之间再做一次显式残差融合，和 50+50_fusion 保持一致
        if not self.use_residual:
            f_2d_fused = f_2d_enhanced
            f_3d_fused = f_3d_enhanced
            f_smiles_fused = f_smiles_enhanced
        else:
            if self.residual_mode == "fixed":
                beta = self.gate
            elif self.residual_mode == "gated":
                beta = torch.sigmoid(self.gate)
            elif self.residual_mode == "learned":
                f_original = torch.cat([f_2d.squeeze(1), f_3d.squeeze(1), f_smiles.squeeze(1)], dim=1)
                beta = self.gate_net(f_original)

            f_2d_orig = f_2d.squeeze(1)
            f_3d_orig = f_3d.squeeze(1)
            f_smiles_orig = f_smiles.squeeze(1)

            f_2d_fused = beta * f_2d_enhanced + (1.0 - beta) * f_2d_orig
            f_3d_fused = beta * f_3d_enhanced + (1.0 - beta) * f_3d_orig
            f_smiles_fused = beta * f_smiles_enhanced + (1.0 - beta) * f_smiles_orig

        return f_2d_fused, f_3d_fused, f_smiles_fused

    def forward(self, f_seq, f_graph, f_semantic):
        """
        前向传播

        输入：
            f_seq: [B, 128] - 2D分子图特征
            f_graph: [B, 128] - 3D分子图特征
            f_semantic: [B, 128] - SMILES特征
        输出：
            F_d: [B, 384] - 融合后的药物特征（拼接形式）
        """
        # ⭐ 注意力瓶颈融合
        f_2d_fused, f_3d_fused, f_smiles_fused = self.bottleneck_fusion(
            f_seq, f_graph, f_semantic
        )

        # ⭐ 拼接融合后的特征（保持与原始接口一致）
        F_d = torch.cat([f_2d_fused, f_3d_fused, f_smiles_fused], dim=1)  # [B, 384]

        return F_d


# ⭐ 保留原始的 Att_FeatureFusion 作为备选（兼容性）
class Att_FeatureFusion(nn.Module):
    def __init__(self, seq_dim, graph_dim, semantic_dim,
                 use_residual=True, residual_mode="gated", fixed_alpha=0.5):
        super(Att_FeatureFusion, self).__init__()
        self.use_residual = use_residual
        self.residual_mode = residual_mode
        # 为每个模态创建独立的线性变换
        self.W1_seq = nn.Linear(seq_dim, 64)
        self.W1_graph = nn.Linear(graph_dim, 64)
        self.W1_semantic = nn.Linear(semantic_dim, 64)
        # 偏置项
        self.b_seq = nn.Parameter(torch.zeros(64))
        self.b_graph = nn.Parameter(torch.zeros(64))
        self.b_semantic = nn.Parameter(torch.zeros(64))
        # 注意力打分层
        self.W2 = nn.Parameter(torch.randn(64, 1))
        self.softmax = nn.Softmax(dim=1)

        if use_residual:
            if residual_mode == "fixed":
                self.register_buffer('gate', torch.tensor(fixed_alpha))
            elif residual_mode == "gated":
                self.gate = nn.Parameter(torch.tensor(0.5))
            elif residual_mode == "learned":
                total_dim = seq_dim + graph_dim + semantic_dim
                self.gate_net = nn.Sequential(
                    nn.Linear(total_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )
            else:
                raise ValueError(f"Unknown residual_mode: {residual_mode}")

    def forward(self, f_seq, f_graph, f_semantic):
        # 输入：f_seq (2D图特征), f_graph (3D图特征), f_semantic (ChemBERTa特征)
        # 计算注意力分数
        w_seq = torch.tanh(self.W1_seq(f_seq) + self.b_seq).mm(self.W2)
        w_graph = torch.tanh(self.W1_graph(f_graph) + self.b_graph).mm(self.W2)
        w_semantic = torch.tanh(self.W1_semantic(f_semantic) + self.b_semantic).mm(self.W2)

        # 三个模态一起softmax，让它们互相竞争
        w_all = torch.cat([w_seq, w_graph, w_semantic], dim=1)  # [batch, 3]
        alpha_all = self.softmax(w_all)  # [batch, 3] 权重和为1

        # 拆分权重
        alpha_seq = alpha_all[:, 0:1]      # [batch, 1]
        alpha_graph = alpha_all[:, 1:2]    # [batch, 1]
        alpha_semantic = alpha_all[:, 2:3]  # [batch, 1]

        if not self.use_residual:
            f_seq_fused = alpha_seq * f_seq
            f_graph_fused = alpha_graph * f_graph
            f_semantic_fused = alpha_semantic * f_semantic
        else:
            if self.residual_mode == "fixed":
                beta = self.gate
            elif self.residual_mode == "gated":
                beta = torch.sigmoid(self.gate)
            elif self.residual_mode == "learned":
                f_original = torch.cat([f_seq, f_graph, f_semantic], dim=1)
                beta = self.gate_net(f_original)

            f_seq_fused = beta * (alpha_seq * f_seq) + (1.0 - beta) * f_seq
            f_graph_fused = beta * (alpha_graph * f_graph) + (1.0 - beta) * f_graph
            f_semantic_fused = beta * (alpha_semantic * f_semantic) + (1.0 - beta) * f_semantic

        # 计算整体药物嵌入（拼接加权后的特征）
        F_d = torch.cat((f_seq_fused, f_graph_fused, f_semantic_fused), dim=1)
        return F_d


def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss

def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent


def cross_entropy_logits(linear_output, label, weights=None, margin=0.0):
    # ⭐ 使用与inductive_mode_32一致的实现，忽略margin参数
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss


class GraphBAN(nn.Module):
    def __init__(self, **config):
        super(GraphBAN, self).__init__()
        drug_in_feats     = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding    = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        drug_padding      = config["DRUG"]["PADDING"]

        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        num_filters     = config["PROTEIN"]["NUM_FILTERS"]
        kernel_size     = config["PROTEIN"]["KERNEL_SIZE"]
        protein_padding = config["PROTEIN"]["PADDING"]

        mlp_in_dim     = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim    = config["DECODER"]["OUT_DIM"]
        out_binary     = config["DECODER"]["BINARY"]

        ban_heads      = config["BCN"]["HEADS"]
        # >>> 新增：交互层类型（ban / cross_mamba / parallel）
        inter_type = config.get("BCN", {}).get("TYPE", "ban").lower()

        # --------- 特征提取与融合 ----------
        self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                           padding=drug_padding, hidden_feats=drug_hidden_feats)

        # ⭐ 新增：3D图编码器（使用边加权GCN）
        self.drug_3d_extractor = MolecularGCN(
            in_feats=drug_in_feats,
            dim_embedding=drug_embedding,
            padding=drug_padding,
            hidden_feats=drug_hidden_feats,
            max_nodes=config.get("DRUG", {}).get("MAX_NODES", 290),  # 290
            use_edge_features=True  # ⭐ 启用边特征
        )

        self.molecule_FCFP = LinearTransform()
        self.protein_esm   = LinearTransform_esm()

        # ⭐ 修改：使用注意力瓶颈融合（Attention Bottleneck Fusion）
        # 从配置中读取融合模块类型，默认使用注意力瓶颈
        fusion_type = config.get("FUSION", {}).get("TYPE", "bottleneck").lower()
        num_latents = config.get("FUSION", {}).get("NUM_LATENTS", 4)  # 瓶颈tokens数量
        fusion_dropout = config.get("FUSION", {}).get("DROPOUT", 0.1)
        use_residual = config.get("FUSION", {}).get("USE_RESIDUAL", True)
        residual_mode = config.get("FUSION", {}).get("RESIDUAL_MODE", "gated")
        fixed_alpha = config.get("FUSION", {}).get("FIXED_ALPHA", 0.5)

        if fusion_type == "bottleneck":
            self.mol_fusion = AttentionBottleneckFusion(
                seq_dim=drug_hidden_feats[-1],     # 2D图：128
                graph_dim=drug_hidden_feats[-1],   # 3D图：128
                semantic_dim=drug_embedding,        # ChemBERTa：128
                num_latents=num_latents,            # 瓶颈数量：4-8
                dropout=fusion_dropout,
                use_residual=use_residual,
                residual_mode=residual_mode,
                fixed_alpha=fixed_alpha
            )
        else:  # 'attention' - 使用原始的注意力融合
            self.mol_fusion = Att_FeatureFusion(
                seq_dim=drug_hidden_feats[-1],     # 2D图：128
                graph_dim=drug_hidden_feats[-1],   # 3D图：128
                semantic_dim=drug_embedding,        # ChemBERTa：128
                use_residual=use_residual,
                residual_mode=residual_mode,
                fixed_alpha=fixed_alpha
            )

        self.pro_fusion    = proFusion()
        self.protein_extractor = ProteinCNN(protein_emb_dim, num_filters, kernel_size, protein_padding)

        # 输出维度（token隐层维度）
        self.d_comp = drug_hidden_feats[-1]  # 典型配置下 = 128
        self.d_prot = num_filters[-1]        # 典型配置下 = 128
        self.d_comp_fused = 384  # ⭐ 融合后的药物特征维度：2D(128) + 3D(128) + ChemBERTa(128)

        # --------- 交互层：BAN / Cross-Mamba / 并联 ----------
        self.inter_type = inter_type
        if inter_type == "ban":
            self.bcn = weight_norm(
                BANLayer(v_dim=self.d_comp_fused, q_dim=self.d_prot, h_dim=mlp_in_dim, h_out=ban_heads),  # ⭐ 使用d_comp_fused
                name='h_mat', dim=None
            )
            self.inter_out_dim = mlp_in_dim

        elif inter_type == "cross_mamba":
            # Cross-Mamba 配置
            cm_cfg_raw = config.get("CROSS_MAMBA", {})
            d_model = cm_cfg_raw.get("D_MODEL", self.d_comp)  # 统一到 d_model

            self.comp_align = nn.Linear(self.d_comp_fused, d_model)  # ⭐ 384 → d_model
            self.prot_align = nn.Linear(self.d_prot, d_model) if self.d_prot != d_model else nn.Identity()

            cm_cfg = CrossMambaConfig(
                d_model=d_model,
                out_dim=mlp_in_dim,
                n_layers=cm_cfg_raw.get("N_LAYERS", 4),
                d_state=cm_cfg_raw.get("D_STATE", 64),
                d_conv=cm_cfg_raw.get("D_CONV", 4),
                expand=cm_cfg_raw.get("EXPAND", 2),
                dropout=cm_cfg_raw.get("DROPOUT", 0.1),
                use_mamba2=cm_cfg_raw.get("USE_MAMBA2", True),
                max_len=cm_cfg_raw.get("MAX_LEN", 8192),
                use_pos=cm_cfg_raw.get("USE_POS", False),
                lowrank=cm_cfg_raw.get("LOWRANK", 64),
                downsample=cm_cfg_raw.get("DOWNSAMPLE", 1),
                compute_map_train=cm_cfg_raw.get("compute_map_train", False),
                bidirectional=cm_cfg_raw.get("bidirectional", False),
                eval_use_map=cm_cfg_raw.get("eval_use_map", True),
            )
            self.cm_head = CrossMambaHead(cm_cfg)
            self.inter_out_dim = mlp_in_dim

        elif inter_type == "parallel":
            # 并联：BAN + Cross-Mamba，然后融合（sum/gated-sum/concat）
            self.bcn = weight_norm(
                BANLayer(v_dim=self.d_comp_fused, q_dim=self.d_prot, h_dim=mlp_in_dim, h_out=ban_heads),  # ⭐ 使用d_comp_fused
                name='h_mat', dim=None
            )
            cm_cfg_raw = config.get("CROSS_MAMBA", {})
            d_model = cm_cfg_raw.get("D_MODEL", self.d_comp)
            self.comp_align = nn.Linear(self.d_comp_fused, d_model)  # ⭐ 384 → d_model
            self.prot_align = nn.Linear(self.d_prot, d_model) if self.d_prot != d_model else nn.Identity()
            cm_cfg = CrossMambaConfig(
                d_model=d_model, out_dim=mlp_in_dim,
                n_layers=cm_cfg_raw.get("N_LAYERS", 4),
                d_state=cm_cfg_raw.get("D_STATE", 64),
                d_conv=cm_cfg_raw.get("D_CONV", 4),
                expand=cm_cfg_raw.get("EXPAND", 2),
                dropout=cm_cfg_raw.get("DROPOUT", 0.1),
                use_mamba2=cm_cfg_raw.get("USE_MAMBA2", True),
                max_len=cm_cfg_raw.get("MAX_LEN", 8192),
                use_pos=cm_cfg_raw.get("USE_POS", False),
                lowrank=cm_cfg_raw.get("LOWRANK", 64),
                downsample=cm_cfg_raw.get("DOWNSAMPLE", 1),
            )
            self.cm_head = CrossMambaHead(cm_cfg)
            self.merge_mode = config.get("BCN", {}).get("PARALLEL_MERGE", "gated-sum")
            if self.merge_mode == "gated-sum":
                self.gate = nn.Parameter(torch.tensor(0.5))
            if self.merge_mode == "concat":
                # concat后投回 mlp_in_dim，确保后端 MLPDecoder 输入不变
                self.parallel_proj = nn.Sequential(
                    nn.Linear(mlp_in_dim * 2, mlp_in_dim),
                    nn.GELU()
                )
            self.inter_out_dim = mlp_in_dim

        else:
            raise ValueError(f"Unknown BCN.TYPE = {inter_type}")

        # --------- 分类头 ----------
        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)
        self.scaler = StandardScaler()

        # ⭐ 新增：BatchNorm层稳定融合后的特征
        self.bn_drug_fusion = nn.BatchNorm1d(384)  # 融合后特征维度

    def forward(self, bg_d, bg_d_3d, bg_smiles, v_p, v_p_esm, device, mode="train"):  # ⭐ 增加bg_d_3d参数
        # ---- Drug tokens ----
        # 1. 提取2D图特征
        v_d = self.drug_extractor(bg_d)  # [B, Nd, 128]

        # 2. ⭐ 提取3D图特征
        if bg_d_3d is not None and bg_d_3d.number_of_nodes() > 0:
            v_d_3d = self.drug_3d_extractor(bg_d_3d)  # [B, Nd, 128]
        else:
            v_d_3d = torch.zeros_like(v_d)

        # 3. 提取ChemBERTa特征
        v_smiles_fcfp = self.molecule_FCFP(bg_smiles)  # [B, 1, 128]

        # 4. 全局池化
        v_d_global = v_d.mean(dim=1)          # [B, 128]
        v_d_3d_global = v_d_3d.mean(dim=1)    # [B, 128]
        v_smiles_global = v_smiles_fcfp.squeeze(1)  # [B, 128]

        # 5. ⭐ 注意力融合三个模态
        v_fusion = self.mol_fusion(v_d_global, v_d_3d_global, v_smiles_global)  # [B, 384]
        v_fusion = self.bn_drug_fusion(v_fusion)  # BatchNorm
        v_fusion = v_fusion.unsqueeze(1)  # [B, 1, 384]

        # ---- Protein tokens ----
        v_p = self.protein_extractor(v_p)       # [B, Np, d_prot]
        v_p_esm = self.protein_esm(v_p_esm)     # [B, 1, 128]
        v_p_fusion = self.pro_fusion(v_p, v_p_esm)  # [B, Np, d_prot]

        # ---- Interaction (BAN / Cross-Mamba / Parallel) ----
        if self.inter_type == "ban":
            f, att = self.bcn(v_fusion, v_p_fusion)  # f:[B, mlp_in_dim]
        elif self.inter_type == "cross_mamba":
            comp_cm = self.comp_align(v_fusion)
            prot_cm = self.prot_align(v_p_fusion)
            want_map = (not self.training) and self.cm_head.cfg.eval_use_map
            f, att = self.cm_head(comp_cm, prot_cm, return_map=want_map)
            # f, att = self.cm_head(comp_cm, prot_cm, return_map=True)  # f:[B, mlp_in_dim], att:[B,Nc,Np]
        else:  # parallel
            f_ban, att_ban = self.bcn(v_fusion, v_p_fusion)  # [B, mlp_in_dim]
            comp_cm = self.comp_align(v_fusion)
            prot_cm = self.prot_align(v_p_fusion)
            f_cm, att_cm = self.cm_head(comp_cm, prot_cm, return_map=True)

            if self.merge_mode == "sum":
                f = f_ban + f_cm
            elif self.merge_mode == "gated-sum":
                alpha = torch.sigmoid(self.gate)
                f = alpha * f_cm + (1.0 - alpha) * f_ban
            else:  # concat
                f = torch.cat([f_ban, f_cm], dim=-1)
                f = self.parallel_proj(f)

            # 融合解释图
            att = att_cm
            if isinstance(att_ban, torch.Tensor):
                if att_ban.dim() == 4:
                    att_ban = att_ban.mean(dim=1)  # [B,h,Nc,Np] -> [B,Nc,Np]
                if att_ban.shape == att_cm.shape:
                    att = 0.5 * (att_cm + att_ban)

        score = self.mlp_classifier(f)

        if mode == "train":
            return v_fusion, v_p_fusion, f, score
        elif mode == "eval":
            return v_fusion, v_p_fusion, score, f
        else:
            return v_fusion, v_p_fusion, f, score 


class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None,
                 activation=None, max_nodes=None, use_edge_features=False):  # ⭐ 新增参数
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        self.max_nodes = max_nodes  # ⭐ 新增：用于3D图padding
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)

        self.output_feats = hidden_feats[-1]

        # ⭐ 新增：边加权GCN分支
        self.use_edge_features = use_edge_features
        if use_edge_features:
            # RBF编码器：0-3埃，间隔0.1，共30个中心
            centers = np.arange(0, 3, 0.1)
            gamma = 10.0
            self.bond_rbf = RBF(centers, gamma)

            # 边加权GCN层
            self.gnn_layers = nn.ModuleList()
            for i in range(len(hidden_feats)):
                in_dim = dim_embedding if i == 0 else hidden_feats[i-1]
                out_dim = hidden_feats[i]
                self.gnn_layers.append(
                    EdgeWeightedGCNLayer(in_dim, out_dim, edge_dim=len(centers))
                )
        else:
            # 2D图使用标准GCN
            self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata['h']
        node_feats = self.init_transform(node_feats)

        # ⭐ 边加权GCN分支
        if self.use_edge_features and 'bond_length' in batch_graph.edata:
            edge_lengths = batch_graph.edata['bond_length']  # [num_edges, 1]

            # 过滤自环边（bond_length=0）
            # 确保mask是1维的 [num_edges]
            non_self_loop_mask = (edge_lengths.squeeze(-1) > 0.01)  # [num_edges]

            if non_self_loop_mask.any():
                edge_feats_full = torch.zeros(edge_lengths.size(0), 30, device=edge_lengths.device)
                # edge_lengths[non_self_loop_mask] 的形状是 [n, 1]，需要squeeze
                edge_feats_full[non_self_loop_mask] = self.bond_rbf(edge_lengths[non_self_loop_mask].squeeze(-1))
            else:
                edge_feats_full = torch.zeros(edge_lengths.size(0), 30, device=edge_lengths.device)

            # 逐层传播
            for layer in self.gnn_layers:
                node_feats = layer(batch_graph, node_feats, edge_feats_full)
        else:
            # 2D图使用标准GCN
            node_feats = self.gnn(batch_graph, node_feats)

        batch_size = batch_graph.batch_size

        # ⭐ 处理变长图（3D图需要padding）
        if self.max_nodes is None:
            node_feats = node_feats.view(batch_size, -1, self.output_feats)
        else:
            import dgl
            graphs = dgl.unbatch(batch_graph)
            padded_feats = []
            for g in graphs:
                num_nodes = g.number_of_nodes()
                start_idx = sum([graphs[i].number_of_nodes() for i in range(graphs.index(g))])
                end_idx = start_idx + num_nodes
                graph_feats = node_feats[start_idx:end_idx, :]

                # Padding到max_nodes
                if num_nodes < self.max_nodes:
                    padding = torch.zeros(self.max_nodes - num_nodes, self.output_feats, device=graph_feats.device)
                    graph_feats = torch.cat([graph_feats, padding], dim=0)
                elif num_nodes > self.max_nodes:
                    graph_feats = graph_feats[:self.max_nodes, :]

                padded_feats.append(graph_feats)

            node_feats = torch.stack(padded_feats, dim=0)

        return node_feats


class LinearTransform(nn.Module):
    def __init__(self):
        super(LinearTransform, self).__init__()
        self.linear1 = nn.Linear(384, 512)  # for seed 12 for better score it was on 384>64>128
        self.linear2 = nn.Linear(512, 128)
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)
        x = x.unsqueeze(1)
        x = torch.relu(self.linear2(x))
        return x


class LinearTransform_esm(nn.Module):
    def __init__(self):
        super(LinearTransform_esm, self).__init__()
        self.linear1 = nn.Linear(1280, 512)
        self.linear2 = nn.Linear(512, 128)
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)
        x = x.unsqueeze(1)
        x = torch.relu(self.linear2(x))
        return x


class molFusion(nn.Module):
    def __init__(self):
        super(molFusion, self).__init__()
    def forward(self, A, B):
        # A:[B,Nd,d], B:[B,1,d] -> 最终:[B,Nd,d]
        result_1 = torch.matmul(A, B.transpose(1,2))
        result_2 = torch.matmul(result_1, B)
        final_result = torch.add(result_2, A)
        return final_result


class proFusion(nn.Module):
    def __init__(self):
        super(proFusion, self).__init__()
    def forward(self, A, B):
        # A:[B,Np,d], B:[B,1,d] -> 最终:[B,Np,d]
        result_1 = torch.matmul(A, B.transpose(1,2))
        result_2 = torch.matmul(result_1, B)
        final_result = torch.add(result_2, A)
        return final_result


class ProteinCNN(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_size, padding=True):
        super(ProteinCNN, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]
        kernels = kernel_size
        self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0])
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1])
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2])
        self.bn3 = nn.BatchNorm1d(in_ch[3])

    def forward(self, v):
        v = self.embedding(v.long())
        v = v.transpose(2, 1)
        v = self.bn1(F.relu(self.conv1(v)))
        v = self.bn2(F.relu(self.conv2(v)))
        v = self.bn3(F.relu(self.conv3(v)))
        v = v.view(v.size(0), v.size(2), -1)
        return v


class CosineLinear(nn.Module):
    def __init__(self, in_dim, out_dim=2, s=16.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(out_dim, in_dim))
        nn.init.xavier_uniform_(self.W)
        self.s = s
    def forward(self, x):
        x = F.normalize(x, dim=-1)          # 归一化特征
        W = F.normalize(self.W, dim=-1)     # 归一化权重
        return self.s * F.linear(x, W)      # 余弦相似度缩放成logits

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        # self.bn1 = nn.BatchNorm1d(hidden_dim)
     
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.bn2 = nn.BatchNorm1d(hidden_dim)
       
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        # self.bn3 = nn.BatchNorm1d(out_dim)
        
        # self.fc4 = nn.Linear(out_dim, binary)
        self.fc4 = CosineLinear(out_dim, binary, s=16.0)

        self.bn1 = nn.LayerNorm(hidden_dim)
        self.bn2 = nn.LayerNorm(hidden_dim)
        self.bn3 = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits


class RandomLayer(nn.Module):
    def __init__(self, input_dim_list, output_dim=256):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]
