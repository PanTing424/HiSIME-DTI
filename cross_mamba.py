# -*- coding: utf-8 -*-

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_HAS_MAMBA = True
# try:
#     from mamba_ssm import Mamba as _Mamba
#     from mamba_ssm import Mamba2 as _Mamba2
# except Exception:
#     try:
#         from mamba_ssm.modules.mamba_simple import Mamba as _Mamba
#         from mamba_ssm.modules.mamba2 import Mamba2 as _Mamba2
#     except Exception:
#         _HAS_MAMBA = False


from mamba_ssm import Mamba as _Mamba
from mamba_ssm import Mamba2 as _Mamba2

# ---------------------------- Config ----------------------------
@dataclass
class CrossMambaConfig:
    d_model: int              # token隐层维度（= 融合后token维度）
    out_dim: int              # 输出到分类器/CDAN的joint维度
    n_layers: int = 4
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.1
    use_mamba2: bool = False
    max_len: int = 8192
    use_pos: bool = False
    # 低秩双线性头
    lowrank: int = 64
    # 交互前可选下采样（降低 Nc×Np）
    downsample: int = 1       # 1=no ds, 2/4=mean-pool stride
    compute_map_train: bool = False   # 训练期是否计算 Nc×Np map（强烈建议 False）
    bidirectional: bool = False       # 使用 Bi-Mamba（默认取消，只保留前向更快）
    eval_use_map: bool = True         # 验证/推理期是否使用 Nc×Np map 输出可解释热图

# ------------------------ Building Blocks ------------------------
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

class BiMambaBlock(nn.Module):
    """单/双向 Mamba；残差 + RMSNorm。"""
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int,
                 dropout: float, use_mamba2: bool, bidirectional: bool = True):
        super().__init__()
        if not _HAS_MAMBA:
            raise ImportError("mamba-ssm 未安装。`pip install mamba-ssm`")
        Block = _Mamba2 if use_mamba2 else _Mamba
        self.bidirectional = bidirectional

        self.fwd = Block(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.bwd = Block(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand) if bidirectional else None

        self.drop = nn.Dropout(dropout)
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fwd(x)
        if self.bidirectional:
            y_b = torch.flip(self.bwd(torch.flip(x, dims=[1])), dims=[1])
            y = 0.5 * (y + y_b)
        y = self.drop(y)
        return self.norm(x + y)



class CrossMambaEncoder(nn.Module):
    """把两模态拼成一条序列：[CLS_C] + comp + [SEP] + [CLS_P] + prot"""
    def __init__(self, cfg: CrossMambaConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # 特殊token参数
        self.cls_c = nn.Parameter(torch.empty(d)); nn.init.normal_(self.cls_c, std=0.02)
        self.cls_p = nn.Parameter(torch.empty(d)); nn.init.normal_(self.cls_p, std=0.02)
        self.sep   = nn.Parameter(torch.empty(d)); nn.init.normal_(self.sep,   std=0.02)

        # 类型嵌入：0=pad/sep, 1=compound, 2=protein, 3=CLS_C, 4=CLS_P
        self.type_embed = nn.Embedding(5, d)
        self.pos_embed = nn.Embedding(cfg.max_len, d) if cfg.use_pos else None

        self.layers = nn.ModuleList([
    BiMambaBlock(d, cfg.d_state, cfg.d_conv, cfg.expand, cfg.dropout, cfg.use_mamba2, bidirectional=cfg.bidirectional)
    for _ in range(cfg.n_layers)
])
        self.final_norm = RMSNorm(d)

    def _pack(self, comp: torch.Tensor, prot: torch.Tensor):
        B, Nc, d = comp.shape
        _, Np, _ = prot.shape
        device = comp.device

        cls_c = self.cls_c.expand(B, 1, d)
        cls_p = self.cls_p.expand(B, 1, d)
        sep   = self.sep  .expand(B, 1, d)

        x = torch.cat([cls_c, comp, sep, cls_p, prot], dim=1)  # [B, L, d]

        t_cls_c = torch.full((B,1), 3, dtype=torch.long, device=device)
        t_comp  = torch.full((B,Nc), 1, dtype=torch.long, device=device)
        t_sep   = torch.full((B,1), 0, dtype=torch.long, device=device)
        t_cls_p = torch.full((B,1), 4, dtype=torch.long, device=device)
        t_prot  = torch.full((B,Np), 2, dtype=torch.long, device=device)
        t = torch.cat([t_cls_c, t_comp, t_sep, t_cls_p, t_prot], dim=1)

        if self.pos_embed is not None:
            pos = torch.arange(x.size(1), device=device).unsqueeze(0).expand(B, -1)
            x = x + self.pos_embed(pos)
        x = x + self.type_embed(t)
        return x, (1, 1+Nc, 1+Nc+1, 1+Nc+1+1, 1+Nc+1+1+prot.size(1))

    def forward(self, comp: torch.Tensor, prot: torch.Tensor):
        # comp:[B,Nc,d], prot:[B,Np,d]
        x, idx = self._pack(comp, prot)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        i0, i1, i2, i3, i4 = idx
        enc_cls_c = x[:, 0:1, :]        # [B,1,d]
        enc_comp  = x[:, 1:i1, :]       # [B,Nc,d]
        enc_cls_p = x[:, i2:i3, :]      # [B,1,d]
        enc_prot  = x[:, i3:i4, :]      # [B,Np,d]
        return enc_comp, enc_prot, enc_cls_c.squeeze(1), enc_cls_p.squeeze(1)


class LowRankBilinearHead(nn.Module):
    """低秩双线性重建 Nc×Np 交互图，并聚合出 joint 表征"""
    def __init__(self, d_model: int, rank: int, out_dim: int, dropout: float):
        super().__init__()
        self.rank = rank
        self.has_lr = rank > 0
        if self.has_lr:
            self.proj_c = nn.Linear(d_model, rank, bias=False)
            self.proj_p = nn.Linear(d_model, rank, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(4 * d_model + (rank if self.has_lr else 0), 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, out_dim)
        )

    def forward(self, comp: torch.Tensor, prot: torch.Tensor,
                cls_c: torch.Tensor, cls_p: torch.Tensor,
                downsample: int = 1, return_map: bool = False, compute_map: bool = False):
        # comp:[B,Nc,d], prot:[B,Np,d]
        if downsample > 1:
            def pool(x, s):
                pad = (-x.size(1)) % s
                if pad:
                    x = F.pad(x, (0,0,0,pad))
                B, L, D = x.shape
                return x.view(B, L//s, s, D).mean(dim=2)
            comp_ds = pool(comp, downsample)
            prot_ds = pool(prot, downsample)
        else:
            comp_ds, prot_ds = comp, prot

       # 仅当允许计算map时才做 O(Nc*Np*r) 的分支
        if self.has_lr and compute_map:
            phi_c = F.relu(self.proj_c(comp_ds))   # [B,Nc,r]
            phi_p = F.relu(self.proj_p(prot_ds))   # [B,Np,r]
            cross = torch.einsum('bcr,bpr->bcp', phi_c, phi_p) / math.sqrt(self.rank)
            attn_c = F.softmax(cross, dim=-1); attn_p = F.softmax(cross.transpose(1,2), dim=-1)
            ctx_c = torch.einsum('bcp,bpd->bcd', attn_c, prot_ds)
            ctx_p = torch.einsum('bpc,bcd->bpd', attn_p, comp_ds)
            lr_vec = (phi_c.mean(dim=1) * phi_p.mean(dim=1))
        else:
            # 轻量路径：不建 Nc×Np map
            ctx_c = prot_ds.mean(dim=1, keepdim=True).expand_as(comp_ds)
            ctx_p = comp_ds.mean(dim=1, keepdim=True).expand_as(prot_ds)
            cross = None
            # 维度对齐到 MLP 的输入期望（4*d_model + rank），即便不计算map也填充零向量
            lr_vec = comp_ds.new_zeros(comp_ds.size(0), self.rank)

        pool_c = (comp_ds + ctx_c).mean(dim=1)
        pool_p = (prot_ds + ctx_p).mean(dim=1)
        joint = torch.cat([cls_c, cls_p, pool_c, pool_p, lr_vec], dim=-1)
        out = self.mlp(joint)
        return (out, cross) if (return_map and cross is not None) else (out, None)
    


class CrossMambaHead(nn.Module):
    """可直接替代 BAN 的交互层：
       forward(comp_tokens, prot_tokens, return_map=False) -> joint[, map]
    """
    def __init__(self, cfg: CrossMambaConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = CrossMambaEncoder(cfg)
        self.head = LowRankBilinearHead(cfg.d_model, cfg.lowrank, cfg.out_dim, cfg.dropout)

    def forward(self, comp_tokens, prot_tokens, return_map: bool=False):
        comp_enc, prot_enc, cls_c, cls_p = self.encoder(comp_tokens, prot_tokens)
        compute_map = (self.training and self.cfg.compute_map_train) or ((not self.training) and self.cfg.eval_use_map)

        return self.head(comp_enc, prot_enc, cls_c, cls_p,
                     downsample=self.cfg.downsample,
                     return_map=return_map and compute_map,
                     compute_map=compute_map)


    