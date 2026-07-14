import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm


class BANLayer(nn.Module):
    """
    双通道注意力网络层 (Bilinear Attention Network Layer)
    
    功能：融合药物和蛋白质特征，通过双线性注意力机制学习两者间的相互作用
    输入：
        v: 药物特征 [batch_size, v_num, v_dim] (来自molFusion)
        q: 蛋白质特征 [batch_size, q_num, q_dim] (来自proFusion)
    输出：
        logits: 融合特征 [batch_size, h_dim]
        att_maps: 注意力图 [batch_size, h_out, v_num, q_num]
    
    核心思想：
        1. 双线性注意力：学习药物和蛋白质特征间的双线性交互
        2. 多头注意力：使用多个注意力头捕获不同的交互模式
        3. 注意力池化：将注意力权重应用到特征上，生成融合表示
    """
    
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=0.2, k=3):
        super(BANLayer, self).__init__()

        # 参数设置
        self.c = 32  # 阈值，决定使用哪种注意力计算方式
        self.k = k   # 注意力头数
        self.v_dim = v_dim  # 药物特征维度
        self.q_dim = q_dim  # 蛋白质特征维度
        self.h_dim = h_dim  # 隐藏层维度
        self.h_out = h_out  # 输出注意力头数

        # 特征变换网络：将输入特征映射到隐藏空间
        # 药物特征变换：v_dim → h_dim * k
        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout)
        # 蛋白质特征变换：q_dim → h_dim * k
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout)
        
        # 池化层：用于多注意力头的聚合
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        # 注意力权重计算：根据输出头数选择不同策略
        if h_out <= self.c:
            # 小输出头数：使用参数化注意力矩阵
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            # 大输出头数：使用线性变换
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

        # 批归一化层
        self.bn = nn.BatchNorm1d(h_dim)

    def attention_pooling(self, v, q, att_map):
        """
        注意力池化函数
        
        功能：使用注意力权重对特征进行加权聚合
        输入：
            v: 变换后的药物特征 [batch_size, v_num, h_dim*k]
            q: 变换后的蛋白质特征 [batch_size, q_num, h_dim*k]
            att_map: 注意力图 [batch_size, v_num, q_num]
        输出：
            fusion_logits: 融合后的特征 [batch_size, h_dim*k]
        """
        # 双线性注意力池化：v^T * att_map * q
        # 使用爱因斯坦求和约定进行高效计算
        fusion_logits = torch.einsum('bvk,bvq,bqk->bk', (v, att_map, q))
        
        # 多注意力头聚合
        if 1 < self.k:
            fusion_logits = fusion_logits.unsqueeze(1)  # [batch_size, 1, h_dim*k]
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k  # 平均池化并缩放
        return fusion_logits

    def forward(self, v, q, softmax=False):
        """
        前向传播过程
        
        输入：
            v: 药物特征 [batch_size, v_num, v_dim]
            q: 蛋白质特征 [batch_size, q_num, q_dim]
            softmax: 是否对注意力图应用softmax
        输出：
            logits: 融合特征 [batch_size, h_dim]
            att_maps: 注意力图 [batch_size, h_out, v_num, q_num]
        """
        
        # ==================== 详细流程图 ====================
        """
        输入：
        v: [batch_size, v_num, v_dim] (药物特征)
        q: [batch_size, q_num, q_dim] (蛋白质特征)
        
        ↓
        ┌─────────────────────────────────────────┐
        │ 步骤1: 特征变换                         │
        │ v_ = self.v_net(v)                     │
        │ q_ = self.q_net(q)                     │
        │ 形状: v_ [B, v_num, h_dim*k], q_ [B, q_num, h_dim*k] │
        │ 说明: 将特征映射到隐藏空间              │
        └─────────────────────────────────────────┘
        ↓
        ┌─────────────────────────────────────────┐
        │ 步骤2: 注意力图计算                     │
        │ 根据h_out大小选择计算方式               │
        │ 小头数: 参数化注意力矩阵                │
        │ 大头数: 线性变换                        │
        │ 输出: att_maps [B, h_out, v_num, q_num] │
        └─────────────────────────────────────────┘
        ↓
        ┌─────────────────────────────────────────┐
        │ 步骤3: 注意力池化                       │
        │ 对每个注意力头进行池化                  │
        │ 使用双线性注意力机制                    │
        │ 输出: logits [B, h_dim]                 │
        └─────────────────────────────────────────┘
        ↓
        ┌─────────────────────────────────────────┐
        │ 步骤4: 批归一化                         │
        │ logits = self.bn(logits)               │
        │ 输出: 最终融合特征 [B, h_dim]           │
        └─────────────────────────────────────────┘
        ↓
        输出：融合特征 [batch_size, h_dim] 和注意力图
        """
        
        # 获取特征维度信息
        v_num = v.size(1)  # 药物特征数量
        q_num = q.size(1)  # 蛋白质特征数量
        
        # 步骤1：特征变换
        # 将药物和蛋白质特征映射到隐藏空间
        v_ = self.v_net(v)  # [batch_size, v_num, h_dim*k]
        q_ = self.q_net(q)  # [batch_size, q_num, h_dim*k]
        
        # 步骤2：注意力图计算
        if self.h_out <= self.c:
            # 小输出头数：使用参数化注意力矩阵
            # 计算双线性注意力：v_^T * h_mat * q_ + bias
            att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
        else:
            # 大输出头数：使用线性变换
            # 重塑特征维度以进行矩阵乘法
            v_ = v_.transpose(1, 2).unsqueeze(3)  # [batch_size, h_dim*k, v_num, 1]
            q_ = q_.transpose(1, 2).unsqueeze(2)  # [batch_size, h_dim*k, 1, q_num]
            
            # 计算双线性交互
            d_ = torch.matmul(v_, q_)  # [batch_size, h_dim*k, v_num, q_num]
            
            # 线性变换得到注意力图
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # [batch_size, v_num, q_num, h_out]
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)  # [batch_size, h_out, v_num, q_num]
        
        # 可选：对注意力图应用softmax
        if softmax:
            p = nn.functional.softmax(att_maps.view(-1, self.h_out, v_num * q_num), 2)
            att_maps = p.view(-1, self.h_out, v_num, q_num)
        
        # 步骤3：注意力池化
        # 对每个注意力头进行池化
        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits_i = self.attention_pooling(v_, q_, att_maps[:, i, :, :])
            logits += logits_i
        
        # 步骤4：批归一化
        logits = self.bn(logits)
        
        return logits, att_maps


class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    Modified from https://github.com/jnhwkim/ban-vqa/blob/master/fc.py
    """

    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class BCNet(nn.Module):
    """Simple class for non-linear bilinear connect network
    Modified from https://github.com/jnhwkim/ban-vqa/blob/master/bc.py
    """

    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=[.2, .5], k=3):
        super(BCNet, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim;
        self.q_dim = q_dim
        self.h_dim = h_dim;
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.dropout = nn.Dropout(dropout[1])  # attention
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if None == h_out:
            pass
        elif h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

    def forward(self, v, q):
        if None == self.h_out:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            logits = torch.einsum('bvk,bqk->bvqk', (v_, q_))
            return logits

        # low-rank bilinear pooling using einsum
        elif self.h_out <= self.c:
            v_ = self.dropout(self.v_net(v))
            q_ = self.q_net(q)
            logits = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
            return logits  # b x h_out x v x q

        # batch outer product, linear projection
        # memory efficient but slow computation
        else:
            v_ = self.dropout(self.v_net(v)).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            logits = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
            return logits.transpose(2, 3).transpose(1, 2)  # b x h_out x v x q

    def forward_with_weights(self, v, q, w):
        v_ = self.v_net(v)  # b x v x d
        q_ = self.q_net(q)  # b x q x d
        logits = torch.einsum('bvk,bvq,bqk->bk', (v_, w, q_))
        if 1 < self.k:
            logits = logits.unsqueeze(1)  # b x 1 x d
            logits = self.p_net(logits).squeeze(1) * self.k  # sum-pooling
        return logits
