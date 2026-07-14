import os
import random
import numpy as np
import torch
import dgl
import logging

from torch._C import ScriptModuleSerializer
#蛋白质氨基酸字符到数字的映射表
CHARPROTSET = {
    "A": 1,
    "C": 2,
    "B": 3,
    "E": 4,
    "D": 5,
    "G": 6,
    "F": 7,
    "I": 8,
    "H": 9,
    "K": 10,
    "M": 11,
    "L": 12,
    "O": 13,
    "N": 14,
    "Q": 15,
    "P": 16,
    "S": 17,
    "R": 18,
    "U": 19,
    "T": 20,
    "W": 21,
    "V": 22,
    "Y": 23,
    "X": 24,
    "Z": 25,
}

CHARPROTLEN = 25


def set_seed(seed=1000):#设定随机种子，保证可复现性
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

#自定义 DataLoader 的打包函数
#将一个 batch 的图和特征列表打包成：
#d：DGL 格式的图集合，方便送进 GNN；
#d_3d：3D分子图的DGL格式集合；
# smile：SMILES 化学式编码（如分子序列）；
# p：蛋白质序列编码；
# esm：ESM（蛋白预训练模型）的向量；
# y：标签。
def graph_collate_func(x):
    d, d_3d, smile, p, esm, y, drug_3d_data = zip(*x)  # ⭐ 增加d_3d和drug_3d_data
    d = dgl.batch(d)
    d_3d = dgl.batch(d_3d)  # ⭐ 批处理3D图
    return d, d_3d, torch.tensor(np.array(smile)), torch.tensor(np.array(p)), torch.tensor(np.array(esm)), torch.tensor(y), drug_3d_data

#graph_collate_func2(x) 多了 teacher_emb：这是用于 蒸馏模型（Teacher-Student）时的版本。
def graph_collate_func2(x):
    d, d_3d, smile, p, esm, y, teacher_emb, drug_3d_data = zip(*x)  # ⭐ 增加d_3d和drug_3d_data
    d = dgl.batch(d)
    d_3d = dgl.batch(d_3d)  # ⭐ 批处理3D图
    return d, d_3d, torch.tensor(np.array(smile)), torch.tensor(np.array(p)), torch.tensor(np.array(esm)), torch.tensor(y), torch.tensor(np.array(teacher_emb)), drug_3d_data
#自动创建目录：在保存模型、结果时自动建文件夹，防止路径不存在出错。
def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)

#蛋白质序列数字编码
#功能是：
# 把字符串蛋白质序列转为固定长度的数字向量；
# 比如 'MKTQ' → [11, 10, 20, 15, 0, 0, ..., 0]（0 是 padding）；
# 用于给 CNN 或 Transformer 模型输入做 embedding 编码。
def integer_label_protein(sequence, max_length=1200):
    """
    Integer encoding for protein string sequence.
    Args:
        sequence (str): Protein string sequence.
        max_length: Maximum encoding length of input protein string.
    """
    encoding = np.zeros(max_length)
    for idx, letter in enumerate(sequence[:max_length]):
        try:
            letter = letter.upper()
            encoding[idx] = CHARPROTSET[letter]
        except KeyError:
            logging.warning(
                f"character {letter} does not exists in sequence category encoding, skip and treat as " f"padding."
            )
    return encoding
