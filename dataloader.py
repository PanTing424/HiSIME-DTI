import pandas as pd
import torch.utils.data as data
import torch
import numpy as np
import dgl  # ⭐ 新增：导入dgl用于构建3D图
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from utils import integer_label_protein
# 它包含三个类：
#
# DTIDataset：标准训练/测试用的数据集类；
#
# DTIDataset2：用于蒸馏（distillation）场景的数据集类（多了 teacher embedding）；
#
# MultiDataLoader：用于组合多个 dataloader，每个 batch 同时从多个源加载。

class DTIDataset(data.Dataset):
    def __init__(self, list_IDs, df, max_drug_nodes=290, drug_3d_features=None):  # ⭐ 新增参数
        self.list_IDs = list_IDs
        self.df = df
        self.max_drug_nodes = max_drug_nodes
        self.drug_3d_features = drug_3d_features  # ⭐ 保存3D特征字典

        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)
        #self.fcfp = MolecularFCFP()
        #self.fcfps = x_batch

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]
        smiles = self.df.iloc[index]['SMILES']  # ⭐ 保存SMILES用于查询3D特征

        # 构建2D图
        v_d = self.fc(smiles=smiles, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
        actual_node_feats = v_d.ndata.pop('h')
        num_actual_nodes = actual_node_feats.shape[0]

        # ⭐ 保存原始特征用于3D图（在padding之前）
        actual_node_feats_for_3d = actual_node_feats.clone()

        # 2D图padding
        num_virtual_nodes = self.max_drug_nodes - num_actual_nodes
        virtual_node_bit = torch.zeros([num_actual_nodes, 1])
        actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
        v_d.ndata['h'] = actual_node_feats
        virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74), torch.ones(num_virtual_nodes, 1)), 1)
        v_d.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
        v_d = v_d.add_self_loop()

        v_p = self.df.iloc[index]['Protein']
        v_p = integer_label_protein(v_p)
        y =   self.df.iloc[index]["Y"]
        fcfps = self.df.iloc[index]["fcfp"]
        esm = self.df.iloc[index]["esm"]

        # ⭐⭐⭐ 新增：构建3D图 ⭐⭐⭐
        drug_3d_data = None
        if self.drug_3d_features is not None:
            drug_3d_data = self.drug_3d_features.get(smiles, None)

        g_3d = None
        if drug_3d_data is not None:
            try:
                num_atoms = drug_3d_data['num_atoms']
                edges = drug_3d_data['edges']  # [num_edges, 2]
                atom_pos = drug_3d_data['atom_pos']  # [num_atoms, 3]

                # 创建DGL图
                g_3d = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=num_atoms)

                # 设置边特征（键长度）
                if 'bond_length' in drug_3d_data:
                    bond_lengths = drug_3d_data['bond_length']
                    if len(bond_lengths) == len(edges):
                        g_3d.edata['bond_length'] = torch.FloatTensor(bond_lengths)
                    else:
                        g_3d.edata['bond_length'] = torch.ones(g_3d.num_edges()) * 1.5
                else:
                    g_3d.edata['bond_length'] = torch.ones(g_3d.num_edges()) * 1.5

                # 记录原始边数量，添加自环
                num_orig_edges = g_3d.num_edges()
                g_3d = dgl.add_self_loop(g_3d)

                # 为自环设置特殊值0.0（在MolecularGCN中会被过滤）
                g_3d.edata['bond_length'][num_orig_edges:] = 0.0

                # 构建节点特征：[x, y, z, 原子特征(72维)]
                atom_pos_tensor = torch.FloatTensor(atom_pos)
                atom_feats_3d = torch.zeros(num_atoms, 75)
                atom_feats_3d[:, :3] = atom_pos_tensor  # 前3维：3D坐标

                min_atoms = min(num_atoms, actual_node_feats_for_3d.shape[0])
                atom_feats_3d[:min_atoms, 3:] = actual_node_feats_for_3d[:min_atoms, :72]  # 后72维：原子特征

                g_3d.ndata['h'] = atom_feats_3d

            except Exception as e:
                print(f"Warning: Failed to build 3D graph: {e}")
                g_3d = None

        # 如果没有3D数据，创建空图占位
        if g_3d is None:
            g_3d = dgl.graph(([], []), num_nodes=1)
            g_3d.ndata['h'] = torch.zeros(1, 75)
            g_3d = dgl.add_self_loop(g_3d)
            g_3d.edata['bond_length'] = torch.zeros(g_3d.num_edges())

        # ⭐ 返回值增加 g_3d 和 drug_3d_data
        return v_d, g_3d, fcfps, v_p, esm, y, drug_3d_data

class DTIDataset2(data.Dataset):
    def __init__(self, list_IDs, df, max_drug_nodes=290, drug_3d_features=None):  # ⭐ 新增参数
        self.list_IDs = list_IDs
        self.df = df
        self.max_drug_nodes = max_drug_nodes
        self.drug_3d_features = drug_3d_features  # ⭐ 保存3D特征字典

        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)
        #self.fcfp = MolecularFCFP()
        #self.fcfps = x_batch

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]
        smiles = self.df.iloc[index]['SMILES']  # ⭐ 保存SMILES用于查询3D特征

        # 构建2D图
        v_d = self.fc(smiles=smiles, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
        actual_node_feats = v_d.ndata.pop('h')
        num_actual_nodes = actual_node_feats.shape[0]

        # ⭐ 保存原始特征用于3D图（在padding之前）
        actual_node_feats_for_3d = actual_node_feats.clone()

        # 2D图padding
        num_virtual_nodes = self.max_drug_nodes - num_actual_nodes
        virtual_node_bit = torch.zeros([num_actual_nodes, 1])
        actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
        v_d.ndata['h'] = actual_node_feats
        virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74), torch.ones(num_virtual_nodes, 1)), 1)
        v_d.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
        v_d = v_d.add_self_loop()

        v_p = self.df.iloc[index]['Protein']
        v_p = integer_label_protein(v_p)
        y =   self.df.iloc[index]["Y"]
        fcfps = self.df.iloc[index]["fcfp"]
        esm = self.df.iloc[index]['esm']
        teacher_emb = self.df.iloc[index]['teacher_emb']

        # ⭐⭐⭐ 新增：构建3D图 ⭐⭐⭐
        drug_3d_data = None
        if self.drug_3d_features is not None:
            drug_3d_data = self.drug_3d_features.get(smiles, None)

        g_3d = None
        if drug_3d_data is not None:
            try:
                num_atoms = drug_3d_data['num_atoms']
                edges = drug_3d_data['edges']  # [num_edges, 2]
                atom_pos = drug_3d_data['atom_pos']  # [num_atoms, 3]

                # 创建DGL图
                g_3d = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=num_atoms)

                # 设置边特征（键长度）
                if 'bond_length' in drug_3d_data:
                    bond_lengths = drug_3d_data['bond_length']
                    if len(bond_lengths) == len(edges):
                        g_3d.edata['bond_length'] = torch.FloatTensor(bond_lengths)
                    else:
                        g_3d.edata['bond_length'] = torch.ones(g_3d.num_edges()) * 1.5
                else:
                    g_3d.edata['bond_length'] = torch.ones(g_3d.num_edges()) * 1.5

                # 记录原始边数量，添加自环
                num_orig_edges = g_3d.num_edges()
                g_3d = dgl.add_self_loop(g_3d)

                # 为自环设置特殊值0.0（在MolecularGCN中会被过滤）
                g_3d.edata['bond_length'][num_orig_edges:] = 0.0

                # 构建节点特征：[x, y, z, 原子特征(72维)]
                atom_pos_tensor = torch.FloatTensor(atom_pos)
                atom_feats_3d = torch.zeros(num_atoms, 75)
                atom_feats_3d[:, :3] = atom_pos_tensor  # 前3维：3D坐标

                min_atoms = min(num_atoms, actual_node_feats_for_3d.shape[0])
                atom_feats_3d[:min_atoms, 3:] = actual_node_feats_for_3d[:min_atoms, :72]  # 后72维：原子特征

                g_3d.ndata['h'] = atom_feats_3d

            except Exception as e:
                print(f"Warning: Failed to build 3D graph: {e}")
                g_3d = None

        # 如果没有3D数据，创建空图占位
        if g_3d is None:
            g_3d = dgl.graph(([], []), num_nodes=1)
            g_3d.ndata['h'] = torch.zeros(1, 75)
            g_3d = dgl.add_self_loop(g_3d)
            g_3d.edata['bond_length'] = torch.zeros(g_3d.num_edges())

        # ⭐ 返回值增加 g_3d 和 drug_3d_data
        return v_d, g_3d, fcfps, v_p, esm, y, teacher_emb, drug_3d_data


class MultiDataLoader(object):
    def __init__(self, dataloaders, n_batches):
        if n_batches <= 0:
            raise ValueError("n_batches should be > 0")
        self._dataloaders = dataloaders
        self._n_batches = np.maximum(1, n_batches)
        self._init_iterators()

    def _init_iterators(self):
        self._iterators = [iter(dl) for dl in self._dataloaders]

    def _get_nexts(self):
        def _get_next_dl_batch(di, dl):
            try:
                batch = next(dl)
            except StopIteration:
                new_dl = iter(self._dataloaders[di])
                self._iterators[di] = new_dl
                batch = next(new_dl)
            return batch

        return [_get_next_dl_batch(di, dl) for di, dl in enumerate(self._iterators)]

    def __iter__(self):
        for _ in range(self._n_batches):
            yield self._get_nexts()
        self._init_iterators()

    def __len__(self):
        return self._n_batches
