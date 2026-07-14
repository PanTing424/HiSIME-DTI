"""
  3D molecular feature extraction tools
  Adapted from EviDTI's compound_tools.py
  Uses RDKit for 3D conformation generation and geometric feature extraction
  """

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdchem

class Compound3DKit(object):
    """the 3Dkit of Compound"""

    @staticmethod
    def get_atom_poses(mol, conf):
        """tbd"""
        atom_poses = []
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetAtomicNum() == 0:
                return [[0.0, 0.0, 0.0]] * len(mol.GetAtoms())
            pos = conf.GetAtomPosition(i)
            atom_poses.append([pos.x, pos.y, pos.z])
        return atom_poses

    @staticmethod
    def get_MMFF_atom_poses(mol, numConfs=None, return_energy=False):
        """the atoms of mol will be changed in some cases."""
        try:
            new_mol = Chem.AddHs(mol)
            res = AllChem.EmbedMultipleConfs(new_mol, numConfs=numConfs)
            ### MMFF generates multiple conformations
            res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
            new_mol = Chem.RemoveHs(new_mol)
            index = np.argmin([x[1] for x in res])
            energy = res[index][1]
            conf = new_mol.GetConformer(id=int(index))
        except:
            new_mol = mol
            AllChem.Compute2DCoords(new_mol)
            energy = 0
            conf = new_mol.GetConformer()

        atom_poses = Compound3DKit.get_atom_poses(new_mol, conf)
        if return_energy:
            return new_mol, atom_poses, energy
        else:
            return new_mol, atom_poses

    @staticmethod
    def get_2d_atom_poses(mol):
        """get 2d atom poses"""
        AllChem.Compute2DCoords(mol)
        conf = mol.GetConformer()
        atom_poses = Compound3DKit.get_atom_poses(mol, conf)
        return atom_poses

    @staticmethod
    def get_bond_lengths(edges, atom_poses):
        """get bond lengths"""
        bond_lengths = []
        for src_node_i, tar_node_j in edges:
            bond_lengths.append(np.linalg.norm(atom_poses[tar_node_j] - atom_poses[src_node_i]))
        bond_lengths = np.array(bond_lengths, 'float32')
        return bond_lengths

    @staticmethod
    def get_superedge_angles(edges, atom_poses, dir_type='HT'):
        """get superedge angles"""

        def _get_vec(atom_poses, edge):
            return atom_poses[edge[1]] - atom_poses[edge[0]]

        def _get_angle(vec1, vec2):
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0
            vec1 = vec1 / (norm1 + 1e-5)  # 1e-5: prevent numerical errors
            vec2 = vec2 / (norm2 + 1e-5)
            angle = np.arccos(np.dot(vec1, vec2))
            return angle

        E = len(edges)
        edge_indices = np.arange(E)
        super_edges = []
        bond_angles = []
        bond_angle_dirs = []
        for tar_edge_i in range(E):
            tar_edge = edges[tar_edge_i]
            if dir_type == 'HT':
                src_edge_indices = edge_indices[edges[:, 1] == tar_edge[0]]
            elif dir_type == 'HH':
                src_edge_indices = edge_indices[edges[:, 1] == tar_edge[1]]
            else:
                raise ValueError(dir_type)
            for src_edge_i in src_edge_indices:
                if src_edge_i == tar_edge_i:
                    continue
                src_edge = edges[src_edge_i]
                src_vec = _get_vec(atom_poses, src_edge)
                tar_vec = _get_vec(atom_poses, tar_edge)
                super_edges.append([src_edge_i, tar_edge_i])
                angle = _get_angle(src_vec, tar_vec)
                bond_angles.append(angle)
                bond_angle_dirs.append(src_edge[1] == tar_edge[0])  # H -> H or H -> T

        if len(super_edges) == 0:
            super_edges = np.zeros([0, 2], 'int64')
            bond_angles = np.zeros([0, ], 'float32')
        else:
            super_edges = np.array(super_edges, 'int64')
            bond_angles = np.array(bond_angles, 'float32')
        return super_edges, bond_angles, bond_angle_dirs
 # ============ 辅助函数 ============

def get_mol_edges(mol):
      """
      从RDKit分子中提取边（化学键）列表
      
      参数:
      - mol: RDKit分子对象
      
      返回:
      - edges: numpy数组，形状 [num_edges, 2]
              每行是 [起始原子索引, 结束原子索引]
              无向图，所以每条化学键会有两条边 (i->j 和 j->i)
      
      示例:
      如果分子有3个原子，键为 0-1, 1-2
      返回: [[0,1], [1,0], [1,2], [2,1]]
      """
      if mol is None or len(mol.GetAtoms()) == 0:
          return None

      edges = []

      # 遍历所有化学键
      for bond in mol.GetBonds():
          i = bond.GetBeginAtomIdx()  # 起始原子的索引
          j = bond.GetEndAtomIdx()    # 结束原子的索引

          # 无向图：双向添加
          edges.append([i, j])  # i -> j
          edges.append([j, i])  # j -> i

      return np.array(edges, dtype=np.int64)
 # ============ 核心转换函数 ============

def mol_to_geognn_graph_data(mol, atom_poses, dir_type='HT'):
      """
      将分子和3D坐标转换为包含几何信息的图数据
      
      参数:
      - mol: RDKit分子对象
      - atom_poses: 原子3D坐标列表 [[x,y,z], [x,y,z], ...]
      - dir_type: 键角图方向类型，'HT' = Head-to-Tail（默认）
      
      返回:
      - data: 字典，包含以下键：
          - 'smiles': SMILES字符串
          - 'num_atoms': 原子数量
          - 'edges': 边列表 [[i,j], ...]
          - 'atom_pos': 原子3D坐标 (numpy数组)
          - 'bond_length': 键长列表 (numpy数组)
          - 'bond_angle': 键角列表 (numpy数组)
          - 'BondAngleGraph_edges': 键角图的边列表
      
      如果输入无效，返回 None
      """
      # 1. 检查输入
      if mol is None or len(mol.GetAtoms()) == 0:
          return None

      data = {}

      # 2. 提取边列表
      edges = get_mol_edges(mol)
      if edges is None:
          return None

      data['edges'] = edges

      # 3. 保存基本信息（用于后续匹配和调试）
      data['smiles'] = Chem.MolToSmiles(mol)  # 转回SMILES
      data['num_atoms'] = len(mol.GetAtoms())

      # 4. 添加3D坐标 ⭐ 核心：3D信息
      data['atom_pos'] = np.array(atom_poses, 'float32')

      # 5. 计算键长 ⭐ 基于3D坐标计算化学键的长度
      data['bond_length'] = Compound3DKit.get_bond_lengths(
          data['edges'],      # 边列表
          data['atom_pos']    # 原子坐标
      )

      # 6. 计算键角 ⭐ 计算三个原子形成的角度
      BondAngleGraph_edges, bond_angles, bond_angle_dirs = \
          Compound3DKit.get_superedge_angles(
              data['edges'],
              data['atom_pos'],
              dir_type=dir_type
          )

      # 保存键角信息
      data['BondAngleGraph_edges'] = BondAngleGraph_edges
      data['bond_angle'] = np.array(bond_angles, 'float32')
      data['bond_angle_dirs'] = bond_angle_dirs

      return data

def mol_to_geognn_graph_data_MMFF3d(mol):
      """
      主函数：使用MMFF力场生成3D构象并提取几何特征
      
      参数:
      - mol: RDKit分子对象
      
      返回:
      - data: 包含3D几何信息的字典
      
      处理逻辑:
      - 如果分子 ≤ 400 个原子：使用MMFF生成3D构象（较慢但准确）
      - 如果分子 > 400 个原子：使用2D坐标（快速，但不是真3D）
      """
      if mol is None or len(mol.GetAtoms()) == 0:
          return None

      # 根据分子大小选择策略
      if len(mol.GetAtoms()) <= 400:
          # 小分子：使用MMFF生成3D
          # numConfs=10: 生成10个不同的构象，选能量最低的
          mol, atom_poses = Compound3DKit.get_MMFF_atom_poses(mol, numConfs=10)
      else:
          # 大分子：使用2D（3D生成太慢）
          atom_poses = Compound3DKit.get_2d_atom_poses(mol)

      # 转换为图数据（包含3D几何信息）
      return mol_to_geognn_graph_data(mol, atom_poses, dir_type='HT')
def smiles_to_3d_graph_data(smiles):
      """
      便捷函数：直接从SMILES字符串生成3D图数据
      
      参数:
      - smiles: SMILES字符串，例如 "CC(C)Cc1ccccc1"
      
      返回:
      - data: 3D图数据字典
      - None: 如果SMILES无效或转换失败
      
      示例:
      >>> data = smiles_to_3d_graph_data("CCO")
      >>> print(data['num_atoms'])  # 3个原子（C, C, O）
      >>> print(data['bond_length'])  # 键长数组
      """
      # 1. SMILES → RDKit分子对象
      mol = AllChem.MolFromSmiles(smiles)

      if mol is None:
          print(f"Warning: Failed to parse SMILES: {smiles}")
          return None

      # 2. 分子对象 → 3D图数据
      return mol_to_geognn_graph_data_MMFF3d(mol)

      # ============ 测试代码 ============

if __name__ == "__main__":
      """
      测试代码：运行这个文件可以测试功能
      
      在命令行运行：
      cd HiSIME-DTI
      python compound_3d_tools.py
      """
      print("Testing 3D feature extraction...")

      # 测试1：简单分子（乙醇）
      smiles = "CCO"
      print(f"\nTest 1: {smiles}")
      data = smiles_to_3d_graph_data(smiles)

      if data is not None:
          print(f"  ✓ Success!")
          print(f"  - Number of atoms: {data['num_atoms']}")
          print(f"  - Number of edges: {len(data['edges'])}")
          print(f"  - Atom positions shape: {data['atom_pos'].shape}")
          print(f"  - Bond lengths: {data['bond_length'][:5]}...")  # 前5个
          print(f"  - Number of bond angles: {len(data['bond_angle'])}")
      else:
          print(f"  ✗ Failed to process SMILES")

      # 测试2：复杂分子（苯环）
      smiles2 = "c1ccccc1"
      print(f"\nTest 2: {smiles2}")
      data2 = smiles_to_3d_graph_data(smiles2)

      if data2 is not None:
          print(f"  ✓ Success!")
          print(f"  - Number of atoms: {data2['num_atoms']}")
          print(f"  - Number of edges: {len(data2['edges'])}")
      else:
          print(f"  ✗ Failed")

      print("\n" + "="*50)
      print("All tests completed!")
