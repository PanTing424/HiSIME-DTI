# HiSIME-DTI

English documentation is provided first. The Chinese version follows after the English section.

---

## English

### 1. Overview

HiSIME-DTI is a standalone DTI repository for inductive drug-target interaction prediction. This project consolidates the code, inductive data splits, precomputed 3D molecular features, and local pretrained weights into one unified workspace so that training and inference do not depend on scattered external paths.

### 2. Main Features

- Repo-local path resolution for data, 3D features, ChemBERTa, and ESM.
- Multimodal DTI modeling with:
  - 2D molecular graphs
  - 3D molecular geometry
  - ChemBERTa SMILES embeddings
  - ESM protein embeddings
- Domain adaptation training for inductive experiments.
- Adaptive model selection that compares two model variants and keeps the better one based on validation performance.
- Inference script for running prediction from a saved checkpoint.

### 3. Repository Structure

```text
HiSIME-DTI/
├── Data/                       # inductive CSV/parquet splits
├── 3D_Features/                # precomputed 3D molecular feature dictionaries
├── ChemBERTa-77M-MTR/          # local ChemBERTa checkpoint
├── pretrained/esm/             # local ESM checkpoint
├── run_model.py                # standard training
├── run_model_adaptive.py       # adaptive training and selection
├── adaptive_trainer.py         # adaptive selection logic
├── predict.py                  # inference
├── dataloader.py               # 2D/3D graph loading
├── compound_3d_tools.py        # 3D conformer and geometry extraction
├── project_paths.py            # repo-local path resolution
├── *.yaml                      # training and model configuration files
├── requirements.txt            # pip-style dependency file
└── environment.exact.yml       # exact conda environment export
```

Important files:

- [`run_model.py`](./run_model.py)
- [`run_model_adaptive.py`](./run_model_adaptive.py)
- [`adaptive_trainer.py`](./adaptive_trainer.py)
- [`dataloader.py`](./dataloader.py)
- [`compound_3d_tools.py`](./compound_3d_tools.py)
- [`project_paths.py`](./project_paths.py)
- [`environment.exact.yml`](./environment.exact.yml)

### 4. Data and Resource Layout

This repository already includes the local resources expected by the code:

- `Data/`
- `3D_Features/`
- `ChemBERTa-77M-MTR/`
- `pretrained/esm/`

Bundled inductive datasets currently include:

- `bindingdb`
- `biosnap`
- `c.elegans`
- `kiba`
- `pdb`

Typical training split layout:

```text
Data/kiba/inductive/seed14/
├── source_train_kiba14.csv
├── target_train_kiba14.csv
├── target_test_kiba14.csv
└── kiba14_inductive_teacher_emb.parquet
```

Matching 3D feature files follow the same relative structure:

```text
3D_Features/kiba/inductive/seed14/
├── source_train_kiba14_3d.npy
├── target_train_kiba14_3d.npy
└── target_test_kiba14_3d.npy
```

The mapping from a CSV path to its 3D feature file is handled in [`project_paths.py`](./project_paths.py) by `resolve_3d_feature_path()`.

### 5. Input Format

#### CSV files

The minimal required columns are:

- `SMILES`
- `Protein`
- `Y`

Formal benchmark CSV files may additionally contain metadata such as:

- `target_id`
- `drug_cluster`
- `target_cluster`

Examples:

- [`Data/sample_data/df_train200.csv`](./Data/sample_data/df_train200.csv)
- [`Data/kiba/inductive/seed14/source_train_kiba14.csv`](./Data/kiba/inductive/seed14/source_train_kiba14.csv)

#### Teacher embedding parquet

The standard training scripts expect a teacher embedding parquet for the source training split.

- loaded in `run_model.py` and `run_model_adaptive.py`
- converted into `teacher_emb`
- aligned row-wise with the training CSV

Example:

- [`Data/kiba/inductive/seed14/kiba14_inductive_teacher_emb.parquet`](./Data/kiba/inductive/seed14/kiba14_inductive_teacher_emb.parquet)

### 6. 3D Feature Representation

3D molecular features are stored as `.npy` files containing a Python dictionary serialized with `allow_pickle=True`.

Top-level format:

- key: canonical or matched SMILES string
- value: geometry dictionary for that molecule

Observed fields in the current repository:

- `smiles`
- `num_atoms`
- `edges`
- `atom_pos`
- `bond_length`
- `BondAngleGraph_edges`
- `bond_angle`
- `bond_angle_dirs`

Typical shapes:

- `edges`: `(num_edges, 2)`
- `atom_pos`: `(num_atoms, 3)`
- `bond_length`: `(num_edges,)`
- `BondAngleGraph_edges`: `(num_angle_edges, 2)`
- `bond_angle`: `(num_angle_edges,)`

How the code uses them:

- [`compound_3d_tools.py`](./compound_3d_tools.py) builds the geometry dictionary.
- [`dataloader.py`](./dataloader.py) converts it into a DGL 3D graph.
- The 3D node feature layout is:
  - first 3 dimensions: xyz coordinates
  - remaining dimensions: copied atom features from the 2D graph
- `bond_length` is attached as an edge feature.

Fallback behavior:

- if a `_3d.npy` file is missing, the run falls back to 2D-only behavior for that split
- if a SMILES entry is missing in the 3D dictionary, an empty placeholder 3D graph is created for that sample

### 7. How 3D Features Are Extracted

3D feature extraction utilities are implemented in [`compound_3d_tools.py`](./compound_3d_tools.py).

Main entry points:

- `smiles_to_3d_graph_data(smiles)`
- `mol_to_geognn_graph_data_MMFF3d(mol)`
- `mol_to_geognn_graph_data(mol, atom_poses, dir_type='HT')`

Extraction logic:

- for molecules with up to 400 atoms:
  - add hydrogens
  - generate multiple conformers with RDKit
  - optimize by MMFF
  - choose the lowest-energy conformer
- for molecules larger than 400 atoms:
  - fall back to 2D coordinates to avoid very expensive conformer generation

Computed geometry includes:

- atom coordinates
- directed bond edges
- bond lengths
- bond-angle graph edges
- bond angles

Suggested workflow for building your own `_3d.npy` files:

1. Read a CSV split and collect unique `SMILES`.
2. Call `smiles_to_3d_graph_data(smiles)` for each molecule.
3. Save a dictionary `{smiles: data_dict}` with `np.save(..., allow_pickle=True)`.
4. Name the file using the same split name with suffix `_3d.npy`.
5. Put the file under the mirrored path in `3D_Features/`.

Minimal example:

```python
import numpy as np
import pandas as pd
from compound_3d_tools import smiles_to_3d_graph_data

df = pd.read_csv("Data/myset/inductive/seed12/source_train_myset12.csv")
smiles_list = df["SMILES"].drop_duplicates().tolist()

feature_dict = {}
for smi in smiles_list:
    data = smiles_to_3d_graph_data(smi)
    if data is not None:
        feature_dict[smi] = data

np.save(
    "3D_Features/myset/inductive/seed12/source_train_myset12_3d.npy",
    feature_dict,
    allow_pickle=True,
)
```

Notes:

- RDKit may canonicalize SMILES during conversion.
- The SMILES keys stored in `_3d.npy` should match the SMILES used at loading time as closely as possible.
- Failed RDKit parsing should be logged or skipped, otherwise those samples will later use empty placeholder 3D graphs.

### 8. Pretrained Models

#### ESM

The code uses a local ESM-1b checkpoint:

- `pretrained/esm/esm1b_t33_650M_UR50S.pt`

It is loaded by `load_local_esm_model()` in [`project_paths.py`](./project_paths.py).

#### ChemBERTa

The code uses the local ChemBERTa checkpoint stored in:

- `ChemBERTa-77M-MTR/`

It is loaded in:

- [`run_model.py`](./run_model.py)
- [`run_model_adaptive.py`](./run_model_adaptive.py)
- [`predict.py`](./predict.py)

### 9. Environment Setup

Two dependency files are kept in the repository:

- [`requirements.txt`](./requirements.txt)
  - lightweight pip-style dependency list
  - suitable when you already manage your own CUDA/PyTorch stack
- [`environment.exact.yml`](./environment.exact.yml)
  - full conda export from a validated local training environment
  - package versions are preserved exactly
  - the exported environment name is `hisime_dti`
  - local machine `prefix` was removed before saving into the repository

Recommended ways to set up the environment:

#### Option A: recreate the exact conda environment

```bash
conda env create -f environment.exact.yml
conda activate hisime_dti
```

#### Option B: install on top of an existing environment

```bash
conda activate your_env
pip install -r requirements.txt
```

Important dependencies in the exact environment include:

- PyTorch 2.3.0
- DGL 2.1.0
- DGLLife 0.3.2
- Transformers 4.47.0
- fair-esm 2.0.0
- ESM 3.2.0
- RDKit 2024.9.6
- mamba-ssm 2.2.4
- CUDA 11.8 runtime packages

Important notes:

- `cross_mamba` depends on `mamba-ssm`
- `cross_mamba` is expected to run on GPU
- sandboxed environments may not expose `/dev/nvidia*`, while the real host environment can still see the GPU correctly

### 10. Standard Training

The standard training entry point is [`run_model.py`](./run_model.py).

Example:

```bash
python run_model.py \
  --train_path Data/kiba/inductive/seed14/source_train_kiba14.csv \
  --val_path Data/kiba/inductive/seed14/target_train_kiba14.csv \
  --test_path Data/kiba/inductive/seed14/target_test_kiba14.csv \
  --teacher_path Data/kiba/inductive/seed14/kiba14_inductive_teacher_emb.parquet \
  --seed 14 \
  --mode inductive \
  --output_dir trained_models/kiba/inductive/seed14/result
```

What this script does:

1. Loads train, validation, and test CSV files.
2. Truncates protein sequences to length 1022 for ESM compatibility.
3. Resolves and loads matching 3D feature files automatically.
4. Computes ESM embeddings for unique proteins on the fly.
5. Computes ChemBERTa embeddings for unique SMILES on the fly.
6. Loads teacher embeddings from parquet.
7. Builds DGL graphs and starts training.

### 11. Adaptive Model Selection

Adaptive training is provided by [`run_model_adaptive.py`](./run_model_adaptive.py). Enable it with `--adaptive`.

Example:

```bash
python run_model_adaptive.py \
  --train_path Data/kiba/inductive/seed14/source_train_kiba14.csv \
  --val_path Data/kiba/inductive/seed14/target_train_kiba14.csv \
  --test_path Data/kiba/inductive/seed14/target_test_kiba14.csv \
  --teacher_path Data/kiba/inductive/seed14/kiba14_inductive_teacher_emb.parquet \
  --seed 14 \
  --mode inductive \
  --adaptive \
  --selection_metric auroc \
  --output_dir trained_models_adaptive/kiba/inductive/seed14/result
```

Adaptive logic is implemented in [`adaptive_trainer.py`](./adaptive_trainer.py).

The current implementation trains and compares two variants:

1. `3d_ban_mfab`
   - `BCN.TYPE = "ban"`
   - `BCN.HEADS = 2`
   - 3D features enabled
   - MFAB bottleneck fusion enabled

2. `crossmamba_3d_mfab`
   - `BCN.TYPE = "cross_mamba"`
   - 3D features enabled
   - MFAB bottleneck fusion enabled

Selection behavior:

- if `selection_metric=auroc`, selection uses validation AUROC
- if `selection_metric=loss`, selection uses validation loss

Implementation note:

- the CLI currently exposes `auroc`, `auprc`, and `loss`
- the present selector logic effectively handles `auroc` and `loss`
- if you want true AUPRC-based selection, extend the selector to compute and compare validation AUPRC explicitly

Outputs:

- selected best configuration
- saved checkpoints
- `adaptive_selection_log.json`

Practical note:

- adaptive mode roughly doubles training cost because it trains two full configurations sequentially

### 12. Prediction

Inference is provided by [`predict.py`](./predict.py).

Example:

```bash
python predict.py \
  --test_path Data/kiba/inductive/seed14/target_test_kiba14.csv \
  --trained_model trained_models_adaptive/kiba/inductive/seed14/result/best_model_epoch_XX.pth \
  --save_dir results/predictions/kiba_seed14.csv \
  --mode inductive
```

The prediction script:

- recomputes ESM and ChemBERTa features for the given test set
- auto-loads the matching 3D feature file if present
- loads a saved checkpoint
- writes prediction probabilities to CSV

### 13. Running with tmux

#### Standard training

```bash
mkdir -p results/kiba/seed14 trained_models/kiba/inductive/seed14/result

tmux new-session -d -s hisime_kiba14 "
source /path/to/miniconda3/etc/profile.d/conda.sh &&
conda activate hisime_dti &&
cd /path/to/HiSIME-DTI &&
CUDA_VISIBLE_DEVICES=0 python -u run_model.py \
  --train_path Data/kiba/inductive/seed14/source_train_kiba14.csv \
  --val_path Data/kiba/inductive/seed14/target_train_kiba14.csv \
  --test_path Data/kiba/inductive/seed14/target_test_kiba14.csv \
  --teacher_path Data/kiba/inductive/seed14/kiba14_inductive_teacher_emb.parquet \
  --seed 14 \
  --mode inductive \
  --output_dir trained_models/kiba/inductive/seed14/result \
  > results/kiba/seed14/output.txt 2>&1"
```

#### Adaptive training

```bash
mkdir -p results_adaptive/kiba/seed14 trained_models_adaptive/kiba/inductive/seed14/result

tmux new-session -d -s hisime_kiba14_adaptive "
source /path/to/miniconda3/etc/profile.d/conda.sh &&
conda activate hisime_dti &&
cd /path/to/HiSIME-DTI &&
CUDA_VISIBLE_DEVICES=0 python -u run_model_adaptive.py \
  --train_path Data/kiba/inductive/seed14/source_train_kiba14.csv \
  --val_path Data/kiba/inductive/seed14/target_train_kiba14.csv \
  --test_path Data/kiba/inductive/seed14/target_test_kiba14.csv \
  --teacher_path Data/kiba/inductive/seed14/kiba14_inductive_teacher_emb.parquet \
  --seed 14 \
  --mode inductive \
  --adaptive \
  --selection_metric auroc \
  --output_dir trained_models_adaptive/kiba/inductive/seed14/result \
  > results_adaptive/kiba/seed14/output.txt 2>&1"
```

Monitoring:

```bash
tmux ls
tmux capture-pane -pt hisime_kiba14 -S -120
tmux capture-pane -pt hisime_kiba14_adaptive -S -120
tail -n 80 results/kiba/seed14/output.txt
tail -n 80 results_adaptive/kiba/seed14/output.txt
```

### 14. Output Files

Typical adaptive output layout:

```text
trained_models_adaptive/kiba/inductive/seed14/result/
├── adaptive_selection_log.json
├── best_model_epoch_*.pth
└── ...

results_adaptive/kiba/seed14/
└── output.txt
```

Useful artifacts:

- `output.txt`
- `adaptive_selection_log.json`
- saved `.pth` checkpoints

### 15. Scope and Limitations

- bundled benchmark data are inductive-only
- ESM and ChemBERTa features are computed online during training/inference, so preprocessing is relatively expensive
- `cross_mamba` depends on `mamba-ssm` and a usable CUDA environment
- the repository is large because it includes datasets, 3D features, and local pretrained checkpoints

### 16. References

- ESM: https://github.com/facebookresearch/esm
- Mamba: https://github.com/state-spaces/mamba
- RDKit: https://www.rdkit.org/
- DGL-LifeSci: https://github.com/awslabs/dgl-lifesci

---

## 中文

### 1. 项目概述

HiSIME-DTI 是一个面向归纳式药物-靶点相互作用预测的独立项目。这个仓库将代码、归纳式数据划分、预计算 3D 分子特征以及本地预训练权重统一组织在同一个工作空间中，便于直接训练和推理。

### 2. 主要特点

- 数据、3D 特征、ChemBERTa、ESM 均使用仓库内本地路径解析。
- 使用多模态 DTI 建模，包括：
  - 2D 分子图
  - 3D 分子几何信息
  - ChemBERTa 的 SMILES 表征
  - ESM 的蛋白表征
- 支持归纳式实验中的域适应训练。
- 支持自适应模型选择，会比较两种模型配置并按照验证集结果自动选优。
- 提供预测脚本，可直接基于训练好的 checkpoint 做推理。

### 3. 仓库结构

```text
HiSIME-DTI/
├── Data/                       # 归纳式 CSV/parquet 数据划分
├── 3D_Features/                # 预计算的 3D 分子特征字典
├── ChemBERTa-77M-MTR/          # 本地 ChemBERTa 权重
├── pretrained/esm/             # 本地 ESM 权重
├── run_model.py                # 标准训练入口
├── run_model_adaptive.py       # 自适应训练与模型选择入口
├── adaptive_trainer.py         # 自适应选择逻辑
├── predict.py                  # 推理入口
├── dataloader.py               # 2D/3D 图加载
├── compound_3d_tools.py        # 3D 构象与几何特征提取
├── project_paths.py            # 仓库内路径解析
├── *.yaml                      # 训练与模型配置文件
├── requirements.txt            # pip 风格依赖文件
└── environment.exact.yml       # 精确 conda 环境导出文件
```

重要文件：

- [`run_model.py`](./run_model.py)
- [`run_model_adaptive.py`](./run_model_adaptive.py)
- [`adaptive_trainer.py`](./adaptive_trainer.py)
- [`dataloader.py`](./dataloader.py)
- [`compound_3d_tools.py`](./compound_3d_tools.py)
- [`project_paths.py`](./project_paths.py)
- [`environment.exact.yml`](./environment.exact.yml)

### 4. 数据与资源组织

当前仓库已经包含代码默认需要的本地资源：

- `Data/`
- `3D_Features/`
- `ChemBERTa-77M-MTR/`
- `pretrained/esm/`

当前保留的归纳式数据集包括：

- `bindingdb`
- `biosnap`
- `c.elegans`
- `kiba`
- `pdb`

典型训练划分结构如下：

```text
Data/kiba/inductive/seed14/
├── source_train_kiba14.csv
├── target_train_kiba14.csv
├── target_test_kiba14.csv
└── kiba14_inductive_teacher_emb.parquet
```

对应的 3D 特征文件路径保持同样的相对结构：

```text
3D_Features/kiba/inductive/seed14/
├── source_train_kiba14_3d.npy
├── target_train_kiba14_3d.npy
└── target_test_kiba14_3d.npy
```

CSV 到 3D 特征文件的映射是在 [`project_paths.py`](./project_paths.py) 中通过 `resolve_3d_feature_path()` 自动完成的。

### 5. 输入文件格式

#### CSV 文件

最少需要以下三列：

- `SMILES`
- `Protein`
- `Y`

正式实验数据还可能带有附加元信息，例如：

- `target_id`
- `drug_cluster`
- `target_cluster`

示例文件：

- [`Data/sample_data/df_train200.csv`](./Data/sample_data/df_train200.csv)
- [`Data/kiba/inductive/seed14/source_train_kiba14.csv`](./Data/kiba/inductive/seed14/source_train_kiba14.csv)

#### Teacher embedding parquet

标准训练脚本要求 source train 对应一个 teacher embedding 的 parquet 文件。

- 在 `run_model.py` 和 `run_model_adaptive.py` 中加载
- 转成 `teacher_emb`
- 与训练 CSV 按行对齐

示例：

- [`Data/kiba/inductive/seed14/kiba14_inductive_teacher_emb.parquet`](./Data/kiba/inductive/seed14/kiba14_inductive_teacher_emb.parquet)

### 6. 3D 特征表示形式

3D 分子特征保存在 `.npy` 文件中，内容是通过 `allow_pickle=True` 序列化的 Python 字典。

顶层格式：

- key：规范化后或可匹配的 SMILES 字符串
- value：该分子的几何信息字典

当前仓库中实际观察到的字段包括：

- `smiles`
- `num_atoms`
- `edges`
- `atom_pos`
- `bond_length`
- `BondAngleGraph_edges`
- `bond_angle`
- `bond_angle_dirs`

常见张量形状：

- `edges`: `(num_edges, 2)`
- `atom_pos`: `(num_atoms, 3)`
- `bond_length`: `(num_edges,)`
- `BondAngleGraph_edges`: `(num_angle_edges, 2)`
- `bond_angle`: `(num_angle_edges,)`

代码中的使用方式：

- [`compound_3d_tools.py`](./compound_3d_tools.py) 负责生成几何字典
- [`dataloader.py`](./dataloader.py) 负责把几何字典转成 DGL 的 3D 图
- 3D 图节点特征的布局为：
  - 前 3 维：xyz 坐标
  - 后续维度：从 2D 图复制过来的原子特征
- `bond_length` 被作为边特征挂到 3D 图上

回退逻辑：

- 如果某个划分对应的 `_3d.npy` 文件不存在，则该划分自动退化为 2D-only 使用方式
- 如果某个样本的 SMILES 在 3D 字典中找不到，则为该样本构造一个空的占位 3D 图

### 7. 3D 特征如何提取

3D 特征提取工具实现在 [`compound_3d_tools.py`](./compound_3d_tools.py) 中。

主要入口函数：

- `smiles_to_3d_graph_data(smiles)`
- `mol_to_geognn_graph_data_MMFF3d(mol)`
- `mol_to_geognn_graph_data(mol, atom_poses, dir_type='HT')`

提取逻辑如下：

- 对于原子数不超过 400 的分子：
  - 先加氢
  - 用 RDKit 生成多个构象
  - 用 MMFF 做优化
  - 选择能量最低的构象
- 对于大于 400 原子的分子：
  - 为避免构象生成代价过高，退化为 2D 坐标

最终计算出的几何信息包括：

- 原子坐标
- 有向化学键边
- 键长
- 键角图边
- 键角

如果你要给自己的新划分生成 `_3d.npy`，建议流程如下：

1. 读取某个 CSV 划分并收集唯一 `SMILES`
2. 对每个分子调用 `smiles_to_3d_graph_data(smiles)`
3. 用 `np.save(..., allow_pickle=True)` 保存 `{smiles: data_dict}` 字典
4. 文件名沿用原划分名并追加 `_3d.npy`
5. 放到 `3D_Features/` 下镜像对应的目录里

最小示例：

```python
import numpy as np
import pandas as pd
from compound_3d_tools import smiles_to_3d_graph_data

df = pd.read_csv("Data/myset/inductive/seed12/source_train_myset12.csv")
smiles_list = df["SMILES"].drop_duplicates().tolist()

feature_dict = {}
for smi in smiles_list:
    data = smiles_to_3d_graph_data(smi)
    if data is not None:
        feature_dict[smi] = data

np.save(
    "3D_Features/myset/inductive/seed12/source_train_myset12_3d.npy",
    feature_dict,
    allow_pickle=True,
)
```

注意：

- RDKit 在处理中可能会对 SMILES 做规范化
- `_3d.npy` 中保存的 key 应尽量与实际加载时使用的 SMILES 保持一致
- 如果某些分子 RDKit 解析失败，最好记录并跳过，否则后续训练时这些样本只能使用空 3D 图

### 8. 预训练模型

#### ESM

代码使用本地 ESM-1b 权重：

- `pretrained/esm/esm1b_t33_650M_UR50S.pt`

加载函数位于 [`project_paths.py`](./project_paths.py) 中的 `load_local_esm_model()`。

#### ChemBERTa

代码使用本地 ChemBERTa 权重目录：

- `ChemBERTa-77M-MTR/`

在以下脚本中会被加载：

- [`run_model.py`](./run_model.py)
- [`run_model_adaptive.py`](./run_model_adaptive.py)
- [`predict.py`](./predict.py)

### 9. 环境配置

仓库中现在保留了两个依赖文件：

- [`requirements.txt`](./requirements.txt)
  - 轻量级 pip 风格依赖列表
  - 适合你已经自己管理好了 CUDA / PyTorch 环境的情况
- [`environment.exact.yml`](./environment.exact.yml)
  - 从一套已验证可运行的本地训练环境完整导出
  - 保留了精确版本号
  - 导出后的环境名为 `hisime_dti`
  - 同时移除了本机专属的 `prefix`

推荐两种环境搭建方式：

#### 方式 A：按完整 conda 环境重建

```bash
conda env create -f environment.exact.yml
conda activate hisime_dti
```

#### 方式 B：在现有环境上补装

```bash
conda activate your_env
pip install -r requirements.txt
```

当前精确环境里的关键版本包括：

- PyTorch 2.3.0
- DGL 2.1.0
- DGLLife 0.3.2
- Transformers 4.47.0
- fair-esm 2.0.0
- ESM 3.2.0
- RDKit 2024.9.6
- mamba-ssm 2.2.4
- CUDA 11.8 运行时相关包

重要说明：

- `cross_mamba` 依赖 `mamba-ssm`
- `cross_mamba` 预期应在 GPU 环境下运行
- 沙箱环境可能看不到 `/dev/nvidia*`，但真实宿主环境中 GPU 仍然可以正常可见

### 10. 标准训练

标准训练入口是 [`run_model.py`](./run_model.py)。

示例命令：

```bash
python run_model.py \
  --train_path Data/kiba/inductive/seed14/source_train_kiba14.csv \
  --val_path Data/kiba/inductive/seed14/target_train_kiba14.csv \
  --test_path Data/kiba/inductive/seed14/target_test_kiba14.csv \
  --teacher_path Data/kiba/inductive/seed14/kiba14_inductive_teacher_emb.parquet \
  --seed 14 \
  --mode inductive \
  --output_dir trained_models/kiba/inductive/seed14/result
```

这个脚本会做的事情：

1. 读取 train、val、test 三个 CSV
2. 将蛋白序列截断到 1022，以兼容 ESM
3. 自动定位并加载对应的 3D 特征文件
4. 在线计算唯一蛋白的 ESM 表征
5. 在线计算唯一 SMILES 的 ChemBERTa 表征
6. 从 parquet 中读取 teacher embedding
7. 构建 DGL 图并开始训练

### 11. 自适应模型选择

自适应训练入口是 [`run_model_adaptive.py`](./run_model_adaptive.py)，通过 `--adaptive` 开启。

示例命令：

```bash
python run_model_adaptive.py \
  --train_path Data/kiba/inductive/seed14/source_train_kiba14.csv \
  --val_path Data/kiba/inductive/seed14/target_train_kiba14.csv \
  --test_path Data/kiba/inductive/seed14/target_test_kiba14.csv \
  --teacher_path Data/kiba/inductive/seed14/kiba14_inductive_teacher_emb.parquet \
  --seed 14 \
  --mode inductive \
  --adaptive \
  --selection_metric auroc \
  --output_dir trained_models_adaptive/kiba/inductive/seed14/result
```

自适应逻辑实现在 [`adaptive_trainer.py`](./adaptive_trainer.py) 中。

当前实现会训练并比较两个配置：

1. `3d_ban_mfab`
   - `BCN.TYPE = "ban"`
   - `BCN.HEADS = 2`
   - 开启 3D 特征
   - 开启 MFAB bottleneck 融合

2. `crossmamba_3d_mfab`
   - `BCN.TYPE = "cross_mamba"`
   - 开启 3D 特征
   - 开启 MFAB bottleneck 融合

选择逻辑：

- `selection_metric=auroc` 时按验证集 AUROC 选
- `selection_metric=loss` 时按验证集 loss 选

实现边界说明：

- 当前命令行虽然暴露了 `auroc`、`auprc`、`loss`
- 但当前选择器逻辑实际上只真正处理了 `auroc` 和 `loss`
- 如果你想严格按 AUPRC 选模型，需要继续扩展选择器，让它显式计算并比较验证集 AUPRC

输出内容包括：

- 最优配置
- 保存的 checkpoint
- `adaptive_selection_log.json`

实践上要注意：

- 自适应模式会顺序训练两套完整模型，所以总训练成本大约翻倍

### 12. 预测

推理入口是 [`predict.py`](./predict.py)。

示例命令：

```bash
python predict.py \
  --test_path Data/kiba/inductive/seed14/target_test_kiba14.csv \
  --trained_model trained_models_adaptive/kiba/inductive/seed14/result/best_model_epoch_XX.pth \
  --save_dir results/predictions/kiba_seed14.csv \
  --mode inductive
```

预测脚本会：

- 为测试集重新计算 ESM 与 ChemBERTa 表征
- 如果存在匹配的 3D 文件则自动加载
- 读取保存好的模型 checkpoint
- 将预测概率写回 CSV

### 13. 使用 tmux 运行

#### 标准训练

```bash
mkdir -p results/kiba/seed14 trained_models/kiba/inductive/seed14/result

tmux new-session -d -s hisime_kiba14 "
source /path/to/miniconda3/etc/profile.d/conda.sh &&
conda activate hisime_dti &&
cd /path/to/HiSIME-DTI &&
CUDA_VISIBLE_DEVICES=0 python -u run_model.py \
  --train_path Data/kiba/inductive/seed14/source_train_kiba14.csv \
  --val_path Data/kiba/inductive/seed14/target_train_kiba14.csv \
  --test_path Data/kiba/inductive/seed14/target_test_kiba14.csv \
  --teacher_path Data/kiba/inductive/seed14/kiba14_inductive_teacher_emb.parquet \
  --seed 14 \
  --mode inductive \
  --output_dir trained_models/kiba/inductive/seed14/result \
  > results/kiba/seed14/output.txt 2>&1"
```

#### 自适应训练

```bash
mkdir -p results_adaptive/kiba/seed14 trained_models_adaptive/kiba/inductive/seed14/result

tmux new-session -d -s hisime_kiba14_adaptive "
source /path/to/miniconda3/etc/profile.d/conda.sh &&
conda activate hisime_dti &&
cd /path/to/HiSIME-DTI &&
CUDA_VISIBLE_DEVICES=0 python -u run_model_adaptive.py \
  --train_path Data/kiba/inductive/seed14/source_train_kiba14.csv \
  --val_path Data/kiba/inductive/seed14/target_train_kiba14.csv \
  --test_path Data/kiba/inductive/seed14/target_test_kiba14.csv \
  --teacher_path Data/kiba/inductive/seed14/kiba14_inductive_teacher_emb.parquet \
  --seed 14 \
  --mode inductive \
  --adaptive \
  --selection_metric auroc \
  --output_dir trained_models_adaptive/kiba/inductive/seed14/result \
  > results_adaptive/kiba/seed14/output.txt 2>&1"
```

监控命令：

```bash
tmux ls
tmux capture-pane -pt hisime_kiba14 -S -120
tmux capture-pane -pt hisime_kiba14_adaptive -S -120
tail -n 80 results/kiba/seed14/output.txt
tail -n 80 results_adaptive/kiba/seed14/output.txt
```

### 14. 输出文件

典型自适应训练输出结构：

```text
trained_models_adaptive/kiba/inductive/seed14/result/
├── adaptive_selection_log.json
├── best_model_epoch_*.pth
└── ...

results_adaptive/kiba/seed14/
└── output.txt
```

主要结果文件：

- `output.txt`
- `adaptive_selection_log.json`
- 保存下来的 `.pth` checkpoint

### 15. 当前范围与限制

- 当前仓库内基准数据只保留了 inductive
- ESM 和 ChemBERTa 特征是在训练/推理时在线计算的，因此前处理开销不小
- `cross_mamba` 依赖 `mamba-ssm` 和可用的 CUDA 环境
- 由于仓库直接包含了数据、3D 特征和本地预训练权重，所以总体体积较大

### 16. 参考项目

- ESM: https://github.com/facebookresearch/esm
- Mamba: https://github.com/state-spaces/mamba
- RDKit: https://www.rdkit.org/
- DGL-LifeSci: https://github.com/awslabs/dgl-lifesci
