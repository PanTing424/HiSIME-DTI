from yacs.config import CfgNode as CN

_C = CN()

# Drug feature extractor
_C.DRUG = CN()
_C.DRUG.NODE_IN_FEATS = 75

_C.DRUG.PADDING = True

_C.DRUG.HIDDEN_LAYERS = [128, 128, 128]
_C.DRUG.NODE_IN_EMBEDDING = 128
_C.DRUG.MAX_NODES = 290

# Protein feature extractor
_C.PROTEIN = CN()
_C.PROTEIN.NUM_FILTERS = [128, 128, 128]
_C.PROTEIN.KERNEL_SIZE = [3, 6, 9]
_C.PROTEIN.EMBEDDING_DIM = 128
_C.PROTEIN.PADDING = True

# BCN setting
_C.BCN = CN()
_C.BCN.HEADS = 2
_C.BCN.TYPE = "ban"                 # 交互层类型默认ban，可在YAML中设为cross_mamba或parallel
_C.BCN.PARALLEL_MERGE = "gated-sum" # parallel模式下的融合策略：[sum, gated-sum, concat]

# CROSS_MAMBA setting
_C.CROSS_MAMBA = CN()
_C.CROSS_MAMBA.D_MODEL = 128
_C.CROSS_MAMBA.N_LAYERS = 4
_C.CROSS_MAMBA.D_STATE = 64
_C.CROSS_MAMBA.D_CONV = 4
_C.CROSS_MAMBA.EXPAND = 2
_C.CROSS_MAMBA.DROPOUT = 0.1
_C.CROSS_MAMBA.USE_MAMBA2 = True
_C.CROSS_MAMBA.MAX_LEN = 8192
_C.CROSS_MAMBA.USE_POS = False
_C.CROSS_MAMBA.LOWRANK = 64
_C.CROSS_MAMBA.DOWNSAMPLE = 1

# MLP decoder
_C.DECODER = CN()
_C.DECODER.NAME = "MLP"
_C.DECODER.IN_DIM = 256
_C.DECODER.HIDDEN_DIM = 512
_C.DECODER.OUT_DIM = 128
_C.DECODER.BINARY = 1

# SOLVER
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCH = 100
_C.SOLVER.BATCH_SIZE = 64
_C.SOLVER.NUM_WORKERS = 0
_C.SOLVER.LR = 5e-5
_C.SOLVER.DA_LR = 1e-3
_C.SOLVER.SEED = 2064

# RESULT
_C.RESULT = CN()
_C.RESULT.OUTPUT_DIR = "./trained_models"
_C.RESULT.SAVE_MODEL = True

# Domain adaptation
_C.DA = CN()
_C.DA.TASK = False
_C.DA.METHOD = "CDAN"
_C.DA.USE = False
_C.DA.INIT_EPOCH = 10
_C.DA.LAMB_DA = 1.0  # 改为float类型以兼容YAML中的0.3
_C.DA.RANDOM_LAYER = False
_C.DA.ORIGINAL_RANDOM = False
_C.DA.RANDOM_DIM = None
_C.DA.USE_ENTROPY = True
_C.DA.ALPHA = 0.2
_C.DA.LAMB_MAX = 0.05

# Multimodal Fusion Setting (注意力瓶颈融合)
_C.FUSION = CN()
_C.FUSION.TYPE = "bottleneck"           # 融合类型：bottleneck (注意力瓶颈) 或 attention (原始注意力)
_C.FUSION.NUM_LATENTS = 4               # 瓶颈tokens数量：推荐 4-8
_C.FUSION.DROPOUT = 0.1                 # Dropout率
_C.FUSION.USE_RESIDUAL = True           # 是否在融合结果和原始模态之间再做一次残差混合
_C.FUSION.RESIDUAL_MODE = "gated"       # 残差融合模式：["fixed", "gated", "learned"]
_C.FUSION.FIXED_ALPHA = 0.5             # fixed模式下的固定权重

# Adaptive Model Selection (方案四)
_C.ADAPTIVE = CN()
_C.ADAPTIVE.ENABLE = False              # 是否启用自适应选择
_C.ADAPTIVE.USE_CROSSMAMBA = True       # 是否使用CrossMamba（手动指定时）
_C.ADAPTIVE.USE_3D_FEATURES = True      # 是否使用3D特征
_C.ADAPTIVE.EARLY_STOP_EPOCHS = 10      # 早期评估的epoch数
_C.ADAPTIVE.SELECTION_METRIC = "auroc"  # 选择指标：auroc, auprc, f1

def get_cfg_defaults():
    return _C.clone()

# Extra defaults for CROSS_MAMBA to avoid YAML merge KeyError
_C.CROSS_MAMBA.compute_map_train = False
_C.CROSS_MAMBA.bidirectional = False
_C.CROSS_MAMBA.eval_use_map = True
