from pathlib import Path

import esm


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "Data"
FEATURES_3D_DIR = PROJECT_ROOT / "3D_Features"
CHEMBERTA_DIR = PROJECT_ROOT / "ChemBERTa-77M-MTR"
ESM_DIR = PROJECT_ROOT / "pretrained" / "esm"
ESM_MODEL_PATH = ESM_DIR / "esm1b_t33_650M_UR50S.pt"


def resolve_3d_feature_path(csv_path: str) -> Path:
    csv_path = Path(csv_path).resolve()

    try:
        rel = csv_path.relative_to(DATA_DIR.resolve())
        return (FEATURES_3D_DIR / rel).with_name(rel.stem + "_3d.npy")
    except ValueError:
        csv_str = str(csv_path)
        if "/Data/" in csv_str:
            return Path(csv_str.replace("/Data/", "/3D_Features/").replace(".csv", "_3d.npy"))
        return csv_path.with_name(csv_path.stem + "_3d.npy")


def load_local_esm_model():
    if ESM_MODEL_PATH.exists() and hasattr(esm.pretrained, "load_model_and_alphabet_local"):
        return esm.pretrained.load_model_and_alphabet_local(str(ESM_MODEL_PATH))
    return esm.pretrained.esm1b_t33_650M_UR50S()

