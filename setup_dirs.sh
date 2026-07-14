#!/bin/bash

# 自适应训练启动脚本
# 确保在正确的目录下运行

# 切换到脚本所在目录
cd "$(dirname "$0")"

echo "Current directory: $(pwd)"
echo "Creating output directories..."

# 创建所有必要的目录
mkdir -p results_adaptive/kiba/seed{12,14,16,18,20}
mkdir -p results_adaptive/bindingdb/seed{12,14,16,18,20}
mkdir -p trained_models_adaptive/kiba/inductive/seed{12,14,16,18,20}/result
mkdir -p trained_models_adaptive/bindingdb/inductive/seed{12,14,16,18,20}/result

echo "Directories created successfully!"
echo ""
echo "You can now run the training commands."
echo ""
echo "Example for KIBA seed12:"
echo "CUDA_VISIBLE_DEVICES=0 python run_model_adaptive.py \\"
echo "  --train_path Data/kiba/inductive/seed12/source_train_kiba12.csv \\"
echo "  --val_path Data/kiba/inductive/seed12/target_train_kiba12.csv \\"
echo "  --test_path Data/kiba/inductive/seed12/target_test_kiba12.csv \\"
echo "  --seed 12 --mode inductive \\"
echo "  --teacher_path Data/kiba/inductive/seed12/kiba12_inductive_teacher_emb.parquet \\"
echo "  --output_dir trained_models_adaptive/kiba/inductive/seed12/result \\"
echo "  --adaptive \\"
echo "  --selection_metric auroc \\"
echo "  > results_adaptive/kiba/seed12/output.txt 2>&1"
