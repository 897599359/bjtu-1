#!/bin/bash
# Transformer训练脚本

echo "======================================"
echo "Decoder-Only Transformer 训练"
echo "======================================"

# 设置随机种子
SEED=42

# 创建目录
mkdir -p data checkpoints results

echo ""
echo "1. 训练基础模型 (4 heads, 30 epochs)"
echo "--------------------------------------"
python src/train.py --config configs/base.yaml --seed $SEED

echo ""
echo "2. 生成文本样本"
echo "--------------------------------------"
python src/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --prompt "ROMEO:" \
    --num_samples 3 \
    --max_len 300

echo ""
echo "======================================"
echo "完成!"
echo "结果保存在 results/ 目录"
echo "======================================"

