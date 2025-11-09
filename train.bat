@echo off
chcp 65001 >nul
echo ======================================
echo Decoder-Only Transformer 训练
echo ======================================

mkdir data 2>nul
mkdir checkpoints 2>nul
mkdir results 2>nul

echo.
echo 正在训练模型...
python src/train.py --config configs/base.yaml --seed 42

echo.
echo 完成！
pause

