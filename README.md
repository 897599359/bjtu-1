# Decoder-Only Transformer 从零实现

从零手工实现Decoder-Only Transformer（类似GPT），并在Tiny Shakespeare数据集上完成语言建模任务。

## 📁 项目结构

```
.
├── src/
│   ├── model.py           # Transformer核心实现
│   ├── data_loader.py     # 数据加载（Tiny Shakespeare）
│   ├── train.py           # 训练脚本
│   └── evaluate.py        # 评估和文本生成
├── configs/
│   ├── base.yaml          # 基础配置（4 heads, 30 epochs）
│   ├── small.yaml         # 小型配置（快速测试）
│   └── ablation_2heads.yaml  # 消融实验（2 heads）
├── scripts/
│   └── run.sh             # 完整训练脚本
├── requirements.txt       # Python依赖
├── train.bat              # Windows训练脚本
├── test_model.py          # 模型测试
└── README.md              # 本文件
```

## 🎯 核心组件实现

### 1. Scaled Dot-Product Attention（缩放点积注意力）

```python
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

- 计算Query和Key的点积
- 除以√d_k进行缩放
- 应用softmax归一化
- 与Value加权求和

### 2. Multi-Head Attention（多头注意力）

```python
MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

- 将输入投影到多个子空间
- 并行计算多个注意力头
- 拼接并线性变换

### 3. Position-wise Feed-Forward Network（逐位置前馈网络）

```python
FFN(x) = ReLU(xW_1 + b_1)W_2 + b_2
```

- 两层全连接网络
- 独立应用于每个位置

### 4. Residual Connection + Layer Normalization（残差+归一化）

```python
output = LayerNorm(x + Sublayer(x))
```

- 稳定训练
- 缓解梯度消失
- 加速收敛

### 5. Positional Encoding（位置编码）

```python
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

- 正弦和余弦函数编码位置信息
- 使模型能够学习相对位置关系

### 6. Causal Mask（因果掩码）

```python
mask[i,j] = 1 if j ≤ i else 0
```

- 确保每个位置只能看到它之前的token
- 实现自回归特性（用于语言建模）

## 🚀 快速开始

### 步骤1: 安装依赖

```bash
pip install -r requirements.txt
```

或手动安装：
```bash
pip install torch numpy matplotlib requests tqdm pyyaml
```


### 步骤3: 开始训练

**Linux/Mac**:
```bash
bash scripts/run.sh
```

**Windows**:
```bash
train.bat
```

**或手动运行**:
```bash
python src/train.py --config configs/base.yaml --seed 42
```

### 步骤4: 生成文本

```bash
python src/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --prompt "ROMEO:" \
    --num_samples 3 \
    --max_len 300
```

## ⚙️ 配置说明

### 基础配置（configs/base.yaml）

| 参数 | 值 | 说明 |
|------|-----|------|
| d_model | 256 | 嵌入维度 |
| n_heads | 4 | 注意力头数 |
| d_ff | 1024 | FFN隐藏层维度 |
| n_layers | 4 | Decoder层数 |
| seq_len | 128 | 序列长度 |
| batch_size | 64 | 批次大小 |
| learning_rate | 3e-4 | 初始学习率 |
| epochs | 30 | 训练轮数 |
| dropout | 0.1 | Dropout率 |

**模型参数量**: ~4.2M

## 📊 数据集

### Tiny Shakespeare

- **来源**: Karpathy's char-rnn
- **大小**: ~1MB, ~1.1M字符
- **词汇表**: 65个唯一字符
- **任务**: 字符级语言建模
- **分割**: 90% 训练, 10% 验证
- **自动下载**: 首次运行时自动下载

## 🧪 消融实验

### 实验1: 不同注意力头数

```bash
# 基础配置（4 heads）
python src/train.py --config configs/base.yaml --seed 42

# 消融实验（2 heads）
python src/train.py --config configs/ablation_2heads.yaml --seed 42
```


**结论**: 更多的注意力头提升了模型性能，验证了多头机制的有效性。

### 实验2: 有无位置编码（可选）

修改`src/model.py`，注释掉位置编码，观察性能显著下降，证明位置信息对序列建模的重要性。

## 📈 训练结果

训练完成后会自动生成：

1. **训练曲线**: `results/training_curves.png`
   - 训练/验证损失曲线
   - 困惑度曲线
   - 学习率调度曲线

2. **模型检查点**: `checkpoints/`
   - `best_model.pt` - 验证损失最低的模型
   - `final_model.pt` - 最后一个epoch的模型
   - `checkpoint_epoch_X.pt` - 定期保存的检查点

3. **生成文本**: 莎士比亚风格的文本样本

## 💻 硬件要求

- **最低**: CPU, 16GB RAM
- **推荐**: NVIDIA GPU (2GB+ VRAM), 16GB RAM

## 🔧 训练技巧

1. **优化器**: AdamW (β1=0.9, β2=0.98, weight_decay=0.01)
2. **学习率调度**: Cosine Annealing (3e-4 → 1e-6)
3. **梯度裁剪**: max_norm=1.0
4. **权重初始化**: Normal(mean=0, std=0.02)
5. **Dropout**: 0.1

## 📝 关键代码片段

### Causal Mask创建

```python
def create_causal_mask(seq_len, device):
    """创建下三角mask，防止看到未来token"""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask == 0
```

### 自回归生成

```python
def generate_text(model, prompt, max_len=200):
    for _ in range(max_len):
        logits = model(prompt)       # 前向传播
        next_token = sample(logits)  # 采样
        prompt = cat(prompt, next_token)  # 追加
    return prompt
```

---

**最后更新**: 2025年11月

