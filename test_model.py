"""
测试模型实现
"""
import torch
from src.model import TransformerLM

print("=" * 60)
print("Decoder-Only Transformer 测试")
print("=" * 60)

vocab_size = 65
batch_size = 4
seq_len = 128

print("\n1. 创建模型...")
model = TransformerLM(
    vocab_size=vocab_size,
    d_model=256,
    n_heads=4,
    d_ff=1024,
    n_layers=4,
    max_len=512,
    dropout=0.1
)

print(f"   ✓ 模型参数量: {sum(p.numel() for p in model.parameters()):,}")

print("\n2. 测试前向传播...")
x = torch.randint(0, vocab_size, (batch_size, seq_len))
logits = model(x)

print(f"   ✓ 输入形状: {x.shape}")
print(f"   ✓ 输出形状: {logits.shape}")
assert logits.shape == (batch_size, seq_len, vocab_size), "输出形状不匹配！"

print("\n3. 验证Causal Mask...")
mask = model.create_causal_mask(5, x.device)
print("   5x5 Causal Mask (下三角矩阵):")
print(mask.int())

print("\n4. 测试反向传播...")
loss = logits.mean()
loss.backward()
print("   ✓ 梯度计算成功")

print("\n5. 测试文本生成...")
model.eval()
with torch.no_grad():
    prompt = torch.randint(0, vocab_size, (1, 10))
    for _ in range(5):
        logits = model(prompt)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        prompt = torch.cat([prompt, next_token], dim=1)
print(f"   ✓ 生成序列长度: {prompt.shape[1]}")

print("\n" + "=" * 60)
print("✅ 所有测试通过!")
print("=" * 60)
print("\n现在可以运行训练:")
print("  python src/train.py --config configs/base.yaml --seed 42")
print("=" * 60)

