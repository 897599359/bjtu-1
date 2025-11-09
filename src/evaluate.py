"""
模型评估和文本生成
"""
import argparse
import torch
import torch.nn.functional as F
from model import TransformerLM
from data_loader import get_dataset


@torch.no_grad()
def generate_text(model, dataset, prompt="ROMEO:", max_len=200, temperature=0.8, device='cpu'):
    """自回归生成文本"""
    model.eval()
    
    # 编码prompt
    if prompt:
        indices = dataset.encode(prompt)
    else:
        indices = [0]
    
    indices = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
    
    # 生成
    for _ in range(max_len):
        logits = model(indices)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1)
        indices = torch.cat([indices, next_idx], dim=1)
    
    # 解码
    generated = dataset.decode(indices[0].cpu().tolist())
    return generated


def load_model(checkpoint_path, device='cpu'):
    """加载训练好的模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # 加载数据集
    dataset = get_dataset(config['dataset'], config.get('data_dir', 'data'))
    
    # 创建模型（使用原来的max_len，位置编码会动态扩展）
    model = TransformerLM(
        vocab_size=dataset.vocab_size,
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        d_ff=config['d_ff'],
        n_layers=config['n_layers'],
        max_len=config['seq_len'],  # 使用原来的seq_len，位置编码会动态扩展
        dropout=0.0
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    return model, dataset, config


def main():
    parser = argparse.ArgumentParser(description='评估Transformer模型并生成文本')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--prompt', type=str, default='ROMEO:', help='生成文本的提示')
    parser.add_argument('--max_len', type=int, default=300, help='生成文本的最大长度')
    parser.add_argument('--temperature', type=float, default=0.8, help='采样温度')
    parser.add_argument('--num_samples', type=int, default=3, help='生成样本数量')
    parser.add_argument('--device', type=str, default='auto', help='设备')
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"正在加载模型: {args.checkpoint}")
    model, dataset, config = load_model(args.checkpoint, device)
    
    print(f"\n模型配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print(f"\n生成 {args.num_samples} 个文本样本:")
    print("=" * 80)
    
    for i in range(args.num_samples):
        print(f"\n样本 {i+1}:")
        print("-" * 80)
        generated = generate_text(
            model, dataset,
            prompt=args.prompt,
            max_len=args.max_len,
            temperature=args.temperature,
            device=device
        )
        print(generated)
        print("-" * 80)


if __name__ == '__main__':
    main()

