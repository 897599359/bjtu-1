"""
Decoder-Only Transformer训练脚本
"""
import os
import time
import math
import argparse
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import TransformerLM
from data_loader import get_dataset


class Trainer:
    """训练器"""
    
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # 优化器
        self.optimizer = AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # 学习率调度器
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'],
            eta_min=config.get('min_lr', 1e-6)
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 梯度裁剪
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.train_perplexities = []
        self.val_perplexities = []
        self.learning_rates = []
        
        # 最佳模型
        self.best_val_loss = float('inf')
        
        # 保存路径
        self.save_dir = config.get('save_dir', 'checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs('results', exist_ok=True)
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_tokens = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(self.device), y.to(self.device)
            
            # 前向传播
            logits = self.model(x)
            
            # 计算损失
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # 更新参数
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'ppl': f'{math.exp(loss.item()):.2f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        avg_loss = total_loss / total_tokens
        avg_ppl = math.exp(avg_loss)
        
        return avg_loss, avg_ppl
    
    @torch.no_grad()
    def evaluate(self):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        for x, y in tqdm(self.val_loader, desc="Evaluating"):
            x, y = x.to(self.device), y.to(self.device)
            
            logits = self.model(x)
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )
            
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
        
        avg_loss = total_loss / total_tokens
        avg_ppl = math.exp(avg_loss)
        
        return avg_loss, avg_ppl
    
    def train(self):
        """完整训练流程"""
        print(f"\n开始训练 Decoder-Only Transformer...")
        print(f"设备: {self.device}")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(1, self.config['epochs'] + 1):
            # 训练
            train_loss, train_ppl = self.train_epoch(epoch)
            
            # 评估
            val_loss, val_ppl = self.evaluate()
            
            # 学习率调度
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_perplexities.append(train_ppl)
            self.val_perplexities.append(val_ppl)
            self.learning_rates.append(current_lr)
            
            # 打印结果
            print(f"\nEpoch {epoch}/{self.config['epochs']}:")
            print(f"  Train Loss: {train_loss:.4f}, Train PPL: {train_ppl:.2f}")
            print(f"  Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")
            print(f"  Learning Rate: {current_lr:.2e}")
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, 'best_model.pt')
                print(f"  ✓ 保存最佳模型 (Val Loss: {val_loss:.4f})")
            
            # 定期保存
            if epoch % self.config.get('save_every', 10) == 0:
                self.save_checkpoint(epoch, f'checkpoint_epoch_{epoch}.pt')
        
        total_time = time.time() - start_time
        print(f"\n训练完成! 总时间: {total_time/60:.2f} 分钟")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")
        
        # 保存最终模型
        self.save_checkpoint(self.config['epochs'], 'final_model.pt')
        
        # 绘制训练曲线
        self.plot_curves()
    
    def save_checkpoint(self, epoch, filename):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, path)
    
    def plot_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss曲线
        axes[0, 0].plot(epochs, self.train_losses, label='Train Loss', marker='o')
        axes[0, 0].plot(epochs, self.val_losses, label='Val Loss', marker='s')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Perplexity曲线
        axes[0, 1].plot(epochs, self.train_perplexities, label='Train PPL', marker='o')
        axes[0, 1].plot(epochs, self.val_perplexities, label='Val PPL', marker='s')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Perplexity')
        axes[0, 1].set_title('Training and Validation Perplexity')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 学习率曲线
        axes[1, 0].plot(epochs, self.learning_rates, marker='o', color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].grid(True)
        axes[1, 0].set_yscale('log')
        
        # Loss对比（最后几个epoch）
        if len(epochs) > 10:
            recent_epochs = list(epochs[-10:])
            axes[1, 1].plot(recent_epochs, self.train_losses[-10:], label='Train Loss', marker='o')
            axes[1, 1].plot(recent_epochs, self.val_losses[-10:], label='Val Loss', marker='s')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_title('Recent Loss (Last 10 Epochs)')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('results/training_curves.png', dpi=300, bbox_inches='tight')
        print(f"\n训练曲线已保存到: results/training_curves.png")
        plt.close()


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='训练Decoder-Only Transformer')
    parser.add_argument('--config', type=str, default='configs/base.yaml', help='配置文件路径')
    parser.add_argument('--device', type=str, default='auto', help='设备 (cuda/cpu/auto)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 加载配置
    config = load_config(args.config)
    print("配置信息:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # 加载数据集
    print(f"\n正在加载数据集: {config['dataset']}...")
    dataset = get_dataset(config['dataset'], config.get('data_dir', 'data'))
    train_loader, val_loader = dataset.get_dataloaders(
        seq_len=config['seq_len'],
        batch_size=config['batch_size'],
        num_workers=config.get('num_workers', 0)
    )
    
    # 创建模型
    print("\n正在创建模型...")
    model = TransformerLM(
        vocab_size=dataset.vocab_size,
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        d_ff=config['d_ff'],
        n_layers=config['n_layers'],
        max_len=config['seq_len'],
        dropout=config['dropout']
    )
    
    # 创建训练器并开始训练
    trainer = Trainer(model, train_loader, val_loader, config, device)
    trainer.train()


if __name__ == '__main__':
    main()

