"""
数据加载和预处理
支持Tiny Shakespeare等字符级语言建模数据集
"""
import os
import requests
import torch
from torch.utils.data import Dataset, DataLoader


class CharDataset(Dataset):
    """字符级数据集"""
    
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y


class TinyShakespeareDataset:
    """Tiny Shakespeare数据集加载器"""
    
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.data_path = os.path.join(data_dir, 'tiny_shakespeare.txt')
        
        # 下载数据（如果不存在）
        if not os.path.exists(self.data_path):
            self.download()
        
        # 加载并处理数据
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        
        # 创建字符到索引的映射
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
        # 将文本转换为索引
        self.data = [self.char_to_idx[ch] for ch in self.text]
        
        # 划分训练集和验证集（90% / 10%）
        split_idx = int(0.9 * len(self.data))
        self.train_data = self.data[:split_idx]
        self.val_data = self.data[split_idx:]
        
        print(f"数据集加载完成:")
        print(f"  总字符数: {len(self.text):,}")
        print(f"  词汇表大小: {self.vocab_size}")
        print(f"  训练集大小: {len(self.train_data):,}")
        print(f"  验证集大小: {len(self.val_data):,}")
        print(f"  字符集示例: {self.chars[:20]}")
    
    def download(self):
        """下载Tiny Shakespeare数据集"""
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        print(f"正在下载Tiny Shakespeare数据集...")
        response = requests.get(url)
        with open(self.data_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"下载完成: {self.data_path}")
    
    def get_dataloaders(self, seq_len=128, batch_size=32, num_workers=0):
        """创建训练和验证数据加载器"""
        train_dataset = CharDataset(self.train_data, seq_len)
        val_dataset = CharDataset(self.val_data, seq_len)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def decode(self, indices):
        """将索引序列解码为文本"""
        return ''.join([self.idx_to_char[idx] for idx in indices])
    
    def encode(self, text):
        """将文本编码为索引序列"""
        return [self.char_to_idx[ch] for ch in text]


def get_dataset(dataset_name='tiny_shakespeare', data_dir='data'):
    """获取数据集"""
    if dataset_name == 'tiny_shakespeare':
        return TinyShakespeareDataset(data_dir)
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")


if __name__ == '__main__':
    # 测试数据加载
    dataset = TinyShakespeareDataset()
    train_loader, val_loader = dataset.get_dataloaders(seq_len=128, batch_size=32)
    
    print("\n测试数据加载:")
    for x, y in train_loader:
        print(f"输入形状: {x.shape}")
        print(f"目标形状: {y.shape}")
        print(f"输入示例: {dataset.decode(x[0].tolist()[:50])}")
        break

