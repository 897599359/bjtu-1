"""
Decoder-Only Transformer模型实现（类似GPT）
适合语言建模任务
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """正弦位置编码
    
    使用正弦和余弦函数为序列中的每个位置编码:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # 如果序列长度超过预计算的位置编码，动态扩展
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            # 动态计算额外的位置编码
            device = x.device
            d_model = self.pe.size(2)
            additional_len = seq_len - self.pe.size(1)
            
            # 计算新的位置编码
            position = torch.arange(self.pe.size(1), seq_len, dtype=torch.float, device=device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * 
                                (-math.log(10000.0) / d_model))
            
            pe_additional = torch.zeros(additional_len, d_model, device=device)
            pe_additional[:, 0::2] = torch.sin(position * div_term)
            pe_additional[:, 1::2] = torch.cos(position * div_term)
            pe_additional = pe_additional.unsqueeze(0)
            
            # 拼接
            self.pe = torch.cat([self.pe, pe_additional], dim=1)
        
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """
    
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        
        # QK^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 应用causal mask（防止看到未来token）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        output = torch.matmul(attn, V)
        return output, attn


class MultiHeadAttention(nn.Module):
    """多头注意力机制
    
    将输入投影到多个子空间，并行计算注意力，最后拼接
    """
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # Self-attention: Q, K, V都来自同一个输入
        Q = self.W_Q(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 注意力计算
        x, attn = self.attention(Q, K, V, mask)
        
        # 合并多头
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_O(x)
        output = self.dropout(output)
        
        return output


class PositionWiseFeedForward(nn.Module):
    """逐位置前馈网络
    
    FFN(x) = ReLU(xW1 + b1)W2 + b2
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class DecoderBlock(nn.Module):
    """Decoder Block
    
    包含:
    1. Masked Multi-Head Self-Attention (带causal mask)
    2. Residual Connection + Layer Normalization
    3. Position-wise Feed-Forward Network  
    4. Residual Connection + Layer Normalization
    """
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        # Masked self-attention + 残差 + LayerNorm
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # FFN + 残差 + LayerNorm  
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class TransformerLM(nn.Module):
    """
    Decoder-Only Transformer语言模型（类似GPT）
    用于自回归语言建模任务
    """
    
    def __init__(self, vocab_size, d_model=512, n_heads=8, d_ff=2048, 
                 n_layers=6, max_len=512, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Decoder blocks
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(d_model)
        
        # Output projection
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """GPT风格的权重初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # Token embedding + scaling
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        
        # Positional encoding
        x = self.pos_encoding(x)
        
        # 创建causal mask (下三角矩阵)
        causal_mask = self.create_causal_mask(seq_len, x.device)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        # 通过所有decoder blocks
        for block in self.blocks:
            x = block(x, causal_mask)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # 输出投影
        logits = self.fc_out(x)
        
        return logits
    
    @staticmethod
    def create_causal_mask(seq_len, device):
        """创建causal mask，防止看到未来的token
        
        返回下三角矩阵：
        [[1, 0, 0],
         [1, 1, 0],
         [1, 1, 1]]
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask == 0


def count_parameters(model):
    """统计模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # 测试模型
    vocab_size = 65
    batch_size = 4
    seq_len = 128
    
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=256,
        n_heads=4,
        d_ff=1024,
        n_layers=4,
        max_len=512,
        dropout=0.1
    )
    
    print(f"模型参数量: {count_parameters(model):,}")
    
    # 随机输入
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 前向传播
    logits = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {logits.shape}")
    print(f"预期形状: ({batch_size}, {seq_len}, {vocab_size})")
    
    # 验证causal mask
    mask = model.create_causal_mask(5, x.device)
    print("\nCausal mask (5x5):")
    print(mask.int())

