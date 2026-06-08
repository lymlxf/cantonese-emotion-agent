import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Xavier initialization for training stability
        for layer in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(layer.weight, gain=1.0 / math.sqrt(2))
            nn.init.zeros_(layer.bias)
        
        self.attn_dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: torch.Tensor = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Clamp scores to prevent extreme values in softmax
        scores = scores.clamp(-50, 50)
        
        if mask is not None:
            # mask: (batch, seq_len) -> expand for multi-head attention
            # mask should be True for valid positions, False for padding
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, -1e4)
        
        attn_weights = F.softmax(scores, dim=-1)
        
        # Fallback: if softmax produced all zeros (washed out by mask), use uniform
        zero_rows = (attn_weights.sum(dim=-1) == 0)
        if zero_rows.any():
            uniform = torch.ones_like(attn_weights) / attn_weights.shape[-1]
            attn_weights = torch.where(zero_rows.unsqueeze(-1), uniform, attn_weights)
        
        attn_weights = self.attn_dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.W_o(context)
        
        return output


class PositionWiseFeedForward(nn.Module):
    
    def __init__(self, d_model: int = 256, d_ff: int = 1024, dropout: float = 0.1):
        super(PositionWiseFeedForward, self).__init__()
        
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CrossModalAttentionBlock(nn.Module):
    
    def __init__(self, d_model: int = 256, num_heads: int = 8, d_ff: int = 1024, dropout: float = 0.1):
        super(CrossModalAttentionBlock, self).__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                query_mask: torch.Tensor = None, key_mask: torch.Tensor = None) -> torch.Tensor:
        # Multi-Head Attention with residual connection
        # Use key_mask for attention (to mask out padding in key/value)
        attn_output = self.mha(query, key, value, mask=key_mask)
        query = self.layer_norm1(query + self.dropout1(attn_output))
        
        # Position-wise Feed Forward with residual connection
        ffn_output = self.ffn(query)
        output = self.layer_norm2(query + self.dropout2(ffn_output))
        
        return output


class CrossModalAttentionModule(nn.Module):
    
    def __init__(self, d_model: int = 256, num_heads: int = 8, num_blocks: int = 2, 
                 d_ff: int = 1024, dropout: float = 0.1):
        super(CrossModalAttentionModule, self).__init__()
        
        self.d_model = d_model
        self.num_blocks = num_blocks
        
        self.blocks = nn.ModuleList([
            CrossModalAttentionBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_blocks)
        ])
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                query_mask: torch.Tensor = None, key_mask: torch.Tensor = None) -> torch.Tensor:
        x = query
        for block in self.blocks:
            x = block(x, key, value, query_mask=query_mask, key_mask=key_mask)
        return x


if __name__ == "__main__":
    batch_size = 4
    seq_len = 50
    d_model = 256
    
    audio_features = torch.randn(batch_size, seq_len, d_model)
    text_features = torch.randn(batch_size, seq_len, d_model)
    
    # CMA-1: Audio as query, Text as key/value
    cma1 = CrossModalAttentionModule(d_model=d_model, num_heads=8, num_blocks=2)
    output1 = cma1(audio_features, text_features, text_features)
    print(f"CMA-1 output shape: {output1.shape}")
    
    # CMA-2: Text as query, Audio as key/value
    cma2 = CrossModalAttentionModule(d_model=d_model, num_heads=8, num_blocks=2)
    output2 = cma2(text_features, audio_features, audio_features)
    print(f"CMA-2 output shape: {output2.shape}")
