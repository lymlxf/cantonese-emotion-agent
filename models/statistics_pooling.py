import torch
import torch.nn as nn


class StatisticsPooling(nn.Module):
    
    def __init__(self):
        super(StatisticsPooling, self).__init__()
        
    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        
        if lengths is not None:
            # Clamp to prevent division by zero for zero-length samples
            safe_lengths = lengths.clamp(min=1).unsqueeze(1).float()
            eps = 1e-8
            
            # Create mask: (batch, seq_len, 1)
            batch_size, max_len, feat_dim = x.shape
            mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            
            # Masked mean
            sum_x = torch.sum(x * mask, dim=1)  # (batch, feat_dim)
            mean = sum_x / (safe_lengths + eps)  # (batch, feat_dim)
            
            # Masked std (unbiased=False for consistency)
            diff = (x - mean.unsqueeze(1)) * mask
            sum_sq = torch.sum(diff ** 2, dim=1)
            std = torch.sqrt(sum_sq / (safe_lengths + eps) + eps)
        else:
            # Fallback: compute over all positions (for backward compatibility)
            mean = torch.mean(x, dim=1)
            std = torch.std(x, dim=1, unbiased=False)
        
        stats = torch.cat([mean, std], dim=-1)
        
        return stats
    
    def get_output_dim(self, input_dim: int) -> int:
        
        return input_dim * 2


if __name__ == "__main__":
    batch_size = 4
    seq_len = 50
    d_model = 256
    
    x = torch.randn(batch_size, seq_len, d_model)
    lengths = torch.tensor([50, 30, 40, 20])  # Variable lengths
    
    pool = StatisticsPooling()
    
    # With masking
    output_masked = pool(x, lengths)
    print(f"With masking - Output shape: {output_masked.shape}")
    
    # Without masking (fallback)
    output_unmasked = pool(x)
    print(f"Without masking - Output shape: {output_unmasked.shape}")
    print(f"Expected output dim: {pool.get_output_dim(d_model)}")
    
    # Verify: masked and unmasked should give different results
    print(f"\nDifference (sample 3, len=20): {torch.abs(output_masked[3] - output_unmasked[3]).mean():.4f}")
