import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class EmotionClassifier(nn.Module):
    
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dims: List[int] = None,
        num_classes: int = 8,
        dropout: float = 0.2
    ):
        super(EmotionClassifier, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
        self.num_classes = num_classes
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(x)
        return logits


if __name__ == "__main__":
    batch_size = 4
    input_dim = 1024
    
    x = torch.randn(batch_size, input_dim)
    
    classifier = EmotionClassifier(input_dim=input_dim, num_classes=8)
    output = classifier(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output classes: {classifier.num_classes}")
    
    # Test with softmax
    probs = F.softmax(output, dim=-1)
    print(f"Probabilities sum: {probs.sum(dim=-1)}")
