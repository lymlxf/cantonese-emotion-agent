"""
CantoBERT Text Encoder: frozen BART-base-cantonese backbone + trainable adapter.

Adapter mirrors TextEncoder's Conv1d + BN + ReLU + BiLSTM stack for fair comparison.
Compatible with checkpoints trained before the bottleneck refactor.
"""

import torch
import torch.nn as nn
from typing import Optional, List


class CantoBertEncoder(nn.Module):
    """
    Contextual text encoder using frozen BART-base-cantonese + trainable Conv1d + BiLSTM.

    Workflow:
        token_ids + attention_mask -> BART encoder (frozen, 768-dim)
        -> Conv1d(k=3, 768->256) + BN + ReLU + Dropout
        -> BiLSTM(256->128x2, bidirectional)
        -> output: (batch, seq_len, 256)

    Args:
        model_name: HuggingFace model ID (default: "Ayaka/bart-base-cantonese")
        d_model: Output dimension (default: 256)
        conv_channels: Conv1d output channels (default: 256)
        hidden_dim: BiLSTM hidden dim per direction (default: 128)
        kernel_size: Conv1d kernel size (default: 3)
        dropout: Dropout rate (default: 0.2)
        freeze_backbone: Whether to freeze BART parameters (default: True)
        unfreeze_layers: Number of top BART encoder layers to unfreeze (default: 2).
                         Set 0 for fully frozen (original behavior).
    """

    def __init__(
        self,
        model_name: str = "Ayaka/bart-base-cantonese",
        d_model: int = 256,
        conv_channels: int = 256,
        hidden_dim: int = 128,
        kernel_size: int = 3,
        dropout: float = 0.2,
        freeze_backbone: bool = True,
        unfreeze_layers: int = 2,
    ):
        super(CantoBertEncoder, self).__init__()

        print(f"Loading CantoBERT model: {model_name}...")
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
        self.bart = AutoModel.from_pretrained(model_name, local_files_only=False)
        self.hidden_size = self.bart.config.hidden_size

        if freeze_backbone:
            for param in self.bart.parameters():
                param.requires_grad = False
            frozen_count = sum(p.numel() for p in self.bart.parameters())

            if unfreeze_layers > 0:
                num_layers = self.bart.config.encoder_layers
                for i in range(max(0, num_layers - unfreeze_layers), num_layers):
                    for param in self.bart.encoder.layers[i].parameters():
                        param.requires_grad = True
                unfrozen = sum(p.numel() for p in self.bart.parameters() if p.requires_grad)
                print(f"  BART backbone: {frozen_count - unfrozen:,} frozen, {unfrozen:,} unfrozen (top {unfreeze_layers}/{num_layers} layers)")
            else:
                print(f"  BART backbone frozen ({frozen_count:,} params)")

        self.conv1d = nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.bn = nn.BatchNorm1d(conv_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )

        self.d_model = d_model

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Adapter: Conv1d({self.hidden_size}->{conv_channels}, k={kernel_size}) + BiLSTM({conv_channels}->{hidden_dim}x2)")
        print(f"  Trainable params: {trainable:,}")
        print("CantoBERT encoder loaded successfully!")

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = (token_ids != self.tokenizer.pad_token_id).long()

        bart_outputs = self.bart(
            input_ids=token_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = bart_outputs.last_hidden_state

        conv_input = hidden.transpose(1, 2)
        conv_out = self.conv1d(conv_input)
        conv_out = self.bn(conv_out)
        conv_out = self.relu(conv_out)
        conv_out = self.dropout(conv_out)

        conv_out = conv_out.transpose(1, 2)
        lstm_out, _ = self.lstm(conv_out)

        return lstm_out

    def get_output_dim(self) -> int:
        return self.d_model

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        self.eval()
        encodings = self.tokenizer(
            texts, max_length=128, truncation=True, padding=True, return_tensors="pt",
        )
        token_ids = encodings["input_ids"].to(next(self.parameters()).device)
        attention_mask = encodings["attention_mask"].to(token_ids.device)
        pad_id = self.tokenizer.pad_token_id
        lengths = (token_ids != pad_id).sum(dim=1)
        embeddings = self.forward(token_ids, attention_mask)
        return embeddings, lengths

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
