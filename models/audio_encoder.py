import torch
import torch.nn as nn
import torch.nn.functional as F


from typing import Optional, List


class AudioEncoder(nn.Module):
    
    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 128,
        num_conv_layers: int = 3,
        conv_channels: Optional[List[int]] = None,
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        super(AudioEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if conv_channels is None:
            conv_channels = [64, 128, 256]

        self.conv_layers = nn.ModuleList()
        in_channels = input_dim
        
        for i, out_channels in enumerate(conv_channels):
            conv_block = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2)
            )
            self.conv_layers.append(conv_block)
            in_channels = out_channels
        
        self.conv_output_dim = conv_channels[-1]

        self.lstm = nn.LSTM(
            input_size=self.conv_output_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        
        for conv_block in self.conv_layers:
            x = conv_block(x)
            x = self.dropout(x)
            # Update lengths after MaxPool (kernel_size=2, stride=2)
            if lengths is not None:
                lengths = lengths // 2

        x = x.transpose(1, 2)  # (batch, time, features)

        if lengths is not None:
            # Pack sequence to skip padding in LSTM
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        lstm_out, _ = self.lstm(x)
        
        if isinstance(lstm_out, nn.utils.rnn.PackedSequence):
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        
        return lstm_out
    
    def get_output_dim(self) -> int:
        return self.hidden_dim * 2


if __name__ == "__main__":
    batch_size = 4
    n_mels = 80
    time_steps = 100
    
    model = AudioEncoder(input_dim=n_mels, hidden_dim=128)
    dummy_input = torch.randn(batch_size, n_mels, time_steps)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output dim: {model.get_output_dim()}")
