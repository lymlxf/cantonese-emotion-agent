import torch
import torch.nn as nn
from text2vec import Word2Vec
from typing import Optional, List
import numpy as np
class TextEncoder(nn.Module):
    def __init__(
        self,
        embedding_model_name: str = "w2v-light-tencent-chinese",
        hidden_dim: int = 128,
        conv_channels: int = 256,
        kernel_size: int = 3,
        dropout: float = 0.2,
        freeze_embeddings: bool = True
    ):
        super(TextEncoder, self).__init__()
        
        self.embedding_model_name = embedding_model_name
        self.hidden_dim = hidden_dim
        self.conv_channels = conv_channels

        print(f"Loading Word2Vec model: {embedding_model_name}...")
        self.w2v_model = Word2Vec(embedding_model_name)

        embedding_dim = 200
        self.conv1d = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm1d(conv_channels)
        self.relu = nn.ReLU(inplace=True)
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text_input: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:

        embeddings = text_input.transpose(1, 2)
        conv_out = self.conv1d(embeddings)
        conv_out = self.bn(conv_out)
        conv_out = self.relu(conv_out)
        conv_out = self.dropout(conv_out)

        conv_out = conv_out.transpose(1, 2)  # (batch, time, features)

        if lengths is not None:
            # Pack sequence to skip padding in LSTM
            conv_out = nn.utils.rnn.pack_padded_sequence(
                conv_out, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        lstm_out, _ = self.lstm(conv_out)

        if isinstance(lstm_out, nn.utils.rnn.PackedSequence):
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        
        return lstm_out
    
    def encode(self, texts: List[str]) -> torch.Tensor:
        """
        Encode texts to word-level embeddings.
        
        Args:
            texts: List of text strings
            
        Returns:
            embeddings: Tensor of shape (batch, max_seq_len, 200)
            lengths: Tensor of shape (batch,)
        """
        import jieba
        
        # Segment each text and encode words
        all_embeddings = []
        lengths = []
        
        for text in texts:
            words = list(jieba.cut(text))
            embeddings = self.w2v_model.encode(words, show_progress_bar=False)
            all_embeddings.append(torch.from_numpy(embeddings).float())
            lengths.append(embeddings.shape[0])
        
        # Pad to same length
        embeddings_tensor = torch.nn.utils.rnn.pad_sequence(
            all_embeddings, batch_first=True, padding_value=0.0
        )
        
        lengths_tensor = torch.tensor(lengths)
        
        return embeddings_tensor, lengths_tensor
    
    def get_output_dim(self) -> int:
        return self.hidden_dim * 2
    
if __name__ == "__main__":
    print("Loading text encoder...")
    model = TextEncoder(hidden_dim=128)
    sample_texts = ["我很开心", "我觉得难过", "你好世界"]
    
    print("\nEncoding texts...")
    embeddings = model.encode(sample_texts)
    
    print(f"Embeddings shape: {embeddings.shape}")
    output = model(embeddings)
    
    print(f"Output shape: {output.shape}")
    print(f"Output dim: {model.get_output_dim()}")