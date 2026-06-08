"""
Collate functions for batching variable-length sequences.
"""

import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch: list) -> dict:
    """
    Collate function for variable-length audio and text sequences.
    
    Args:
        batch: List of samples from CantoneseSERDataset.__getitem__
        
    Returns:
        dict with padded tensors and lengths:
            - 'audio': (batch, n_mels, max_time)
            - 'audio_lengths': (batch,)
            - 'text': (batch, max_seq, 200)
            - 'text_lengths': (batch,)
            - 'labels': (batch,)
    """
    # Separate fields
    audios = [item['audio'] for item in batch]  # List of (n_mels, time_i)
    audio_lengths = torch.tensor([item['audio_length'] for item in batch])
    
    texts = [item['text'] for item in batch]  # List of (1, 200)
    text_lengths = torch.tensor([item['text_length'] for item in batch])
    
    labels = torch.stack([item['label'] for item in batch])
    
    # Pad audio: (n_mels, time) -> pad time dimension
    # Transpose to (time, n_mels) for pad_sequence
    audios_transposed = [a.transpose(0, 1) for a in audios]
    audios_padded = pad_sequence(
        audios_transposed,
        batch_first=True,
        padding_value=0.0
    )
    # Transpose back to (batch, n_mels, time)
    audios_padded = audios_padded.transpose(1, 2)
    
    # Pad text: (1, 200) -> pad if needed (usually all same length)
    texts_padded = pad_sequence(
        texts,
        batch_first=True,
        padding_value=0.0
    )
    
    return {
        'audio': audios_padded,              # (batch, n_mels, max_time)
        'audio_lengths': audio_lengths,      # (batch,)
        'text': texts_padded,                # (batch, max_seq, 200)
        'text_lengths': text_lengths,        # (batch,)
        'labels': labels                     # (batch,)
    }


def collate_fn_audio_only(batch: list) -> dict:
    """
    Collate function for audio-only mode.
    
    Args:
        batch: List of samples
        
    Returns:
        dict with padded audio and labels:
            - 'audio': (batch, n_mels, max_time)
            - 'audio_lengths': (batch,)
            - 'labels': (batch,)
    """
    audios = [item['audio'] for item in batch]
    audio_lengths = torch.tensor([item['audio_length'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    # Pad audio
    audios_transposed = [a.transpose(0, 1) for a in audios]
    audios_padded = pad_sequence(
        audios_transposed,
        batch_first=True,
        padding_value=0.0
    )
    audios_padded = audios_padded.transpose(1, 2)
    
    return {
        'audio': audios_padded,
        'audio_lengths': audio_lengths,
        'labels': labels
    }


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from .dataset import CantoneseSERDataset
    
    print("Testing collate_fn...")
    
    # Create dataset
    dataset = CantoneseSERDataset("data/train-00000-of-00045.parquet")
    
    # Create dataloader with collate_fn
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Get a batch
    batch = next(iter(dataloader))
    
    print(f"\nBatch contents:")
    print(f"  Audio shape: {batch['audio'].shape}")
    print(f"  Audio lengths: {batch['audio_lengths']}")
    print(f"  Text shape: {batch['text'].shape}")
    print(f"  Text lengths: {batch['text_lengths']}")
    print(f"  Labels: {batch['labels']}")
    
    # Test audio-only collate
    print("\n\nTesting collate_fn_audio_only...")
    dataloader_audio = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn_audio_only
    )
    
    batch_audio = next(iter(dataloader_audio))
    print(f"  Audio shape: {batch_audio['audio'].shape}")
    print(f"  Audio lengths: {batch_audio['audio_lengths']}")
    print(f"  Labels: {batch_audio['labels']}")
