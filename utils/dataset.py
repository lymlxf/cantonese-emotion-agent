"""
Dataset class for Cantonese Speech Emotion Recognition.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path

from .audio_utils import AudioPreprocessor
from .text_utils import TextPreprocessor


class CantoneseSERDataset(Dataset):
    """
    Dataset for Cantonese multimodal emotion recognition.
    
    Args:
        parquet_path: Path to parquet file or directory of parquet files
        audio_preprocessor: AudioPreprocessor instance
        text_preprocessor: TextPreprocessor instance
        label_map: Dictionary mapping label strings to integers
        confidence_threshold: Minimum confidence to include sample (default: 0.0)
    """
    
    def __init__(
        self,
        parquet_path: str,
        audio_preprocessor: AudioPreprocessor = None,
        text_preprocessor: TextPreprocessor = None,
        label_map: dict = None,
        confidence_threshold: float = 0.0
    ):
        self.parquet_path = Path(parquet_path)
        
        # Load parquet file(s)
        if self.parquet_path.is_dir():
            parquet_files = sorted(self.parquet_path.glob("*.parquet"))
            self.df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
        else:
            self.df = pd.read_parquet(self.parquet_path)
        
        # Filter by confidence if threshold > 0
        if confidence_threshold > 0:
            self.df = self.df[self.df['confidence'] >= confidence_threshold].reset_index(drop=True)
        
        # Initialize preprocessors
        self.audio_preprocessor = audio_preprocessor or AudioPreprocessor()
        self.text_preprocessor = text_preprocessor or TextPreprocessor()
        
        # Label mapping
        if label_map is None:
            # Default 8-class mapping
            label_map = {
                'angry': 0,
                'disgusted': 1,
                'fearful': 2,
                'happy': 3,
                'neutral': 4,
                'other': 5,
                'sad': 6,
                'surprised': 7,
                '<unk>': 5  # Map unknown to 'other'
            }
        self.label_map = label_map
        
        # Build integer label array for efficient sampling
        self.labels = self.df['label'].map(
            lambda s: self.label_map.get(s, -1)
        ).to_numpy(dtype=np.int64)
        
        # Print dataset info
        print(f"Loaded {len(self.df)} samples")
        print(f"Labels: {self.df['label'].value_counts().to_dict()}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample.
        
        Returns:
            dict with keys:
                - 'audio': Tensor (n_mels, time_steps)
                - 'audio_length': int
                - 'text': Tensor (1, 200) - sentence embedding
                - 'text_length': int (always 1)
                - 'label': Tensor (1,)
        """
        row = self.df.iloc[idx]
        
        # Process audio
        audio_bytes = row['audio_file']['bytes']
        mel_spec, audio_length = self.audio_preprocessor(audio_bytes)
        
        # Process text
        text = row['text']
        text_embeddings, text_length = self.text_preprocessor(text)
        
        # Convert to tensors
        text_tensor = torch.tensor(text_embeddings, dtype=torch.float32)
        
        # Map label
        label_str = row['label']
        label = self.label_map.get(label_str, -1)  # -1 for unknown labels
        
        return {
            'audio': mel_spec,                          # (n_mels, time_steps)
            'audio_length': audio_length,               # int
            'text': text_tensor,                        # (1, 200)
            'text_length': text_length,                 # int (always 1)
            'label': torch.tensor(label, dtype=torch.long)
        }


if __name__ == "__main__":
    print("Testing CantoneseSERDataset...")
    
    # Create dataset
    dataset = CantoneseSERDataset(
        parquet_path="data/train-00000-of-00045.parquet",
        confidence_threshold=0.0
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Audio shape: {sample['audio'].shape}")
    print(f"  Audio length: {sample['audio_length']}")
    print(f"  Text shape: {sample['text'].shape}")
    print(f"  Text length: {sample['text_length']}")
    print(f"  Label: {sample['label']}")
    
    # Test with confidence filtering
    print("\n\nTesting with confidence threshold 0.9...")
    dataset_filtered = CantoneseSERDataset(
        parquet_path="data/train-00000-of-00045.parquet",
        confidence_threshold=0.9
    )
    print(f"Filtered dataset size: {len(dataset_filtered)}")
