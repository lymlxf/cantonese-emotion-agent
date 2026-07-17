#!/usr/bin/env python
"""
Validation script to test the complete preprocessing and model pipeline.
"""

import torch
from torch.utils.data import DataLoader

from utils import CantoneseSERDataset, AudioPreprocessor, TextPreprocessor
from utils.collate_fn import collate_fn
from models import MultimodalSER


def validate_preprocessing():
    """Test preprocessing pipeline."""
    print("="*60)
    print("1. Testing Preprocessing Pipeline")
    print("="*60)
    
    # Create dataset
    dataset = CantoneseSERDataset("data/train-00000-of-00045.parquet")
    
    # Check a few samples
    print("\nChecking 5 samples...")
    for i in range(5):
        sample = dataset[i]
        
        # Validate shapes
        assert sample['audio'].shape[0] == 80, f"Audio should have 80 mels, got {sample['audio'].shape[0]}"
        assert sample['text'].shape == (1, 200), f"Text should be (1, 200), got {sample['text'].shape}"
        assert 0 <= sample['label'].item() < 8, f"Label should be 0-7, got {sample['label']}"
        
        # Check for NaN/Inf
        assert not torch.isnan(sample['audio']).any(), f"Audio has NaN at index {i}"
        assert not torch.isinf(sample['audio']).any(), f"Audio has Inf at index {i}"
        
        print(f"  Sample {i}: audio={sample['audio'].shape}, text={sample['text'].shape}, "
              f"label={sample['label'].item()}, audio_len={sample['audio_length']}")
    
    print("[OK] All samples valid")
    

def validate_batching():
    """Test batching with variable lengths."""
    print("\n" + "="*60)
    print("2. Testing Batching with Variable Lengths")
    print("="*60)
    
    dataset = CantoneseSERDataset("data/train-00000-of-00045.parquet")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Get a batch
    batch = next(iter(dataloader))
    
    print(f"\nBatch shapes:")
    print(f"  Audio: {batch['audio'].shape}")
    print(f"  Audio lengths: {batch['audio_lengths']}")
    print(f"  Text: {batch['text'].shape}")
    print(f"  Text lengths: {batch['text_lengths']}")
    print(f"  Labels: {batch['labels'].shape}")
    
    # Check that lengths are different (variable length)
    assert len(torch.unique(batch['audio_lengths'])) > 1, "Audio lengths should vary"
    
    print("[OK] Batching works correctly")


def validate_model_forward():
    """Test model forward pass with real data."""
    print("\n" + "="*60)
    print("3. Testing Model Forward Pass")
    print("="*60)
    
    # Create dataset and dataloader
    dataset = CantoneseSERDataset("data/train-00000-of-00045.parquet")
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    
    # Create model
    model = MultimodalSER(modality="both", num_classes=8)
    model.eval()
    
    # Get a batch
    batch = next(iter(dataloader))
    
    # Forward pass
    with torch.no_grad():
        logits = model(
            batch['audio'],
            batch['text'],
            batch['audio_lengths'],
            batch['text_lengths']
        )
    
    print(f"\nInput shapes:")
    print(f"  Audio: {batch['audio'].shape}")
    print(f"  Text: {batch['text'].shape}")
    
    print(f"\nOutput:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Expected: (4, 8)")
    
    assert logits.shape == (4, 8), f"Expected (4, 8), got {logits.shape}"
    
    print("[OK] Model forward pass successful")
    
    # Test with masking
    print("\nTesting with and without masking...")
    with torch.no_grad():
        logits_masked = model(
            batch['audio'],
            batch['text'],
            batch['audio_lengths'],
            batch['text_lengths']
        )
        logits_unmasked = model(batch['audio'], batch['text'])
    
    diff = torch.abs(logits_masked - logits_unmasked).mean().item()
    print(f"  Mean difference (masked vs unmasked): {diff:.4f}")
    
    if diff > 1e-6:
        print("  [OK] Masking has effect (outputs differ)")
    else:
        print("  [WARN] Masking has no effect (outputs identical)")


def validate_audio_only():
    """Test audio-only mode."""
    print("\n" + "="*60)
    print("4. Testing Audio-Only Mode")
    print("="*60)
    
    from utils.collate_fn import collate_fn_audio_only
    
    dataset = CantoneseSERDataset("data/train-00000-of-00045.parquet")
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn_audio_only)
    
    model = MultimodalSER(modality="audio", num_classes=8)
    model.eval()
    
    batch = next(iter(dataloader))
    
    with torch.no_grad():
        logits = model(batch['audio'], audio_lengths=batch['audio_lengths'])
    
    print(f"\nInput: {batch['audio'].shape}")
    print(f"Output: {logits.shape}")
    
    assert logits.shape == (4, 8), f"Expected (4, 8), got {logits.shape}"
    
    print("[OK] Audio-only mode works")


def validate_class_weights():
    """Test class weight computation."""
    print("\n" + "="*60)
    print("5. Testing Class Weight Computation")
    print("="*60)
    
    from train import compute_class_weights
    
    dataset = CantoneseSERDataset("data/train-00000-of-00045.parquet")
    weights = compute_class_weights(dataset, num_classes=8)
    
    print(f"\nClass weights: {weights}")
    print(f"Min weight: {weights.min():.4f}")
    print(f"Max weight: {weights.max():.4f}")
    
    # Check that minority classes have higher weights
    print("\nClass distribution:")
    for i in range(8):
        count = sum(1 for j in range(len(dataset)) if dataset[j]['label'].item() == i)
        print(f"  Class {i}: {count} samples, weight: {weights[i]:.4f}")
    
    print("[OK] Class weights computed")


def main():
    print("\n" + "="*60)
    print("PREPROCESSING VALIDATION")
    print("="*60)
    
    try:
        validate_preprocessing()
        validate_batching()
        validate_model_forward()
        validate_audio_only()
        validate_class_weights()
        
        print("\n" + "="*60)
        print("[OK] ALL VALIDATIONS PASSED")
        print("="*60)
        
    except Exception as e:
        print(f"\n[FAIL] VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
