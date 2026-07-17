#!/usr/bin/env python
"""
Training script for Cantonese Speech Emotion Recognition.
Supports multimodal, audio-only, and text-only modes.
"""

import argparse
import os
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from tqdm import tqdm
import numpy as np

from models import MultimodalSER
from utils import CantoneseSERDataset, AudioPreprocessor, TextPreprocessor
from utils.collate_fn import collate_fn, collate_fn_audio_only
from utils.sampler import BalancedBatchSampler


def compute_sample_weights(dataset, num_classes=8):
    label_counts = torch.zeros(num_classes)
    for i in range(len(dataset)):
        label = dataset[i]['label'].item()
        if 0 <= label < num_classes:
            label_counts[label] += 1
    
    label_counts = label_counts.clamp(min=1.0)
    weights = torch.sqrt(label_counts.max() / label_counts)
    weights = weights.clamp(min=0.5, max=2.0)
    
    return weights


class WarmupScheduler:
    """Linear warmup LR scheduler: ramps LR from 0 to target over warmup_epochs."""
    
    def __init__(self, optimizer, warmup_epochs, target_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.target_lr = target_lr
        self.current_epoch = 0
    
    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            lr = self.target_lr * self.current_epoch / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        return self.target_lr


def train_epoch(model, dataloader, criterion, optimizer, device, modality="both",
                scaler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    nan_count = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Move to device
        audio = batch['audio'].to(device)
        audio_lengths = batch['audio_lengths'].to(device).clamp(min=1)
        labels = batch['labels'].to(device)
        
        # Forward pass based on modality
        if modality == "both":
            text = batch['text'].to(device)
            text_lengths = batch['text_lengths'].to(device).clamp(min=1)
            logits = model(audio, text, audio_lengths, text_lengths)
        elif modality == "audio":
            logits = model(audio, audio_lengths=audio_lengths)
        elif modality == "text":
            text = batch['text'].to(device)
            text_lengths = batch['text_lengths'].to(device).clamp(min=1)
            logits = model(text_input=text, text_lengths=text_lengths)
        
        # Compute loss
        loss = criterion(logits, labels)
        
        # NaN detection: skip batch if loss is NaN
        if torch.isnan(loss):
            nan_count += 1
            if nan_count > 10:
                raise RuntimeError(f"Training aborted: {nan_count} consecutive NaN batches")
            pbar.set_postfix({'loss': 'NaN', 'nan_batches': nan_count})
            continue
        
        # Backward pass
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device, modality="both"):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            audio = batch['audio'].to(device)
            audio_lengths = batch['audio_lengths'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass based on modality
            if modality == "both":
                text = batch['text'].to(device)
                text_lengths = batch['text_lengths'].to(device)
                logits = model(audio, text, audio_lengths, text_lengths)
            elif modality == "audio":
                logits = model(audio, audio_lengths=audio_lengths)
            elif modality == "text":
                text = batch['text'].to(device)
                text_lengths = batch['text_lengths'].to(device)
                logits = model(text_input=text, text_lengths=text_lengths)
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train Cantonese SER model")
    
    # Data arguments
    parser.add_argument("--data", required=True, help="Path to parquet file or directory")
    parser.add_argument("--cache_dir", default=None, help="Directory to cache preprocessed data")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--confidence_threshold", type=float, default=0.0, help="Confidence threshold")
    
    # Model arguments
    parser.add_argument("--modality", default="both", choices=["both", "audio", "text"],
                       help="Modality: both, audio, or text")
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_classes", type=int, default=8, help="Number of emotion classes")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--early_stop", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--use_weighted_sampling", action="store_true", help="Use weighted random sampling to handle class imbalance")
    parser.add_argument("--use_balanced_sampler", action="store_true", help="Use BalancedBatchSampler for class-balanced batches")
    parser.add_argument("--warmup_epochs", type=int, default=0, help="Linear LR warmup epochs")
    parser.add_argument("--use_amp", action="store_true", help="Use Automatic Mixed Precision training")
    
    # Other arguments
    parser.add_argument("--output_dir", default="checkpoints", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save config
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Load data
    print("\nLoading data...")
    
    # Check if data is a cache directory (contains .pt files, flat or nested)
    is_cache = os.path.isdir(args.data) and (
        any(f.endswith('.pt') for f in os.listdir(args.data))
        or any(
            os.path.isdir(os.path.join(args.data, d)) and
            any(f.endswith('.pt') for f in os.listdir(os.path.join(args.data, d)))
            for d in os.listdir(args.data)
        )
    )
    
    if is_cache:
        # Load from cache
        from preprocess import CachedDataset
        dataset = CachedDataset(args.data)
    elif args.cache_dir and os.path.exists(args.cache_dir):
        # Load from specified cache_dir
        from preprocess import CachedDataset
        dataset = CachedDataset(args.cache_dir)
    else:
        # Load from parquet
        audio_prep = AudioPreprocessor()
        text_prep = TextPreprocessor()
        dataset = CantoneseSERDataset(
            parquet_path=args.data,
            audio_preprocessor=audio_prep,
            text_preprocessor=text_prep,
            confidence_threshold=args.confidence_threshold
        )
    
    # Split train/val
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Create dataloaders
    collate = collate_fn if args.modality == "both" else collate_fn_audio_only
    
    # Create sampler if balanced mode enabled
    train_sampler = None
    if args.use_balanced_sampler:
        # Extract labels, handling Subset wrapping from random_split
        if isinstance(train_dataset, torch.utils.data.Subset):
            train_labels = train_dataset.dataset.labels[train_dataset.indices]
        else:
            train_labels = train_dataset.labels
        
        train_sampler = BalancedBatchSampler(
            labels=train_labels,
            batch_size=args.batch_size,
            num_classes=args.num_classes,
        )
        print(f"BalancedBatchSampler: {train_sampler.num_batches} batches/epoch "
              f"({train_sampler.samples_per_class} per class, {train_sampler.remainder} remainder)")
    
    # DataLoader kwargs that only apply with multiprocessing
    dl_kwargs = {}
    if args.num_workers > 0:
        dl_kwargs = dict(persistent_workers=True, prefetch_factor=4)
    
    if args.use_balanced_sampler:
        # batch_sampler returns complete batches — mutually exclusive with batch_size/shuffle/sampler
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=args.num_workers,
            collate_fn=collate,
            pin_memory=True,
            **dl_kwargs
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate,
            pin_memory=True,
            **dl_kwargs
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
        **{k: v for k, v in dl_kwargs.items() if k != 'prefetch_factor'}
    )
    
    # Create model
    print(f"\nCreating model (modality={args.modality})...")
    model = MultimodalSER(
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_classes=args.num_classes,
        dropout=args.dropout,
        modality=args.modality
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    class_weights = compute_sample_weights(train_dataset, args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # Warmup scheduler (linear ramp from 0 to target LR over warmup_epochs)
    warmup = WarmupScheduler(optimizer, args.warmup_epochs, args.lr) if args.warmup_epochs > 0 else None
    
    # AMP scaler (only on CUDA)
    scaler = torch.cuda.amp.GradScaler() if args.use_amp and device.type == 'cuda' else None
    if args.use_amp:
        print(f"AMP: {'enabled' if scaler else 'disabled (requires CUDA)'}")
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-"*60)
        
        # Warmup: linearly increase LR from 0 to target
        if warmup is not None:
            current_lr = warmup.step()
            print(f"  LR (warmup): {current_lr:.2e}")
        
        # Toggle augmentation mode on shared preprocessor
        # NOTE: On Windows with num_workers>0 (spawn multiprocessing),
        # workers have their own copy and won't see this toggle.
        # Set num_workers=0 for correct augmentation behavior on Windows.
        if not is_cache and hasattr(dataset, 'audio_preprocessor'):
            dataset.audio_preprocessor.train()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, args.modality, scaler=scaler
        )
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Switch to eval mode (no augmentations) for validation
        if not is_cache and hasattr(dataset, 'audio_preprocessor'):
            dataset.audio_preprocessor.eval()
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, args.modality
        )
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'config': vars(args)
            }
            
            checkpoint_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(checkpoint, checkpoint_path)
            print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{args.early_stop}")
        
        # Early stopping
        if patience_counter >= args.early_stop:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
