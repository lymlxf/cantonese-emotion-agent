#!/usr/bin/env python
"""
High-speed training script with automatic format selection.

Supports four data formats (auto-detected by --data path):
  1. .sermerged file   --data cache.sermerged   (RECOMMENDED - preprocess directly to this)
  2. Merged .pt file   --data cache_merged.pt    (all in RAM, fastest if RAM > data)
  3. LMDB directory    --data cache.lmdb/        (RAM < data, memory-mapped)
  4. Cache directory   --data cache/             (individual .pt files, slowest)

RECOMMENDED workflow (preprocess once, train fast, no wasted space):
    # Step 1: Preprocess directly to .sermerged (no intermediate .pt files)
    python preprocess.py --input "data/*.parquet" --merged_output cache.sermerged

    # Step 2: Train (memory-mapped, OS caches hot pages, no RAM pressure)
    python train_fast.py --data cache.sermerged --batch_size 32 --lr 4e-4 ...

Alternative workflows:
    # If RAM > data: load everything into RAM
    python inmemory_dataset.py merge --input cache/ --output cache_merged.pt
    python train_fast.py --data cache_merged.pt --batch_size 32 ...

    # If RAM < data and you already have .pt files: convert to LMDB
    python lmdb_dataset.py convert --input cache/ --output cache.lmdb
    python train_fast.py --data cache.lmdb/ --batch_size 32 ...
"""

import argparse
import os
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np

from models import MultimodalSER
from utils.collate_fn import collate_fn, collate_fn_audio_only
from utils.sampler import BalancedBatchSampler

# Lazy imports for optional dataset formats (avoids ModuleNotFoundError when unused)
# InMemoryDataset = None  # loaded on demand
# LMDBDataset = None
# LMDBDatasetWorker = None


# =============================================================================
# Warmup Scheduler
# =============================================================================

class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, base_scheduler=None):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.base_scheduler = base_scheduler
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self, val_loss=None):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            warmup_factor = self.current_step / self.warmup_steps
            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                group['lr'] = base_lr * warmup_factor
        elif self.base_scheduler is not None:
            if isinstance(self.base_scheduler, ReduceLROnPlateau) and val_loss is not None:
                self.base_scheduler.step(val_loss)
            else:
                self.base_scheduler.step()

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


# =============================================================================
# Safe Collate Functions
# =============================================================================

def collate_fn_safe(batch):
    filtered = []
    for item in batch:
        text_len = item.get('text_length', item['text'].shape[0] if item['text'].dim() > 0 else 0)
        audio_len = item.get('audio_length', item['audio'].shape[-1] if item['audio'].dim() > 0 else 0)
        if text_len > 0 and audio_len > 0:
            filtered.append(item)
    if len(filtered) == 0:
        raise ValueError("All samples in batch have zero length!")
    return collate_fn(filtered)


def collate_fn_audio_only_safe(batch):
    filtered = []
    for item in batch:
        audio_len = item.get('audio_length', item['audio'].shape[-1] if item['audio'].dim() > 0 else 0)
        if audio_len > 0:
            filtered.append(item)
    if len(filtered) == 0:
        raise ValueError("All samples in batch have zero length!")
    return collate_fn_audio_only(filtered)


# =============================================================================
# Sample Weights
# =============================================================================

def compute_sample_weights(labels_array, num_classes=8):
    label_counts = torch.zeros(num_classes)
    for label in labels_array:
        if 0 <= label < num_classes:
            label_counts[int(label)] += 1
    label_counts = label_counts.clamp(min=1.0)
    weights = torch.sqrt(label_counts.max() / label_counts)
    weights = weights.clamp(min=0.5, max=2.0)
    return weights


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device, modality="both",
                scaler=None, grad_clip_norm=1.0, epoch=0):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    nan_count = 0

    pbar = tqdm(dataloader, desc=f"Train Ep{epoch+1}")
    for batch_idx, batch in enumerate(pbar):
        audio = batch['audio'].to(device, non_blocking=True)
        audio_lengths = batch['audio_lengths'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=scaler is not None):
            if modality == "both":
                text = batch['text'].to(device, non_blocking=True)
                text_lengths = batch['text_lengths'].to(device, non_blocking=True)
                text_lengths = text_lengths.clamp(min=1)
                audio_lengths = audio_lengths.clamp(min=1)
                logits = model(audio, text, audio_lengths, text_lengths)
            elif modality == "audio":
                audio_lengths = audio_lengths.clamp(min=1)
                logits = model(audio, audio_lengths=audio_lengths)
            elif modality == "text":
                text = batch['text'].to(device, non_blocking=True)
                text_lengths = batch['text_lengths'].to(device, non_blocking=True)
                text_lengths = text_lengths.clamp(min=1)
                logits = model(text_input=text, text_lengths=text_lengths)

            loss = criterion(logits, labels)

        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            if nan_count <= 3:  # Only print first few
                print(f"\n[WARNING] NaN loss at batch {batch_idx}, skipping")
            continue

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })

    avg_loss = total_loss / max(len(dataloader) - nan_count, 1)
    accuracy = 100 * correct / max(total, 1)
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device, modality="both"):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            audio = batch['audio'].to(device, non_blocking=True)
            audio_lengths = batch['audio_lengths'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)

            if modality == "both":
                text = batch['text'].to(device, non_blocking=True)
                text_lengths = batch['text_lengths'].to(device, non_blocking=True)
                text_lengths = text_lengths.clamp(min=1)
                audio_lengths = audio_lengths.clamp(min=1)
                logits = model(audio, text, audio_lengths, text_lengths)
            elif modality == "audio":
                audio_lengths = audio_lengths.clamp(min=1)
                logits = model(audio, audio_lengths=audio_lengths)
            elif modality == "text":
                text = batch['text'].to(device, non_blocking=True)
                text_lengths = batch['text_lengths'].to(device, non_blocking=True)
                text_lengths = text_lengths.clamp(min=1)
                logits = model(text_input=text, text_lengths=text_lengths)

            loss = criterion(logits, labels)
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return total_loss / len(dataloader), 100 * correct / total


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fast training with InMemoryDataset")
    parser.add_argument("--data", required=True, help="Cache dir OR merged .pt file")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--modality", default="both", choices=["both", "audio", "text"])
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--use_balanced_sampler", action="store_true")
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--use_amp", action="store_true", help="Use Automatic Mixed Precision")
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4,
                       help="DataLoader workers. With LMDB, 0 is safest; 4+ ok with per-worker handles")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    scaler = torch.amp.GradScaler('cuda') if args.use_amp and device.type == "cuda" else None
    if scaler:
        print("AMP enabled")

    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.output_dir, "config.json"), "w"), indent=2)

    # ========== LOAD DATASET ==========
    print("\n" + "=" * 60)
    data_path = Path(args.data)

    def is_lmdb_path(p: Path) -> bool:
        """Detect if path points to an LMDB database."""
        if p.is_dir():
            return any(p.glob("data.mdb")) or str(p).endswith('.lmdb')
        return False

    def is_sermerged(p: Path) -> bool:
        """Detect .sermerged single-file format."""
        return p.is_file() and p.suffix == '.sermerged'

    if is_sermerged(data_path):
        # ===== .sermerged: memory-mapped single file (RECOMMENDED) =====
        print(f"[Format] .sermerged memory-mapped file: {data_path}")
        print(f"[Format] Zero I/O after page cache warmup, low memory footprint")

        from merged_format import MergedDataset
        dataset = MergedDataset(str(data_path))
        lmdb_path_str = None
        use_lmdb = False

    elif is_lmdb_path(data_path):
        # ===== LMDB: memory-mapped, out-of-core =====
        from lmdb_dataset import LMDBDataset, LMDBDatasetWorker
        print(f"[Format] LMDB memory-mapped database: {data_path}")
        print(f"[Format] Data stays on disk, OS caches hot pages in RAM")
        print(f"[Format] num_workers should be 0 or use worker_init_fn")

        dataset = LMDBDataset(str(data_path), readonly=True, lock=True)
        lmdb_path_str = str(data_path)
        use_lmdb = True

        if args.num_workers > 0:
            print(f"[Note] LMDB with num_workers={args.num_workers}: using per-worker handles")

    elif data_path.is_file() and data_path.suffix == '.pt':
        # ===== Merged .pt file: all in RAM =====
        from inmemory_dataset import InMemoryDataset
        print(f"[Format] Merged .pt file (InMemoryDataset): {data_path}")
        dataset = InMemoryDataset.from_merged(str(data_path), device='cpu')
        lmdb_path_str = None
        use_lmdb = False

    else:
        # ===== Cache directory: try InMemoryDataset =====
        from inmemory_dataset import InMemoryDataset
        print(f"[Format] Cache directory: {data_path}")

        total_pt_size = sum(
            f.stat().st_size
            for pattern in ['**/*.pt', '*.pt']
            for f in data_path.glob(pattern)
        )
        import psutil
        available_ram = psutil.virtual_memory().available
        print(f"  Data size: {total_pt_size / 1024**3:.1f} GB")
        print(f"  Available RAM: {available_ram / 1024**3:.1f} GB")

        if total_pt_size < available_ram * 0.8:
            print(f"  -> Loading into RAM (InMemoryDataset)")
            dataset = InMemoryDataset(str(data_path), device='cpu')
            merged_path = Path(args.output_dir) / "cache_merged.pt"
            if not merged_path.exists():
                print(f"\n[Tip] Saving merged file for faster next startup...")
                dataset.save_merged(str(merged_path))
                print(f"[Tip] Next time use: --data {merged_path}")
        else:
            print(f"  [WARNING] Data ({total_pt_size/1024**3:.1f}GB) > 80% of available RAM")
            print(f"  [WARNING] Loading into RAM may cause swapping!")
            print(f"  -> Recommend: convert to LMDB first:")
            print(f"     python lmdb_dataset.py convert --input {data_path} --output cache.lmdb")
            print(f"  -> Then use:  --data cache.lmdb/")
            print(f"\n  Attempting InMemoryDataset anyway...")
            dataset = InMemoryDataset(str(data_path), device='cpu')

        lmdb_path_str = None
        use_lmdb = False

    # ========== SPLIT ==========
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"Train: {len(train_dataset):,}, Val: {len(val_dataset):,}")

    # ========== DATALOADERS ==========
    collate = collate_fn_safe if args.modality == "both" else collate_fn_audio_only_safe

    # Worker init for LMDB: each worker opens its own LMDB handle
    if use_lmdb and args.num_workers > 0:
        def lmdb_worker_init(worker_id):
            """Each worker gets its own LMDB connection (no lock contention)."""
            from lmdb_dataset import LMDBDatasetWorker
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                worker_info.dataset = LMDBDatasetWorker(lmdb_path_str)
        worker_init_fn = lmdb_worker_init
    else:
        worker_init_fn = None

    train_sampler = None
    if args.use_balanced_sampler:
        if isinstance(train_dataset, torch.utils.data.Subset):
            train_labels = train_dataset.dataset.labels[train_dataset.indices]
        else:
            train_labels = train_dataset.labels
        train_sampler = BalancedBatchSampler(
            labels=train_labels,
            batch_size=args.batch_size,
            num_classes=args.num_classes,
        )
        print(f"BalancedBatchSampler: {train_sampler.num_batches} batches/epoch")

    # NOTE: persistent_workers disabled for large mmap files to avoid
    # OOM during worker spawn. Workers are created lazily on first iter.
    dl_common = dict(
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=False,  # Safer for large files + limited RAM
        prefetch_factor=2 if (args.num_workers > 0 and not use_lmdb) else None,
    )

    if args.use_balanced_sampler:
        train_loader = DataLoader(
            train_dataset, batch_sampler=train_sampler, **dl_common
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, **dl_common
        )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0 if use_lmdb else args.num_workers,  # Val safer with 0 workers for LMDB
        collate_fn=collate, pin_memory=True,
    )

    # ========== MODEL ==========
    print(f"\nCreating model (modality={args.modality})...")
    model = MultimodalSER(
        d_model=args.d_model, num_heads=args.num_heads,
        num_classes=args.num_classes, dropout=args.dropout,
        modality=args.modality
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ========== OPTIMIZER ==========
    if isinstance(train_dataset, torch.utils.data.Subset):
        train_labels = train_dataset.dataset.labels[train_dataset.indices]
    else:
        train_labels = train_dataset.labels
    class_weights = compute_sample_weights(train_labels, args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    base_scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    scheduler = WarmupScheduler(optimizer, warmup_steps=args.warmup_epochs, base_scheduler=base_scheduler)

    print(f"\nTraining: lr={args.lr}, warmup={args.warmup_epochs}, batch={args.batch_size}")
    print(f"          amp={scaler is not None}, grad_clip={args.grad_clip_norm}")

    # ========== TRAINING LOOP ==========
    print("\n" + "=" * 60)
    print("Training started")
    print("=" * 60)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        lr = scheduler.get_last_lr()[0]
        print(f"\nEpoch {epoch+1}/{args.epochs} | LR: {lr:.6f}")
        print("-" * 40)

        epoch_start = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            modality=args.modality, scaler=scaler,
            grad_clip_norm=args.grad_clip_norm, epoch=epoch
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device, args.modality)
        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start
        print(f"Train: loss={train_loss:.4f}, acc={train_acc:.2f}%")
        print(f"Val:   loss={val_loss:.4f}, acc={val_acc:.2f}%")
        print(f"Time:  {epoch_time:.1f}s ({epoch_time/60:.1f} min)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'config': vars(args)
            }, os.path.join(args.output_dir, "best_model.pt"))
            print(f"[Saved best model]")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{args.early_stop}")

        if patience_counter >= args.early_stop:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print("\n" + "=" * 60)
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
