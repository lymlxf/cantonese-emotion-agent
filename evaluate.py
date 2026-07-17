#!/usr/bin/env python
"""
Evaluate a trained model on test data.

Supports multiple data formats (auto-detected):
  - .sermerged file    --data cache.sermerged     (memory-mapped, RECOMMENDED)
  - Cache directory    --data cache/              (individual .pt files)
  - Parquet file/dir   --data data/test.parquet   (raw data, on-the-fly preprocessing)

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pt --data cache.sermerged
    python evaluate.py --checkpoint checkpoints/best_model.pt --data cache/
    python evaluate.py --checkpoint checkpoints/best_model.pt --data data/test.parquet --batch_size 64
"""

import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)

from models import MultimodalSER
from utils.collate_fn import collate_fn, collate_fn_audio_only


LABEL_NAMES = {
    0: 'angry', 1: 'disgusted', 2: 'fearful', 3: 'happy',
    4: 'neutral', 5: 'other', 6: 'sad', 7: 'surprised',
}


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = ckpt['config']
    modality = config.get('modality', 'both')

    model = MultimodalSER(
        d_model=config.get('d_model', 256),
        num_heads=config.get('num_heads', 8),
        num_classes=config.get('num_classes', 8),
        dropout=config.get('dropout', 0.2),
        modality=modality,
    ).to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"  Modality: {modality}")
    print(f"  Val acc (training): {ckpt.get('val_acc', 'N/A'):.2f}%")
    print(f"  Epoch: {ckpt.get('epoch', 'N/A')}")

    return model, config


def load_dataset(data_path: str, config: dict):
    """Load dataset from .sermerged, cache directory, or parquet."""
    modality = config.get('modality', 'both')
    collate = collate_fn if modality in ("both",) else collate_fn_audio_only

    data_path_obj = Path(data_path)

    # ---- 1. .sermerged file (memory-mapped, RECOMMENDED) ----
    if data_path_obj.is_file() and data_path_obj.suffix == '.sermerged':
        print(f"[Format] .sermerged memory-mapped file: {data_path}")
        from merged_format import MergedDataset
        dataset = MergedDataset(str(data_path))

    # ---- 2. Cache directory (individual .pt files) ----
    elif os.path.isdir(data_path) and (
        any(f.endswith('.pt') for f in os.listdir(data_path))
        or any(
            os.path.isdir(os.path.join(data_path, d)) and
            any(f.endswith('.pt') for f in os.listdir(os.path.join(data_path, d)))
            for d in os.listdir(data_path)
        )
    ):
        print(f"[Format] Cache directory: {data_path}")
        from preprocess import CachedDataset
        dataset = CachedDataset(data_path)

    # ---- 3. Parquet (on-the-fly preprocessing) ----
    else:
        print(f"[Format] Parquet (on-the-fly preprocessing): {data_path}")
        from utils import CantoneseSERDataset, AudioPreprocessor, TextPreprocessor
        dataset = CantoneseSERDataset(
            parquet_path=data_path,
            audio_preprocessor=AudioPreprocessor(),
            text_preprocessor=TextPreprocessor(),
            confidence_threshold=config.get('confidence_threshold', 0.0),
        )

    loader = DataLoader(
        dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
        pin_memory=True,
    )

    print(f"  Samples: {len(dataset)}")
    print(f"  Batches: {len(loader)}")
    return loader


@torch.no_grad()
def evaluate(model, dataloader, device, modality):
    """Run inference and collect predictions."""
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        audio = batch['audio'].to(device)
        audio_lengths = batch['audio_lengths'].to(device)
        labels = batch['labels'].to(device)

        if modality == "both":
            text = batch['text'].to(device)
            text_lengths = batch['text_lengths'].to(device)
            # Safety clamp (same as train_fast.py)
            text_lengths = text_lengths.clamp(min=1)
            audio_lengths = audio_lengths.clamp(min=1)
            logits = model(audio, text, audio_lengths, text_lengths)
        elif modality == "audio":
            audio_lengths = audio_lengths.clamp(min=1)
            logits = model(audio, audio_lengths=audio_lengths)
        elif modality == "text":
            text = batch['text'].to(device)
            text_lengths = batch['text_lengths'].to(device)
            text_lengths = text_lengths.clamp(min=1)
            logits = model(text_input=text, text_lengths=text_lengths)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return torch.tensor(all_labels), torch.tensor(all_preds)


def print_results(y_true, y_pred, num_classes):
    """Print evaluation metrics."""
    y_true_np = y_true.cpu().numpy() if torch.is_tensor(y_true) else y_true
    y_pred_np = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else y_pred

    acc = accuracy_score(y_true_np, y_pred_np)
    print(f"\n{'='*60}")
    print(f"Overall Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"{'='*60}")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_np, y_pred_np, labels=range(num_classes), zero_division=0
    )

    print(f"\n{'Class':<12} {'Samples':>8} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 55)
    for i in range(num_classes):
        name = LABEL_NAMES.get(i, f'class_{i}')
        print(f"{name:<12} {support[i]:>8} {precision[i]:>10.4f} {recall[i]:>10.4f} {f1[i]:>10.4f}")

    print("-" * 55)
    print(f"{'macro avg':<12} {sum(support):>8} "
          f"{np.mean(precision):>10.4f} {np.mean(recall):>10.4f} {np.mean(f1):>10.4f}")

    weighted_f1 = np.average(f1, weights=support) if sum(support) > 0 else 0
    print(f"{'weighted avg':<12} {sum(support):>8} {'':>10} {'':>10} {weighted_f1:>10.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_true_np, y_pred_np, labels=range(num_classes))
    print(f"\nConfusion Matrix (rows=true, cols=pred):")
    header = "".join(f"{LABEL_NAMES.get(i, str(i)):>8}" for i in range(num_classes))
    print(f"{'':>10}{header}")
    for i in range(num_classes):
        row = "".join(f"{cm[i][j]:>8}" for j in range(num_classes))
        print(f"{LABEL_NAMES.get(i, str(i)):>10}{row}")

    return {
        'accuracy': float(acc),
        'per_class': {
            LABEL_NAMES.get(i, f'class_{i}'): {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i]),
            }
            for i in range(num_classes)
        },
        'macro_f1': float(np.mean(f1)),
        'weighted_f1': float(weighted_f1),
        'confusion_matrix': cm.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained SER model")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--data", required=True, help="Path to test data (.sermerged, cache dir, or parquet)")
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (override)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, config = load_model(args.checkpoint, device)
    if args.batch_size:
        config['batch_size'] = args.batch_size

    loader = load_dataset(args.data, config)

    print("\nRunning evaluation...")
    y_true, y_pred = evaluate(model, loader, device, config['modality'])

    results = print_results(y_true, y_pred, config.get('num_classes', 8))

    if args.output:
        results['config'] = config
        results['checkpoint'] = args.checkpoint
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
