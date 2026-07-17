#!/usr/bin/env python
"""
Preprocess parquet files and cache as PyTorch tensors.

Two output modes:
  1. Cache directory (default): save as individual .pt files
  2. Merged file (--merged_output): stream all samples into a single .sermerged file

Usage:
    # Mode 1: Individual .pt files (existing behavior)
    python preprocess.py --input "data/*.parquet" --output cache/

    # Mode 2: Single merged file (no intermediate small files, saves space)
    python preprocess.py --input "data/*.parquet" --merged_output cache.sermerged
"""

import argparse
import glob
import os
import random
import sys
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd

from utils.dataset import CantoneseSERDataset
from utils.audio_utils import AudioPreprocessor
from utils.text_utils import TextPreprocessor

# Import merged format writer
from merged_format import MergedStreamWriter


def preprocess_and_cache(
    input_pattern: str,
    output_dir: str,
    confidence_threshold: float = 0.0,
    sample_rate: int = 22050,
    n_mels: int = 80,
    organize_by_label: bool = False,
    balance_target: str = None,
    max_aug_ratio: float = None,
    resume: bool = False,
):
    """
    Preprocess all parquet files matching the pattern and save as .pt files.

    Args:
        input_pattern: Glob pattern for parquet files (e.g., "data/*.parquet")
        output_dir: Output directory for cached files
        confidence_threshold: Minimum confidence to include sample
        sample_rate: Audio sample rate
        n_mels: Number of mel filter banks
        organize_by_label: If True, save into label subdirectories
        balance_target: Target samples per class ('median' or int string, e.g. '5000')
        max_aug_ratio: Max augmented copies per original sample (e.g. 2.0 = at most 2x augmentation)
        resume: If True, skip parquet files that already have cached output
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create preprocessors
    audio_prep = AudioPreprocessor(
        sample_rate=sample_rate,
        n_mels=n_mels
    )
    text_prep = TextPreprocessor()

    # Find all parquet files
    parquet_files = sorted(glob.glob(input_pattern))

    if not parquet_files:
        print(f"No parquet files found matching: {input_pattern}")
        return

    print(f"Found {len(parquet_files)} parquet files")

    # If resume, skip files that already have cached output
    if resume:
        remaining = []
        for pf in parquet_files:
            basename = Path(pf).stem
            # Check if any file with this basename exists in output_dir (flat or nested)
            if organize_by_label:
                # Check subdirectories for files with this basename
                already_cached = False
                for subdir in os.listdir(output_dir):
                    subdir_path = os.path.join(output_dir, subdir)
                    if os.path.isdir(subdir_path):
                        if any(f.startswith(basename) for f in os.listdir(subdir_path)):
                            already_cached = True
                            break
            else:
                already_cached = any(f.startswith(basename) for f in os.listdir(output_dir))

            if already_cached:
                print(f"  Skipping {pf} (already cached)")
            else:
                remaining.append(pf)

        if not remaining:
            print("All parquet files already cached. Nothing to do.")
            return
        print(f"Resuming with {len(remaining)} remaining files")
        parquet_files = remaining

    # --- Pass 1: count labels and determine augmentation needs ---
    if balance_target is not None:
        print("\n[Pass 1] Counting class distribution...")

        # Use same label map as CantoneseSERDataset
        label_map = {
            'angry': 0, 'disgusted': 1, 'fearful': 2, 'happy': 3,
            'neutral': 4, 'other': 5, 'sad': 6, 'surprised': 7,
            '<unk>': 5,
        }

        all_labels = []
        for parquet_file in parquet_files:
            df = pd.read_parquet(parquet_file)
            if confidence_threshold > 0 and 'confidence' in df.columns:
                df = df[df['confidence'] >= confidence_threshold]
            file_labels = df['label'].map(lambda s: label_map.get(s, -1)).values
            all_labels.extend(file_labels)

        all_labels = np.array(all_labels, dtype=np.int64)
        all_labels = all_labels[all_labels >= 0]  # filter unknown

        unique_classes = np.unique(all_labels)
        class_counts = {int(c): int(np.sum(all_labels == c)) for c in unique_classes}

        # Parse target
        if balance_target.lower() == 'median':
            target = int(np.median(list(class_counts.values())))
        else:
            target = int(balance_target)

        print(f"  Balance target: {target} per class")
        print(f"  Original counts: {class_counts}")

        # Determine which classes need augmentation (count < target)
        need_augmentation = {c for c, cnt in class_counts.items() if cnt < target}
        need_downsample = {c for c, cnt in class_counts.items() if cnt > target}
        if need_augmentation:
            info = {}
            for c in need_augmentation:
                orig = class_counts[c]
                n_aug = target - orig
                if max_aug_ratio is not None:
                    max_allowed = int(orig * max_aug_ratio)
                    n_aug = min(n_aug, max_allowed)
                info[c] = f"{orig}\u2192{orig + n_aug} ({'+' + str(n_aug)})"
            print(f"  Augmenting: {info}")
        if need_downsample:
            print(f"  Downsampling: { {c: class_counts[c] for c in need_downsample} } -> {target}")
        if max_aug_ratio is not None:
            print(f"  Augmentation cap: {max_aug_ratio}x per original")
    else:
        target = None
        need_augmentation = set()
        class_counts = {}

    # --- Pass 2: process and save ---
    total_samples = 0
    per_class_counts = {}

    for parquet_file in parquet_files:
        print(f"\nProcessing {parquet_file}...")

        # Create dataset for this file
        dataset = CantoneseSERDataset(
            parquet_path=parquet_file,
            audio_preprocessor=audio_prep,
            text_preprocessor=text_prep,
            confidence_threshold=confidence_threshold
        )
        labels = dataset.labels  # numpy array of ints

        # Build reverse label map for subdirectory naming
        reverse_label_map = {v: k for k, v in dataset.label_map.items()}
        reverse_label_map[5] = 'other'  # prefer 'other' over '<unk>'

        basename = Path(parquet_file).stem

        # Determine which local indices belong to each class
        local_indices_by_class = {}
        for idx, label_id in enumerate(labels):
            label_id = int(label_id)
            if label_id < 0:
                continue
            local_indices_by_class.setdefault(label_id, []).append(idx)

        # Prepare indices to process
        indices_to_process = []
        augment_requests = {}  # {label_id: num_augmented_copies_needed}

        for label_id, local_indices in local_indices_by_class.items():
            count = len(local_indices)
            if balance_target is not None:
                if count > target:
                    # Downsample: randomly select 'target' samples
                    selected = random.sample(local_indices, target)
                    indices_to_process.extend((idx, False) for idx in selected)
                elif count == target:
                    indices_to_process.extend((idx, False) for idx in local_indices)
                else:
                    # Need augmentation: save all originals + generate augmented copies (with cap)
                    indices_to_process.extend((idx, False) for idx in local_indices)
                    n_augmented = target - count
                    if max_aug_ratio is not None:
                        max_allowed = int(count * max_aug_ratio)
                        n_augmented = min(n_augmented, max(max_allowed, 1))
                    if n_augmented > 0:
                        augment_requests[label_id] = n_augmented
            else:
                # Process all originals
                indices_to_process.extend((idx, False) for idx in local_indices)

        # Distribute augmentation copies across originals of each class
        for label_id, n_augmented in augment_requests.items():
            originals_of_class = [idx for idx, is_aug in indices_to_process if labels[idx] == label_id and not is_aug]
            if originals_of_class:
                copies_per_original = n_augmented // len(originals_of_class)
                extra = n_augmented % len(originals_of_class)
                for j, orig_idx in enumerate(originals_of_class):
                    n_copies = copies_per_original + (1 if j < extra else 0)
                    for aug_i in range(n_copies):
                        indices_to_process.append((orig_idx, True))

        # Shuffle to interleave classes
        random.shuffle(indices_to_process)

        # Process and save
        aug_counter = 0  # global counter for augmented files
        for idx, is_augmented in tqdm(indices_to_process, desc=f"Processing {basename}"):
            if is_augmented:
                # Enable augmentations, re-process the same audio/text
                audio_prep.train()
                sample = dataset[idx]
                audio_prep.eval()
                aug_suffix = f"_aug{aug_counter:04d}"
                aug_counter += 1
            else:
                sample = dataset[idx]
                aug_suffix = ""

            label_id = sample['label'].item()
            label_name = reverse_label_map.get(label_id, f'class_{label_id}')

            # Determine save path
            if organize_by_label:
                label_dir = os.path.join(output_dir, label_name)
                os.makedirs(label_dir, exist_ok=True)
                cache_path = os.path.join(label_dir, f"{basename}_{idx:06d}{aug_suffix}.pt")
            else:
                cache_path = os.path.join(output_dir, f"{basename}_{idx:06d}{aug_suffix}.pt")

            torch.save(sample, cache_path)
            total_samples += 1
            per_class_counts[label_name] = per_class_counts.get(label_name, 0) + 1

    print(f"\n{'='*60}")
    print(f"Done! Preprocessed {total_samples} samples")
    if per_class_counts:
        print(f"\nPer-class counts:")
        for label_name, count in sorted(per_class_counts.items()):
            print(f"  {label_name}: {count}")
    print(f"Cached to: {output_dir}")
    print(f"{'='*60}")


def preprocess_and_merge(
    input_pattern: str,
    merged_output: str,
    confidence_threshold: float = 0.0,
    sample_rate: int = 22050,
    n_mels: int = 80,
    balance_target: str = None,
    max_aug_ratio: float = None,
):
    """
    Preprocess all parquet files and STREAM directly into a single .sermerged file.

    No intermediate .pt files are created. Samples are appended to the output
    file as they are processed, keeping memory usage minimal.

    Args:
        input_pattern: Glob pattern for parquet files (e.g., "data/*.parquet")
        merged_output: Output .sermerged file path
        confidence_threshold: Minimum confidence to include sample
        sample_rate: Audio sample rate
        n_mels: Number of mel filter banks
        balance_target: Target samples per class ('median' or int string)
        max_aug_ratio: Max augmented copies per original sample
    """

    # Create preprocessors
    audio_prep = AudioPreprocessor(sample_rate=sample_rate, n_mels=n_mels)
    text_prep = TextPreprocessor()

    # Find all parquet files
    parquet_files = sorted(glob.glob(input_pattern))
    if not parquet_files:
        print(f"No parquet files found matching: {input_pattern}")
        return
    print(f"Found {len(parquet_files)} parquet files")

    # --- Pass 1: count labels and determine augmentation needs ---
    if balance_target is not None:
        print("\n[Pass 1] Counting class distribution...")
        label_map = {
            'angry': 0, 'disgusted': 1, 'fearful': 2, 'happy': 3,
            'neutral': 4, 'other': 5, 'sad': 6, 'surprised': 7,
            '<unk>': 5,
        }
        all_labels = []
        for parquet_file in parquet_files:
            df = pd.read_parquet(parquet_file)
            if confidence_threshold > 0 and 'confidence' in df.columns:
                df = df[df['confidence'] >= confidence_threshold]
            file_labels = df['label'].map(lambda s: label_map.get(s, -1)).values
            all_labels.extend(file_labels)
        all_labels = np.array(all_labels, dtype=np.int64)
        all_labels = all_labels[all_labels >= 0]
        unique_classes = np.unique(all_labels)
        class_counts = {int(c): int(np.sum(all_labels == c)) for c in unique_classes}
        if balance_target.lower() == 'median':
            target = int(np.median(list(class_counts.values())))
        else:
            target = int(balance_target)
        print(f"  Balance target: {target} per class")
        print(f"  Original counts: {class_counts}")
        need_augmentation = {c for c, cnt in class_counts.items() if cnt < target}
        if need_augmentation:
            info = {}
            for c in need_augmentation:
                orig = class_counts[c]
                n_aug = target - orig
                if max_aug_ratio is not None:
                    max_allowed = int(orig * max_aug_ratio)
                    n_aug = min(n_aug, max(max_allowed, 1))
                info[c] = f"{orig}\u2192{orig + n_aug} ({'+' + str(n_aug)})"
            print(f"  Augmenting: {info}")
    else:
        target = None
        need_augmentation = set()

    # --- Pass 2: process and stream to merged file ---
    writer = MergedStreamWriter(merged_output)
    total_samples = 0
    per_class_counts = {}

    # Build reverse label map
    _tmp_ds = CantoneseSERDataset(
        parquet_path=parquet_files[0],
        audio_preprocessor=audio_prep,
        text_preprocessor=text_prep,
        confidence_threshold=confidence_threshold
    )
    reverse_label_map = {v: k for k, v in _tmp_ds.label_map.items()}
    reverse_label_map[5] = 'other'
    del _tmp_ds

    for parquet_file in parquet_files:
        print(f"\nProcessing {parquet_file}...")

        dataset = CantoneseSERDataset(
            parquet_path=parquet_file,
            audio_preprocessor=audio_prep,
            text_preprocessor=text_prep,
            confidence_threshold=confidence_threshold
        )
        labels = dataset.labels

        # Class grouping
        local_indices_by_class = {}
        for idx, label_id in enumerate(labels):
            label_id = int(label_id)
            if label_id < 0:
                continue
            local_indices_by_class.setdefault(label_id, []).append(idx)

        # Prepare indices
        indices_to_process = []
        augment_requests = {}
        for label_id, local_indices in local_indices_by_class.items():
            count = len(local_indices)
            if balance_target is not None:
                if count > target:
                    selected = random.sample(local_indices, target)
                    indices_to_process.extend((idx, False) for idx in selected)
                elif count == target:
                    indices_to_process.extend((idx, False) for idx in local_indices)
                else:
                    indices_to_process.extend((idx, False) for idx in local_indices)
                    n_augmented = target - count
                    if max_aug_ratio is not None:
                        max_allowed = int(count * max_aug_ratio)
                        n_augmented = min(n_augmented, max(max_allowed, 1))
                    if n_augmented > 0:
                        augment_requests[label_id] = n_augmented
            else:
                indices_to_process.extend((idx, False) for idx in local_indices)

        # Distribute augmentations
        for label_id, n_augmented in augment_requests.items():
            originals_of_class = [idx for idx, is_aug in indices_to_process
                                  if labels[idx] == label_id and not is_aug]
            if originals_of_class:
                copies_per_original = n_augmented // len(originals_of_class)
                extra = n_augmented % len(originals_of_class)
                for j, orig_idx in enumerate(originals_of_class):
                    n_copies = copies_per_original + (1 if j < extra else 0)
                    for _ in range(n_copies):
                        indices_to_process.append((orig_idx, True))

        random.shuffle(indices_to_process)

        # Process and stream to merged file
        aug_counter = 0
        for idx, is_augmented in tqdm(indices_to_process, desc=f"Streaming {Path(parquet_file).stem}"):
            if is_augmented:
                audio_prep.train()
                sample = dataset[idx]
                audio_prep.eval()
                aug_counter += 1
            else:
                sample = dataset[idx]

            label_id = sample['label'].item()
            label_name = reverse_label_map.get(label_id, f'class_{label_id}')

            writer.write_sample(sample)
            total_samples += 1
            per_class_counts[label_name] = per_class_counts.get(label_name, 0) + 1

    # Finalize
    writer.finalize(meta={'source_files': parquet_files})

    print(f"\n{'='*60}")
    print(f"Done! Preprocessed {total_samples} samples")
    if per_class_counts:
        print(f"\nPer-class counts:")
        for label_name, count in sorted(per_class_counts.items()):
            print(f"  {label_name}: {count}")
    print(f"Output: {merged_output}")
    print(f"{'='*60}")


class CachedDataset(torch.utils.data.Dataset):
    """
    Dataset that loads preprocessed samples from cache.

    Supports:
      - Flat directory structure (all .pt in one dir)
      - Nested structure (subdirectories per label, e.g., cache/angry/*.pt)
      - Single merged .sermerged file

    Args:
        cache_source: Path to cache directory OR .sermerged file
        include_labels: Optional set of label names to include (nested mode)
    """

    def __init__(self, cache_source: str, include_labels: set = None):
        cache_path = Path(cache_source)

        # Auto-detect format
        if cache_path.is_file():
            # Single file: could be .sermerged or old-style .pt
            if cache_path.suffix == '.sermerged':
                # Use MergedDataset
                from merged_format import MergedDataset
                self._impl = MergedDataset(str(cache_path), prefetch_labels=True)
                self._mode = 'sermerged'
            else:
                raise ValueError(f"Unknown single file format: {cache_path}")
        elif cache_path.is_dir():
            # Directory: traditional .pt files
            self._impl = None
            self._mode = 'dir'
            self.cache_dir = cache_path

            # Detect structure: nested or flat
            subdirs = [d for d in self.cache_dir.iterdir() if d.is_dir()]
            if subdirs:
                self.cache_files = []
                self._label_dirs = {}
                for subdir in sorted(subdirs):
                    label_name = subdir.name
                    if include_labels is None or label_name in include_labels:
                        files = sorted(subdir.glob("*.pt"))
                        self.cache_files.extend(files)
                        self._label_dirs[label_name] = files
                if include_labels:
                    print(f"  Filtered by labels: {include_labels}")
            else:
                self.cache_files = sorted(self.cache_dir.glob("*.pt"))
                self._label_dirs = None

            if not self.cache_files:
                raise ValueError(f"No .pt files found in {cache_source}")

            print(f"Loaded {len(self.cache_files)} cached samples from {cache_source}")
            self._labels = None
        else:
            raise ValueError(f"Cache source not found: {cache_source}")

    @property
    def labels(self):
        if self._mode == 'sermerged':
            return self._impl.labels
        # Directory mode
        if self._labels is None:
            self._labels = np.array([
                torch.load(f)['label'].item()
                for f in self.cache_files
            ], dtype=np.int64)
        return self._labels

    def __len__(self) -> int:
        if self._mode == 'sermerged':
            return len(self._impl)
        return len(self.cache_files)

    def __getitem__(self, idx: int) -> dict:
        if self._mode == 'sermerged':
            return self._impl[idx]
        return torch.load(self.cache_files[idx])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess parquet files")
    parser.add_argument("--input", required=True, help="Input parquet file or glob pattern")

    # Output mode: exactly one of --output or --merged_output must be given
    parser.add_argument("--output", default=None, help="Output cache directory (individual .pt files)")
    parser.add_argument("--merged_output", default=None, help="Output .sermerged file (single merged file, saves space)")

    parser.add_argument("--confidence", type=float, default=0.0, help="Confidence threshold")
    parser.add_argument("--sample_rate", type=int, default=22050, help="Audio sample rate")
    parser.add_argument("--n_mels", type=int, default=80, help="Number of mel filters")
    parser.add_argument("--organize_by_label", action="store_true", help="Save into label subdirectories (dir mode only)")
    parser.add_argument("--balance_target", type=str, default=None,
                        help="Target samples per class ('median' or int)")
    parser.add_argument("--max_aug_ratio", type=float, default=None,
                        help="Max augmented copies per original (e.g. 2.0)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip parquet files that already have cached output (dir mode only)")
    args = parser.parse_args()

    # Validate: exactly one output mode
    if args.output is None and args.merged_output is None:
        parser.error("Must specify one of --output or --merged_output")
    if args.output is not None and args.merged_output is not None:
        parser.error("Cannot specify both --output and --merged_output")

    if args.merged_output is not None:
        # Merged mode
        preprocess_and_merge(
            input_pattern=args.input,
            merged_output=args.merged_output,
            confidence_threshold=args.confidence,
            sample_rate=args.sample_rate,
            n_mels=args.n_mels,
            balance_target=args.balance_target,
            max_aug_ratio=args.max_aug_ratio,
        )
    else:
        # Directory mode (existing behavior)
        preprocess_and_cache(
            input_pattern=args.input,
            output_dir=args.output,
            confidence_threshold=args.confidence,
            sample_rate=args.sample_rate,
            n_mels=args.n_mels,
            organize_by_label=args.organize_by_label,
            balance_target=args.balance_target,
            max_aug_ratio=args.max_aug_ratio,
            resume=args.resume,
        )
