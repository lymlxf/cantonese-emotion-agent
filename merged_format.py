"""
Streamable Merged Dataset Format (.sermerged) — v3

Single-file format for efficient storage of preprocessed samples.
Designed for out-of-core training: memory-mapped random access
without loading all data into RAM.

File Layout v3:
    [0:10]          magic     b"SERMERGED\x00"
    [10:11]         version   uint8 (3)
    [11:19]         num_samples      int64
    [19:27]         data_start       int64  (byte offset where DATA region begins)
    [27:35]         index_offset     int64  (byte offset where INDEX region begins)
    [35:43]         labels_offset    int64  (byte offset where LABELS region begins)
    [43:RESERVED]   META: JSON-encoded metadata (padded with zeros to RESERVED)
    [data_start:index_offset]   DATA: consecutive pickle.dumps(sample) blocks
    [index_offset:labels_offset] INDEX: N entries of (offset:int64, size:int64)
    [labels_offset:end]         LABELS: numpy int64 array (N elements)

Key fixes in v3:
    - RESERVED_HEADER_SIZE (4KB) prevents header+meta from overwriting DATA
    - labels stored as compact binary (not in JSON)
    - No data shifting during finalize

No external dependencies. Uses only Python stdlib + pickle.
"""

import os
import json
import pickle
import struct
import mmap
from pathlib import Path
from typing import Optional, BinaryIO

import torch
import numpy as np


# ==== Constants ====
MAGIC = b"SERMERGED\x00"
VERSION = 3
HEADER_SIZE = 43  # fixed binary header (before META)
RESERVED_HEADER_SIZE = 4096  # total reserved space: header + meta + padding
INDEX_ENTRY_SIZE = 16  # (offset: int64, size: int64)


def write_header(f: BinaryIO, num_samples: int, data_start: int,
                 index_offset: int, labels_offset: int, meta: dict):
    """Write file header + meta at position 0, padded to data_start."""
    f.seek(0)
    f.write(MAGIC)                                          # 10 bytes
    f.write(struct.pack("<B", VERSION))                     # 1 byte
    f.write(struct.pack("<q", num_samples))                 # 8 bytes
    f.write(struct.pack("<q", data_start))                  # 8 bytes
    f.write(struct.pack("<q", index_offset))                # 8 bytes
    f.write(struct.pack("<q", labels_offset))               # 8 bytes
    # META: JSON (small, fits in reserved space)
    meta_bytes = json.dumps(meta, ensure_ascii=False).encode('utf-8')
    max_meta = data_start - HEADER_SIZE
    if len(meta_bytes) > max_meta:
        # Truncate meta if too large (shouldn't happen in practice)
        meta_bytes = meta_bytes[:max_meta]
    f.write(meta_bytes)
    # Pad remainder with zeros
    padding = data_start - f.tell()
    if padding > 0:
        f.write(b'\x00' * padding)
    assert f.tell() == data_start, f"Header write mismatch: {f.tell()} != {data_start}"


def read_header(f: BinaryIO) -> dict:
    """Read and return header info."""
    f.seek(0)
    magic = f.read(10)
    if magic != MAGIC:
        raise ValueError(f"Invalid file format: magic mismatch (got {magic!r})")
    version = struct.unpack("<B", f.read(1))[0]
    if version != VERSION:
        raise ValueError(
            f"File format version mismatch: file=v{version}, code=v{VERSION}. "
            f"Please regenerate the .sermerged file with the current script."
        )
    num_samples = struct.unpack("<q", f.read(8))[0]
    data_start = struct.unpack("<q", f.read(8))[0]
    index_offset = struct.unpack("<q", f.read(8))[0]
    labels_offset = struct.unpack("<q", f.read(8))[0]

    # META: between HEADER_SIZE and data_start
    meta_size = data_start - HEADER_SIZE
    if meta_size > 0:
        meta_bytes = f.read(meta_size).rstrip(b'\x00')
        if meta_bytes:
            meta = json.loads(meta_bytes.decode('utf-8'))
        else:
            meta = {}
    else:
        meta = {}

    return {
        'num_samples': num_samples,
        'data_start': data_start,
        'index_offset': index_offset,
        'labels_offset': labels_offset,
        'meta': meta,
    }


class MergedStreamWriter:
    """
    Stream writer for .sermerged format.

    Usage:
        writer = MergedStreamWriter("output.sermerged")
        for sample in samples:
            writer.write_sample(sample)
        writer.finalize()
    """

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(output_path, 'wb')

        # Reserve header space (will be filled during finalize)
        self.data_start = RESERVED_HEADER_SIZE
        self.f.write(b'\x00' * self.data_start)

        self.num_samples = 0
        self.data_blocks = []  # List of (offset, size) relative to data_start
        self.labels = []

    def write_sample(self, sample: dict):
        """Append a single sample. Call in preprocessing loop."""
        serializable = {}
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                serializable[k] = v.cpu().numpy()
            else:
                serializable[k] = v

        data_bytes = pickle.dumps(serializable, protocol=pickle.HIGHEST_PROTOCOL)
        offset = self.f.tell() - self.data_start
        size = len(data_bytes)

        self.f.write(data_bytes)
        self.data_blocks.append((offset, size))
        self.labels.append(int(sample['label']))
        self.num_samples += 1

    def finalize(self, meta: dict = None):
        """Finalize: write index, labels, and header."""
        if meta is None:
            meta = {}

        # 1. DATA region ends here
        index_offset = self.f.tell()

        # 2. Write INDEX region
        for offset, size in self.data_blocks:
            self.f.write(struct.pack("<qq", offset, size))

        # 3. Write LABELS region (compact binary, not JSON)
        labels_offset = self.f.tell()
        labels_array = np.array(self.labels, dtype=np.int64)
        self.f.write(labels_array.tobytes())

        # 4. Rewrite header at position 0 (overwrites reserved zeros, NOT data)
        write_header(
            self.f,
            num_samples=self.num_samples,
            data_start=self.data_start,
            index_offset=index_offset,
            labels_offset=labels_offset,
            meta=meta,
        )

        self.f.close()

        # Stats
        file_size = self.output_path.stat().st_size
        print(f"[MergedStreamWriter] Finalized: {self.num_samples:,} samples")
        print(f"  File: {self.output_path} ({file_size / 1024**3:.2f} GB)")
        print(f"  Reserved header: {self.data_start} bytes")
        print(f"  Data: {self.data_start}..{index_offset} ({(index_offset - self.data_start) / 1024**3:.2f} GB)")
        print(f"  Index: {index_offset}..{labels_offset} ({len(self.data_blocks):,} entries)")
        print(f"  Labels: {labels_offset}..{file_size} ({labels_array.nbytes / 1024**2:.1f} MB)")


class MergedDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset backed by a .sermerged file with memory mapping.

    No data is pre-loaded. Samples fetched via memory-mapped random access.
    OS automatically caches frequently accessed pages in RAM.

    Args:
        file_path: Path to .sermerged file
    """

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        self.file_size = self.file_path.stat().st_size

        # Open file and read header
        self._f = open(file_path, 'rb')
        header = read_header(self._f)
        self.num_samples = header['num_samples']
        self.data_start = header['data_start']
        self.index_offset = header['index_offset']
        self.labels_offset = header['labels_offset']
        self.meta = header['meta']

        # Memory map the entire file
        self._mmap = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)

        # Pre-read index into memory (small: N * 16 bytes)
        self._index = np.empty((self.num_samples, 2), dtype=np.int64)
        idx_start = self.index_offset
        for i in range(self.num_samples):
            off = idx_start + i * INDEX_ENTRY_SIZE
            self._index[i, 0] = struct.unpack("<q", self._mmap[off:off+8])[0]
            self._index[i, 1] = struct.unpack("<q", self._mmap[off+8:off+16])[0]

        # Read labels from compact binary region
        labels_size = self.num_samples * 8  # int64 = 8 bytes each
        labels_bytes = self._mmap[self.labels_offset:self.labels_offset + labels_size]
        self._labels = np.frombuffer(labels_bytes, dtype=np.int64).copy()

        print(f"[MergedDataset] Loaded {self.num_samples:,} samples from {file_path}")
        print(f"  File size: {self.file_size / 1024**3:.2f} GB")
        print(f"  Index memory: {self._index.nbytes / 1024**2:.1f} MB")
        print(f"  Labels memory: {self._labels.nbytes / 1024**2:.1f} MB")

    @property
    def labels(self):
        return self._labels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        """Fetch sample via memory-mapped random access."""
        offset, size = self._index[idx]
        data_start_abs = self.data_start + offset
        data_bytes = self._mmap[data_start_abs:data_start_abs + size]
        sample = pickle.loads(data_bytes)

        # Convert numpy arrays back to tensors
        for k, v in sample.items():
            if isinstance(v, np.ndarray):
                sample[k] = torch.from_numpy(v)
        return sample

    def close(self):
        self._mmap.close()
        self._f.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self):
        """Support for DataLoader pickling (multiprocessing)."""
        return {'file_path': str(self.file_path)}

    def __setstate__(self, state):
        """Re-open file after pickling."""
        self.__init__(state['file_path'])


def benchmark(sermerged_path: str, num_random: int = 10000):
    """Benchmark random/sequential read speed."""
    import time

    ds = MergedDataset(sermerged_path)
    n = min(num_random, len(ds))

    print(f"\nSequential access ({n:,} samples)...")
    start = time.time()
    for i in range(n):
        _ = ds[i]
    t_seq = time.time() - start
    print(f"  {t_seq:.1f}s ({n/t_seq:,.0f} samples/sec)")

    print(f"\nRandom access ({n:,} samples)...")
    indices = np.random.permutation(len(ds))[:n]
    start = time.time()
    for idx in indices:
        _ = ds[int(idx)]
    t_rand = time.time() - start
    print(f"  {t_rand:.1f}s ({n/t_rand:,.0f} samples/sec)")

    est_epoch = (len(ds) / (n / t_rand)) / 60
    print(f"\nEstimated epoch time (data only): {est_epoch:.1f} min")
    ds.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", help="Benchmark a .sermerged file")
    parser.add_argument("--num-samples", type=int, default=10000)
    args = parser.parse_args()

    if args.benchmark:
        benchmark(args.benchmark, args.num_samples)
