"""
Balanced batch sampler for handling extreme class imbalance.
Ensures each batch contains roughly equal samples per class.
"""

import numpy as np
from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler):
    """
    Yields batches where each class has roughly equal representation.
    
    For classes with fewer than samples_per_class samples, indices are
    sampled with replacement. For classes with more, they are sampled
    without replacement, producing different samples in each batch.
    
    Args:
        labels: Array-like of integer labels for each sample
        batch_size: Total number of samples per batch
        num_classes: Number of classes (auto-detected if None)
        replacement: Allow replacement for minority classes (default: True)
    
    Example:
        >>> labels = [0, 0, 0, 1, 1, 2]  # 3:2:1 imbalance
        >>> sampler = BalancedBatchSampler(labels, batch_size=3, num_classes=3)
        >>> batch = next(iter(sampler))
        >>> # batch contains 1 sample from each class
    """
    
    def __init__(
        self,
        labels,
        batch_size: int,
        num_classes: int = None,
        replacement: bool = True,
    ):
        if isinstance(labels, list):
            labels = np.array(labels, dtype=np.int64)
        elif isinstance(labels, np.ndarray):
            labels = labels.astype(np.int64)
        
        self.batch_size = batch_size
        self.replacement = replacement
        
        # Filter out unknown labels (typically -1)
        valid_mask = labels >= 0
        labels = labels[valid_mask]
        # Keep original indices for final output
        self._valid_indices = np.where(valid_mask)[0]
        
        # Determine number of classes
        unique_labels = np.unique(labels)
        if num_classes is None:
            self.num_classes = len(unique_labels)
        else:
            self.num_classes = num_classes
        
        # Group valid indices by class label
        self.class_indices = {}
        for c in unique_labels:
            # Map back to original index space
            class_mask = labels == c
            self.class_indices[int(c)] = self._valid_indices[class_mask]
        
        # Samples per class per batch
        self.samples_per_class = batch_size // self.num_classes
        self.remainder = batch_size % self.num_classes
        
        # Epoches: determined by the class with the most samples
        # Each epoch covers all samples from the largest class
        max_class_size = max(len(v) for v in self.class_indices.values())
        self.num_batches = max(1, max_class_size // self.samples_per_class)
    
    def __iter__(self):
        """Yield balanced batches for one epoch."""
        class_keys = list(self.class_indices.keys())
        
        for _ in range(self.num_batches):
            batch = []
            
            # Which classes get the remainder slots (random each batch)
            if self.remainder > 0:
                extra_classes = np.random.choice(
                    class_keys,
                    size=self.remainder,
                    replace=False,
                ).tolist()
            else:
                extra_classes = []
            
            for c in class_keys:
                n = self.samples_per_class
                if c in extra_classes:
                    n += 1
                    extra_classes.remove(c)  # each class gets at most 1 extra
                
                indices = self.class_indices[c]
                if len(indices) >= n:
                    sampled = np.random.choice(indices, size=n, replace=False)
                elif self.replacement:
                    sampled = np.random.choice(indices, size=n, replace=True)
                else:
                    sampled = indices.copy()
                
                batch.extend(sampled.tolist())
            
            # Shuffle within batch so classes are interleaved
            np.random.shuffle(batch)
            yield batch
    
    def __len__(self) -> int:
        """Number of batches per epoch."""
        return self.num_batches
