"""
Experience Replay Buffer for Continual Learning in Federated Drift Adaptation

This module implements reservoir sampling-based experience replay to mitigate
catastrophic forgetting across multiple time slots.

Key differences from naive rehearsal:
- Naive Rehearsal: Only merges D_{t-1} + D_t (2-slot memory)
- Experience Replay: Maintains buffer from ALL past slots (long-term memory)

Usage:
    buffer = ExperienceReplayBuffer(max_size=500, sample_mode='reservoir')
    buffer.add_samples(new_data)  # Reservoir sampling update
    replay_data = buffer.get_all()  # Retrieve for training
"""

import random
import copy
import numpy as np
from torch.utils.data import Dataset, Subset, ConcatDataset


class ExperienceReplayBuffer:
    """
    Fixed-size buffer with reservoir sampling for continual learning.
    
    Attributes
    ----------
    max_size : int
        Maximum buffer capacity (samples)
    sample_mode : str
        'reservoir' (Algorithm R) or 'uniform' (random replacement)
    buffer : list
        Stored samples [(data, label), ...]
    n_seen : int
        Total samples seen (for reservoir probability)
    
    Methods
    -------
    add_samples(dataset, time_slot)
        Add new data using reservoir sampling
    get_all()
        Return all buffered samples as dataset
    get_sample(n)
        Randomly sample n items from buffer
    clear()
        Empty the buffer
    """
    
    def __init__(self, max_size=500, sample_mode='reservoir', store_val=False):
        """
        Initialize experience replay buffer.
        
        Parameters
        ----------
        max_size : int
            Buffer capacity (default: 500 samples)
        sample_mode : str
            'reservoir' - Reservoir sampling (Algorithm R)
            'uniform' - Uniform random replacement
        store_val : bool
            Whether to also store validation samples (default: False)
            Set to True for fair comparison with naive rehearsal
        """
        self.max_size = max_size
        self.sample_mode = sample_mode
        self.store_val = store_val
        self.buffer = []  # List of (data, label) tuples for training
        self.val_buffer = [] if store_val else None  # Separate validation buffer
        self.n_seen = 0   # Total training samples processed
        self.n_seen_val = 0  # Total validation samples processed (if store_val)
        
        if sample_mode not in ['reservoir', 'uniform']:
            raise ValueError(f"Invalid sample_mode: {sample_mode}. Choose 'reservoir' or 'uniform'")
    
    def _add_to_buffer(self, samples, buffer, n_seen):
        """
        Helper to add samples to a buffer using reservoir sampling.
        
        Parameters
        ----------
        samples : list
            New samples to add
        buffer : list
            Target buffer (self.buffer or self.val_buffer)
        n_seen : int
            Current count of seen samples
        
        Returns
        -------
        int
            Updated n_seen count
        """
        if self.sample_mode == 'reservoir':
            # Reservoir sampling (Algorithm R)
            for sample in samples:
                n_seen += 1
                
                if len(buffer) < self.max_size:
                    # Buffer not full - always add
                    buffer.append(sample)
                else:
                    # Replace with probability k/n
                    replace_prob = self.max_size / n_seen
                    if random.random() < replace_prob:
                        # Random replacement
                        replace_idx = random.randint(0, self.max_size - 1)
                        buffer[replace_idx] = sample
        
        elif self.sample_mode == 'uniform':
            # Uniform random replacement (simpler baseline)
            for sample in samples:
                if len(buffer) < self.max_size:
                    buffer.append(sample)
                else:
                    # Always replace random item
                    replace_idx = random.randint(0, self.max_size - 1)
                    buffer[replace_idx] = sample
        
        return n_seen
    
    def add_samples(self, train_dataset, val_dataset=None, time_slot=None):
        """
        Add samples from new dataset(s) using reservoir sampling.
        
        Reservoir Sampling (Algorithm R):
        For each new sample s with index i (global):
          - If buffer not full: append s
          - Else: with probability k/i, replace random buffer item
        
        This ensures uniform probability for ALL historical samples.
        
        Parameters
        ----------
        train_dataset : torch.utils.data.Dataset
            Training data to add (from current time slot)
        val_dataset : torch.utils.data.Dataset, optional
            Validation data to add (only if store_val=True)
        time_slot : int, optional
            Current time slot (for logging/debugging)
        """
        # Extract training samples
        train_samples = []
        for idx in range(len(train_dataset)):
            data, label = train_dataset[idx]
            train_samples.append((data, label))
        
        # Add to training buffer
        self.n_seen = self._add_to_buffer(train_samples, self.buffer, self.n_seen)
        
        # Add to validation buffer if enabled
        if self.store_val and val_dataset is not None:
            val_samples = []
            for idx in range(len(val_dataset)):
                data, label = val_dataset[idx]
                val_samples.append((data, label))
            
            self.n_seen_val = self._add_to_buffer(val_samples, self.val_buffer, self.n_seen_val)
    
    def get_all(self):
        """
        Return all buffered samples as PyTorch dataset(s).
        
        Returns
        -------
        BufferedDataset or tuple
            If store_val=False: returns training BufferedDataset (or None)
            If store_val=True: returns (train_dataset, val_dataset) tuple
        """
        if not self.store_val:
            # Original behavior: return only training buffer
            if len(self.buffer) == 0:
                return None
            return BufferedDataset(self.buffer)
        else:
            # New behavior: return both buffers
            train_data = BufferedDataset(self.buffer) if len(self.buffer) > 0 else None
            val_data = BufferedDataset(self.val_buffer) if len(self.val_buffer) > 0 else None
            return train_data, val_data
    
    def get_sample(self, n):
        """
        Randomly sample n items from buffer (with replacement).
        
        Parameters
        ----------
        n : int
            Number of samples to draw
        
        Returns
        -------
        BufferedDataset or None
            Dataset with n samples (or None if buffer empty)
        """
        if len(self.buffer) == 0:
            return None
        
        sampled_items = random.choices(self.buffer, k=min(n, len(self.buffer)))
        return BufferedDataset(sampled_items)
    
    def clear(self):
        """Reset buffer (for ablation studies)."""
        self.buffer = []
        self.n_seen = 0
        if self.store_val:
            self.val_buffer = []
            self.n_seen_val = 0
    
    def size(self):
        """Return current buffer size."""
        return len(self.buffer)
    
    def __len__(self):
        """Return current buffer size."""
        return len(self.buffer)
    
    def __repr__(self):
        """String representation for debugging."""
        base = (f"ExperienceReplayBuffer(max_size={self.max_size}, "
                f"current_size={len(self.buffer)}, "
                f"n_seen={self.n_seen}, "
                f"mode={self.sample_mode}")
        if self.store_val:
            base += f", val_size={len(self.val_buffer)}, n_seen_val={self.n_seen_val}"
        return base + ")"


class BufferedDataset(Dataset):
    """
    PyTorch Dataset wrapper for experience replay buffer.
    
    This converts the buffer's list of (data, label) tuples into
    a standard PyTorch Dataset compatible with DataLoader.
    """
    
    def __init__(self, samples):
        """
        Initialize dataset from buffer samples.
        
        Parameters
        ----------
        samples : list
            List of (data, label) tuples
        """
        self.samples = samples
    
    def __len__(self):
        """Return number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get sample by index.
        
        Parameters
        ----------
        idx : int
            Sample index
        
        Returns
        -------
        tuple
            (data, label) for the requested sample
        """
        return self.samples[idx]


def merge_buffer_with_current(buffer, current_dataset):
    """
    Merge experience replay buffer with current time slot data.
    
    This is the key function for continual learning:
    - Training data = Buffer (past slots) + Current (new slot)
    
    Parameters
    ----------
    buffer : ExperienceReplayBuffer
        Buffer containing historical samples
    current_dataset : torch.utils.data.Dataset
        Current time slot dataset
    
    Returns
    -------
    torch.utils.data.ConcatDataset or Dataset
        Merged dataset for training
    """
    buffered_data = buffer.get_all()
    
    if buffered_data is None:
        # Buffer empty (first time slot) - use only current data
        return current_dataset
    
    # Concatenate buffer + current
    merged_dataset = ConcatDataset([buffered_data, current_dataset])
    return merged_dataset


# ============================================================================
# Statistics and Monitoring
# ============================================================================

def compute_buffer_statistics(buffer):
    """
    Compute statistics about buffer contents (for logging/analysis).
    
    Parameters
    ----------
    buffer : ExperienceReplayBuffer
        Buffer to analyze
    
    Returns
    -------
    dict
        Statistics: size, label distribution, etc.
    """
    if len(buffer) == 0:
        return {
            'size': 0,
            'label_distribution': {},
            'n_seen_total': buffer.n_seen
        }
    
    # Extract labels
    labels = [label for _, label in buffer.buffer]
    label_counts = {}
    for label in labels:
        # Handle both int and tensor labels
        label_val = int(label.item()) if hasattr(label, 'item') else int(label)
        label_counts[label_val] = label_counts.get(label_val, 0) + 1
    
    return {
        'size': len(buffer),
        'label_distribution': label_counts,
        'n_seen_total': buffer.n_seen,
        'fill_ratio': len(buffer) / buffer.max_size
    }


def log_buffer_stats(buffer, time_slot, logger=None):
    """
    Log buffer statistics (for TensorBoard or console).
    
    Parameters
    ----------
    buffer : ExperienceReplayBuffer
        Buffer to log
    time_slot : int
        Current time slot
    logger : SummaryWriter, optional
        TensorBoard logger
    """
    stats = compute_buffer_statistics(buffer)
    
    print(f"[T={time_slot}] ER Buffer: {stats['size']}/{buffer.max_size} samples "
          f"({stats['fill_ratio']:.1%} full), {stats['n_seen_total']} seen total")
    
    if logger is not None:
        logger.add_scalar('ER/buffer_size', stats['size'], time_slot)
        logger.add_scalar('ER/fill_ratio', stats['fill_ratio'], time_slot)
        
        # Log label distribution entropy (diversity measure)
        if stats['label_distribution']:
            dist = np.array(list(stats['label_distribution'].values())) / stats['size']
            entropy = -np.sum(dist * np.log(dist + 1e-10))
            logger.add_scalar('ER/label_entropy', entropy, time_slot)
