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
    
    def __init__(self, max_size=500, sample_mode='reservoir'):
        """
        Initialize experience replay buffer.
        
        Parameters
        ----------
        max_size : int
            Buffer capacity (default: 500 samples)
        sample_mode : str
            'reservoir' - Reservoir sampling (Algorithm R)
            'uniform' - Uniform random replacement
        """
        self.max_size = max_size
        self.sample_mode = sample_mode
        self.buffer = []  # List of (data, label) tuples
        self.n_seen = 0   # Total samples processed
        
        if sample_mode not in ['reservoir', 'uniform']:
            raise ValueError(f"Invalid sample_mode: {sample_mode}. Choose 'reservoir' or 'uniform'")
    
    def add_samples(self, dataset, time_slot=None):
        """
        Add samples from new dataset using reservoir sampling.
        
        Reservoir Sampling (Algorithm R):
        For each new sample s with index i (global):
          - If buffer not full: append s
          - Else: with probability k/i, replace random buffer item
        
        This ensures uniform probability for ALL historical samples.
        
        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            New data to add (from current time slot)
        time_slot : int, optional
            Current time slot (for logging/debugging)
        """
        # Extract samples from dataset
        new_samples = []
        for idx in range(len(dataset)):
            data, label = dataset[idx]
            new_samples.append((data, label))
        
        if self.sample_mode == 'reservoir':
            # Reservoir sampling (Algorithm R)
            for sample in new_samples:
                self.n_seen += 1
                
                if len(self.buffer) < self.max_size:
                    # Buffer not full - always add
                    self.buffer.append(sample)
                else:
                    # Replace with probability k/n
                    replace_prob = self.max_size / self.n_seen
                    if random.random() < replace_prob:
                        # Random replacement
                        replace_idx = random.randint(0, self.max_size - 1)
                        self.buffer[replace_idx] = sample
        
        elif self.sample_mode == 'uniform':
            # Uniform random replacement (simpler baseline)
            for sample in new_samples:
                if len(self.buffer) < self.max_size:
                    self.buffer.append(sample)
                else:
                    # Always replace random item
                    replace_idx = random.randint(0, self.max_size - 1)
                    self.buffer[replace_idx] = sample
    
    def get_all(self):
        """
        Return all buffered samples as a PyTorch dataset.
        
        Returns
        -------
        BufferedDataset
            Custom dataset containing all buffer samples
        """
        if len(self.buffer) == 0:
            return None
        return BufferedDataset(self.buffer)
    
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
    
    def size(self):
        """Return current buffer size."""
        return len(self.buffer)
    
    def __len__(self):
        """Return current buffer size."""
        return len(self.buffer)
    
    def __repr__(self):
        """String representation for debugging."""
        return (f"ExperienceReplayBuffer(max_size={self.max_size}, "
                f"current_size={len(self.buffer)}, "
                f"n_seen={self.n_seen}, "
                f"mode={self.sample_mode})")


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
