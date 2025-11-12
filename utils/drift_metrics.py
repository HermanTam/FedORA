"""
Fine-grained drift detection metrics using permutation analysis.

This module provides metrics to distinguish between:
- Label drift: P(Y) changes (label distribution shift)
- Feature drift: P(X|Y) changes (covariate shift)
- Concept drift: P(Y|X) changes (label remapping)
- No drift: all stable

Based on prototype comparison and permutation gain analysis.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque


def label_hist(iterator, C):
    """
    Extract label histogram from iterator.
    
    Args:
        iterator: Data iterator with dataset.labels or dataset.targets
        C: Number of classes
        
    Returns:
        Normalized histogram (probabilities)
    """
    ds = getattr(iterator, "dataset", None)
    if ds is not None and hasattr(ds, "labels"):
        labels = np.asarray(ds.labels)
    elif ds is not None and hasattr(ds, "targets"):
        labels = np.asarray(ds.targets)
    else:
        # Iterate once to collect labels
        ys = []
        for batch in iterator:
            if len(batch) >= 2:
                _, y = batch[0], batch[1]
                ys.append(np.asarray(y))
        labels = np.concatenate(ys) if ys else np.array([])
    
    if labels.size == 0:
        return np.zeros(C, dtype=np.float64)
    
    hist = np.bincount(labels.astype(int), minlength=C).astype(np.float64)
    return hist / max(hist.sum(), 1e-12)


def identity_feature_shift(mu_last, mu_curr):
    """
    Mean L2 distance between class prototypes (identity-aligned).
    
    Args:
        mu_last: Last prototypes (C x D)
        mu_curr: Current prototypes (C x D)
        
    Returns:
        Mean L2 distance for non-empty classes
    """
    # Mask classes where either row norm is zero (empty classes)
    mask_last = (np.linalg.norm(mu_last, axis=1) > 0)
    mask_curr = (np.linalg.norm(mu_curr, axis=1) > 0)
    mask = mask_last & mask_curr
    
    if not np.any(mask):
        return 0.0
    
    diffs = mu_last[mask] - mu_curr[mask]
    return float(np.mean(np.linalg.norm(diffs, axis=1)))


def cosine(a, b, eps=1e-12):
    """Cosine similarity between two vectors."""
    na = np.linalg.norm(a) + eps
    nb = np.linalg.norm(b) + eps
    return float(np.dot(a, b) / (na * nb))


def permutation_gain(mu_last, mu_curr):
    """
    Compute permutation gain to detect label remapping.
    
    If labels were remapped (y → π(y)), then current prototypes under
    the best permutation will align better than identity alignment.
    
    Args:
        mu_last: Last prototypes (C x D)
        mu_curr: Current prototypes (C x D)
        
    Returns:
        sid: Identity alignment score (diagonal of similarity matrix)
        sperm: Best permutation alignment score
        gain: sperm - sid (positive if remapping exists)
    """
    C = mu_last.shape[0]
    S = np.zeros((C, C), dtype=np.float64)
    
    # Build similarity matrix
    for c in range(C):
        for d in range(C):
            S[c, d] = cosine(mu_last[c], mu_curr[d])
    
    sid = float(np.mean(np.diag(S)))
    
    # Hungarian algorithm for optimal permutation
    try:
        r, c_idx = linear_sum_assignment(-S)  # Maximize similarity
        sperm = float(np.mean(S[r, c_idx]))
    except Exception:
        # Greedy fallback
        used = set()
        vals = []
        S_copy = S.copy()
        for c in range(C):
            d = int(np.argmax(S_copy[c]))
            while d in used and len(used) < C:
                S_copy[c, d] = -1e9
                d = int(np.argmax(S_copy[c]))
            used.add(d)
            vals.append(S[c, d])
        sperm = float(np.mean(vals)) if vals else sid
    
    return sid, sperm, float(sperm - sid)


def diagnose_drift_multiclass(
    mu_last, mu_curr,
    last_iterator, curr_iterator,
    C=10,
    tau_label=0.3,
    tau_feat=0.3,
    tau_perm=0.1
):
    """
    Diagnose drift type based on fine-grained metrics.
    
    Args:
        mu_last: Last prototypes (C x D)
        mu_curr: Current prototypes (C x D)
        last_iterator: Last validation iterator
        curr_iterator: Current validation iterator
        C: Number of classes
        tau_label: Threshold for label shift
        tau_feat: Threshold for feature shift
        tau_perm: Threshold for permutation gain
        
    Returns:
        drift_type: 'none', 'label', 'feature', or 'concept'
        metrics: Dict of computed values
    """
    # Compute metrics
    h_last = label_hist(last_iterator, C)
    h_curr = label_hist(curr_iterator, C)
    label_shift = float(np.abs(h_last - h_curr).sum())
    
    feat_shift = identity_feature_shift(mu_last, mu_curr)
    
    sid, sperm, perm_gain_val = permutation_gain(mu_last, mu_curr)
    
    metrics = {
        'label_shift': label_shift,
        'feature_shift': feat_shift,
        'perm_sid': sid,
        'perm_sbest': sperm,
        'perm_gain': perm_gain_val,
    }
    
    # Decision rules
    if label_shift < tau_label and feat_shift < tau_feat and perm_gain_val < tau_perm:
        drift_type = 'none'
    elif label_shift >= tau_label and feat_shift < tau_feat and perm_gain_val < tau_perm:
        drift_type = 'label'
    elif label_shift < tau_label and feat_shift >= tau_feat and perm_gain_val < tau_perm:
        drift_type = 'feature'
    elif label_shift >= tau_label and perm_gain_val >= tau_perm:
        drift_type = 'concept'
    else:
        # Tie-break: prefer feature if perm_gain is low
        drift_type = 'feature' if feat_shift >= tau_feat else 'concept'
    
    return drift_type, metrics


class AdaptiveThresholdManager:
    """
    Manage adaptive thresholds for drift detection using rolling statistics.
    
    Supports multiple adaptation strategies:
    - 'ewma': Exponentially weighted moving average
    - 'median_mad': Robust median + MAD (Median Absolute Deviation)
    - 'quantile': Per-slot quantile-based outlier detection
    """
    
    def __init__(self, method='ewma', window_size=5, k=2.0, warm_up=2):
        """
        Args:
            method: 'ewma', 'median_mad', or 'quantile'
            window_size: Number of slots to keep in history
            k: Multiplier for threshold (k * std or k * MAD)
            warm_up: Number of slots before thresholds become active
        """
        self.method = method
        self.window_size = window_size
        self.k = k
        self.warm_up = warm_up
        
        # Per-client metric histories
        self.label_history = {}  # client_id -> deque of values
        self.feat_history = {}
        self.perm_history = {}
        
        self.slot_count = 0
    
    def update(self, client_id, label_shift, feat_shift, perm_gain):
        """
        Update history with new metrics.
        
        Args:
            client_id: Client identifier
            label_shift, feat_shift, perm_gain: Current slot metrics
        """
        if client_id not in self.label_history:
            self.label_history[client_id] = deque(maxlen=self.window_size)
            self.feat_history[client_id] = deque(maxlen=self.window_size)
            self.perm_history[client_id] = deque(maxlen=self.window_size)
        
        self.label_history[client_id].append(label_shift)
        self.feat_history[client_id].append(feat_shift)
        self.perm_history[client_id].append(perm_gain)
    
    def get_thresholds(self, client_id, default_tau_label=0.3, default_tau_feat=0.3, default_tau_perm=0.1):
        """
        Compute adaptive thresholds for a client.
        
        Args:
            client_id: Client identifier
            default_tau_*: Fixed fallback thresholds
            
        Returns:
            (tau_label, tau_feat, tau_perm)
        """
        # During warm-up, use default thresholds
        if client_id not in self.label_history or len(self.label_history[client_id]) < self.warm_up:
            return default_tau_label, default_tau_feat, default_tau_perm
        
        if self.method == 'ewma':
            return self._ewma_thresholds(client_id, default_tau_label, default_tau_feat, default_tau_perm)
        elif self.method == 'median_mad':
            return self._median_mad_thresholds(client_id, default_tau_label, default_tau_feat, default_tau_perm)
        else:
            return default_tau_label, default_tau_feat, default_tau_perm
    
    def _ewma_thresholds(self, client_id, default_label, default_feat, default_perm):
        """EWMA-based thresholds: mean + k * std"""
        label_vals = np.array(self.label_history[client_id])
        feat_vals = np.array(self.feat_history[client_id])
        perm_vals = np.array(self.perm_history[client_id])
        
        tau_label = max(np.mean(label_vals) + self.k * np.std(label_vals), default_label)
        tau_feat = max(np.mean(feat_vals) + self.k * np.std(feat_vals), default_feat)
        tau_perm = max(np.mean(perm_vals) + self.k * np.std(perm_vals), default_perm)
        
        return tau_label, tau_feat, tau_perm
    
    def _median_mad_thresholds(self, client_id, default_label, default_feat, default_perm):
        """Median + k * MAD (robust to outliers)"""
        label_vals = np.array(self.label_history[client_id])
        feat_vals = np.array(self.feat_history[client_id])
        perm_vals = np.array(self.perm_history[client_id])
        
        def mad_threshold(vals, default_val):
            median = np.median(vals)
            mad = np.median(np.abs(vals - median))
            return max(median + self.k * mad, default_val)
        
        tau_label = mad_threshold(label_vals, default_label)
        tau_feat = mad_threshold(feat_vals, default_feat)
        tau_perm = mad_threshold(perm_vals, default_perm)
        
        return tau_label, tau_feat, tau_perm
    
    def advance_slot(self):
        """Increment slot counter."""
        self.slot_count += 1


def compute_confusion_matrix(predictions, ground_truth, classes=None):
    """
    Compute confusion matrix for drift detection.
    
    Args:
        predictions: Dict[client_id, predicted_drift_type]
        ground_truth: Dict[client_id, true_drift_type]
        classes: List of class names (default: ['none', 'label', 'feature', 'concept'])
        
    Returns:
        confusion_matrix: 2D numpy array
        classes: List of class names
    """
    if classes is None:
        classes = ['none', 'label', 'feature', 'concept']
    
    n_classes = len(classes)
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    for client_id in ground_truth:
        true_type = ground_truth[client_id]
        pred_type = predictions.get(client_id, 'none')
        
        if true_type in class_to_idx and pred_type in class_to_idx:
            matrix[class_to_idx[true_type], class_to_idx[pred_type]] += 1
    
    return matrix, classes


def print_confusion_matrix(matrix, classes, title="Confusion Matrix"):
    """
    Pretty-print confusion matrix.
    
    Args:
        matrix: 2D numpy array
        classes: List of class names
        title: Title for the matrix
    """
    print("\n" + "="*80)
    print(title)
    print("="*80)
    
    # Header
    header_label = "True \\ Pred"
    print(f"{header_label:<12}", end="")
    for cls in classes:
        print(f"{cls:>12}", end="")
    print()
    
    print("-" * (12 + 12 * len(classes)))
    
    # Rows
    for i, true_cls in enumerate(classes):
        print(f"{true_cls:<12}", end="")
        for j in range(len(classes)):
            print(f"{matrix[i, j]:>12}", end="")
        print()
    
    # Metrics
    print("\n" + "-"*80)
    print("Per-Class Metrics:")
    print("-"*80)
    
    for i, cls in enumerate(classes):
        tp = matrix[i, i]
        fp = matrix[:, i].sum() - tp
        fn = matrix[i, :].sum() - tp
        tn = matrix.sum() - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"{cls:>12}: Precision={precision:6.2%}  Recall={recall:6.2%}  F1={f1:6.2%}")
    
    # Overall accuracy
    accuracy = np.trace(matrix) / matrix.sum() if matrix.sum() > 0 else 0.0
    print(f"\n{'Overall':>12}: Accuracy={accuracy:6.2%}")
    print("="*80)
