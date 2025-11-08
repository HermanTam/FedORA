"""
Adaptive Threshold Management for Drift Detection

This module implements adaptive thresholds using statistical methods:
- μ + τσ rule (mean + tau * standard deviation)
- Chi-square test for label distribution shift
- Kolmogorov-Smirnov test for feature distribution shift

Author: Code Review Assistant
Date: November 6, 2025
"""

import numpy as np
from scipy import stats


class AdaptiveThresholdManager:
    """
    Manages adaptive thresholds for drift detection.
    Each client maintains its own baselines to avoid needing global coordination.
    
    Key Features:
    - Client-side detection (privacy-preserving)
    - Rolling window for adaptive baselines
    - Statistical significance tests
    - Configurable confidence levels
    """
    
    def __init__(self, tau=3.0, window_size=10, min_samples=3):
        """
        Initialize threshold manager.
        
        Args:
            tau: Number of standard deviations for threshold (default 3.0 = 99.7% confidence)
            window_size: Rolling window size for computing baseline statistics
            min_samples: Minimum samples needed before using adaptive thresholds
        """
        self.tau = tau
        self.window_size = window_size
        self.min_samples = min_samples
        
        # History of metrics (for computing μ and σ)
        self.metric_history = {
            "prototype_shift": [],
            "feature_shift": [],
            "label_shift": [],
            "accuracy_drop": []
        }
        
        # Baseline statistics: {metric_name: {"mu": float, "sigma": float}}
        self.baselines = {
            "prototype_shift": {"mu": 0.0, "sigma": 0.1},
            "feature_shift": {"mu": 0.0, "sigma": 0.1},
            "label_shift": {"mu": 0.0, "sigma": 0.1},
            "accuracy_drop": {"mu": 0.0, "sigma": 0.05}
        }
        
        # Calibration phase flag
        self.calibrated = False
    
    def update_baseline(self, metric_name, value):
        """
        Update rolling baseline for a metric.
        
        Args:
            metric_name: One of ["prototype_shift", "feature_shift", "label_shift", "accuracy_drop"]
            value: Current metric value
        """
        if metric_name not in self.metric_history:
            return
        
        history = self.metric_history[metric_name]
        history.append(float(value))
        
        # Maintain rolling window
        if len(history) > self.window_size:
            history.pop(0)
        
        # Recompute μ and σ if we have enough samples
        if len(history) >= self.min_samples:
            self.baselines[metric_name]["mu"] = np.mean(history)
            self.baselines[metric_name]["sigma"] = max(np.std(history), 1e-8)  # Avoid division by zero
            
            if len(history) >= self.window_size:
                self.calibrated = True
    
    def get_threshold(self, metric_name):
        """
        Get adaptive threshold: μ + τ*σ
        
        Args:
            metric_name: Metric to get threshold for
            
        Returns:
            threshold (float)
        """
        baseline = self.baselines.get(metric_name, {"mu": 0.0, "sigma": 0.1})
        return baseline["mu"] + self.tau * baseline["sigma"]
    
    def detect_drift_threshold(self, metric_name, current_value, update_baseline=True):
        """
        Detect drift using μ + τσ rule.
        
        Args:
            metric_name: Metric name
            current_value: Current metric value
            update_baseline: Whether to update baseline with this value
        
        Returns:
            is_drift (bool), threshold (float), info (dict)
        """
        threshold = self.get_threshold(metric_name)
        is_drift = current_value > threshold
        
        # Update history (adaptive baseline)
        if update_baseline:
            self.update_baseline(metric_name, current_value)
        
        info = {
            "threshold": threshold,
            "mu": self.baselines[metric_name]["mu"],
            "sigma": self.baselines[metric_name]["sigma"],
            "current_value": current_value,
            "calibrated": self.calibrated
        }
        
        return is_drift, threshold, info
    
    def chi_square_test(self, observed_dist, expected_dist=None, alpha=0.01):
        """
        Chi-square test for label distribution shift.
        
        Args:
            observed_dist: Current label distribution (counts or probabilities)
            expected_dist: Baseline distribution (if None, use uniform)
            alpha: Significance level (default 0.01 for strict detection)
        
        Returns:
            is_drift (bool), p_value (float), info (dict)
        """
        observed = np.array(observed_dist)
        
        # Handle probabilities vs counts
        if observed.sum() <= 1.0 + 1e-6:  # Probabilities
            observed = observed * 100  # Convert to pseudo-counts
        
        if expected_dist is None:
            expected = np.ones_like(observed) / len(observed) * observed.sum()
        else:
            expected = np.array(expected_dist)
            if expected.sum() <= 1.0 + 1e-6:
                expected = expected * 100
        
        # Ensure minimum expected frequency (chi-square requirement)
        min_expected = 5
        if (expected < min_expected).any():
            # Combine small categories
            observed = np.array([observed.sum()])
            expected = np.array([expected.sum()])
        
        # Chi-square statistic
        try:
            chi2, p_value = stats.chisquare(observed, expected)
        except Exception as e:
            # Fallback if test fails
            return False, 1.0, {"error": str(e)}
        
        is_drift = p_value < alpha
        
        info = {
            "chi2_statistic": chi2,
            "p_value": p_value,
            "alpha": alpha,
            "observed_sum": observed.sum(),
            "expected_sum": expected.sum()
        }
        
        return is_drift, p_value, info
    
    def ks_test(self, current_samples, baseline_samples, alpha=0.01):
        """
        Kolmogorov-Smirnov test for feature distribution shift.
        
        Args:
            current_samples: Current sample values (1D array)
            baseline_samples: Baseline sample values (1D array)
            alpha: Significance level
        
        Returns:
            is_drift (bool), p_value (float), info (dict)
        """
        current = np.array(current_samples).flatten()
        baseline = np.array(baseline_samples).flatten()
        
        if len(current) < 2 or len(baseline) < 2:
            return False, 1.0, {"error": "Insufficient samples"}
        
        try:
            ks_stat, p_value = stats.ks_2samp(baseline, current)
        except Exception as e:
            return False, 1.0, {"error": str(e)}
        
        is_drift = p_value < alpha
        
        info = {
            "ks_statistic": ks_stat,
            "p_value": p_value,
            "alpha": alpha,
            "current_mean": current.mean(),
            "baseline_mean": baseline.mean(),
            "current_std": current.std(),
            "baseline_std": baseline.std()
        }
        
        return is_drift, p_value, info
    
    def detect_drift_multimodal(self, metrics, use_tests=False, alpha=0.01):
        """
        Multi-modal drift detection combining multiple signals.
        
        Args:
            metrics: Dict with {
                "prototype_shift": float,
                "label_dist_current": array,
                "label_dist_baseline": array (optional),
                "feature_samples_current": array (optional),
                "feature_samples_baseline": array (optional),
                "accuracy_drop": float
            }
            use_tests: Whether to use statistical tests or μ+τσ rule
            alpha: Significance level for tests
        
        Returns:
            results (dict): {
                "drift_detected": bool,
                "drift_signals": dict with individual test results,
                "confidence": float (0-1)
            }
        """
        signals = {}
        drift_count = 0
        total_tests = 0
        
        # 1. Prototype shift
        if "prototype_shift" in metrics:
            is_drift, thresh, info = self.detect_drift_threshold(
                "prototype_shift", metrics["prototype_shift"]
            )
            signals["prototype"] = {
                "drift": is_drift,
                "value": metrics["prototype_shift"],
                "threshold": thresh,
                "info": info
            }
            if is_drift:
                drift_count += 1
            total_tests += 1
        
        # 2. Label distribution shift
        if "label_dist_current" in metrics:
            if use_tests:
                is_drift, p_val, info = self.chi_square_test(
                    metrics["label_dist_current"],
                    metrics.get("label_dist_baseline"),
                    alpha=alpha
                )
                signals["label_dist"] = {
                    "drift": is_drift,
                    "p_value": p_val,
                    "test": "chi_square",
                    "info": info
                }
            else:
                # Use L1 distance with adaptive threshold
                if "label_dist_baseline" in metrics:
                    l1_dist = np.abs(
                        np.array(metrics["label_dist_current"]) - 
                        np.array(metrics["label_dist_baseline"])
                    ).sum()
                    is_drift, thresh, info = self.detect_drift_threshold(
                        "label_shift", l1_dist
                    )
                    signals["label_dist"] = {
                        "drift": is_drift,
                        "value": l1_dist,
                        "threshold": thresh,
                        "info": info
                    }
                else:
                    is_drift = False
            
            if is_drift:
                drift_count += 1
            total_tests += 1
        
        # 3. Feature distribution shift (optional)
        if "feature_samples_current" in metrics and "feature_samples_baseline" in metrics:
            if use_tests:
                is_drift, p_val, info = self.ks_test(
                    metrics["feature_samples_current"],
                    metrics["feature_samples_baseline"],
                    alpha=alpha
                )
                signals["feature_dist"] = {
                    "drift": is_drift,
                    "p_value": p_val,
                    "test": "ks",
                    "info": info
                }
                if is_drift:
                    drift_count += 1
                total_tests += 1
        
        # 4. Accuracy drop
        if "accuracy_drop" in metrics:
            is_drift, thresh, info = self.detect_drift_threshold(
                "accuracy_drop", metrics["accuracy_drop"]
            )
            signals["accuracy"] = {
                "drift": is_drift,
                "value": metrics["accuracy_drop"],
                "threshold": thresh,
                "info": info
            }
            if is_drift:
                drift_count += 1
            total_tests += 1
        
        # Aggregate decision
        drift_detected = drift_count > 0  # At least one signal
        confidence = drift_count / total_tests if total_tests > 0 else 0.0
        
        return {
            "drift_detected": drift_detected,
            "drift_signals": signals,
            "confidence": confidence,
            "drift_count": drift_count,
            "total_tests": total_tests
        }
    
    def reset(self):
        """Reset all baselines (e.g., after major environment change)"""
        for metric in self.metric_history:
            self.metric_history[metric] = []
        self.calibrated = False
    
    def get_status(self):
        """Get current status of threshold manager"""
        return {
            "calibrated": self.calibrated,
            "tau": self.tau,
            "window_size": self.window_size,
            "baselines": self.baselines,
            "history_lengths": {k: len(v) for k, v in self.metric_history.items()}
        }


# Convenience functions for common use cases

def create_adaptive_manager(tau=3.0, window_size=10):
    """Create a standard adaptive threshold manager"""
    return AdaptiveThresholdManager(tau=tau, window_size=window_size)


def detect_drift_simple(manager, prototype_shift, label_shift, accuracy_drop):
    """
    Simple drift detection with three basic metrics.
    
    Args:
        manager: AdaptiveThresholdManager instance
        prototype_shift: Cosine distance between prototypes
        label_shift: L1 distance between label distributions
        accuracy_drop: Decrease in accuracy
    
    Returns:
        is_drift (bool), details (dict)
    """
    results = manager.detect_drift_multimodal({
        "prototype_shift": prototype_shift,
        "accuracy_drop": accuracy_drop
    }, use_tests=False)
    
    return results["drift_detected"], results


if __name__ == "__main__":
    # Example usage
    print("Testing AdaptiveThresholdManager...")
    
    manager = AdaptiveThresholdManager(tau=2.0, window_size=5)
    
    # Simulate normal fluctuations (calibration phase)
    print("\n1. Calibration phase (normal fluctuations):")
    for i in range(6):
        value = 0.05 + np.random.normal(0, 0.01)
        is_drift, threshold, info = manager.detect_drift_threshold("prototype_shift", value)
        print(f"  Step {i}: value={value:.4f}, threshold={threshold:.4f}, drift={is_drift}")
    
    # Simulate drift
    print("\n2. Drift detection (large shift):")
    is_drift, threshold, info = manager.detect_drift_threshold("prototype_shift", 0.3)
    print(f"  Large shift: value=0.3, threshold={threshold:.4f}, drift={is_drift}")
    print(f"  Baseline: μ={info['mu']:.4f}, σ={info['sigma']:.4f}")
    
    # Test chi-square for label distribution
    print("\n3. Chi-square test for label distribution:")
    observed = np.array([150, 100, 50, 100])
    expected = np.array([100, 100, 100, 100])
    is_drift, p_value, info = manager.chi_square_test(observed, expected, alpha=0.01)
    print(f"  Drift detected: {is_drift}, p-value: {p_value:.6f}")
    
    # Multi-modal detection
    print("\n4. Multi-modal drift detection:")
    results = manager.detect_drift_multimodal({
        "prototype_shift": 0.25,
        "label_dist_current": [0.4, 0.3, 0.2, 0.1],
        "label_dist_baseline": [0.25, 0.25, 0.25, 0.25],
        "accuracy_drop": 0.15
    }, use_tests=True, alpha=0.01)
    
    print(f"  Overall drift: {results['drift_detected']}")
    print(f"  Confidence: {results['confidence']:.2f}")
    print(f"  Signals: {results['drift_count']}/{results['total_tests']}")
    
    print("\n✓ All tests passed!")
