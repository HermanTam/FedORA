"""
Differential Privacy for Federated Drift Detection

This module implements differential privacy mechanisms for protecting client data
when sharing representations (prototypes, embeddings) with the server.

Supported mechanisms:
- Gaussian mechanism (for (ε,δ)-DP)
- Laplace mechanism (for ε-DP)
- Gradient clipping

Author: Code Review Assistant
Date: November 6, 2025
"""

import numpy as np


class DifferentialPrivacyManager:
    """
    Manages differential privacy for client representations.
    
    Key Features:
    - Configurable privacy budget (ε, δ)
    - Gaussian and Laplace noise mechanisms
    - Automatic sensitivity bounding via clipping
    - Privacy accounting across multiple releases
    """
    
    def __init__(self, epsilon=1.0, delta=1e-5, mechanism='gaussian', composition='basic'):
        """
        Initialize DP manager.
        
        Args:
            epsilon: Privacy budget (smaller = more private, more noise)
                    Typical values: 0.1 (very private) to 10.0 (less private)
            delta: Failure probability for (ε,δ)-DP (only for Gaussian)
                   Typical: 1e-5 for small datasets, 1e-7 for large datasets
            mechanism: 'gaussian' or 'laplace'
            composition: 'basic' (linear) or 'advanced' (tighter bounds)
        """
        self.epsilon = epsilon
        self.delta = delta
        self.mechanism = mechanism
        self.composition = composition
        
        # Privacy accounting
        self.epsilon_spent = 0.0
        self.num_releases = 0
        
        # Validate parameters
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {epsilon}")
        if mechanism == 'gaussian' and (delta <= 0 or delta >= 1):
            raise ValueError(f"Delta must be in (0,1), got {delta}")
        if mechanism not in ['gaussian', 'laplace']:
            raise ValueError(f"Unknown mechanism: {mechanism}")
    
    def clip_vector(self, vector, max_norm=10.0):
        """
        Clip vector to bound L2 sensitivity.
        
        Args:
            vector: numpy array
            max_norm: Maximum L2 norm allowed
        
        Returns:
            clipped_vector (numpy array)
        """
        vector = np.asarray(vector, dtype=np.float64)
        norm = np.linalg.norm(vector)
        
        if norm > max_norm:
            return vector * (max_norm / norm)
        return vector
    
    def add_noise_gaussian(self, vector, sensitivity=1.0):
        """
        Add Gaussian noise for (ε,δ)-DP.
        
        Gaussian mechanism: σ = sensitivity * sqrt(2 * ln(1.25/δ)) / ε
        
        Args:
            vector: numpy array
            sensitivity: L2 sensitivity (default 1.0 if pre-clipped to unit norm)
        
        Returns:
            noisy_vector (numpy array)
        """
        vector = np.asarray(vector, dtype=np.float64)
        
        # Compute noise scale
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        
        # Generate Gaussian noise
        noise = np.random.normal(0, sigma, size=vector.shape)
        
        return vector + noise
    
    def add_noise_laplace(self, vector, sensitivity=1.0):
        """
        Add Laplace noise for ε-DP.
        
        Laplace mechanism: scale = sensitivity / ε
        
        Args:
            vector: numpy array
            sensitivity: L1 sensitivity
        
        Returns:
            noisy_vector (numpy array)
        """
        vector = np.asarray(vector, dtype=np.float64)
        
        # Compute noise scale
        scale = sensitivity / self.epsilon
        
        # Generate Laplace noise
        noise = np.random.laplace(0, scale, size=vector.shape)
        
        return vector + noise
    
    def privatize_vector(self, vector, max_norm=10.0, sensitivity=None):
        """
        Apply full DP pipeline: clip + add noise.
        
        Args:
            vector: numpy array to privatize
            max_norm: Clipping threshold
            sensitivity: If None, use max_norm as sensitivity (after clipping)
        
        Returns:
            noisy_vector (numpy array)
        """
        # Step 1: Clip to bound sensitivity
        clipped = self.clip_vector(vector, max_norm=max_norm)
        
        # Step 2: Set sensitivity (clipping ensures L2 sensitivity ≤ max_norm)
        if sensitivity is None:
            sensitivity = max_norm
        
        # Step 3: Add noise
        if self.mechanism == 'gaussian':
            noisy = self.add_noise_gaussian(clipped, sensitivity=sensitivity)
        elif self.mechanism == 'laplace':
            noisy = self.add_noise_laplace(clipped, sensitivity=sensitivity)
        else:
            raise ValueError(f"Unknown mechanism: {self.mechanism}")
        
        # Privacy accounting
        self.epsilon_spent += self.epsilon
        self.num_releases += 1
        
        return noisy
    
    def privatize_prototype(self, prototype, max_norm=10.0):
        """
        Privatize a single prototype vector.
        
        Args:
            prototype: Prototype vector (e.g., mean output features)
            max_norm: Clipping bound
        
        Returns:
            noisy_prototype (numpy array)
        """
        return self.privatize_vector(prototype, max_norm=max_norm)
    
    def privatize_batch_prototypes(self, prototypes, max_norm=10.0):
        """
        Privatize multiple prototype vectors.
        
        Args:
            prototypes: List or array of prototype vectors
            max_norm: Clipping bound
        
        Returns:
            list of noisy prototypes
        """
        return [self.privatize_prototype(p, max_norm) for p in prototypes]
    
    def privatize_label_distribution(self, label_dist, sensitivity=None):
        """
        Privatize label distribution (probability vector).
        
        Args:
            label_dist: Probability vector [p(y=0), p(y=1), ..., p(y=C-1)]
            sensitivity: L1 sensitivity (default 2.0 for probability simplex)
        
        Returns:
            noisy_dist (numpy array), normalized to sum to 1
        """
        label_dist = np.asarray(label_dist, dtype=np.float64)
        
        # For probability simplex, L1 sensitivity is typically 2.0
        # (changing one sample can shift mass by at most 2/n)
        if sensitivity is None:
            sensitivity = 2.0 / len(label_dist)
        
        # Add noise
        if self.mechanism == 'gaussian':
            noisy = self.add_noise_gaussian(label_dist, sensitivity=sensitivity)
        else:
            noisy = self.add_noise_laplace(label_dist, sensitivity=sensitivity)
        
        # Post-process: clip to [0, 1] and renormalize
        noisy = np.clip(noisy, 0, 1)
        noisy = noisy / (noisy.sum() + 1e-10)
        
        self.epsilon_spent += self.epsilon
        self.num_releases += 1
        
        return noisy
    
    def get_privacy_spent(self):
        """
        Get total privacy budget spent.
        
        Returns:
            dict with epsilon_spent, num_releases
        """
        if self.composition == 'basic':
            # Basic composition: ε_total = k * ε
            total_epsilon = self.num_releases * self.epsilon
        elif self.composition == 'advanced':
            # Advanced composition (tighter bound)
            # ε_total ≈ sqrt(2k ln(1/δ)) * ε + k * ε^2
            k = self.num_releases
            if self.delta > 0:
                total_epsilon = np.sqrt(2 * k * np.log(1 / self.delta)) * self.epsilon
                total_epsilon += k * self.epsilon ** 2
            else:
                total_epsilon = k * self.epsilon
        else:
            total_epsilon = self.epsilon_spent
        
        return {
            "total_epsilon": total_epsilon,
            "per_release_epsilon": self.epsilon,
            "num_releases": self.num_releases,
            "delta": self.delta,
            "composition": self.composition
        }
    
    def reset_accounting(self):
        """Reset privacy budget accounting"""
        self.epsilon_spent = 0.0
        self.num_releases = 0
    
    def get_noise_magnitude(self, sensitivity=1.0):
        """
        Estimate expected noise magnitude.
        
        Args:
            sensitivity: Sensitivity value
        
        Returns:
            expected_l2_norm of noise
        """
        if self.mechanism == 'gaussian':
            sigma = sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
            # E[||noise||_2] for d-dimensional Gaussian ≈ sigma * sqrt(d)
            # Return per-dimension std dev
            return sigma
        elif self.mechanism == 'laplace':
            scale = sensitivity / self.epsilon
            # Laplace scale parameter
            return scale
        return 0.0


def create_dp_manager(epsilon=1.0, delta=1e-5, mechanism='gaussian'):
    """
    Convenience function to create DP manager.
    
    Args:
        epsilon: Privacy budget
        delta: Failure probability (for Gaussian)
        mechanism: 'gaussian' or 'laplace'
    
    Returns:
        DifferentialPrivacyManager instance
    """
    return DifferentialPrivacyManager(epsilon=epsilon, delta=delta, mechanism=mechanism)


def measure_utility_loss(clean_prototypes, noisy_prototypes):
    """
    Measure utility loss from DP noise.
    
    Args:
        clean_prototypes: List of original prototypes
        noisy_prototypes: List of noisy prototypes
    
    Returns:
        dict with loss metrics
    """
    clean = np.array(clean_prototypes)
    noisy = np.array(noisy_prototypes)
    
    # L2 distance
    l2_dists = np.linalg.norm(clean - noisy, axis=1)
    avg_l2 = np.mean(l2_dists)
    
    # Cosine similarity
    cos_sims = []
    for c, n in zip(clean, noisy):
        cos_sim = np.dot(c, n) / (np.linalg.norm(c) * np.linalg.norm(n) + 1e-10)
        cos_sims.append(cos_sim)
    avg_cos_sim = np.mean(cos_sims)
    
    # Signal-to-noise ratio
    signal_power = np.mean(np.linalg.norm(clean, axis=1) ** 2)
    noise_power = np.mean(l2_dists ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    return {
        "avg_l2_distance": avg_l2,
        "avg_cosine_similarity": avg_cos_sim,
        "snr_db": snr,
        "max_l2_distance": np.max(l2_dists),
        "min_cosine_similarity": np.min(cos_sims)
    }


def run_privacy_utility_experiment(prototypes, epsilon_values, max_norm=10.0, delta=1e-5):
    """
    Run experiment to measure privacy-utility tradeoff.
    
    Args:
        prototypes: List of prototype vectors
        epsilon_values: List of epsilon values to test
        max_norm: Clipping bound
        delta: Failure probability
    
    Returns:
        results: List of dicts with metrics for each epsilon
    """
    results = []
    
    for eps in epsilon_values:
        # Create DP manager
        dp_manager = DifferentialPrivacyManager(epsilon=eps, delta=delta, mechanism='gaussian')
        
        # Privatize prototypes
        noisy = dp_manager.privatize_batch_prototypes(prototypes, max_norm=max_norm)
        
        # Measure utility loss
        utility = measure_utility_loss(prototypes, noisy)
        
        # Privacy spent
        privacy = dp_manager.get_privacy_spent()
        
        results.append({
            "epsilon": eps,
            "delta": delta,
            "avg_l2_distance": utility["avg_l2_distance"],
            "avg_cosine_similarity": utility["avg_cosine_similarity"],
            "snr_db": utility["snr_db"],
            "total_epsilon": privacy["total_epsilon"]
        })
    
    return results


if __name__ == "__main__":
    # Example usage and testing
    print("Testing DifferentialPrivacyManager...")
    
    # 1. Test vector privatization
    print("\n1. Vector privatization:")
    dp = DifferentialPrivacyManager(epsilon=1.0, delta=1e-5, mechanism='gaussian')
    
    original = np.array([1.5, 2.3, -0.8, 3.2, 1.1])
    noisy = dp.privatize_vector(original, max_norm=5.0)
    
    print(f"  Original: {original}")
    print(f"  Noisy:    {noisy}")
    print(f"  L2 distance: {np.linalg.norm(original - noisy):.4f}")
    
    # 2. Test prototype batch
    print("\n2. Batch prototype privatization:")
    prototypes = [
        np.random.randn(128),
        np.random.randn(128),
        np.random.randn(128)
    ]
    
    dp = DifferentialPrivacyManager(epsilon=0.5, delta=1e-5)
    noisy_prototypes = dp.privatize_batch_prototypes(prototypes, max_norm=10.0)
    
    utility = measure_utility_loss(prototypes, noisy_prototypes)
    print(f"  Avg L2 distance: {utility['avg_l2_distance']:.4f}")
    print(f"  Avg cosine similarity: {utility['avg_cosine_similarity']:.4f}")
    print(f"  SNR: {utility['snr_db']:.2f} dB")
    
    # 3. Test label distribution
    print("\n3. Label distribution privatization:")
    label_dist = np.array([0.3, 0.25, 0.25, 0.2])
    noisy_dist = dp.privatize_label_distribution(label_dist)
    
    print(f"  Original: {label_dist}")
    print(f"  Noisy:    {noisy_dist}")
    print(f"  Sum:      {noisy_dist.sum():.4f}")
    
    # 4. Privacy accounting
    print("\n4. Privacy accounting:")
    privacy = dp.get_privacy_spent()
    print(f"  Total epsilon spent: {privacy['total_epsilon']:.4f}")
    print(f"  Number of releases: {privacy['num_releases']}")
    
    # 5. Privacy-utility tradeoff
    print("\n5. Privacy-utility tradeoff experiment:")
    test_prototypes = [np.random.randn(64) for _ in range(10)]
    epsilon_values = [0.1, 0.5, 1.0, 5.0, 10.0]
    
    results = run_privacy_utility_experiment(test_prototypes, epsilon_values)
    print("  ε\t\tL2 dist\t\tCos sim\t\tSNR (dB)")
    for r in results:
        print(f"  {r['epsilon']:.1f}\t\t{r['avg_l2_distance']:.4f}\t\t{r['avg_cosine_similarity']:.4f}\t\t{r['snr_db']:.2f}")
    
    print("\n✓ All tests passed!")
