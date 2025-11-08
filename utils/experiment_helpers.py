from collections import defaultdict
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.metrics_utils import sample_stats


def parse_seed_argument(arg_value, default_seed):
    if not arg_value:
        return [default_seed]
    seeds = []
    for chunk in arg_value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        seeds.append(int(chunk))
    return seeds or [default_seed]


def evaluate_iterator_accuracy(learner_ensemble, iterator):
    if iterator is None:
        return None
    _, acc = learner_ensemble.evaluate_iterator(iterator)
    return acc


def build_bar_plot(p_stats: Dict[str, float], g_stats: Dict[str, float]):
    fig, ax = plt.subplots(figsize=(6, 4))
    categories = ['P-Score', 'G-Score']
    means = [p_stats.get('mean', 0.0), g_stats.get('mean', 0.0)]
    stds = [p_stats.get('std', 0.0), g_stats.get('std', 0.0)]
    ax.bar(categories, means, yerr=stds, capsize=5, color=['#1f77b4', '#ff7f0e'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Accuracy')
    ax.set_title('Objective Scores')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    return fig


def aggregate_summary(seed_metrics: List[Dict]) -> Dict[str, Dict[str, float]]:
    summary_lists = defaultdict(list)
    for metrics in seed_metrics:
        stats = metrics.get("stats", {})
        if "p_scores" in stats:
            summary_lists["p_scores"].append(stats["p_scores"]["mean"])
        if "g_scores" in stats and stats["g_scores"]["mean"] is not None:
            summary_lists["g_scores"].append(stats["g_scores"]["mean"])
        if "forgetting" in stats and stats["forgetting"]["mean"] is not None:
            summary_lists["forgetting"].append(stats["forgetting"]["mean"])
        if "overall_current" in stats and stats["overall_current"]["mean"] is not None:
            summary_lists["overall_current"].append(stats["overall_current"]["mean"])

    return {key: sample_stats(values) for key, values in summary_lists.items()}

