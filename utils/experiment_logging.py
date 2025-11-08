import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python <3.9 not supported
    ZoneInfo = None  # type: ignore


def _uk_timestamp() -> str:
    tz = ZoneInfo("Europe/London") if ZoneInfo else None
    now = datetime.now(tz=tz)
    return now.strftime("%Y%m%dT%H%M%S")


def _sanitize(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in value)


@dataclass
class ExperimentLogger:
    args: object
    seeds: List[int]
    base_dir: Path = field(init=False)
    seed_dirs: Dict[int, Path] = field(init=False, default_factory=dict)
    summary_path: Path = field(init=False)

    def __post_init__(self):
        timestamp = _uk_timestamp()
        diag = getattr(self.args, "diagnosis_mode", "binary")
        obj = "obj1" if getattr(self.args, "objective_aware", False) else "obj0"
        experiment = _sanitize(str(getattr(self.args, "experiment", "exp")))
        method = _sanitize(str(getattr(self.args, "method", "method")))
        self.base_dir = Path("experiments") / f"{timestamp}__{experiment}__{method}__diag-{diag}__{obj}"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.base_dir / "summary.json"
        self._write_command(self.base_dir)
        self._write_config(self.base_dir, self.args)

    def _write_command(self, directory: Path):
        command_file = directory / "command.txt"
        command_file.write_text(" ".join(sys.argv))

    def _write_config(self, directory: Path, args_obj: object):
        config_file = directory / "config.json"
        config_file.write_text(json.dumps(vars(args_obj), indent=2, default=str))

    def get_seed_dir(self, seed: int) -> Path:
        if seed in self.seed_dirs:
            return self.seed_dirs[seed]
        seed_dir = self.base_dir / f"seed_{seed:05d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = seed_dir / "logs"
        plots_dir = seed_dir / "plots"
        metrics_dir = seed_dir / "metrics"
        for directory in (logs_dir, plots_dir, metrics_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self._write_command(seed_dir)
        # config with seed override
        seed_args = dict(vars(self.args))
        seed_args["seed"] = seed
        seed_dir.joinpath("config.json").write_text(json.dumps(seed_args, indent=2, default=str))
        self.seed_dirs[seed] = seed_dir
        return seed_dir

    def seed_logs_dir(self, seed: int) -> Path:
        return self.get_seed_dir(seed) / "logs"

    def seed_metrics_dir(self, seed: int) -> Path:
        return self.get_seed_dir(seed) / "metrics"

    def seed_plots_dir(self, seed: int) -> Path:
        return self.get_seed_dir(seed) / "plots"

    def write_metrics(self, seed: int, metrics: Dict):
        metrics_path = self.seed_metrics_dir(seed) / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2, default=float))

    def write_plot(self, seed: int, filename: str, figure):
        plots_dir = self.seed_plots_dir(seed)
        figure_path = plots_dir / filename
        figure.savefig(figure_path, bbox_inches="tight")

    def save_summary(self, runs: List[Dict]):
        self.summary_path.write_text(json.dumps(runs, indent=2, default=float))

