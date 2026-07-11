"""Training metrics: local JSONL always, W&B stream when enabled.

The JSONL file is the source of truth (metrics must survive session death and
SIGKILL, and the resume test replays it); W&B is an additive stream. wandb is
imported only when enabled so the package works without the `track` extra,
degrading with a warning if enabled but unavailable.
"""

import json
import warnings
from pathlib import Path


class MetricsLogger:
    def __init__(self, out_dir: Path, wandb_config: dict | None = None):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._file = (out_dir / "metrics.jsonl").open("a", encoding="utf-8")
        self._run = None
        if wandb_config and wandb_config.get("enabled"):
            try:
                import wandb
            except ImportError:
                warnings.warn(
                    "wandb enabled in config but not importable; "
                    "logging to metrics.jsonl only",
                    stacklevel=2,
                )
            else:
                self._run = wandb.init(
                    project=wandb_config.get("project", "tinystories-v2"),
                    name=wandb_config.get("run_name"),
                    resume="allow",
                )

    def log(self, metrics: dict, step: int) -> None:
        self._file.write(json.dumps({"step": step, **metrics}) + "\n")
        self._file.flush()
        if self._run is not None:
            self._run.log(metrics, step=step)

    def finish(self) -> None:
        self._file.close()
        if self._run is not None:
            self._run.finish()
