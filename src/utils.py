"""Seeds, devices, run logs and plots."""

import json
import logging
import random
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

log = logging.getLogger("cifar")
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
RUNS_DIR = Path("results/runs")
FIG_DIR = Path("results/figures")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return CUDA, then MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    """Wait for queued GPU work so timers read true elapsed time."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def git_commit() -> str:
    """Return the short commit hash, or 'none' before the first commit."""
    out = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    )
    return out.stdout.strip() or "none"


def save_run(run: dict, runs_dir: Path = RUNS_DIR) -> Path:
    """Write one run log as JSON through a temp file."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{run['run_id']}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(run, indent=1))
    tmp.replace(path)
    return path


def load_runs(phase: str, runs_dir: Path = RUNS_DIR) -> list[dict]:
    """Load every run log of one phase, sorted by run id."""
    runs = [json.loads(p.read_text()) for p in sorted(runs_dir.glob("*.json"))]
    return [r for r in runs if r.get("phase") == phase]


def plot_history(runs: list[dict], labels: list[str], path: Path | None = None):
    """Plot val loss and val accuracy of several runs against epoch and against time."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for run, label in zip(runs, labels, strict=True):
        h = run["history"]
        epochs = range(1, len(h["val_acc"]) + 1)
        elapsed = np.cumsum(h["epoch_time_s"])
        for x, col in ((epochs, 0), (elapsed, 1)):
            axes[0, col].plot(x, h["val_loss"], label=label)
            axes[1, col].plot(x, h["val_acc"], label=label)
    for col, name in enumerate(("epoch", "training time (s)")):
        axes[1, col].set_xlabel(name)
    axes[0, 0].set_ylabel("val loss")
    axes[1, 0].set_ylabel("val accuracy")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120)
    return fig


def plot_confusion(cm, title: str, path: Path | None = None):
    """Plot a confusion matrix with class names."""
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.imshow(cm, cmap="Blues")
    ticks = range(len(CLASSES))
    ax.set_xticks(ticks, CLASSES, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(ticks, CLASSES, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    for i in ticks:
        for j in ticks:
            ax.text(j, i, int(cm[i][j]), ha="center", va="center", fontsize=6)
    fig.tight_layout()
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120)
    return fig
