"""Metrics, confusion matrices, timing, model size and the one-shot test evaluation."""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.data import get_test_loader
from src.models import load_run, read_run
from src.utils import (
    CLASSES,
    FIG_DIR,
    RUNS_DIR,
    get_device,
    git_commit,
    plot_confusion,
    save_run,
    sync,
)

log = logging.getLogger("cifar")


@torch.no_grad()
def predict(model, loader, amp: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Return logits and labels for a loader, with the model in eval mode."""
    model.eval()
    device = next(model.parameters()).device
    logits, ys = [], []
    for x, y in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            logits.append(model(x).float().cpu())
        ys.append(y.cpu())
    return torch.cat(logits), torch.cat(ys)


def classification_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    """Return accuracy, macro F1, per-class precision/recall/F1 and the confusion matrix."""
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=range(10), zero_division=0)
    per_class = {
        c: {"precision": a, "recall": b, "f1": d}
        for c, a, b, d in zip(CLASSES, p, r, f, strict=True)
    }
    return {
        "acc": float((y == pred).mean()),
        "macro_f1": float(f.mean()),
        "per_class": per_class,
        "confusion": confusion_matrix(y, pred, labels=range(10)).tolist(),
    }


def top_confused(cm, k: int = 5) -> list[tuple[str, str, int]]:
    """Return the k most frequent (true, predicted, count) errors."""
    cm = np.array(cm)
    np.fill_diagonal(cm, 0)
    order = np.argsort(cm, axis=None)[::-1][:k]
    return [(CLASSES[i // 10], CLASSES[i % 10], int(cm[i // 10, i % 10])) for i in order]


def make_predictor(model, clf=None):
    """Return a function from an image batch to labels. A classical head replaces the softmax."""
    if clf is None:
        return lambda x: model(x).argmax(1)
    return lambda x: clf.predict(model.embed(x).float().cpu().numpy())


@torch.no_grad()
def time_ms(fn, x, device, warmup: int = 200, iters: int = 500) -> float:
    """Return the mean milliseconds of fn(x) after warm-up, with the device synchronized."""
    for _ in range(warmup):
        fn(x)
    sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn(x)
    sync(device)
    return (time.perf_counter() - start) / iters * 1000


def inference_times(run: dict, device, res: int, warmup: int = 200, iters: int = 500) -> dict:
    """Return batch-1 latency on CPU and device, and batch-128 throughput on the device."""
    out = {}
    for dev in dict.fromkeys([torch.device("cpu"), device]):
        model, clf = load_run(run, dev)
        fn = make_predictor(model, clf)
        x1 = torch.randn(1, 3, res, res, device=dev)
        out[f"latency_b1_{dev.type}_ms"] = time_ms(fn, x1, dev, warmup, iters)
        if dev == device:
            x128 = torch.randn(128, 3, res, res, device=dev)
            out["throughput_b128_img_s"] = 128000 / time_ms(fn, x128, dev, warmup // 4, iters // 5)
    return out


def size_mb(run: dict) -> float:
    """Return the size in MB of the saved weights plus any classical head."""
    files = [run["weights"]] + ([run["clf"]] if run.get("clf") else [])
    return sum(Path(f).stat().st_size for f in files) / 1e6


def evaluate_run(run_id: str, device, time_it: bool, runs_dir: Path = RUNS_DIR) -> dict:
    """Score a finished run on the test set once and log it as a 'final' run."""
    out_id = f"final_{run_id}"
    if (runs_dir / f"{out_id}.json").exists():
        raise FileExistsError(f"{out_id} already saw the test set")
    run = read_run(runs_dir / f"{run_id}.json")
    cfg = run["config"]
    model, clf = load_run(run, device)
    loader = get_test_loader(device, cfg["norm"], cfg["res"])
    if clf is None:
        logits, y = predict(model, loader)
        pred = logits.argmax(1).numpy()
    else:
        fn = make_predictor(model, clf)
        with torch.no_grad():
            parts = [(fn(x), yb.cpu()) for x, yb in loader]
        pred, y = np.concatenate([p for p, _ in parts]), torch.cat([b for _, b in parts])
    res = classification_metrics(y.numpy(), np.asarray(pred))
    final = {
        "run_id": out_id,
        "phase": "final",
        "source_run": run_id,
        "git_commit": git_commit(),
        "params": run["params"],
        "size_mb": size_mb(run),
        "train_time_s": run.get("train_time_s", run.get("fit_time_s")),
        **res,
    }
    if time_it:
        final.update(inference_times(run, device, cfg["res"]))
    save_run(final, runs_dir)
    return final


def _fmt(values: list[float], scale: float = 100) -> str:
    arr = np.array(values) * scale
    spread = f" ± {arr.std():.2f}" if len(arr) > 1 else ""
    return f"{arr.mean():.2f}{spread} (n={len(arr)})"


def final_table(rows: list[dict], runs_dir: Path = RUNS_DIR) -> str:
    """Build the markdown results table from the 'final' run logs of each row."""
    head = "| Model | Optimizer | Input | Params | Test acc % (mean ± std) | Macro F1 % |"
    head += " Latency b=1 CPU/GPU (ms) | Size (MB) | Train time (min) | Notes |"
    lines = [head, "|" + "---|" * 10]
    for row in rows:
        runs = [read_run(runs_dir / f"final_{r}.json") for r in row["runs"]]
        first = runs[0]
        gpu = next(
            (v for k, v in first.items() if k.startswith("latency_b1_") and "cpu" not in k), None
        )
        minutes = np.mean([r["train_time_s"] for r in runs]) / 60
        cells = [
            row["label"], row["optimizer"], str(row["input"]), f"{first['params']:,}",
            _fmt([r["acc"] for r in runs]), _fmt([r["macro_f1"] for r in runs]),
            f"{first['latency_b1_cpu_ms']:.2f} / {'-' if gpu is None else f'{gpu:.2f}'}", f"{first['size_mb']:.1f}",
            f"{minutes:.1f}", row.get("notes", ""),
        ]  # fmt: skip
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score runs on the test set, once each.")
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--table", action="store_true", help="only rebuild results/final_table.md")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rows = yaml.safe_load(Path(args.config).read_text())["rows"]
    if args.table:
        Path("results/final_table.md").write_text(final_table(rows))
        return
    device = get_device()
    for row in rows:
        for i, run_id in enumerate(row["runs"]):
            final = evaluate_run(run_id, device, time_it=i == 0)
            log.info("%s %s test acc %.4f", row["label"], run_id, final["acc"])
            plot_confusion(final["confusion"], row["label"], FIG_DIR / f"cm_{run_id}.png")
    Path("results/final_table.md").write_text(final_table(rows))


if __name__ == "__main__":
    main()
