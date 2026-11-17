"""Penultimate-layer features with LogReg and XGBoost heads."""

import argparse
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data import get_loaders
from src.evaluate import inference_times
from src.models import build_model, read_run
from src.train import load_config
from src.utils import get_device, git_commit, save_run, set_seed

log = logging.getLogger("cifar")


@torch.no_grad()
def extract(model, loader) -> tuple[np.ndarray, np.ndarray]:
    """Return penultimate features and labels. Run the loader without augmentation."""
    model.eval()
    feats, ys = [], []
    for x, y in loader:
        feats.append(model.embed(x).float().cpu().numpy())
        ys.append(y.cpu().numpy())
    return np.concatenate(feats), np.concatenate(ys)


def _score(clf, fva, yva) -> tuple[float, float]:
    proba = clf.predict_proba(fva)
    return float((proba.argmax(1) == yva).mean()), float(log_loss(yva, proba, labels=range(10)))


def _best(fits: list[tuple]) -> tuple:
    """Pick (clf, params, acc, loss) by val accuracy, then val loss."""
    return max(fits, key=lambda f: (f[2], -f[3]))


def fit_logreg(ftr, ytr, fva, yva, cs, max_iter: int, seed: int):
    """Fit StandardScaler + LogisticRegression for each C. Return the best by val accuracy."""
    fits = []
    for c in cs:
        clf = make_pipeline(
            StandardScaler(), LogisticRegression(C=c, max_iter=max_iter, random_state=seed)
        )
        clf.fit(ftr, ytr)
        fits.append((clf, {"C": c}, *_score(clf, fva, yva)))
    return _best(fits)


def fit_xgb(ftr, ytr, fva, yva, grid: dict, device: str, seed: int, rounds: int, patience: int):
    """Fit XGBoost over a depth and learning-rate grid with early stopping on val."""
    fits = []
    for depth in grid["max_depth"]:
        for lr in grid["learning_rate"]:
            clf = XGBClassifier(
                tree_method="hist",
                device=device,
                max_depth=depth,
                learning_rate=lr,
                n_estimators=rounds,
                early_stopping_rounds=patience,
                eval_metric="mlogloss",
                random_state=seed,
            )
            clf.fit(ftr, ytr, eval_set=[(fva, yva)], verbose=False)
            fits.append((clf, {"max_depth": depth, "learning_rate": lr}, *_score(clf, fva, yva)))
    return _best(fits)


def _source_model(cfg: dict, device):
    """Build the feature extractor: ImageNet weights or a trained run's best weights."""
    model = build_model(cfg).to(device)
    if cfg.get("source_run"):
        run = read_run(Path("results/runs") / f"{cfg['source_run'].format(**cfg)}.json")
        model.load_state_dict(torch.load(run["weights"], map_location=device))
        cfg["source_weights"] = run["weights"]
    return model.eval()


def run_hybrid(cfg: dict, device, ckpt_root: str = "models") -> list[dict]:
    """Fit both classical heads on one source's features. Log and return one run per head."""
    set_seed(cfg["seed"])
    loaders = get_loaders(256, device, cfg["norm"], cfg["res"], seed=cfg["seed"])
    model = _source_model(cfg, device)
    ftr, ytr = extract(model, loaders["train_eval"])
    fva, yva = extract(model, loaders["val"])
    fits = {}
    for kind in cfg["heads"]:
        t0 = time.perf_counter()
        if kind == "logreg":
            fits[kind] = fit_logreg(
                ftr, ytr, fva, yva, cfg["logreg_c"], cfg["max_iter"], cfg["seed"]
            )
        else:
            args = (
                cfg["xgb_grid"],
                cfg["xgb_device"],
                cfg["seed"],
                cfg["xgb_rounds"],
                cfg["xgb_patience"],
            )
            fits[kind] = fit_xgb(ftr, ytr, fva, yva, *args)
        fits[kind] += (time.perf_counter() - t0,)
    runs = []
    for kind, (clf, params, acc, loss, fit_s) in fits.items():
        run_id = f"{cfg['name']}_{kind}_s{cfg['seed']}"
        out = Path(ckpt_root) / run_id
        out.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out / "model.pt")
        joblib.dump(clf, out / "clf.joblib")
        run = {
            "run_id": run_id,
            "phase": "hybrid",
            "model": f"{cfg['name']}+{kind}",
            "seed": cfg["seed"],
            "input_res": cfg["res"],
            "best_val_acc": acc,
            "val_loss": loss,
            "best_params": params,
            "fit_time_s": fit_s,
            "feature_dim": ftr.shape[1],
            "git_commit": git_commit(),
            "params": sum(p.numel() for p in model.parameters()),
            "weights": str(out / "model.pt"),
            "clf": str(out / "clf.joblib"),
            "config": cfg,
        }
        run.update(inference_times(run, device, cfg["res"], iters=cfg.get("time_iters", 500)))
        save_run(run)
        runs.append(run)
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit LogReg and XGBoost on CNN features.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for run in run_hybrid(load_config(args.config, args.set), get_device()):
        log.info("%s val acc %.4f (%s)", run["run_id"], run["best_val_acc"], run["best_params"])


if __name__ == "__main__":
    main()
