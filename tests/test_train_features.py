import numpy as np
import pytest
import torch

from src.evaluate import classification_metrics, evaluate_run, final_table, top_confused
from src.features import fit_logreg, fit_xgb
from src.train import fit


def test_fit_writes_log_and_resumes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.data.VAL_IDX", tmp_path / "val.npy")
    cfg = {
        "name": "t", "phase": "test", "model": "scratch", "optimizer": "sgd", "lr": 0.05,
        "momentum": 0.9, "batch_size": 64, "epochs": 2, "seed": 1, "res": 32, "norm": "cifar",
        "subset": 256, "ckpt_dir": str(tmp_path / "models"),
    }  # fmt: skip
    monkeypatch.setattr("src.data.load_cifar", lambda root="data": _tiny())
    run = fit(cfg, torch.device("cpu"), tmp_path / "runs")
    assert len(run["history"]["val_acc"]) == 2
    assert (tmp_path / "runs" / f"{run['run_id']}.json").exists()
    again = fit(cfg, torch.device("cpu"), tmp_path / "runs")
    assert again["history"]["val_acc"] == run["history"]["val_acc"]


def _tiny():
    rng = np.random.default_rng(0)
    y = np.arange(6000) % 10
    x = rng.integers(0, 255, (6000, 3, 32, 32), dtype=np.uint8)
    return x, y, x[:100], y[:100]


def _blobs():
    rng = np.random.default_rng(0)
    y = np.arange(600) % 10
    return rng.normal(size=(600, 8)) + np.eye(10)[y][:, :8] * 4, y


def test_classical_heads_learn_separable_features():
    f, y = _blobs()
    _, params, acc, _ = fit_logreg(f[:400], y[:400], f[400:], y[400:], [0.1, 1], 500, 0)
    assert acc > 0.5 and "C" in params
    grid = {"max_depth": [2], "learning_rate": [0.3]}
    _, params, acc, _ = fit_xgb(f[:400], y[:400], f[400:], y[400:], grid, "cpu", 0, 30, 5)
    assert acc > 0.5


def test_metrics_and_confusion_pairs():
    y = np.array([3, 3, 3, 5, 5, 0])
    pred = np.array([3, 5, 5, 5, 3, 0])
    m = classification_metrics(y, pred)
    assert m["acc"] == 3 / 6
    assert m["confusion"][3][5] == 2
    assert top_confused(m["confusion"], 1) == [("cat", "dog", 2)]


def test_final_evaluation_runs_once_and_builds_table(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.data.load_cifar", lambda root="data": _tiny())
    cfg = {
        "name": "t", "phase": "test", "model": "scratch", "optimizer": "adam", "lr": 0.001,
        "batch_size": 64, "epochs": 1, "seed": 1, "res": 32, "norm": "cifar", "subset": 128,
    }  # fmt: skip
    monkeypatch.setattr("src.evaluate.time_ms", lambda *a, **k: 1.0)
    cpu, runs = torch.device("cpu"), tmp_path / "runs"
    run = fit(cfg, cpu, runs)
    final = evaluate_run(run["run_id"], cpu, time_it=True, runs_dir=runs)
    assert 0 <= final["acc"] <= 1 and len(final["confusion"]) == 10
    assert final["size_mb"] > 1
    with pytest.raises(FileExistsError):
        evaluate_run(run["run_id"], cpu, time_it=False, runs_dir=runs)
    row = {"label": "Scratch", "optimizer": "Adam", "input": 32, "runs": [run["run_id"]]}
    assert "| Scratch | Adam | 32 |" in final_table([row], runs)
