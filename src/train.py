"""Training loop with checkpoint-resume, run logs and a CLI."""

import argparse
import logging
import time
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from src.data import get_loaders
from src.evaluate import predict
from src.models import build_model
from src.utils import RUNS_DIR, get_device, git_commit, save_run, set_seed

log = logging.getLogger("cifar")


def run_id_of(cfg: dict) -> str:
    """Return cfg['run_id'], or an id built from the settings that change between runs."""
    if cfg.get("run_id"):
        return cfg["run_id"].format(**cfg)
    plain = cfg["optimizer"] == "sgd" and not cfg.get("momentum")
    optimizer = "sgdplain" if plain else cfg["optimizer"]
    parts = [cfg["name"], optimizer, f"lr{cfg['lr']}", f"bs{cfg['batch_size']}"]
    if cfg.get("accum", 1) > 1:
        parts.append(f"acc{cfg['accum']}")
    parts += [cfg.get("schedule", "none"), f"e{cfg['epochs']}", f"s{cfg['seed']}"]
    return "_".join(parts)


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """Build SGD or Adam with one lr for the body and cfg['head_lr'] for the head."""
    body = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head")]
    head = [p for n, p in model.named_parameters() if n.startswith("head")]
    groups = [
        {"params": g, "lr": lr}
        for g, lr in ((body, cfg["lr"]), (head, cfg.get("head_lr", cfg["lr"])))
        if g
    ]
    wd = cfg.get("weight_decay", 0.0)
    if cfg["optimizer"] == "adam":
        return torch.optim.Adam(groups, weight_decay=wd)
    return torch.optim.SGD(groups, momentum=cfg.get("momentum", 0.0), weight_decay=wd)


def build_scheduler(opt, cfg: dict, total: int):
    """Return a per-update scheduler, or None."""
    kind = cfg.get("schedule", "none")
    if kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
    if kind == "onecycle":
        max_lr = [g["lr"] for g in opt.param_groups]
        return torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, total_steps=total)
    if kind == "step":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=max(total // 3, 1), gamma=0.1)
    return None


def train_epoch(model, loader, loss_fn, opt, sched, accum: int, scaler, amp: bool):
    """Run one epoch with `accum` micro-batches per update. Return mean loss and accuracy."""
    model.train()
    device = next(model.parameters()).device
    total_loss = correct = seen = 0
    opt.zero_grad(set_to_none=True)
    for i, (x, y) in enumerate(loader):
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            out = model(x)
            loss = loss_fn(out, y)
        scaler.scale(loss * len(x) / (accum * loader.batch_size)).backward()
        total_loss += loss.item() * len(x)
        correct += (out.argmax(1) == y).sum().item()
        seen += len(x)
        if (i + 1) % accum == 0 or i + 1 == len(loader):
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            if sched:
                sched.step()
    return total_loss / seen, correct / seen


def _eval(model, loader, loss_fn, amp):
    logits, y = predict(model, loader, amp)
    return loss_fn(logits, y).item(), (logits.argmax(1) == y).float().mean().item()


def _is_better(acc, loss, best):
    return best is None or acc > best[0] or (acc == best[0] and loss < best[1])


def _prepare(cfg: dict, device, ckpt_root: Path):
    """Build loaders, model, optimizer and scheduler. Load init_from weights if set."""
    loaders = get_loaders(
        cfg["batch_size"], device, cfg["norm"], cfg["res"], cfg.get("subset"), cfg["seed"]
    )
    model = build_model(cfg).to(device)
    if cfg.get("init_from"):
        src = ckpt_root / cfg["init_from"].format(**cfg) / "best.pt"
        model.load_state_dict(torch.load(src, map_location=device))
    opt = build_optimizer(model, cfg)
    updates = -(-len(loaders["train"]) // cfg.get("accum", 1))
    return loaders, model, opt, build_scheduler(opt, cfg, cfg["epochs"] * updates)


def _resume(last: Path, model, opt, sched, device):
    """Restore training state from last.pt. Return history, best and the next epoch."""
    state = torch.load(last, map_location=device)
    model.load_state_dict(state["model"])
    opt.load_state_dict(state["opt"])
    if sched:
        sched.load_state_dict(state["sched"])
    log.info("resumed %s at epoch %d", last.parent.name, state["epoch"])
    return state["hist"], state["best"], state["epoch"]


def _run_log(cfg, run_id, model, hist, best, ckpt_dir, device) -> dict:
    return {
        "run_id": run_id, "phase": cfg["phase"], "model": cfg["model"],
        "optimizer": cfg["optimizer"], "lr": cfg["lr"], "batch_size": cfg["batch_size"],
        "epochs": cfg["epochs"], "seed": cfg["seed"], "input_res": cfg["res"],
        "best_val_acc": best[0], "best_epoch": best[2], "train_time_s": sum(hist["epoch_time_s"]),
        "history": hist, "git_commit": git_commit(),
        "params": sum(p.numel() for p in model.parameters()),
        "weights": str(ckpt_dir / "best.pt"), "config": cfg, "device": device.type,
    }  # fmt: skip


def fit(cfg: dict, device: torch.device, runs_dir: Path = RUNS_DIR) -> dict:
    """Train per cfg, keep the best epoch by val accuracy, write the run log and return it."""
    set_seed(cfg["seed"])
    run_id, ckpt_root = run_id_of(cfg), Path(cfg.get("ckpt_dir", "models"))
    ckpt_dir = ckpt_root / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    loaders, model, opt, sched = _prepare(cfg, device, ckpt_root)
    amp = cfg.get("amp", False) and device.type == "cuda"
    scaler, loss_fn = torch.amp.GradScaler(enabled=amp), nn.CrossEntropyLoss()
    keys = ("train_loss", "train_acc", "val_loss", "val_acc", "epoch_time_s")
    hist, best, start = {k: [] for k in keys}, None, 0
    if (ckpt_dir / "last.pt").exists():
        hist, best, start = _resume(ckpt_dir / "last.pt", model, opt, sched, device)
    writer = SummaryWriter(f"runs/{run_id}")
    for epoch in range(start, cfg["epochs"]):
        t0 = time.perf_counter()
        tl, ta = train_epoch(
            model, loaders["train"], loss_fn, opt, sched, cfg.get("accum", 1), scaler, amp
        )
        elapsed = time.perf_counter() - t0
        vl, va = _eval(model, loaders["val"], loss_fn, amp)
        for k, v in zip(keys, (tl, ta, vl, va, elapsed), strict=True):
            hist[k].append(v)
        writer.add_scalars("acc", {"train": ta, "val": va}, epoch)
        if _is_better(va, vl, best):
            best = (va, vl, epoch + 1)
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
        state = {"model": model.state_dict(), "opt": opt.state_dict(), "hist": hist}
        state |= {"best": best, "epoch": epoch + 1, "sched": sched.state_dict() if sched else None}
        torch.save(state, ckpt_dir / "last.pt")
        log.info(
            "%s ep %d/%d train %.3f/%.3f val %.3f/%.3f %.0fs",
            run_id, epoch + 1, cfg["epochs"], tl, ta, vl, va, elapsed,
        )  # fmt: skip
    writer.close()
    run = _run_log(cfg, run_id, model, hist, best, ckpt_dir, device)
    save_run(run, runs_dir)
    return run


def load_config(path: str, overrides: list[str]) -> dict:
    """Read a YAML config and apply key=value overrides."""
    cfg = yaml.safe_load(Path(path).read_text())
    for item in overrides:
        key, value = item.split("=", 1)
        cfg[key] = yaml.safe_load(value)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one model from a YAML config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run = fit(load_config(args.config, args.set), get_device())
    log.info("best val acc %.4f at epoch %d", run["best_val_acc"], run["best_epoch"])


if __name__ == "__main__":
    main()
