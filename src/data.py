"""CIFAR-10 in memory: split, normalization and batch iterators with augmentation."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torchvision.datasets import CIFAR10

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
VAL_IDX = Path("results/val_idx.npy")
STATS_FILE = Path("results/cifar_stats.json")


def load_cifar(root: str = "data") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return uint8 NCHW train and test images with their labels."""
    out = []
    for train in (True, False):
        ds = CIFAR10(root, train=train, download=True)
        out += [ds.data.transpose(0, 3, 1, 2), np.array(ds.targets)]
    return out[0], out[1], out[2], out[3]


def split_indices(
    y: np.ndarray, val_size: int = 5000, seed: int = 42, path: Path = VAL_IDX
) -> tuple[np.ndarray, np.ndarray]:
    """Return stratified train and val indices. Save the val indices on first use."""
    idx = np.arange(len(y))
    if path.exists():
        val = np.load(path)
    else:
        _, val = train_test_split(idx, test_size=val_size, stratify=y, random_state=seed)
        val = np.sort(val)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, val)
    return np.setdiff1d(idx, val), val


def channel_stats(x: np.ndarray) -> tuple[list[float], list[float]]:
    """Return per-channel mean and std of uint8 NCHW images on the 0-1 scale."""
    f = x.astype(np.float64) / 255
    return f.mean(axis=(0, 2, 3)).tolist(), f.std(axis=(0, 2, 3)).tolist()


class Batches:
    """Iterate over batches of an in-memory uint8 image set, with optional train augmentation."""

    def __init__(self, x, y, batch_size, stats, device, res=32, train=False, seed=0, pad=4):
        self.x = torch.as_tensor(x).to(device)
        self.y = torch.as_tensor(y).to(device)
        self.batch_size, self.res, self.train, self.pad = batch_size, res, train, pad
        self.device = device
        self.mean = torch.tensor(stats[0], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(stats[1], device=device).view(1, 3, 1, 1)
        self.gen = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return -(-len(self.x) // self.batch_size)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        n, size = len(x), x.shape[-1]
        xp = F.pad(x, (self.pad,) * 4)
        oy, ox = torch.randint(0, 2 * self.pad + 1, (2, n), generator=self.gen).to(self.device)
        span = torch.arange(size, device=self.device)
        rows = (oy[:, None] + span)[:, :, None]
        cols = (ox[:, None] + span)[:, None, :]
        out = xp[torch.arange(n, device=self.device)[:, None, None], :, rows, cols]
        out = out.permute(0, 3, 1, 2)
        flip = (torch.rand(n, generator=self.gen) < 0.5).to(self.device)
        return torch.where(flip[:, None, None, None], out.flip(-1), out)

    def _prepare(self, xb: torch.Tensor) -> torch.Tensor:
        xb = xb.float() / 255
        if self.train:
            xb = self._augment(xb)
        if self.res != xb.shape[-1]:
            xb = F.interpolate(xb, size=self.res, mode="bilinear", align_corners=False)
        return (xb - self.mean) / self.std

    def __iter__(self):
        n = len(self.x)
        order = torch.randperm(n, generator=self.gen).to(self.device) if self.train else None
        for i in range(0, n, self.batch_size):
            idx = order[i : i + self.batch_size] if self.train else slice(i, i + self.batch_size)
            yield self._prepare(self.x[idx]), self.y[idx]


def get_stats(norm: str, x_train: np.ndarray | None = None, path: Path = STATS_FILE):
    """Return (mean, std) for 'imagenet' or 'cifar'. The CIFAR values come from the 45k split."""
    if norm == "imagenet":
        return IMAGENET
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(channel_stats(x_train)))
    return tuple(json.loads(path.read_text()))


def get_loaders(
    batch_size: int,
    device,
    norm: str = "cifar",
    res: int = 32,
    subset: int | None = None,
    seed: int = 42,
    root: str = "data",
) -> dict[str, Batches]:
    """Build 'train' (augmented), 'train_eval' (plain, ordered) and 'val' loaders."""
    x, y, _, _ = load_cifar(root)
    tr, va = split_indices(y)
    stats = get_stats(norm, x[tr])
    if subset:
        tr, _ = train_test_split(tr, train_size=subset, stratify=y[tr], random_state=seed)
    kw = dict(stats=stats, device=device, res=res)
    return {
        "train": Batches(x[tr], y[tr], batch_size, train=True, seed=seed, **kw),
        "train_eval": Batches(x[tr], y[tr], 256, **kw),
        "val": Batches(x[va], y[va], 256, **kw),
    }


def get_test_loader(device, norm: str = "cifar", res: int = 32, root: str = "data") -> Batches:
    """Build the test loader. Statistics still come from the 45k train split."""
    x, y, xt, yt = load_cifar(root)
    tr, _ = split_indices(y)
    return Batches(xt, yt, 256, get_stats(norm, x[tr]), device, res=res)
