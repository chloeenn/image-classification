import numpy as np
import pytest
import torch

from src.data import Batches, get_loaders, get_test_loader, load_cifar, split_indices

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def cifar():
    return load_cifar("data")


def test_split_sizes_overlap_and_strata(cifar, tmp_path):
    _, y, x_test, _ = cifar
    tr, va = split_indices(y, path=tmp_path / "val.npy")
    assert (len(tr), len(va), len(x_test)) == (45000, 5000, 10000)
    assert not set(tr) & set(va)
    assert (np.bincount(y[va]) == 500).all()


def test_split_is_saved_and_reused(cifar, tmp_path):
    _, y, _, _ = cifar
    path = tmp_path / "val.npy"
    first = split_indices(y, path=path)[1]
    assert (np.load(path) == first).all()
    assert (split_indices(y, seed=1, path=path)[1] == first).all()


def _images(n=20):
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (n, 3, 32, 32), dtype=np.uint8), np.arange(n) % 10


def test_eval_transform_is_deterministic():
    x, y = _images()
    loader = Batches(x, y, 8, ([0.5] * 3, [0.2] * 3), CPU, res=64, train=False)
    a, b = next(iter(loader))[0], next(iter(loader))[0]
    assert torch.equal(a, b)
    assert a.shape == (8, 3, 64, 64)


def test_train_transform_changes_images_and_is_seeded():
    x, y = _images()
    stats = ([0.5] * 3, [0.2] * 3)
    plain = next(iter(Batches(x, y, 20, stats, CPU)))[0]
    aug1 = next(iter(Batches(x, y, 20, stats, CPU, train=True, seed=1)))
    aug2 = next(iter(Batches(x, y, 20, stats, CPU, train=True, seed=1)))
    assert torch.equal(aug1[0], aug2[0])
    assert not torch.equal(aug1[0], plain[aug1[1].argsort().argsort()])


def test_real_loaders_val_and_test_have_no_randomness():
    loaders = get_loaders(64, CPU, subset=500)
    assert len(loaders["train"].x) == 500
    assert len(loaders["val"].x) == 5000
    val = loaders["val"]
    assert torch.equal(next(iter(val))[0], next(iter(val))[0])
    test = get_test_loader(CPU)
    assert len(test.x) == 10000
    assert torch.equal(next(iter(test))[0], next(iter(test))[0])
