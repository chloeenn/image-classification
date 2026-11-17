import torch

from src.data import Batches
from src.evaluate import make_predictor, predict
from src.models import ScratchCNN, build_model
from src.train import build_optimizer, run_id_of
from src.utils import set_seed

CPU = torch.device("cpu")


def test_scratch_output_and_embedding_shapes():
    model = ScratchCNN()
    x = torch.randn(4, 3, 32, 32)
    assert model(x).shape == (4, 10)
    assert model.embed(x).shape == (4, 128)


def test_resnet_shapes_and_frozen_backbone():
    model = build_model({"model": "resnet18", "pretrained": False, "freeze_backbone": True})
    x = torch.randn(2, 3, 64, 64)
    assert model(x).shape == (2, 10)
    assert model.embed(x).shape == (2, 512)
    assert not any(p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.head.parameters())


def test_frozen_backbone_keeps_batchnorm_stats():
    model = build_model({"model": "resnet18", "pretrained": False, "freeze_backbone": True})
    model.train()
    bn = model.backbone.bn1
    before = bn.running_mean.clone()
    model(torch.randn(4, 3, 64, 64))
    assert not bn.training
    assert torch.equal(bn.running_mean, before)


def test_same_seed_gives_same_first_batch_loss():
    losses = []
    for _ in range(2):
        set_seed(7)
        model = ScratchCNN()
        x, y = torch.randn(16, 3, 32, 32), torch.randint(0, 10, (16,))
        losses.append(torch.nn.functional.cross_entropy(model(x), y).item())
    assert losses[0] == losses[1]


def test_checkpoint_roundtrip_gives_same_predictions(tmp_path):
    model = ScratchCNN().eval()
    x = torch.randint(0, 255, (12, 3, 32, 32), dtype=torch.uint8).numpy()
    loader = Batches(x, torch.zeros(12, dtype=torch.long), 6, ([0.5] * 3, [0.2] * 3), CPU)
    before, _ = predict(model, loader)
    torch.save(model.state_dict(), tmp_path / "m.pt")
    restored = ScratchCNN()
    restored.load_state_dict(torch.load(tmp_path / "m.pt"))
    after, _ = predict(restored, loader)
    assert torch.equal(before, after)
    assert (make_predictor(restored)(torch.randn(3, 3, 32, 32)).shape) == (3,)


def test_optimizer_uses_head_lr_and_skips_frozen_params():
    model = build_model({"model": "resnet18", "pretrained": False, "freeze_backbone": False})
    opt = build_optimizer(model, {"optimizer": "adam", "lr": 1e-4, "head_lr": 1e-3})
    assert [g["lr"] for g in opt.param_groups] == [1e-4, 1e-3]
    model.freeze_backbone(True)
    assert len(build_optimizer(model, {"optimizer": "sgd", "lr": 0.1}).param_groups) == 1


def test_run_id_changes_with_each_swept_setting():
    base = {"name": "o", "optimizer": "sgd", "lr": 0.1, "batch_size": 128, "epochs": 10, "seed": 1}
    assert run_id_of(base) != run_id_of({**base, "lr": 0.01})
    assert run_id_of({**base, "accum": 10}) != run_id_of(base)
    assert run_id_of({**base, "momentum": 0.9}) != run_id_of(base)
