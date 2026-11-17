"""Scratch CNN and ResNet18 classifiers. Both expose embed() and a `head` linear layer."""

import json
from pathlib import Path

import joblib
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def _block(c_in: int, c_out: int, dropout: float) -> nn.Sequential:
    layers = []
    for i in range(2):
        layers += [nn.Conv2d(c_in if i == 0 else c_out, c_out, 3, padding=1, bias=False)]
        layers += [nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers, nn.MaxPool2d(2), nn.Dropout(dropout))


class ScratchCNN(nn.Module):
    """Three conv blocks (64, 128, 256 channels), FC(128), then a 10-way head."""

    def __init__(self, dropouts=(0.1, 0.2, 0.3), head_dropout: float = 0.5, fc_dim: int = 128):
        super().__init__()
        chans = (3, 64, 128, 256)
        self.blocks = nn.Sequential(
            *[_block(chans[i], chans[i + 1], dropouts[i]) for i in range(3)]
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(256 * 4 * 4, fc_dim), nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(fc_dim, 10))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 128-value penultimate features."""
        return self.fc(self.blocks(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))


class Resnet18(nn.Module):
    """ResNet18 backbone (512 features) with a new linear head."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(512, 10)
        self.backbone_frozen = False

    def freeze_backbone(self, frozen: bool) -> None:
        """Freeze or unfreeze the backbone. A frozen backbone keeps BatchNorm in eval mode."""
        self.backbone_frozen = frozen
        for p in self.backbone.parameters():
            p.requires_grad = not frozen
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone_frozen:
            self.backbone.eval()
        return self

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 512-value penultimate features."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))


def build_model(cfg: dict) -> nn.Module:
    """Build the model named in cfg and apply its freeze setting."""
    if cfg["model"] == "scratch":
        return ScratchCNN(tuple(cfg.get("dropouts", (0.1, 0.2, 0.3))), cfg.get("head_dropout", 0.5))
    model = Resnet18(cfg.get("pretrained", True))
    model.freeze_backbone(cfg.get("freeze_backbone", False))
    return model


def load_run(run: dict, device: torch.device):
    """Rebuild the model of a run log in eval mode. Return it with its classical head or None."""
    cfg = dict(run["config"], pretrained=False)
    model = build_model(cfg)
    model.load_state_dict(torch.load(run["weights"], map_location="cpu"))
    clf = joblib.load(run["clf"]) if run.get("clf") else None
    return model.to(device).eval(), clf


def read_run(path: str | Path) -> dict:
    """Read a run log from disk."""
    return json.loads(Path(path).read_text())
