"""Gradio demo: classify an image with the scratch CNN or the fine-tuned ResNet18."""

import json
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.data import get_stats, load_cifar
from src.models import load_run, read_run
from src.utils import CLASSES

CONFIG = json.loads(Path("configs/demo.json").read_text())
NOTE = (
    "Trained on 32x32 images of 10 classes. Photos of other subjects, or at higher "
    "resolution, often get wrong labels."
)


@lru_cache(maxsize=2)
def _load(name: str):
    run = read_run(Path("results/runs") / f"{CONFIG[name]}.json")
    model, _ = load_run(run, torch.device("cpu"))
    return model, run["config"]


def classify(image: Image.Image, name: str) -> dict[str, float]:
    """Return the top 3 class probabilities for an image."""
    model, cfg = _load(name)
    small = image.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)
    x = torch.from_numpy(np.asarray(small).copy()).permute(2, 0, 1)[None].float() / 255
    mean, std = (torch.tensor(v).view(1, 3, 1, 1) for v in get_stats(cfg["norm"]))
    if cfg["res"] != 32:
        x = F.interpolate(x, size=cfg["res"], mode="bilinear", align_corners=False)
    with torch.no_grad():
        probs = model((x - mean) / std).softmax(1)[0]
    top = probs.topk(3)
    return {CLASSES[i]: float(p) for p, i in zip(top.values, top.indices, strict=True)}


def _examples(n: int = 6) -> list[str]:
    _, _, x_test, y_test = load_cifar()
    out = Path("data/examples")
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        path = out / f"{CLASSES[y_test[i]]}_{i}.png"
        Image.fromarray(x_test[i].transpose(1, 2, 0)).resize((128, 128)).save(path)
        paths.append([str(path), list(CONFIG)[0]])
    return paths


def build() -> gr.Blocks:
    """Build the demo page."""
    with gr.Blocks(title="CIFAR-10 classifier") as demo:
        gr.Markdown(f"# CIFAR-10 classifier\n{NOTE}")
        image = gr.Image(type="pil", label="Image")
        model = gr.Dropdown(list(CONFIG), value=list(CONFIG)[0], label="Model")
        label = gr.Label(num_top_classes=3)
        gr.Button("Classify").click(classify, [image, model], label)
        gr.Examples(_examples(), [image, model])
    return demo


if __name__ == "__main__":
    build().launch()
