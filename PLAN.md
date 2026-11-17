# CIFAR-10 Image Classification: Project Plan (v3)

Tick boxes as you go. To change a decision, edit the Decision Log (section 3) first.

## 1. Goal

Train image classifiers on CIFAR-10 (10 classes, 60k 32×32 RGB images). Compare them on accuracy, inference time and model size.

Questions the README answers:
1. How far does a small CNN trained from scratch get, and how much does transfer learning add?
2. How do batch gradient descent, mini-batch SGD with momentum, and Adam differ in convergence and final accuracy?
3. Do CNN features with logistic regression or XGBoost match the CNN's own softmax head?
4. What accuracy, speed and size does each model trade?

Deliverables: GitHub repo, results table, confusion matrices, error analysis, Gradio demo, README.

Pass/fail targets:

| Model | Target test accuracy |
|---|---|
| Scratch CNN (Adam, augmented) | at least 85% (floor 80%) |
| ResNet18, frozen backbone, new head, 224 px | at least 85% |
| ResNet18, fine-tuned, 224 px | at least 94% |
| Hybrid (features + LogReg or XGBoost) | within 1.5 points of the model that supplies the features |

---

## 2. Feasibility

The project is feasible on a free Colab or Kaggle T4 GPU in 6–7 working days. These problems in the first draft are fixed below.

| # | Problem | Fix |
|---|---|---|
| 1 | Batch GD on 45k images makes one update per epoch. It needs thousands of epochs, and the full batch does not fit in GPU memory. | Run it as a small experiment: gradient accumulation on a 5k subset, 200 updates. Compare with mini-batch at the same number of gradient evaluations. |
| 2 | The optimizer grid had about 27 runs. | Cut to about 14 runs of 10 epochs (Phase 3). |
| 3 | "Hybrid close to the CNN" had no measure. A fine-tuned ResNet plus LogReg matches its own linear head by construction. | Extract features from three sources (frozen ImageNet ResNet18, fine-tuned ResNet18, scratch CNN). The frozen ImageNet source is the informative one. |
| 4 | The 90% target for fine-tuned ResNet18 was low. At 224 px it reaches 95% or more. | Raised to 94%. |
| 5 | The scratch CNN uses 32 px input and ResNet uses 224 px, so their speeds do not compare directly. | The results table lists input size and parameter count. |
| 6 | A single run cannot show a 0.5-point gap. | Final models run with 3 seeds. Report mean ± std. |
| 7 | No timing method. | Fixed protocol in section 7. |
| 8 | Colab sessions end and delete local files. | Save a checkpoint every epoch to Google Drive. Training resumes from it. |
| 9 | Notebooks and `src/` duplicate code. | Logic lives in `src/`. Notebooks call it and plot. |
| 10 | The demo takes any photo, but the models trained on 32×32 CIFAR images. | The demo shows its limits and includes CIFAR examples. |
| 11 | No run log, so tables would be built by hand. | Each run writes one JSON file. Scripts build tables and plots from those files. |
| 12 | Nothing kept the code small or clean. | Section 5 sets code and writing rules, enforced by `make check`. |

---

## 3. Decision Log

| Topic | Decision |
|---|---|
| Split | Official 50k train and 10k test. Split train 45k/5k (train/val), stratified, seed 42. Save val indices to `results/val_idx.npy`. Use the test set only in Phase 6. |
| Seeds | 42 for development. Final models use 42, 43, 44. |
| Epochs and batch size | Scratch CNN: 30 epochs, batch 128. ResNet: 5 epochs head only, then 10 epochs fine-tuning, batch 64–128. |
| Normalization | Scratch CNN: mean and std from the 45k train split. ResNet: ImageNet mean and std. |
| Augmentation | Random crop (32 px, padding 4) and horizontal flip, train only. For ResNet, crop and flip at 32 px, then resize to 224. |
| Pretrained input | Resize 32 to 224 (bilinear). A 32 px stem is a stretch goal. |
| Mixed precision | `torch.autocast` (fp16) for ResNet runs. |
| Model selection | Best val accuracy. Ties go to lower val loss. |
| Tracking | JSON run logs plus TensorBoard. No account needed. |
| Hybrid features | Penultimate layer: 512 values for ResNet18, 128 for the scratch CNN. Extract without augmentation. |
| Dependencies | `requirements.txt` with pinned versions, tested on Colab. |

---

## 4. Repo Structure

```
image-classification/
├── README.md
├── PLAN.md
├── Makefile                   # check, test, prose
├── pyproject.toml             # ruff, vulture, pytest settings
├── .pre-commit-config.yaml
├── requirements.txt
├── .gitignore                 # see section 5.3
├── configs/                   # one YAML per experiment
├── data/                      # downloaded on first run, gitignored
├── notebooks/                 # call src/, plot, explain
│   ├── 01_eda.ipynb
│   ├── 02_baseline_cnn.ipynb
│   ├── 03_optimizers.ipynb
│   ├── 04_transfer_learning.ipynb
│   ├── 05_hybrid_models.ipynb
│   └── 06_evaluation.ipynb
├── src/
│   ├── data.py                # datasets, transforms, split, loaders
│   ├── models.py              # ScratchCNN, build_resnet18
│   ├── train.py               # train and eval loops, checkpoint, resume, CLI
│   ├── features.py            # feature extraction, LogReg and XGBoost fits
│   ├── evaluate.py            # metrics, confusion matrix, timing, size
│   └── utils.py               # seed, plotting, run logging
├── tests/
├── results/
│   ├── runs/                  # one JSON per run
│   ├── figures/
│   └── final_table.md
├── models/                    # weights, gitignored; best ones go to HF Hub
└── app.py                     # Gradio demo
```

Start with this layout and add no files beyond it. Create a new module only when an existing one passes 300 lines.

---

## 5. Code and Writing Standards

These rules apply to every phase. `make check` enforces them and must pass before a phase counts as done.

### 5.1 Keep the code small
- Write the simplest code that works. Add an abstraction only when two places already need it.
- Use plain functions. Use a class only when the object holds state (a model, a dataset).
- Config is a YAML file read into a dict, with `argparse` for the CLI. Add no config framework.
- Limits: functions of 40 lines or fewer, cyclomatic complexity 10 or less, files of 300 lines or fewer.
- Write no code for cases that cannot happen. Catch an exception only when you handle it, and never write `except: pass`.
- Do not leave switches, options or parameters that no run uses.
- Type-hint public function signatures. Skip hints on local variables.
- Write a comment only to say why. Delete comments that restate the code.
- Give a public function a docstring of one line that says what it returns or does. Skip docstrings on private helpers.
- Use `pathlib`, f-strings, `logging` (no `print` in `src/`) and fixed seeds. Never hard-code paths or magic numbers: put them in the config.

### 5.2 Remove dead code automatically
- `ruff` (lint and format) rules: `F` (unused imports, variables), `ERA` (commented-out code), `C90` (complexity), `B`, `UP`, `SIM`, `I`. Run `ruff check --fix` to delete unused imports and variables.
- `vulture src tests app.py --min-confidence 80` lists unused functions, classes and arguments. Review the list and delete each item. A script cannot judge which deletions are safe, so a person confirms each one.
- `nbstripout` clears notebook outputs before every commit. Notebooks hold no logic that `src/` needs.
- Delete a failed experiment's code once its result is in `results/runs/`. Git history keeps the old code.
- Before each phase closes, run `make check`, then read `git diff --stat` and confirm every new function has a caller.

### 5.3 Keep dummy and generated files out of git
Create `.gitignore` in Phase 0, before the first commit. It lists:
```
# data and weights
data/
models/
*.pt
*.pth
*.npy
*.pkl
*.joblib
*.ckpt

# dummy, scratch and temporary files
tmp/
scratch/
dummy*
fake*
sample_data/
*.tmp
*.bak
*.log
.DS_Store

# experiment output
runs/
lightning_logs/
wandb/
results/runs/*.tmp

# python and tools
__pycache__/
*.pyc
.venv/
.ipynb_checkpoints/
.pytest_cache/
.ruff_cache/
.vulture_cache/
.mypy_cache/
.env
```
Rules:
- Put dummy data, throwaway scripts and smoke-test output in `tmp/` or `scratch/`. Tests build their own fake data in memory or in pytest's `tmp_path`.
- Keep `results/val_idx.npy` in git: add `!results/val_idx.npy` after the `*.npy` line. Also keep `results/runs/*.json`, `results/figures/` and `results/final_table.md`, because the README cites them.
- Never commit secrets. Keep API tokens (Hugging Face, W&B) in `.env`, which is ignored.
- Before each commit run `git status` and confirm that no untracked file is dummy or generated. Add it to `.gitignore` or delete it.
- If a dummy file was committed, run `git rm --cached <file>` and add its pattern to `.gitignore`.
- `make check` includes `git ls-files | grep -E '(^|/)(tmp|scratch|dummy|fake)'` and fails on any match.

### 5.4 Make targets
```
make check   # ruff check, ruff format --check, vulture, pytest, tracked-dummy-file scan
make fix     # ruff check --fix, ruff format
make prose   # scan README, PLAN, docstrings and comments for banned phrases
```
`pre-commit` runs `make fix` and `make prose` on each commit.

### 5.5 Writing rules (README, docs, docstrings, comments, commit messages, notebook text)
1. Do not use a metaphor, simile or other figure of speech you have often seen in print.
2. Use a short word where it does the job of a long one.
3. If you can cut a word, cut it.
4. Use the active voice, not the passive.
5. Use an everyday English word in place of a foreign phrase, scientific word or jargon word. Where a technical term is the standard name (softmax, batch norm, confusion matrix), use it and define it once.
6. Break any of these rules before you write something awkward or wrong.

Also:
- State facts directly. Write "Adam reached 84% in 10 epochs", with no padding such as "it is worth noting that".
- Avoid the "not A but B" and "rather than A, B" forms. Say what is true: write "The model overfits after epoch 12", not "The problem is not underfitting but overfitting".
- Use a literal phrase when one exists.
- Every number in the README comes from a file in `results/`. Do not write a result before the run exists.
- Read each paragraph aloud before you commit it. If a sentence sounds like a slogan, rewrite it as a plain fact.
- `make prose` greps for these patterns and fails on any hit: `not (just|only|merely) .* but`, `rather than`, `it'?s worth noting`, `delve`, `leverage`, `seamless`, `robust`, `cutting-edge`, `powerful`, `comprehensive`, `in the realm of`, `plays a (crucial|key|vital) role`, `a testament`, `landscape`, `journey`, and the em dash (—). A hit is a prompt to reword, and a person decides the fix.

### 5.6 Review checklist (before each commit)
- [ ] `make check` and `make prose` pass.
- [ ] `git status` shows no dummy, scratch or generated file.
- [ ] Every new function has a caller or a test.
- [ ] No commented-out code, no stray `print`, no unused config keys.
- [ ] Tests cover the new behavior.
- [ ] The commit message states what changed and why in one or two plain sentences.

---

## 6. Phases

Each phase lists tasks, output and a "Done when" check. Finish the check before you start the next phase. Every "Done when" also requires `make check` to pass.

### Phase 0: Setup (about 2 hours)
- [ ] Create the repo, folders, pinned `requirements.txt`, and `.gitignore` (section 5.3) before the first commit
- [ ] Add `pyproject.toml` (ruff, vulture, pytest), `Makefile`, `.pre-commit-config.yaml`
- [ ] Open a Colab or Kaggle notebook with a GPU. Mount Drive for checkpoints.
- [ ] `utils.set_seed()` sets Python, NumPy and PyTorch seeds
- [ ] Smoke test: load one batch and run a forward pass on the GPU

**Done when:** `python -m src.train --config configs/smoke.yaml` runs 1 epoch, and `make check` passes on the empty skeleton.

### Phase 1: Data and EDA (about half a day)
- [ ] Download CIFAR-10. Plot a sample grid for each class.
- [ ] Check class counts (5,000 per class in train, 1,000 in test)
- [ ] Compute per-channel mean and std on the 45k train split
- [ ] Split 45k/5k, stratified. Save val indices.
- [ ] `data.py`: `get_loaders(...)` with CIFAR and ImageNet normalization modes
- [ ] Show 8 augmented versions of one image to check the transforms

**Output:** EDA notebook and `data.py`.
**Done when:** tests confirm the split sizes, no train/val overlap, and that val and test transforms use no randomness.

### Phase 2: Baseline CNN (about half a day)
- Architecture: 3 blocks of [Conv-BN-ReLU ×2, MaxPool, Dropout 0.1–0.3] with 64, 128 and 256 channels. Then flatten, FC(128), Dropout 0.5, FC(10). The FC(128) layer supplies hybrid features.
- Loss: cross-entropy. Optimizer: Adam, lr 1e-3, OneCycle or cosine schedule, 30 epochs.
- Log train and val loss and accuracy each epoch. Save the best checkpoint by val accuracy.
- If the train–val accuracy gap exceeds 10 points, raise dropout or weight decay (5e-4).

**Output:** baseline val accuracy, learning curves, checkpoint.
**Done when:** val accuracy is at least 82%. If lower, debug the pipeline before you continue.

### Phase 3: Optimizer Experiments (about 1 day)
Same CNN, same seed, 10 epochs per run. Change one variable at a time.

| Step | Runs |
|---|---|
| A. Optimizer and LR (batch 128) | SGD + momentum 0.9 with lr 1e-1, 1e-2, 1e-3. Adam with lr 1e-2, 1e-3, 1e-4. Plain SGD at the best SGD+momentum lr. |
| B. Batch size (best lr for each optimizer) | 64, 128, 256 |
| C. Batch GD | Full batch by gradient accumulation on a 5k subset, lr 1e-1 and 1e-2, 200 updates. Compare with mini-batch at equal gradient evaluations. |
| D. Scheduler (optional) | StepLR and OneCycle on the best optimizer |

- [ ] Plot val loss and accuracy against epochs and against wall-clock time
- [ ] Pick the best config by val accuracy. If it beats Adam, retrain the 30-epoch baseline with it.

**Output:** optimizer table and convergence plot.
**Done when:** every run has a JSON log, and the notebook explains the ranking in plain words.

### Phase 4: Transfer Learning (about 1 day)
- [ ] `build_resnet18(pretrained=True)` with `fc` replaced by `Linear(512, 10)`
- [ ] Preprocess: random crop and flip at 32 px, resize to 224, ImageNet normalize
- [ ] Stage 1, head only: freeze the backbone and keep its BatchNorm layers in `eval()` mode. Adam lr 1e-3, 5 epochs.
- [ ] Stage 2, fine-tune: unfreeze all layers. Use lr 1e-4 for the backbone and 1e-3 for the head, cosine schedule, 10 epochs, AMP.
- [ ] Record time per epoch and total training time
- [ ] Save the checkpoint from each stage

**Output:** two ResNet18 models with accuracy and training time.
**Done when:** head-only val accuracy is at least 85% and fine-tuned val accuracy is at least 94%.
**Speed:** a T4 with AMP takes about 1–2 minutes per epoch. If slower, set `num_workers` to 2–4, `pin_memory=True` and use `channels_last`.

### Phase 5: Hybrid Models (about half a day)
- [ ] `features.py` extracts features with `eval()` and `no_grad`, on un-augmented images. Save train and val features as `.npy`. Extract test features in Phase 6.
- [ ] Sources: (a) frozen ImageNet ResNet18, (b) fine-tuned ResNet18, (c) scratch CNN
- [ ] LogReg: `StandardScaler` then `LogisticRegression`, tune C in {0.01, 0.1, 1, 10} on val, `max_iter` at least 1000
- [ ] XGBoost: `tree_method="hist"`, `device="cuda"`, early stopping on val, tune `max_depth` in {4, 6, 8} and `learning_rate` in {0.05, 0.1}
- [ ] Record fit time and inference time. Inference time includes the feature extractor.

**Output:** 6 hybrid results (3 sources × 2 classifiers), each next to its source model's softmax head.
**Done when:** all val accuracies are logged, and the notebook says whether the classical heads add anything.

### Phase 6: Final Evaluation (about 1 day)
- [ ] Freeze configs and checkpoints. Write the chosen hyperparameters to `results/final_configs.md`.
- [ ] Retrain final models with 3 seeds. If time runs short, use one seed and say so in the README.
- [ ] Evaluate each model on the test set once. Report mean ± std.
- [ ] Per-class precision, recall and F1. One confusion matrix per model.
- [ ] List the most confused pairs (expect cat/dog, deer/horse, automobile/truck). Review 16–25 misclassified images and group the causes.
- [ ] Measure inference time and model size (section 7).

**Output:** `results/final_table.md` and figures in `results/figures/`.
**Done when:** the final table has no empty cells and each final model saw the test set once.

### Phase 7: Demo and Packaging (about 1 day)
- [ ] `app.py`: upload an image, resize it, show the top 3 predictions with confidence. Model dropdown (scratch CNN or ResNet18). A few CIFAR example images.
- [ ] Demo note: "Trained on 32×32 images of 10 classes. Photos of other subjects, or at higher resolution, often get wrong labels."
- [ ] README: overview, setup, exact reproduction commands, results table, key plots, lessons learned, limits. Follow section 5.5.
- [ ] Final cleanup: run `make fix`, `make check`, `make prose`. Delete everything `vulture` flags and nothing uses.
- [ ] Optional: deploy to Hugging Face Spaces (CPU). Host weights on the HF Hub.

**Output:** public repo and demo.
**Done when:** a fresh clone, `pip install -r requirements.txt` and `python app.py` work, and the README commands reproduce the table.

---

## 7. Evaluation Protocol

Run log (one JSON per run in `results/runs/`):
```json
{"run_id": "", "phase": "", "model": "", "optimizer": "", "lr": 0, "batch_size": 0,
 "epochs": 0, "seed": 0, "input_res": 32, "best_val_acc": 0, "best_epoch": 0,
 "train_time_s": 0, "history": {"train_loss": [], "val_loss": [], "val_acc": []},
 "git_commit": ""}
```

Metrics: accuracy, per-class precision/recall/F1, confusion matrix, train time, inference time, model size, parameter count.

Inference timing:
- 200 warm-up iterations, then at least 500 timed iterations.
- On GPU, call `torch.cuda.synchronize()` before each clock read.
- Report latency at batch size 1 (CPU and GPU) and throughput at batch size 128 (GPU).
- Exclude data loading and preprocessing. Hybrids include the feature extractor and the classical head.
- State the hardware (for example, Colab T4 and the CPU model).

Model size: file size of the saved `state_dict` in MB. For hybrids, add the sklearn or XGBoost model file.

## 8. Tests

Run `pytest` before each phase closes:
- Split sizes (45k/5k/10k), no index overlap, stratified.
- Val and test transforms are deterministic.
- Model output shape is `(B, 10)`. Frozen ResNet parameters have `requires_grad=False`.
- Setting the seed twice gives the same first-batch loss.
- Saving and loading a checkpoint gives the same predictions.

---

## 9. Results Table Template

| Model | Optimizer | Input | Params | Test acc (mean ± std) | Macro F1 | Latency b=1 CPU/GPU (ms) | Size (MB) | Train time | Notes |
|---|---|---|---|---|---|---|---|---|---|
| Scratch CNN | Adam | 32 | | | | | | | baseline |
| Scratch CNN | SGD + momentum | 32 | | | | | | | |
| Scratch CNN | Batch GD | 32 | | | | | | | 5k subset |
| ResNet18 (head only) | Adam | 224 | | | | | | | frozen backbone |
| ResNet18 (fine-tuned) | Adam | 224 | | | | | | | |
| Frozen ResNet18 + LogReg | none | 224 | | | | | | | |
| Frozen ResNet18 + XGBoost | none | 224 | | | | | | | |
| Fine-tuned ResNet18 + LogReg | none | 224 | | | | | | | matches own head by construction |
| Fine-tuned ResNet18 + XGBoost | none | 224 | | | | | | | |
| Scratch CNN + LogReg | none | 32 | | | | | | | |
| Scratch CNN + XGBoost | none | 32 | | | | | | | |

---

## 10. Timeline (6–7 working days)

| Day | Work | Check |
|---|---|---|
| 1 | Phases 0 and 1 | split tests pass |
| 2 | Phase 2, start Phase 3 | baseline at least 82% val |
| 3 | Finish Phase 3 | optimizer table complete |
| 4 | Phase 4 | fine-tuned at least 94% val |
| 5 | Phase 5, start Phase 6 | hybrid table complete |
| 6 | Finish Phase 6 | final table full |
| 7 | Phase 7 | fresh-clone run works |

If you fall more than a day behind, drop these in order: 3-seed runs for exploration models, the Phase 3 scheduler step, the HF Spaces deploy.

---

## 11. Common Pitfalls
- **Test leakage:** select models on val. Fit scalers and classifiers on train only.
- **Augmenting val or test:** only the train transform uses randomness.
- **Normalization mismatch:** CIFAR statistics for the scratch CNN, ImageNet statistics for ResNet, and the same ones again in the demo.
- **Frozen backbone with BatchNorm in train mode:** the running statistics still change. Put the backbone in `eval()` during stage 1.
- **Missing `model.eval()` or `torch.no_grad()`** during evaluation and feature extraction.
- **Hybrid features taken from augmented images.** Use the deterministic transform.
- **Wrong timing:** warm-up, asynchronous GPU calls or data loading inside the timer. Follow section 7.
- **Several changes per experiment:** change one variable and log every run.
- **A single-seed gap under 1 point:** treat it as noise.
- **Large weights, dummy files or secrets in git.** Follow section 5.3.
- **Dead code left behind:** run `make check` each phase.

---

## 12. Stretch Goals (after Phase 7)
- Cutout, MixUp or AutoAugment
- ResNet18 with a 32 px stem (3×3 first conv, no max-pool). It runs faster and often scores higher than 224 px upscaling.
- MobileNetV3 or EfficientNet-B0
- Quantization or pruning, with a size and speed comparison
- Grad-CAM
- CIFAR-100 with the same pipeline

---

## 13. Resources
- CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html
- PyTorch CIFAR-10 tutorial: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
- torchvision models: https://pytorch.org/vision/stable/models.html
- XGBoost GPU: https://xgboost.readthedocs.io/en/stable/gpu/index.html
- Gradio docs: https://www.gradio.app/docs
- Hugging Face Spaces: https://huggingface.co/docs/hub/spaces
- Ruff: https://docs.astral.sh/ruff/
- Vulture: https://github.com/jendrikseipp/vulture
