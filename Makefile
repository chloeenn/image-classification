.PHONY: check fix prose test baseline optim-a optim-b optim-c transfer hybrid

PY ?= python
TRAIN = $(PY) -m src.train
OPTIM = $(TRAIN) --config configs/optim.yaml --set

baseline:
	$(TRAIN) --config configs/baseline.yaml

optim-a:
	for lr in 0.1 0.01 0.001; do $(OPTIM) optimizer=sgd momentum=0.9 lr=$$lr; done
	for lr in 0.01 0.001 0.0001; do $(OPTIM) optimizer=adam lr=$$lr; done




# Best learning rates from step A of the optimizer experiments (see results/runs).
SGD_LR ?= 0.01
ADAM_LR ?= 0.0001
BATCHGD = $(TRAIN) --config configs/optim.yaml --set phase=batchgd subset=5000 epochs=200 name=

optim-b:
	$(OPTIM) optimizer=sgd lr=$(SGD_LR)
	for bs in 64 256; do $(OPTIM) optimizer=sgd momentum=0.9 lr=$(SGD_LR) batch_size=$$bs; done
	for bs in 64 256; do $(OPTIM) optimizer=adam lr=$(ADAM_LR) batch_size=$$bs; done

optim-c:
	for lr in 0.1 0.01; do $(BATCHGD)batchgd optimizer=sgd lr=$$lr batch_size=500 accum=10; done
	$(BATCHGD)minibatch optimizer=sgd momentum=0.9 lr=$(SGD_LR) batch_size=128

transfer:
	$(TRAIN) --config configs/resnet_head.yaml
	$(TRAIN) --config configs/resnet_ft.yaml

hybrid:
	$(PY) -m src.features --config configs/hybrid_frozen.yaml
	$(PY) -m src.features --config configs/hybrid_finetuned.yaml
	$(PY) -m src.features --config configs/hybrid_scratch.yaml

test:
	python -m pytest

check:
	python -m ruff check .
	python -m ruff format --check .
	python -m vulture
	python -m pytest
	@! git ls-files | grep -E '(^|/)(tmp|scratch|dummy|fake)' || (echo "tracked dummy file"; exit 1)

fix:
	python -m ruff check --fix .
	python -m ruff format .

prose:
	@! grep -rnEi 'not (just|only|merely) .* but|rather than|it.?s worth noting|delve|leverage|seamless|robust|cutting-edge|powerful|comprehensive|in the realm of|plays a (crucial|key|vital) role|a testament|landscape|journey|—' README.md src tests app.py configs || (echo "reword the lines above"; exit 1)
