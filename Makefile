COMP := biohub-cell-tracking-during-development
RAW  ?= /Volumes/extSSD/kaggle/biohub/raw/train
GT   ?= /Volumes/extSSD/kaggle/biohub/gt/train

.PHONY: setup data train baseline lint test clean

setup:
	uv sync --extra dev

data:
	bash scripts/local/fetch_data.sh

# Train the detector locally on MPS. usage: make train FOLD=0 EPOCHS=12
train:
	python scripts/local/train_detector.py --raw $(RAW) --gt $(GT) --fold $(FOLD) --epochs $(EPOCHS)

# Classical baseline pipeline over train volumes (blob -> link -> score).
baseline:
	python scripts/local/run_baseline.py --root $(dir $(RAW)) --gt-root $(GT) --split train --limit 5

lint:
	ruff check src tests scripts

test:
	pytest -q

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
