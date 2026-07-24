COMP := biohub-cell-tracking-during-development

.PHONY: setup data score lint test clean

setup:
	uv sync --extra dev

data:
	bash scripts/fetch_data.sh

# Score a CTC-format prediction directory against ground truth.
# usage: make score GT=data/raw/<seq>_GT PRED=outputs/<run>/RES
score:
	python -m ctrack.eval.ctc --gt $(GT) --pred $(PRED)

lint:
	ruff check src tests

test:
	pytest -q

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
