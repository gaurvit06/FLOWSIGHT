.PHONY: install sample train serve dashboard replay test lint experiments docker

install:
	pip install -r requirements.lock && pip install --no-deps -e .

sample:            ## synthetic dataset (no download needed)
	python scripts/make_sample_data.py

train:
	python -m nids.models.train --config configs/train.yaml

serve:
	uvicorn nids.api.main:app --reload --port 8000

dashboard:
	streamlit run dashboard/Overview.py

replay:
	python -m nids.capture.replay --n 500 --delay 0.05

test:
	pytest

lint:
	ruff check src tests dashboard scripts

experiments:       ## all report scripts except cross_dataset (needs --test-dir)
	python scripts/cv_report.py
	python scripts/feature_selection_study.py
	python scripts/imbalance_ablation.py
	python scripts/shap_global.py
	python scripts/zero_day_experiment.py

docker:
	docker compose up --build
