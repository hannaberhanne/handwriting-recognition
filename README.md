# Handwriting Recognition (Ghana OCR)

This repo currently has two tracks:
- A working ML inference pipeline for letter-level recognition (`src/inference/predict_yamfo_matt.py`).
- A web app stack (`api/` + `app/`) with API inference wired to the ML model.

## Quick Start

1. Create and activate env:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install deps:
```bash
pip install -r requirements.txt
```

3. Run a town prediction from `data/<folder>`:
```bash
python src/inference/predict_town.py --folder YAMFO
python src/inference/predict_town.py --folder Bechem
python src/inference/predict_town.py --folder Ataneata
```

## What Works Right Now

- `src/inference/predict_yamfo_matt.py`
  - Best current inference path.
  - Loads `emnist_cnn_corrected.pth`.
  - Uses multi-binarization + TTA + O/Q/D disambiguation + lexicon match.

- `src/inference/predict_town.py`
  - Unified runner for any `data/<folder>` containing letter PNGs.

- `src/inference/predict_bechem.py`, `src/inference/predict_Ataneata.py`
  - Thin wrappers over `src/inference/predict_town.py`.

## Legacy / Experimental

- `experiments/ind_study_emnist/`
  - Training and experiments. Useful for research, not deployment.
- `experiments/matt-crop-attempt/`, `src/experiments/archive/`
  - Older segmentation/cropping experiments.
- `src/inference/predict_yamfo_letters.py`
  - Alternative inference approach (kept for comparison).

## Project Layout (Current)

```text
handwriting-recognition/
├── api/                    # FastAPI app scaffold (currently stub predictions)
├── app/                    # React/Vite frontend scaffold
├── data/                   # Input letters, raw scans, processed outputs
├── experiments/            # Research/training sandboxes moved out of root
├── src/
│   ├── inference/
│   ├── preprocess/
│   └── experiments/
├── outputs/                # generated figures/debug images
├── docs/research/          # notes and status reports
├── emnist_cnn_corrected.pth
└── requirements.txt
```

Detailed path map: `docs/REPO_LAYOUT.md`.

## API

Run the backend:
```bash
uvicorn api.main:app --reload --port 8000
```

The `POST /predict` endpoint now calls the real model via `src/inference/predict_yamfo_matt.py` and returns top-3 predictions.

## Run It (Quick)

From repo root:

```bash
source .venv/bin/activate
uvicorn api.main:app --reload --port 8000
```

In a second terminal for the frontend:

```bash
cd app
npm run dev
```

Useful links:
- API docs: `http://localhost:8000/docs`
- Frontend: `http://localhost:5173`
