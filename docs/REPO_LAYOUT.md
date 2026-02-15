# Repo Layout

## Canonical Paths

- `src/inference/`
  - Production inference scripts.
  - Primary entrypoint: `src/inference/predict_town.py`.

- `src/preprocess/`
  - PDF conversion, segmentation, and cropping utilities.

- `src/experiments/`
  - Archived script experiments that are not production paths.

- `api/`
  - FastAPI service wired to model inference.

- `app/`
  - React/Vite frontend.

- `data/`
  - Runtime inputs and OCR outputs.

- `experiments/`
  - Top-level research/training sandboxes moved out of root:
    - `experiments/ind_study_emnist/`
    - `experiments/matt-crop-attempt/`

- `outputs/figures/`
  - Generated debug/result images (moved out of root clutter).

- `docs/research/`
  - Research notes and status report.
