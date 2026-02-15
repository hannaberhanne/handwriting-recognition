# Working Status

## Production Candidate

- `src/inference/predict_yamfo_matt.py`
  - Current best inference script.
  - Depends on `emnist_cnn_corrected.pth`.

## Working Entry Points

- `python src/inference/predict_town.py --folder <FolderName>`
- `python src/inference/predict_bechem.py`
- `python src/inference/predict_Ataneata.py`
- `uvicorn api.main:app --reload --port 8000`

## Known Gaps

- Folder-based predictions require actual PNG letters under `data/<folder>`.
- `src/inference/predict_Edaso.py` is empty.
- Inference code is still script-shaped and should be extracted into a dedicated module for cleaner API integration.

## Refactor Targets

1. Extract reusable inference functions into a dedicated module (instead of importing script internals).
2. Add pytest smoke tests for model load and one-image prediction through API.
3. Add a stable response schema with confidence calibration notes.
