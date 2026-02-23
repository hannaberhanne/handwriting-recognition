from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime
import shutil
import sys

ROOT_DIR = (Path(__file__).parent / "..").resolve()
INFERENCE_DIR = ROOT_DIR / "src" / "inference"
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

import predict_yamfo_matt as inference

app = FastAPI(title="Ghana OCR API", version="0.1")

# CORS for local dev (Vite default runs on 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = (Path(__file__).parent / ".." / "data").resolve()
RAW_UPLOADS = DATA_DIR / "raw_uploads"
RAW_UPLOADS.mkdir(parents=True, exist_ok=True)


class Prediction(BaseModel):
    label: str
    confidence: float


class PredictResponse(BaseModel):
    filename: str
    predictions: list[Prediction]


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    # Save upload
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = file.filename.replace(" ", "_")
    out_path = RAW_UPLOADS / f"{ts}_{safe_name}"

    with out_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = inference.predict_letter_with_details(str(out_path))
    tensor = result["tensor"]
    probs = result["probs"]
    if tensor is None or probs is None:
        raise HTTPException(status_code=422, detail="Could not read/parse input image for OCR.")
    labels = [chr(i + 65) for i in range(26)]

    top_indices = probs.argsort()[::-1][:3]
    preds = [
        {
            "label": labels[int(idx)],
            "confidence": round(float(probs[int(idx)]), 4),
        }
        for idx in top_indices
    ]

    return {
        "filename": out_path.name,
        "predictions": preds
    }
