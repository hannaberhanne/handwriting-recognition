#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    inference_dir = root / "src" / "inference"
    sys.path.insert(0, str(inference_dir))

    import predict_yamfo_matt as inference  # type: ignore

    sample_img = root / "experiments" / "ind_study_emnist" / "letters" / "H.jpg"
    if not sample_img.is_file():
        print(f"Sample image not found: {sample_img}")
        return 1

    result = inference.predict_letter_with_details(str(sample_img))
    tensor = result["tensor"]
    letter = result["letter"]
    probs = result["probs"]
    if tensor is None or probs is None:
        print("Inference failed: image could not be processed.")
        return 1
    labels = [chr(i + 65) for i in range(26)]
    top_idx = probs.argsort()[::-1][:3]
    top3 = [
        {"label": labels[int(i)], "confidence": round(float(probs[int(i)]), 4)}
        for i in top_idx
    ]

    payload = {
        "image": str(sample_img.relative_to(root)),
        "predicted_letter": letter,
        "top3": top3,
    }

    outputs_dir = root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    out_file = outputs_dir / "test_prediction.json"
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Inference test passed. Wrote: {out_file}")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
