#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

from fastapi.testclient import TestClient


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    from api.main import app

    sample_img = root / "experiments" / "ind_study_emnist" / "letters" / "H.jpg"
    if not sample_img.is_file():
        print(f"Sample image not found: {sample_img}")
        return 1

    client = TestClient(app)
    with sample_img.open("rb") as f:
        response = client.post(
            "/predict",
            files={"file": (sample_img.name, f, "image/jpeg")},
        )

    if response.status_code != 200:
        print(f"API predict failed: {response.status_code} {response.text}")
        return 1

    data = response.json()
    outputs_dir = root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    out_file = outputs_dir / "test_api_predict.json"
    out_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    print(f"API test passed. Wrote: {out_file}")
    print(json.dumps(data, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
