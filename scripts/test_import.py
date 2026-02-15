#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    inference_dir = root / "src" / "inference"
    sys.path.insert(0, str(inference_dir))

    try:
        import predict_yamfo_matt as inference  # type: ignore
    except Exception as exc:  # noqa: BLE001
        print(f"Import failed: {exc}")
        return 1

    outputs_dir = root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    out_file = outputs_dir / "import_test.txt"
    out_file.write_text(
        "Import test worked: predict_yamfo_matt loaded and model is available.\n",
        encoding="utf-8",
    )

    print(f"Import test passed. Wrote: {out_file}")
    # Touch symbols so this actually verifies module shape.
    _ = inference.model
    _ = inference.predict_letter
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
