#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def ok(msg: str) -> None:
    print(f"[OK]   {msg}")


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def check_exists(path: Path, kind: str) -> bool:
    if kind == "file" and path.is_file():
        ok(f"file exists: {path}")
        return True
    if kind == "dir" and path.is_dir():
        ok(f"dir exists: {path}")
        return True
    fail(f"missing {kind}: {path}")
    return False


def check_no_symlink(path: Path) -> bool:
    if not path.exists():
        return True
    if path.is_symlink():
        fail(f"legacy symlink still present: {path} -> {path.resolve()}")
        return False
    ok(f"not a symlink: {path}")
    return True


def import_module(module_path: Path) -> bool:
    spec = importlib.util.spec_from_file_location("repo_check_module", module_path)
    if spec is None or spec.loader is None:
        fail(f"cannot create import spec: {module_path}")
        return False
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ok(f"module import succeeded: {module_path}")
        return True
    except Exception as exc:  # noqa: BLE001
        fail(f"module import failed: {module_path} ({exc})")
        return False


def main() -> int:
    print("Running repository health checks...")
    checks_ok = True

    required_dirs = [
        ROOT / "src" / "inference",
        ROOT / "src" / "preprocess",
        ROOT / "src" / "experiments",
        ROOT / "api",
        ROOT / "app",
        ROOT / "data",
        ROOT / "outputs",
        ROOT / "docs",
        ROOT / "experiments",
    ]
    required_files = [
        ROOT / "emnist_cnn_corrected.pth",
        ROOT / "requirements.txt",
        ROOT / "api" / "main.py",
        ROOT / "src" / "inference" / "predict_yamfo_matt.py",
        ROOT / "src" / "inference" / "predict_town.py",
    ]

    for d in required_dirs:
        checks_ok &= check_exists(d, "dir")
    for f in required_files:
        checks_ok &= check_exists(f, "file")

    # Ensure legacy convenience links are gone for a clean slate.
    legacy_paths = [
        ROOT / "ind_study_emnist",
        ROOT / "matt-crop-attempt",
        ROOT / "Status_Report.md",
        ROOT / "src" / "predict_town.py",
        ROOT / "src" / "predict_yamfo_matt.py",
        ROOT / "src" / "segmenting",
        ROOT / "src" / "cropping",
        ROOT / "src" / "archive",
    ]
    for lp in legacy_paths:
        checks_ok &= check_no_symlink(lp)

    # Import and lightweight function check for inference module.
    inference_mod = ROOT / "src" / "inference" / "predict_yamfo_matt.py"
    if import_module(inference_mod):
        sys.path.insert(0, str(ROOT / "src" / "inference"))
        try:
            import predict_yamfo_matt as inference  # type: ignore

            _ = inference.AHAFO_TOWNS
            _ = inference.predict_letter
            ok("inference API symbols available")
        except Exception as exc:  # noqa: BLE001
            checks_ok = False
            fail(f"inference API symbols unavailable ({exc})")
    else:
        checks_ok = False

    print("")
    if checks_ok:
        print("Repository check: PASS")
        return 0
    print("Repository check: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
