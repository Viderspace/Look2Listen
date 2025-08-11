#!/usr/bin/env python3
# scripts/bootstrap.py
"""
Bootstrap Script for Environment Setup

Auto-installs project dependencies, choosing the correct requirements file:

- requirements-local.txt → Full local dev/demo environment
- requirements-colab.txt → Minimal Colab setup for training/inference

Detects environment, upgrades pip, installs packages, and prints key package/backend info.

Usage:
    Local:  python scripts/bootstrap.py
    Colab:  !python /content/avspeech_project/scripts/bootstrap.py
"""
import os
import sys
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(f"→ {' '.join(cmd)}")
    subprocess.check_call(cmd)


def detect_env() -> tuple[bool, Path, Path]:
    # Project root = parent of this scripts/ directory
    scripts_dir = Path(__file__).resolve().parent
    root = scripts_dir.parent

    # Detect Colab
    is_colab = (Path("/content").exists()) or ("google.colab" in sys.modules)

    req_name = "requirements-colab.txt" if is_colab else "requirements-local.txt"
    req_file = root / req_name
    if not req_file.exists():
        raise FileNotFoundError(
            f"Expected {req_name} at project root: {req_file}\n"
            f"Make sure you have both requirements-local.txt and requirements-colab.txt at {root}"
        )
    return is_colab, root, req_file


def sanity_report():
    print("\n=== Sanity report ===")

    def safe_import(name, attr=None):
        try:
            mod = __import__(name)
            if attr:
                for part in attr.split("."):
                    mod = getattr(mod, part)
            return mod
        except Exception as e:
            print(f"{name}: not available ({e.__class__.__name__}: {e})")
            return None

    np = safe_import("numpy", "version.version")
    if np: print(f"numpy: {np}")

    cv2 = safe_import("cv2")
    if cv2: print(f"opencv-python: {cv2.__version__}")

    torch = safe_import("torch")
    if torch:
        print(f"torch: {torch.__version__}")
        # Backend info
        mps = getattr(torch.backends, "mps", None)
        cuda = getattr(torch, "cuda", None)
        if mps and hasattr(mps, "is_available"):
            print(f"MPS available: {mps.is_available()}")
        if cuda and hasattr(cuda, "is_available"):
            print(f"CUDA available: {cuda.is_available()}")
            if cuda.is_available():
                print(f"CUDA device: {cuda.get_device_name(0)}")

    librosa = safe_import("librosa", "version.__version__")
    if librosa: print(f"librosa: {librosa}")

    print("=====================\n")


def main():
    is_colab, root, req_file = detect_env()
    print(f"Project root: {root}")
    print(f"Detected environment: {'Colab' if is_colab else 'Local'}")
    print(f"Using requirements file: {req_file.name}\n")

    # Always upgrade pip first (helps resolver)
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

    # Install exact requirements
    run([sys.executable, "-m", "pip", "install", "-r", str(req_file)])

    sanity_report()
    print(f"✓ Installed deps from {req_file.name} (is_colab={is_colab})")

if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Command failed with exit code {e.returncode}: {e.cmd}", file=sys.stderr)
        sys.exit(e.returncode)
    except Exception as e:
        print(f"\n✗ {e.__class__.__name__}: {e}", file=sys.stderr)
        sys.exit(1)