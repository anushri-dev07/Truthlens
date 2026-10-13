from __future__ import annotations

import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Download ONNX models for TruthLens")
    p.add_argument("--out", default="artifacts/models", help="output dir")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Lightweight placeholder: create a dummy onnx filename expected? or skip
    print("Models dir ready:", out.resolve())


if __name__ == "__main__":
    main()
