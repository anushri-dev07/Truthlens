from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path

MODELS = {
    "community_forensics": {
        "repo": "onnx-community/CommunityForensics-DeepfakeDet-ViT-ONNX",
        "files": {
            "onnx/model_int8.onnx": "model_int8.onnx",
            "config.json": "config.json",
            "preprocessor_config.json": "preprocessor_config.json",
        },
    },
    "deepfake_v2": {
        "repo": "onnx-community/Deep-Fake-Detector-v2-Model-ONNX",
        "files": {
            "onnx/model_int8.onnx": "model_int8.onnx",
            "config.json": "config.json",
            "preprocessor_config.json": "preprocessor_config.json",
        },
    },
}


def _download(url: str, dest: Path, retries: int = 4) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "truthlens"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                dest.write_bytes(resp.read())
            if dest.stat().st_size > 0:
                return True
        except Exception:
            time.sleep(2 * attempt)
    return False


def main():
    parser = argparse.ArgumentParser(description="Download ONNX models for TruthLens")
    parser.add_argument("--out", default="artifacts/models")
    parser.add_argument("--model", choices=list(MODELS), default="community_forensics")
    args = parser.parse_args()
    spec = MODELS[args.model]
    base = f"https://huggingface.co/{spec['repo']}/resolve/main/"
    out = Path(args.out) / args.model
    for remote, local in spec["files"].items():
        dest = out / local
        if dest.exists() and dest.stat().st_size > 0:
            print(f"skip {dest}")
            continue
        print(f"downloading {remote} -> {dest}")
        if not _download(base + remote, dest):
            print(f"FAILED {remote}")
    print("done:", out.resolve())


if __name__ == "__main__":
    main()
