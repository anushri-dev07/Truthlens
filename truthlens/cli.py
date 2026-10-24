# Simple CLI to analyze a video
from __future__ import annotations

import argparse

from truthlens.inference.pipeline import run_pipeline


def main():
    p = argparse.ArgumentParser(description="TruthLens CLI")
    p.add_argument("video", help="Path to video file")
    args = p.parse_args()
    res = run_pipeline(args.video)
    print(res)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
