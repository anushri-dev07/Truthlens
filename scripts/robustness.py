from __future__ import annotations

import argparse
import json

from truthlens.eval.robustness import run_robustness


def main():
    parser = argparse.ArgumentParser(description="Verdict stability under transforms")
    parser.add_argument("--video", required=True, help="path to a video file")
    args = parser.parse_args()
    report = run_robustness(args.video)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
