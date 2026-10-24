from __future__ import annotations

import argparse

from truthlens.eval.harness import evaluate_predictions, evaluate_videos, read_manifest
from truthlens.inference.pipeline import run_pipeline


def _pred_video(p: str) -> float:
    res = run_pipeline(p)
    return float(res["ai_probability"]) / 100.0


def main():
    parser = argparse.ArgumentParser(description="TruthLens benchmark (AUC/PR/FPR/ECE)")
    parser.add_argument("--manifest", help="CSV/JSON with columns/keys: path,label")
    parser.add_argument("--real", nargs="+", help="paths to real videos")
    parser.add_argument("--fake", nargs="+", help="paths to AI-generated videos")
    args = parser.parse_args()

    if args.manifest:
        items = read_manifest(args.manifest)
        report = evaluate_videos(items, predictor=_pred_video)
    else:
        y_true: list[int] = []
        y_pred: list[float] = []
        for rp in args.real or []:
            y_true.append(0)
            y_pred.append(_pred_video(rp))
        for fp in args.fake or []:
            y_true.append(1)
            y_pred.append(_pred_video(fp))
        if not y_true:
            print({"msg": "provide --manifest or --real/--fake"})
            return
        report = evaluate_predictions(y_true, y_pred)

    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
