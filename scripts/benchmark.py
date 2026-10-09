from __future__ import annotations

import argparse
from pathlib import Path

from truthlens.eval.calibration import expected_calibration_error
from truthlens.eval.metrics import _accuracy, _auc_roc
from truthlens.inference.pipeline import run_pipeline


def _pred_video(p: Path) -> float:
    res = run_pipeline(p)
    return float(res["ai_probability"]) / 100.0


def main():
    p = argparse.ArgumentParser(description="TruthLens benchmark (with ECE)")
    p.add_argument("--real", nargs="+", required=False)
    p.add_argument("--fake", nargs="+", required=False)
    args = p.parse_args()
    y_true: list[int] = []
    y_pred: list[float] = []
    for rp in args.real or []:
        y_true.append(0)
        y_pred.append(_pred_video(rp))
    for fp in args.fake or []:
        y_true.append(1)
        y_pred.append(_pred_video(fp))
    if y_true:
        print({
            "auc": round(_auc_roc(y_true, y_pred), 4),
            "acc": round(_accuracy(y_true, y_pred), 4),
            "ece": round(expected_calibration_error(y_true, y_pred), 4),
        })
    else:
        print({"msg": "no videos provided"})


if __name__ == "__main__":
    main()
