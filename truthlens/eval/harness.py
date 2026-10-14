from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path

from truthlens.eval.calibration import expected_calibration_error
from truthlens.eval.metrics import (
    _accuracy,
    _auc_roc,
    best_threshold,
    confusion_matrix,
    false_positive_rate,
    precision_recall_f1,
    true_positive_rate,
)


def read_manifest(path: str | Path) -> list[tuple[str, int]]:
    """Read a manifest of (video_path, label).

    Supports CSV with columns `path,label` (label: 1=AI, 0=real) or a JSON
    list of {"path": ..., "label": ...} objects.
    """
    p = Path(path)
    items: list[tuple[str, int]] = []
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        for row in data:
            items.append((str(row["path"]), int(row["label"])))
        return items
    with p.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append((str(row["path"]), int(row["label"])))
    return items


def default_predictor(video_path: str) -> float:
    from truthlens.inference.pipeline import run_pipeline

    res = run_pipeline(video_path)
    return float(res["ai_probability"]) / 100.0


def evaluate_predictions(
    y_true: list[int], y_pred: list[float], decisions: list[str] | None = None
) -> dict:
    """Compute the full report; optionally account for inconclusive decisions.

    If `decisions` is given (values "likely_ai"/"likely_real"/"inconclusive"),
    coverage = fraction not inconclusive and selective_accuracy = accuracy over
    decided cases only.
    """
    report: dict = {
        "n": len(y_true),
        "positives": sum(y_true),
        "negatives": len(y_true) - sum(y_true),
        "auc": round(_auc_roc(y_true, y_pred), 4),
        "accuracy": round(_accuracy(y_true, y_pred), 4),
        "ece": round(expected_calibration_error(y_true, y_pred), 4),
        "confusion@0.5": confusion_matrix(y_true, y_pred, 0.5),
        "tpr@0.5": round(true_positive_rate(y_true, y_pred, 0.5), 4),
        "fpr@0.5": round(false_positive_rate(y_true, y_pred, 0.5), 4),
        "prf@0.5": {k: round(v, 4) for k, v in precision_recall_f1(y_true, y_pred, 0.5).items()},
        "best_threshold": best_threshold(y_true, y_pred),
    }
    if decisions is not None:
        decided = [
            (t, p)
            for t, p, d in zip(y_true, y_pred, decisions)
            if d in ("likely_ai", "likely_real")
        ]
        report["coverage"] = round(len(decided) / len(y_true), 4) if y_true else 0.0
        report["selective_accuracy"] = (
            round(
                sum(1 for t, p in decided if (p >= 0.5) == bool(t)) / len(decided),
                4,
            )
            if decided
            else None
        )
    return report


def evaluate_videos(
    items: list[tuple[str, int]],
    predictor: Callable[[str], float] = default_predictor,
    decider: Callable[[float], str] | None = None,
) -> dict:
    y_true: list[int] = []
    y_pred: list[float] = []
    decisions: list[str] | None = [] if decider else None
    for path, label in items:
        p = predictor(path)
        y_true.append(label)
        y_pred.append(p)
        if decisions is not None and decider is not None:
            decisions.append(decider(p))
    return evaluate_predictions(y_true, y_pred, decisions)
