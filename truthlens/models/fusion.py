from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from truthlens.config import fusion_model_path, weights


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + float(np.exp(-x)))


def _conf_from_p(p: float) -> str:
    if p < 0.4:
        return "low"
    if p < 0.7:
        return "medium"
    return "high"


def _extract_vector(scores: list[dict]) -> np.ndarray:
    order = ["s1_frame", "s2_temporal", "s3_freq", "s4_audio", "s5_meta", "s6_semantic"]
    return np.array(
        [float(next((s["prob_ai"] for s in scores if s["name"] == k), 0.5)) for k in order],
        dtype=float,
    )


class Fusion:
    def __init__(self, weights_map: dict[str, float] | None = None, clf_path: Path | None = None):
        self.weights = weights_map or weights()
        self.clf_path = Path(clf_path) if clf_path else fusion_model_path()
        self.clf: LogisticRegression | None = None
        if self.clf_path.exists():
            try:
                import joblib

                self.clf = joblib.load(self.clf_path)
            except Exception:
                self.clf = None

    def predict_proba_pct(self, scores: list[dict]) -> float:
        active = [s for s in scores if s.get("status", "completed") == "completed"]
        if self.clf is not None and len(active) == len(scores):
            X = _extract_vector(scores).reshape(1, -1)
            try:
                p = float(self.clf.predict_proba(X)[0, 1])
                return float(np.clip(p * 100.0, 0.0, 100.0))
            except Exception:
                pass
        # weighted logit over completed signals only (weights renormalized)
        logit = 0.0
        total_w = 0.0
        for s in active:
            name = s.get("name")
            w = float(self.weights.get(name or "", 0.0))
            if w <= 0:
                continue
            p = float(s.get("prob_ai", 0.5))
            eps = 1e-6
            p_cl = max(eps, min(1 - eps, p))
            logit += w * float(np.log(p_cl / (1 - p_cl)))
            total_w += w
        if total_w <= 0:
            return 50.0
        return float(_sigmoid(logit / total_w)) * 100.0
