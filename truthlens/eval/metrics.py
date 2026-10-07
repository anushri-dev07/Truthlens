from __future__ import annotations


def _accuracy(y_true: list[int], y_pred: list[float], thr: float = 0.5) -> float:
    if not y_true:
        return 0.0
    y_hat = [1 if p >= thr else 0 for p in y_pred]
    correct = sum(1 for a, b in zip(y_true, y_hat) if a == b)
    return correct / len(y_true)


def _auc_roc(y_true: list[int], y_pred: list[float]) -> float:
    if not y_true:
        return 0.5
    pairs = sorted(zip(y_pred, y_true), key=lambda x: x[0])
    pos = sum(y_true)
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return 0.5
    tpr = 0.0
    fpr = 0.0
    prev_t = 0.0
    prev_f = 0.0
    prev_s = -1.0
    auc = 0.0
    for score_v, lbl in pairs:
        if lbl == 1:
            tpr += 1.0 / pos
        else:
            fpr += 1.0 / neg
        if score_v != prev_s and prev_s >= 0:
            auc += (fpr - prev_f) * (tpr + prev_t) / 2.0
            prev_t, prev_f, prev_s = tpr, fpr, score_v
    auc += (fpr - prev_f) * (tpr + prev_t) / 2.0
    return float(max(0.0, min(1.0, auc)))
