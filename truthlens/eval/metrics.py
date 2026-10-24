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
    pos = sum(1 for t in y_true if t == 1)
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return 0.5
    pairs = sorted(zip(y_pred, y_true), key=lambda x: x[0])
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    sum_pos_ranks = sum(r for r, (_, lbl) in zip(ranks, pairs) if lbl == 1)
    auc = (sum_pos_ranks - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(max(0.0, min(1.0, auc)))


def confusion_matrix(
    y_true: list[int], y_pred: list[float], thr: float = 0.5
) -> dict[str, int]:
    tp = fp = tn = fn = 0
    for t, p in zip(y_true, y_pred):
        pred = 1 if p >= thr else 0
        if t == 1 and pred == 1:
            tp += 1
        elif t == 0 and pred == 1:
            fp += 1
        elif t == 0 and pred == 0:
            tn += 1
        else:
            fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def precision_recall_f1(y_true: list[int], y_pred: list[float], thr: float = 0.5) -> dict:
    c = confusion_matrix(y_true, y_pred, thr)
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def false_positive_rate(y_true: list[int], y_pred: list[float], thr: float = 0.5) -> float:
    c = confusion_matrix(y_true, y_pred, thr)
    denom = c["fp"] + c["tn"]
    return c["fp"] / denom if denom else 0.0


def true_positive_rate(y_true: list[int], y_pred: list[float], thr: float = 0.5) -> float:
    c = confusion_matrix(y_true, y_pred, thr)
    denom = c["tp"] + c["fn"]
    return c["tp"] / denom if denom else 0.0


def best_threshold(y_true: list[int], y_pred: list[float]) -> dict:
    """Pick threshold maximizing Youden's J (tpr - fpr)."""
    candidates = sorted({round(p, 4) for p in y_pred})
    if not candidates:
        return {"threshold": 0.5, "youden_j": 0.0}
    best = {"threshold": 0.5, "youden_j": -1.0}
    for thr in candidates:
        j = true_positive_rate(y_true, y_pred, thr) - false_positive_rate(y_true, y_pred, thr)
        if j > best["youden_j"]:
            best = {"threshold": float(thr), "youden_j": float(j)}
    return best
