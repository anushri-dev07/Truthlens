from __future__ import annotations


def expected_calibration_error(y_true: list[int], y_pred: list[float], n_bins: int = 10) -> float:
    if not y_true:
        return 0.0
    bins = [[] for _ in range(n_bins)]
    for y, p in zip(y_true, y_pred):
        idx = int(p * n_bins)
        if idx == n_bins:
            idx = n_bins - 1
        bins[idx].append((y, p))
    ece = 0.0
    m = len(y_true)
    for b in bins:
        if not b:
            continue
        acc = sum(y for y, _ in b) / len(b)
        conf = sum(p for _, p in b) / len(b)
        ece += (len(b) / m) * abs(acc - conf)
    return float(ece)
