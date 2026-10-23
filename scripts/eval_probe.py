"""Repeated video-disjoint evaluation of the frame probe.

Reports held-out performance across many random splits, where a video is
never present in both the training and the test portion. This is the honest
generalisation estimate for the detector; a single split on 61 videos is far
too noisy to quote on its own.

Usage:
    python scripts/eval_probe.py --embeddings artifacts/emb_probe_cf2/embeddings.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from truthlens.eval.metrics import _auc_roc


def _make_clf():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced"),
    )


def _video_scores(prob, groups, y):
    acc: dict = {}
    for g, p, t in zip(groups, prob, y):
        acc.setdefault((int(g), int(t)), []).append(p)
    return [k[1] for k in acc], [float(np.mean(v)) for v in acc.values()]


def evaluate(embeddings: str, splits: int = 30, test_size: float = 0.3, seed: int = 42):
    d = np.load(embeddings)
    X, y, g = d["X"], d["y"], d["groups"]
    frame_aucs, video_aucs = [], []
    for s in range(splits):
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=s)
        tr, te = next(splitter.split(np.arange(len(y)), y, groups=g))
        mask = np.zeros(len(y), dtype=bool)
        mask[te] = True
        clf = _make_clf().fit(X[~mask], y[~mask])
        prob = clf.predict_proba(X[mask])[:, 1]
        frame_aucs.append(_auc_roc(list(y[mask]), list(prob)))
        vy, vp = _video_scores(prob, g[mask], y[mask])
        video_aucs.append(_auc_roc(vy, vp))
    return {
        "splits": splits,
        "test_size": test_size,
        "n_videos": int(len(set(int(x) for x in g))),
        "n_frames": int(len(y)),
        "held_out_frame_auc": {
            "mean": round(float(np.mean(frame_aucs)), 4),
            "median": round(float(np.median(frame_aucs)), 4),
            "min": round(float(np.min(frame_aucs)), 4),
            "max": round(float(np.max(frame_aucs)), 4),
            "std": round(float(np.std(frame_aucs)), 4),
        },
        "held_out_video_auc": {
            "mean": round(float(np.mean(video_aucs)), 4),
            "median": round(float(np.median(video_aucs)), 4),
            "min": round(float(np.min(video_aucs)), 4),
            "max": round(float(np.max(video_aucs)), 4),
            "std": round(float(np.std(video_aucs)), 4),
        },
        "splits_above_chance": int(sum(1 for v in video_aucs if v > 0.5)),
        "note": (
            "Every split is video-disjoint: a video is never in both the "
            "training and the test portion."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Repeated held-out evaluation of the probe")
    parser.add_argument("--embeddings", default="artifacts/emb_probe_cf2/embeddings.npz")
    parser.add_argument("--splits", type=int, default=30)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--out", default="artifacts/probe_eval.json")
    args = parser.parse_args()
    report = evaluate(args.embeddings, args.splits, args.test_size)
    print(json.dumps(report, indent=2))
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
