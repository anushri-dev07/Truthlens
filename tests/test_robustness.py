from __future__ import annotations

import os
import tempfile


def _make_video(path: str, n: int = 12, size: int = 48) -> None:
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (size, size))
    for i in range(n):
        writer.write(np.ones((size, size, 3), np.uint8) * ((i * 9) % 255))
    writer.release()


def test_robustness_runs():
    from truthlens.eval.robustness import run_robustness

    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        _make_video(path)
        calls = {"n": 0}

        def fake_predictor(p: str) -> float:
            calls["n"] += 1
            return 0.5

        report = run_robustness(path, predictor=fake_predictor)
        assert report["baseline"] == 0.5
        assert report["stable"] is True
        assert report["max_delta"] == 0.0
    finally:
        os.unlink(path)
