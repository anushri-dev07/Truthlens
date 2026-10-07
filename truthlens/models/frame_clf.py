from __future__ import annotations

from pathlib import Path

import numpy as np


class FrameClassifier:
    def __init__(self, model_path: str | Path | None = None):
        self.model_path = model_path
        self.session = None
        if model_path and Path(model_path).exists():
            try:
                import onnxruntime as ort

                self.session = ort.InferenceSession(str(model_path))
            except Exception:
                self.session = None

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        # img HWC RGB 0-255
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[None, ...]  # NCHW
        return img

    def predict(self, img: np.ndarray) -> float:
        if not self.session:
            return 0.5
        input_name = self.session.get_inputs()[0].name
        x = self._preprocess(img)
        ort_outs = self.session.run(None, {input_name: x})
        out = ort_outs[0]
        if out.ndim == 2 and out.shape[1] == 2:
            p_ai = float(out[0, 1])
        else:
            p_ai = float(out.flatten()[0]) if out.size else 0.5
        return max(0.0, min(1.0, p_ai))
