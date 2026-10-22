from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULT_MODEL_PATH = Path("artifacts/models/community_forensics/model_int8.onnx")
DEFAULT_PROBE_DIR = Path("artifacts/models/community_probe")
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]

_SESSIONS: dict[str, object] = {}
_HEADS: dict[str, object] = {}


def _run_session(session, x: np.ndarray) -> list:
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: x})


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + float(np.exp(-x)))


class FrameClassifier:
    """ONNX image classifier for deepfake detection.

    Supports both 2-logit (softmax) and 1-logit (sigmoid) heads. Preprocessing is
    read from the sibling ``preprocessor_config.json`` when present.
    """

    def __init__(self, model_path: str | Path | None = None):
        if model_path is None:
            try:
                from truthlens.config import frame_model_path

                model_path = frame_model_path()
            except Exception:
                model_path = DEFAULT_MODEL_PATH
        self.model_path = Path(model_path)
        self.session = None
        self.cfg = self._load_pre_cfg()
        key = str(self.model_path)
        if key in _SESSIONS:
            self.session = _SESSIONS[key]
        elif self.model_path.exists():
            try:
                import onnxruntime as ort

                sess = ort.InferenceSession(
                    str(self.model_path), providers=["CPUExecutionProvider"]
                )
                _SESSIONS[key] = sess
                self.session = sess
            except Exception:
                self.session = None

    def _load_pre_cfg(self) -> dict:
        p = self.model_path.parent / "preprocessor_config.json"
        cfg = {}
        if p.exists():
            try:
                cfg = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                cfg = {}
        size = cfg.get("size") or {}
        self._size = (
            int(size.get("width") or size.get("shortest_edge") or 384),
            int(size.get("height") or size.get("shortest_edge") or 384),
        )
        self._crop = int(cfg.get("crop_size") or 0)
        self._mean = np.array(cfg.get("image_mean") or DEFAULT_MEAN, dtype=np.float32)
        self._std = np.array(cfg.get("image_std") or DEFAULT_STD, dtype=np.float32)
        self._rescale = float(cfg.get("rescale_factor") or 1.0 / 255.0)
        self._resample = int(cfg.get("resample") or 1)
        return cfg

    @property
    def available(self) -> bool:
        return self.session is not None

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        import cv2

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        interp = cv2.INTER_CUBIC if self._resample == 3 else cv2.INTER_LINEAR
        img = cv2.resize(img, self._size, interpolation=interp)
        if self._crop and self._crop < img.shape[0]:
            h, w = img.shape[:2]
            top = (h - self._crop) // 2
            left = (w - self._crop) // 2
            img = img[top: top + self._crop, left: left + self._crop]
        x = img.astype(np.float32) * self._rescale
        x = (x - self._mean) / self._std
        return x.transpose(2, 0, 1)[None, ...]

    def predict(self, img_rgb: np.ndarray) -> float:
        """Return P(deepfake) for an HWC image (any size)."""
        if not self.session:
            return 0.5
        x = self._preprocess(img_rgb)
        out = _run_session(self.session, x)[0]
        logits = np.asarray(out, dtype=np.float32).reshape(-1)
        if logits.size >= 2:
            probs = _softmax(logits[:2])
            return float(np.clip(probs[1], 0.0, 1.0))
        val = float(logits[0]) if logits.size else 0.0
        return float(np.clip(_sigmoid(val), 0.0, 1.0))


class ProbeClassifier:
    """Linear probe on frozen ViT embeddings.

    Uses an ONNX model that exposes the pre-classifier (CLS) embedding plus a
    scikit-learn logistic head trained on labelled data. Adapts the decision
    boundary to the target domain; requires only CPU and trains in seconds.
    """

    def __init__(self, probe_dir: str | Path | None = None):
        if probe_dir is None:
            try:
                from truthlens.config import probe_dir as _cfg_probe

                probe_dir = _cfg_probe()
            except Exception:
                probe_dir = DEFAULT_PROBE_DIR
        self.dir = Path(probe_dir)
        self._base = FrameClassifier(self.dir / "embed_model.onnx")
        self.session = self._base.session
        self._head = None
        self._emb_index = 0
        head_path = self.dir / "probe.pkl"
        if self.session is not None and head_path.exists():
            key = str(head_path)
            if key in _HEADS:
                self._head = _HEADS[key]
            else:
                try:
                    import joblib

                    _HEADS[key] = joblib.load(head_path)
                    self._head = _HEADS[key]
                except Exception:
                    self._head = None
        if self.session is not None:
            outs = [o.name for o in self.session.get_outputs()]
            for i, name in enumerate(outs):
                if "logits" not in name:
                    self._emb_index = i
                    break

    @property
    def available(self) -> bool:
        return self.session is not None and self._head is not None

    def predict(self, img_rgb: np.ndarray) -> float:
        if not self.available:
            return 0.5
        x = self._base._preprocess(img_rgb)
        out = _run_session(self.session, x)[self._emb_index]
        emb = np.asarray(out, dtype=np.float32).reshape(1, -1)
        prob = self._head.predict_proba(emb)[0][1]
        return float(np.clip(prob, 0.0, 1.0))


def get_classifier(model_path: str | Path | None = None):
    """Return the best available S1 classifier: probe > ONNX model > None."""
    try:
        probe = ProbeClassifier()
        if probe.available:
            return probe, "cnn_probe"
    except Exception:
        pass
    clf = FrameClassifier(model_path)
    if clf.available:
        return clf, "cnn_vit"
    return clf, "face-heuristic"
