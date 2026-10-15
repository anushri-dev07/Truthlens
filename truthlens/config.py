from __future__ import annotations

from functools import lru_cache
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")

_DEFAULTS = {
    "weights": {
        "s1_frame": 0.30,
        "s2_temporal": 0.25,
        "s3_freq": 0.20,
        "s4_audio": 0.10,
        "s5_meta": 0.10,
        "s6_semantic": 0.05,
    },
    "sample_frames": 8,
    "thresholds": {
        "ai": 70.0,
        "real": 30.0,
        "min_signals": 2,
        "strong_ai": 65.0,
        "strong_real": 35.0,
        "stable_delta": 0.15,
    },
    "fusion_model": "artifacts/models/fusion_lr.pkl",
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = {k: (v.copy() if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None) -> dict:
    """Load config from YAML, merged over defaults. Missing file -> defaults."""
    cfg = dict(_deep_merge(_DEFAULTS, {}))
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if cfg_path.exists():
        try:
            import yaml

            data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            cfg = _deep_merge(cfg, data)
        except Exception:
            pass
    return cfg


@lru_cache(maxsize=1)
def get_config() -> dict:
    return load_config()


def weights() -> dict:
    return get_config()["weights"]


def thresholds() -> dict:
    return get_config()["thresholds"]


def sample_frames() -> int:
    return int(get_config()["sample_frames"])


def fusion_model_path() -> Path:
    return Path(get_config()["fusion_model"])
