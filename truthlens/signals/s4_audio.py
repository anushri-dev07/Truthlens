from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from truthlens.data.io import probe_video
from truthlens.signals.base import SignalScore


def _extract_audio(video_path: str | Path, out_wav: str) -> bool:
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(video_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                out_wav,
                "-y",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def score(video_path: str | Path) -> SignalScore:
    info = probe_video(video_path)
    streams = info.get("streams", []) if isinstance(info, dict) else []
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    if not has_audio:
        return SignalScore("s4_audio", prob_ai=0.5, evidence=["no-audio-stream"])
    fd, wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        ok = _extract_audio(video_path, wav)
        if not ok or not os.path.exists(wav) or os.path.getsize(wav) == 0:
            return SignalScore("s4_audio", prob_ai=0.5, evidence=["audio-extract-failed"])
        try:
            import librosa

            y, sr = librosa.load(wav, sr=16000)
            if len(y) < 512:
                return SignalScore("s4_audio", prob_ai=0.5, evidence=["audio-too-short"])
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_std = float(np.mean(mfcc.std(axis=1)))
            zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
            spec_flat = float(np.mean(librosa.feature.spectral_flatness(y=y)))
            p = 0.5 + (mfcc_std - 25.0) * 0.01 + (spec_flat - 0.1) * 0.5
            p = float(np.clip(p, 0.25, 0.85))
            evidence = [
                f"s4_mfccstd={mfcc_std:.2f}",
                f"s4_zcr={zcr:.4f}",
                f"s4_flat={spec_flat:.4f}",
            ]
            return SignalScore("s4_audio", prob_ai=p, evidence=evidence)
        except Exception:
            return SignalScore("s4_audio", prob_ai=0.5, evidence=["librosa-unavailable"])
    finally:
        try:
            os.unlink(wav)
        except Exception:
            pass
