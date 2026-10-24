from __future__ import annotations

import base64
import threading
from pathlib import Path

import cv2

from truthlens.config import sample_frames as _cfg_sample_frames
from truthlens.data.faces import extract_faces
from truthlens.data.video import uniform_sample_indices
from truthlens.signals.base import SignalScore

FACE_FOUND_P = 0.4
NO_FACE_P = 0.6

PREVIEW_MAX_WIDTH = 480
PREVIEW_JPEG_QUALITY = 70

_clf = None
_clf_method: str | None = None
_clf_lock = threading.Lock()


def _classifier():
    global _clf, _clf_method
    if _clf is None:
        with _clf_lock:
            if _clf is None:
                from truthlens.models.frame_clf import get_classifier

                _clf, _clf_method = get_classifier()
    return _clf, _clf_method


def _preview_data_url(frame) -> str | None:
    """Encode a sampled frame as a small JPEG data URL for the evidence viewer."""
    try:
        h, w = frame.shape[:2]
        if w > PREVIEW_MAX_WIDTH:
            scale = PREVIEW_MAX_WIDTH / float(w)
            frame = cv2.resize(
                frame, (PREVIEW_MAX_WIDTH, max(1, int(h * scale))), interpolation=cv2.INTER_AREA
            )
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), PREVIEW_JPEG_QUALITY])
        if not ok:
            return None
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        return None


def _fps_duration(cap: cv2.VideoCapture) -> tuple[float, float]:
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    duration = total_frames / fps if fps > 0 else 0.0
    return float(fps), float(duration)


def score(
    video_path: str | Path,
    sample_frames: int | None = None,
    sampled: dict | None = None,
) -> SignalScore:
    if sample_frames is None:
        sample_frames = _cfg_sample_frames()
    cap = None
    if sampled is None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            return SignalScore(
                "s1_frame",
                prob_ai=0.5,
                evidence=["cannot-open-video"],
                status="failed",
            )
        fps, duration = _fps_duration(cap)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        idxs = uniform_sample_indices(duration, fps, sample_frames)
    else:
        fps = float(sampled.get("fps") or 0.0)
        duration = float(sampled.get("duration") or 0.0)
        total_frames = int(sampled.get("total_frames") or 0)
        idxs = uniform_sample_indices(duration, fps, sample_frames)
        first_frame = next(
            (
                sampled["frames"].get(index)
                for index in idxs
                if sampled["frames"].get(index) is not None
            ),
            None,
        )
        height, width = first_frame.shape[:2] if first_frame is not None else (0, 0)
    clf, clf_method = _classifier()
    use_model = clf.available
    if use_model:
        method = clf_method or "cnn_vit"
    else:
        method = "face-heuristic"
    timeline: list[dict] = []
    frame_probs: list[float] = []
    crops_count = 0

    for i in idxs:
        if sampled is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
        else:
            frame = sampled["frames"].get(i)
            ret = frame is not None
        t = round(i / fps, 3) if fps > 0 else 0.0
        if not ret or frame is None:
            timeline.append({"t": t, "score": 0.5, "processed": False, "frame": i})
            continue
        faces = extract_faces(frame)
        crops_count += len(faces)
        if use_model:
            if faces:
                x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                crop = frame[y : y + h, x : x + w]
            else:
                crop = frame
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            p = clf.predict(rgb)
        else:
            p = FACE_FOUND_P if faces else NO_FACE_P
        frame_probs.append(p)
        timeline.append(
            {
                "t": t,
                "score": round(p, 4),
                "processed": True,
                "frame": i,
                "faces": len(faces),
                "preview": _preview_data_url(frame),
            }
        )

    if cap is not None:
        cap.release()
    if not frame_probs:
        return SignalScore(
            "s1_frame",
            prob_ai=0.5,
            evidence=["frames-unreadable"],
            status="failed",
            details={"timeline": timeline},
        )

    p = float(sum(frame_probs) / len(frame_probs))
    processed = [t for t in timeline if t.get("processed")]
    sampled_times = [t["t"] for t in processed]
    sampling_interval = None
    if len(sampled_times) > 1:
        gaps = [sampled_times[i + 1] - sampled_times[i] for i in range(len(sampled_times) - 1)]
        sampling_interval = round(sum(gaps) / len(gaps), 3)
    elif len(sampled_times) == 1:
        sampling_interval = 0.0
    evidence = [
        f"s1_method={method}",
        f"s1_frames={len(frame_probs)}",
        f"s1_faces={crops_count}",
    ]
    return SignalScore(
        "s1_frame",
        prob_ai=p,
        evidence=evidence,
        status="completed",
        details={
            "timeline": timeline,
            "faces": crops_count,
            "method": method,
            "video": {
                "duration_sec": round(duration, 3),
                "fps": round(fps, 3),
                "total_frames": int(total_frames),
                "width": int(width),
                "height": int(height),
            },
            "sampling": {
                "requested_frames": sample_frames,
                "scored_frames": len(frame_probs),
                "interval_sec": sampling_interval,
                "sampled_times": sampled_times,
            },
        },
    )
