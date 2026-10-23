import numpy as np
import pytest

from truthlens.signals import s2_temporal, s3_freq


class FakeCapture:
    def __init__(self, opened=True, fps=10.0, frame_count=10, frames=None):
        self.opened = opened
        self.fps = fps
        self.frame_count = frame_count
        self.frames = iter(frames or [])

    def isOpened(self):
        return self.opened

    def get(self, prop):
        if prop == s2_temporal.cv2.CAP_PROP_FPS:
            return self.fps
        if prop == s2_temporal.cv2.CAP_PROP_FRAME_COUNT:
            return self.frame_count
        return 0

    def set(self, _prop, _value):
        return True

    def read(self):
        try:
            return True, next(self.frames)
        except StopIteration:
            return False, None

    def release(self):
        pass


@pytest.mark.parametrize("signal", [s2_temporal, s3_freq])
def test_unopenable_video_is_failed(monkeypatch, signal):
    monkeypatch.setattr(signal.cv2, "VideoCapture", lambda _path: FakeCapture(opened=False))

    result = signal.score("unreadable.mp4")

    assert result.status == "failed"
    assert result.completed is False


@pytest.mark.parametrize("signal", [s2_temporal, s3_freq])
def test_missing_sampling_metadata_is_not_available(monkeypatch, signal):
    monkeypatch.setattr(
        signal.cv2,
        "VideoCapture",
        lambda _path: FakeCapture(fps=0.0, frame_count=10),
    )

    result = signal.score("unknown-duration.mp4")

    assert result.status == "not_available"
    assert result.completed is False


@pytest.mark.parametrize("signal", [s2_temporal, s3_freq])
def test_unreadable_sampled_frames_are_failed(monkeypatch, signal):
    monkeypatch.setattr(
        signal.cv2,
        "VideoCapture",
        lambda _path: FakeCapture(fps=10.0, frame_count=10),
    )

    result = signal.score("corrupt-frames.mp4")

    assert result.status == "failed"
    assert result.completed is False


def test_temporal_check_is_unavailable_with_only_one_decoded_frame(monkeypatch):
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    monkeypatch.setattr(
        s2_temporal.cv2,
        "VideoCapture",
        lambda _path: FakeCapture(fps=10.0, frame_count=1, frames=[frame]),
    )

    result = s2_temporal.score("single-frame.mp4")

    assert result.status == "not_available"
    assert result.evidence == ["no-temporal-pairs"]