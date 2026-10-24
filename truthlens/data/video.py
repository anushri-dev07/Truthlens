from __future__ import annotations

import math
from typing import List


def uniform_sample_indices(duration_sec: float, fps: float, target_frames: int) -> List[int]:
    if (
        not math.isfinite(duration_sec)
        or not math.isfinite(fps)
        or duration_sec <= 0
        or fps <= 0
        or target_frames <= 0
    ):
        return []
    total = math.floor(duration_sec * fps)
    if total <= 0:
        return []
    if target_frames >= total:
        return list(range(total))
    if target_frames == 1:
        return [0]
    return [round(i * (total - 1) / (target_frames - 1)) for i in range(target_frames)]
