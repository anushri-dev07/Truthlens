from __future__ import annotations
import json
import subprocess
from pathlib import Path


def probe_video(path: str | Path) -> dict:
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError, Exception):
        return {}
    try:
        return json.loads(res.stdout) or {}
    except Exception:
        return {}
