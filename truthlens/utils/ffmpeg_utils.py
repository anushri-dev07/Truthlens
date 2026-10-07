import subprocess


def has_ffprobe() -> bool:
    try:
        subprocess.run(
            ["ffprobe", "-version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False
