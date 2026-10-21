from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

DATASET = "gonnerthetooner/deepfake_video_data"
API = f"https://huggingface.co/api/datasets/{DATASET}/tree/main?recursive=true"
BASE = f"https://huggingface.co/datasets/{DATASET}/resolve/main/"

FOLDERS = {"real": "celebdf_v2/Celeb-real", "ai": "celebdf_v2/Celeb-synthesis"}


def list_files(folder: str) -> list[dict]:
    with urllib.request.urlopen(API, timeout=90) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [
        f
        for f in data
        if f.get("type") == "file"
        and f["path"].startswith(folder + "/")
        and f["path"].lower().endswith(".mp4")
    ]


def download(url: str, dest: Path, retries: int = 4) -> bool:
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "truthlens-eval"})
            with urllib.request.urlopen(req, timeout=180) as resp:
                dest.write_bytes(resp.read())
            if dest.stat().st_size > 0:
                return True
        except Exception:
            time.sleep(2 * attempt)
    return False


def main():
    parser = argparse.ArgumentParser(description="Fetch Celeb-DF eval subset")
    parser.add_argument("--per-class", type=int, default=40)
    parser.add_argument("--root", default="data/eval")
    args = parser.parse_args()
    root = Path(args.root)
    manifest = ["path,label"]
    for label_name, folder in FOLDERS.items():
        out = root / label_name
        out.mkdir(parents=True, exist_ok=True)
        files = list_files(folder)
        have = {p.name for p in out.glob("*.mp4")}
        need = args.per_class - len(have)
        print(f"{label_name}: have {len(have)}, need {need} more")
        for f in files:
            if need <= 0:
                break
            name = Path(f["path"]).name
            if name in have:
                continue
            dest = out / name
            if download(BASE + f["path"], dest):
                print(f"  got {name}")
                have.add(name)
                need -= 1
            else:
                print(f"  FAIL {name}")
            if need <= 0:
                break
    label = {"real": 0, "ai": 1}
    for name, label_name in label.items():
        for p in sorted((root / name).glob("*.mp4")):
            manifest.append(f"{p.resolve()},{label_name}")
    (root / "manifest.csv").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(f"manifest rows: {len(manifest) - 1}")


if __name__ == "__main__":
    main()
