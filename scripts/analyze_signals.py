from __future__ import annotations

import argparse
import json
from pathlib import Path

from truthlens.eval.harness import read_manifest
from truthlens.eval.metrics import _auc_roc
from truthlens.inference.pipeline import run_pipeline

ORDER = ["s1_frame", "s2_temporal", "s3_freq", "s4_audio", "s5_meta", "s6_semantic"]


def collect(manifest: str, cache: Path) -> list[dict]:
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    rows = []
    for path, label in read_manifest(manifest):
        res = run_pipeline(path)
        sig = {s["key"]: s for s in res["signals"]}
        rows.append(
            {
                "path": path,
                "label": label,
                "ai_probability": res["ai_probability"],
                "signals": {
                    k: {"prob": sig[k]["score"] / 100.0, "status": sig[k]["status"]}
                    for k in ORDER
                    if k in sig
                },
            }
        )
        print(f"done {Path(path).name} label={label}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def report(rows: list[dict]) -> None:
    y = [r["label"] for r in rows]
    overall = [r["ai_probability"] / 100.0 for r in rows]
    print(f"\noverall AUC = {_auc_roc(y, overall):.4f}  (n={len(y)})")
    print("\nper-signal AUC (1.0 = perfect, 0.5 = chance, <0.5 = inverted):")
    results = []
    for k in ORDER:
        pairs = [(r["signals"][k]["prob"], r["label"]) for r in rows
                 if k in r["signals"] and r["signals"][k]["status"] == "completed"]
        if len(pairs) < 3 or len({lbl for _, lbl in pairs}) < 2:
            print(f"  {k:14s} skipped (insufficient coverage)")
            continue
        auc = _auc_roc([lbl for _, lbl in pairs], [p for p, _ in pairs])
        results.append((k, auc, len(pairs)))
    for k, auc, n in sorted(results, key=lambda x: -x[1]):
        flag = "  <-- inverted" if auc < 0.5 else ""
        print(f"  {k:14s} AUC={auc:.4f}  n={n}{flag}")


def main():
    parser = argparse.ArgumentParser(description="Per-signal AUC diagnostics")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache", default="artifacts/eval_scores.json")
    args = parser.parse_args()
    rows = collect(args.manifest, Path(args.cache))
    report(rows)


if __name__ == "__main__":
    main()
