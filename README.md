# TruthLens

Multi-signal AI video forensics. TruthLens fuses forensic signals (frame,
temporal, frequency, audio, metadata, semantic) into one AI-probability plus an
explainable evidence report.

## What it does

- **Verdicts with uncertainty**: `likely_ai`, `likely_real`, or
  `inconclusive` — no forced binary answer.
- **Signal breakdown**: each signal reports status (`completed`,
  `not_available`, `failed`) and its findings. Failed/unavailable checks are
  excluded from the verdict — never treated as suspicious.
- **Forensic timeline**: sampled frames with per-frame scores.
- **Cross-signal disagreement**: flags when signals point in opposite
  directions (mixed evidence) instead of pretending they agree.
- **Digital evidence**: SHA-256 hash, file size, detector version, timestamp.
- **PDF report** and **case history** (SQLite) and **batch upload**.

## Quickstart (dev)

```powershell
pip install -r requirements.txt
uvicorn truthlens.api.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000

## Pages

- `/` upload (single or multiple for batch)
- `/history` case history
- `/report/{id}` reopen a saved report
- `/report/{id}/pdf` download the PDF report

## Evaluation

Score the detector on a labelled set (CSV/JSON manifest with `path,label`,
where `1` = AI-generated) and calibrate thresholds:

```powershell
python scripts/benchmark.py --manifest data/eval_manifest.example.csv
```

Report includes AUC, accuracy, precision/recall/F1, TPR/FPR, ECE, and the
Youden-J best threshold.

Check verdict stability under resize/re-encode:

```powershell
python scripts/robustness.py --video path\to\clip.mp4
```

## Testing / lint

```powershell
python -m pytest -q
ruff check .
```

## Notes

- CPU-only friendly (ONNX optional). ffmpeg is required for audio and ffprobe
  metadata paths; without it those signals report `not_available`.
- Research-oriented. Scores are heuristic unless a validated model is loaded.
  Calibrate on your own splits before making strong accuracy claims.
