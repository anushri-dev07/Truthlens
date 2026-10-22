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

## Detector

The primary frame signal (`s1_frame`) is a **linear probe on frozen ViT
embeddings** (`ProbeClassifier`). A pretrained deepfake ViT (Community
Forensics) exposes its pre-classifier CLS embedding; a scikit-learn logistic
head is trained on labelled data to adapt the decision boundary to the target
domain. This is CPU-only and trains in seconds.

Get the models and train the probe:

```powershell
python scripts/download_models.py --model community_forensics   # base ViT
python scripts/fetch_eval.py --per-class 40                     # labelled eval set
python scripts/train_probe.py --model artifacts/models/community_forensics/model.onnx
```

`ProbeClassifier` is used automatically when `artifacts/models/community_probe/`
exists; otherwise TruthLens falls back to the raw ONNX classifier, then to a
face heuristic.

### Results (Celeb-DF v2 subset, 61 videos, 40 real / 21 AI)

| Method | AUC |
| --- | --- |
| Zero-shot ViT (raw ONNX) | 0.53 |
| Face heuristic | 0.50 |
| **Linear probe (default)** | **0.82** |

Overall pipeline at the calibrated threshold (0.55): accuracy 0.75, TPR 0.86,
FPR 0.30.

## Evaluation

Score the detector on a labelled set (CSV/JSON manifest with `path,label`,
where `1` = AI-generated) and calibrate thresholds:

```powershell
python scripts/benchmark.py --manifest data/eval/manifest.csv
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
- Cross-dataset generalisation is hard: raw off-the-shelf deepfake models score
  near chance on Celeb-DF. The probe fixes this by adapting to the target domain.
  Retrain the probe on your own labelled data for best results.
