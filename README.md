# TruthLens

TruthLens is an experimental video-forensics project. It combines available
frame, temporal, frequency, audio, and metadata checks into an **uncalibrated
detection score** and an evidence report. The score is not the probability
that a video is AI-generated, and results are not proof of authenticity.

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
- **Upload limits**: MP4, WebM, or MOV; 100 MiB per video; up to five files and
  250 MiB total per batch.

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

The primary frame signal (`s1_frame`) uses the best available option: a linear
probe on frozen ViT embeddings (`ProbeClassifier`), the raw ONNX frame model,
or a fixed face-presence heuristic when neither model can load. The fallback
is not manipulation detection. The probe uses a pretrained Community Forensics
ViT embedding with a scikit-learn logistic head trained on labelled data. The
score is a model output, not a calibrated probability.

Other checks are mostly experimental handcrafted statistics and rules: optical
flow/frame differences, spatial-frequency measurements, audio features, and a
weak metadata-tag heuristic. Their thresholds have not been validated as
reliable indicators of synthetic media. Semantic analysis is a placeholder and
is excluded. Audio analysis does not check audio/video synchronization. An
optional learned fusion model can be loaded; otherwise completed signals are
combined using configured fixed weights.

Get the models and train the probe:

```powershell
python scripts/download_models.py --model community_forensics   # base ViT
python scripts/fetch_eval.py --per-class 40                     # labelled eval set
python scripts/train_probe.py --model artifacts/models/community_forensics/model.onnx
```

The probe is used automatically when the probe model and head load; otherwise
TruthLens tries the raw ONNX classifier and then the face heuristic. Training
and runtime use the same preprocessing configuration. The training script
stores that configuration and metadata alongside its embedding cache and
rejects cache reuse when the run settings differ.

### Results (Celeb-DF v2 subset, 61 videos, 40 real / 21 AI)

Measured with **video-disjoint** splits: a video is never in both the training
and the test portion (`scripts/eval_probe.py`, 30 random 70/30 splits).

| Method | AUC |
| --- | --- |
| Zero-shot ViT (raw ONNX) | 0.53 |
| Face heuristic | 0.50 |
| **Linear probe, held-out videos** | **0.65** (mean, σ=0.10) |

Held-out video AUC across splits: mean 0.653, median 0.657, range 0.464–0.885;
28 of 30 splits above chance.

Earlier figures of ~0.82 for this detector were **contaminated**: the probe had
been fitted on all 61 evaluation videos, so that number measured training-set
performance (in-sample AUC is 1.00). The 0.65 figure above is the honest
held-out estimate and is what should be quoted.

This is weak but above chance. It is not a general-purpose deepfake detector,
and it has only been measured on one dataset of face-swap deepfakes.

## Evaluation

Score the detector on a labelled set (CSV/JSON manifest with `path,label`,
where `1` = AI-generated) and calibrate thresholds:

```powershell
python scripts/benchmark.py --manifest data/eval/manifest.csv
```

The evaluation report includes AUC, accuracy, precision/recall/F1, TPR/FPR,
ECE, and a threshold selected on the supplied evaluation data. That selected
threshold is exploratory; do not report its metrics as independent test
performance. Keep threshold selection separate from final evaluation.

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
- Reported held-out evaluation is limited to a small 61-video Celeb-DF v2
  subset and repeated splits of that same subset. It is limited evidence, not
  proof of general-purpose performance. The detector may be wrong on other
  generators, edits, sources, or compression settings.
- The web app is intended for local demonstration. Stored case reports may
  include sampled frame previews; consider this before analyzing sensitive
  videos.
