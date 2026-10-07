# TruthLens - Overview
TruthLens is a multi-signal AI video detector. It fuses forensic signals (frame, temporal, frequency, audio, metadata, semantic) to estimate AI probability.

## Quickstart (dev)
1. `uv sync` / `pip install -e .`
2. `uvicorn truthlens.api.main:app --reload` or `python -m truthlens`
3. Open http://127.0.0.1:8000

## Testing
`python -m pytest -q`

## Lint
`ruff check .`

## Notes
- CPU-only friendly (ONNX optional). ffmpeg required for audio/ffprobe paths.
- This is research-oriented; calibrate with your own splits before making strong claims.
