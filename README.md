# AI-Generated Music Detector

A two-branch prototype classifier that distinguishes AI-generated music from human recordings. Combines CLAP audio embeddings with a log-mel spectrogram CNN, trained on FakeMusicCaps (synthetic) and FMA (human) audio.

## Architecture

```
Audio file
  ├── CLAP branch  → CLAP audio embeddings → Logistic Regression
  └── CNN branch   → Log-mel spectrogram   → Compact CNN
        ↓
    Fused prediction (average / configurable ensemble)
        ↓
    "ai_generated" | "human"  (at calibrated threshold)
```

No captions required — the CLAP branch uses audio embeddings only.

## Results (test set, `artifacts/all/`)

| Branch | Accuracy | AUROC | AUPRC |
|--------|----------|-------|-------|
| CNN (log-mel spectrogram) | **88.2%** | **0.991** | **0.996** |
| Baseline (duration / loudness / ZCR) | 86.8% | 0.853 | 0.869 |

The CNN branch substantially outperforms the hand-crafted feature baseline, particularly on ranking metrics (AUROC/AUPRC), showing that spectral artifact patterns are highly discriminative for AI detection.

## Datasets

| Dataset | Source | Role |
|---------|--------|------|
| FakeMusicCaps | [Kaggle — awsaf49/fmc-dataset](https://www.kaggle.com/datasets/awsaf49/fmc-dataset) | AI-generated audio (label = 1) |
| FMA (medium) | [Kaggle — imsparsh/fma-free-music-archive-small-medium](https://www.kaggle.com/datasets/imsparsh/fma-free-music-archive-small-medium) | Human recordings (label = 0) |

> Raw audio is not included in this repo (too large). Download via the provided Kaggle script — see [Quick Start](#quick-start).

**Citation — FakeMusicCaps:**
> Comanducci, L., Bestagini, P., and Tubaro, S. "FakeMusicCaps: a Dataset for Detection and Attribution of Synthetic Music Generated via Text-to-Music Models." arXiv:2409.10684, 2024.

**Citation — FMA:**
> Defferrard, M., Benzi, K., Vandergheynst, P., Bresson, X. "FMA: A Dataset for Music Analysis." ISMIR, 2017.

## Quick Start

```bash
git clone https://github.com/NinaSo/ai-music-detector.git
cd ai-music-detector
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

**Apple Silicon (M-series):**
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Download datasets (optional — skip if using provided manifest)

```bash
bash scripts/download_kaggle.sh /ABS/PATH/data/raw
```

### Build manifest

```bash
python scripts/build_manifest.py \
  --fake_csv /PATH/fakemusiccaps_metadata.csv \
  --fake_root /PATH/fakemusiccaps_audio \
  --fma_root /PATH/fma_audio \
  --fma_tracks_csv /PATH/fma_tracks.csv \
  --fma_track_col track_id \
  --fma_artist_col artist_id \
  --fma_limit 12000 \
  --output data/manifests/combined.csv
```

### Train all branches

```bash
python -m aidetector.train_all \
  --manifest data/manifests/combined.csv \
  --output_dir artifacts/run1 \
  --group_col group_id \
  --calibration_metric f1 \
  --cnn_epochs 8
```

### Predict on a new audio file

```bash
python -m aidetector.predict \
  --audio /PATH/new_track.wav \
  --cnn_model artifacts/all/cnn/cnn_best.pt \
  --baseline_model artifacts/all/baseline/baseline_logreg.joblib \
  --threshold_json artifacts/all/cnn/threshold.json \
  --output_json predictions/new_track.json
```

Output fields: `cnn_prob`, `baseline_prob`, `fused_prob`, `pred_label_name` (`ai_generated` or `human`).

### Streamlit UI

```bash
streamlit run src/aidetector/ui_app.py
```

Two tabs:
- **Predict** — upload WAV/MP3, get AI probability + label. Optional high-recall mode and ensemble selector (CNN-only / CLAP-only / weighted).
- **Further Training** — continue training from a checkpoint, compare old vs new metrics, export dashboard figures.

## Holdout Evaluation (unseen AI generators)

Train with one AI source held out to measure generalization to unseen generators:

```bash
python -m aidetector.run_holdout_suite \
  --manifest data/manifests/combined.csv \
  --output_root artifacts/holdout_suite \
  --holdout_sources musicldm,audioldm2,mustango,stable_audio_open \
  --group_col group_id \
  --human_test_ratio 1.0 \
  --cnn_epochs 8
```

Aggregated mean/std report → `artifacts/holdout_suite/holdout_summary.json`

## File Structure

```
repo_ai_music_detector/
├── src/aidetector/
│   ├── audio_io.py              # Audio loading utilities
│   ├── clap_audio.py            # CLAP embedding extraction
│   ├── cnn_detector.py          # Log-mel CNN architecture
│   ├── data.py                  # Dataset and manifest handling
│   ├── train_alignment.py       # CLAP branch training
│   ├── train_cnn.py             # CNN branch training
│   ├── train_baseline.py        # Baseline (hand-crafted features) training
│   ├── train_all.py             # Train all branches in one command
│   ├── train_fmc_source_logreg.py  # Generator attribution (multiclass)
│   ├── predict.py               # Single-file prediction
│   ├── predict_clap_audio.py    # CLAP-only prediction
│   ├── predict_fmc_source.py    # Generator attribution prediction
│   ├── run_holdout_suite.py     # Holdout evaluation suite
│   ├── metrics.py               # Shared metrics utilities
│   ├── device.py                # Device selection (CPU/MPS/CUDA)
│   └── ui_app.py                # Streamlit web UI
├── scripts/
│   ├── build_manifest.py        # Build CSV manifest from audio folders
│   ├── download_kaggle.sh       # Download FMC + FMA datasets
│   └── render_holdout_report.py # Render holdout summary report
├── data/manifests/
│   └── combined.csv             # Pre-built manifest (paths relative)
├── artifacts/all/               # Trained model weights (main run)
│   ├── cnn/cnn_best.pt          # CNN weights (364 KB)
│   ├── baseline/baseline_logreg.joblib
│   └── combined_metrics.json
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Notes

- This is a prototype, not SOTA. Designed for clarity and reproducibility over peak performance.
- Group-based train/val/test splits (`--group_col group_id`) prevent artist/source leakage.
- CLAP branch is audio-only and does not require text captions.
- Start with short clips (10s) and balanced sampling for faster iteration.
