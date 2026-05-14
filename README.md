# CCAC_MAPS

Official baseline repository for the Multimodal Adolescent Psychological States Challenge for CCAC.

Official Website shows [here](https://ccacmaps.hai-lab.cn/)

## Background

This challenge studies longitudinal prediction of adolescent psychological states from multimodal audio-video recordings and DASS scale history. The dataset follows adolescents across four phases, `T1` to `T4`, and provides induced-paradigm audio-visual features together with psychological scale annotations.

## Task

Use subject history from `T1`, `T2`, and `T3` to predict the target psychological state at `T4`.

This baseline predicts `t4_anxiety_level` from audio-video features. It does not use DASS history features by default, so it should be treated as a simple reference system rather than a performance ceiling.

## Dataset Layout

Place the released dataset under `datasets/`:

```text
datasets/
├── train_val/
│   ├── labels.csv
│   ├── audio_wavlm_base/
│   ├── video_dinov2_small/
│   └── ...
├── test/
│   ├── subjects.csv
│   ├── audio_wavlm_base/
│   ├── video_dinov2_small/
│   └── ...
└── metadata/
```

Feature archives may be distributed as `.7z` files. Extract each archive into a directory with the same feature name before running the baseline, for example `train_val/audio_wavlm_base.7z` should become `train_val/audio_wavlm_base/`.

Feature files are organized as:

```text
<split>/<feature_name>/<anon_school>/<anon_class>/<anon_person>/<stage>/<clip_type>/
```

where `split` is `train_val` or `test`, `stage` is one of `T1`, `T2`, `T3`, and `clip_type` is one of:

| Clip | Description |
|---|---|
| `A01` | Standardized reading passage, "The North Wind and the Sun" |
| `B01` | Free response about yesterday |
| `B02` | Free response about the happiest memory from the past week |
| `B03` | Free response about the saddest memory from the past week |

Each clip feature directory can contain:

| File | Meaning |
|---|---|
| `sequence.npz` | Time or chunk-level feature sequence with `features`, `timestamps_ms`, `feature_names`, and `source_hz` |
| `pooled.npy` | Numeric pooled vector used by this baseline |
| `pooled.json` | Metadata for the pooled vector |

## Features Description

We focuses on pretrained audio and video representations plus a lightweight video descriptor. All SSL features use mean and standard deviation pooling, so the pooled vector dimension is twice the encoder embedding dimension.

### Audio Features

| Feature name | Source model | Sequence shape example | Pooled dimension | Notes |
|---|---|---:|---:|---|
| `audio_wavlm_base` | `microsoft/wavlm-base` | `(num_chunks, 1536)` | `3072` | Default baseline audio feature. Each chunk vector stores mean and standard deviation over WavLM frame states; `pooled.npy` again stores mean and standard deviation over chunk vectors. |
| `audio_chinese_hubert_base` | `TencentGameMate/chinese-hubert-base` | `(num_chunks, 1536)` | `3072` | Optional released audio SSL feature when available. |
| `audio_wav2vec2_chinese_base` | `TencentGameMate/chinese-wav2vec2-base` | `(num_chunks, 1536)` | `3072` | Optional released audio SSL feature when available. |
| `audio_wav2vec2_xlsr_chinese` | `jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn` | `(num_chunks, 2048)` | `4096` | Optional released audio SSL feature when available. |

Audio SSL extraction uses enhanced 16 kHz mono audio. The sequence time axis corresponds to fixed-length audio chunks, and the pooled vector is intended for clip-level modeling.

### Video Features

| Feature name | Source model or method | Sequence shape example | Pooled dimension | Notes |
|---|---|---:|---:|---|
| `video_dinov2_small` | `facebook/dinov2-small` | `(num_frames, 384)` | `768` | Default baseline video feature. |
| `video_dinov2_base` | `facebook/dinov2-base` | `(num_frames, 768)` | `1536` | Larger DINOv2 representation. |
| `video_siglip_base` | `google/siglip-base-patch16-224` | `(num_frames, 768)` | `1536` | Vision-language image representation. |
| `video_vit_mae_base` | `facebook/vit-mae-base` | `(num_frames, 768)` | `1536` | Masked-autoencoder visual representation. |
| `video_clip_base` | `openai/clip-vit-base-patch32` | `(num_frames, 512)` | `1024` | CLIP base image representation. |
| `video_clip_large` | `openai/clip-vit-large-patch14` | `(num_frames, 768)` | `1536` | CLIP large image representation. |
| `video_basic` | brightness, blur, frame motion | `(num_frames, 3)` | `13` | Lightweight handcrafted video statistics. |

Video SSL features are extracted from sparsely sampled frames. The default sampling configuration is up to 64 frames at 1 FPS before pooling.

### Baseline Tensor

For each subject, this baseline reads one audio feature and one video feature for every `T1/T2/T3` and `A01/B01/B02/B03` combination. Missing clips are zero-filled and tracked by a mask.

Default tensor shape:

```text
[3 stages, 4 clips, 3072 audio dims + 768 video dims]
```

The default input dimension is therefore `3840`.

## Baseline Model

The baseline has three components:

1. A shared clip encoder projects concatenated audio-video pooled vectors.
2. Attention pooling fuses `A01/B01/B02/B03` clips within each stage.
3. A bidirectional GRU models the `T1 -> T2 -> T3` trajectory and predicts `T4` anxiety level.

Training uses cross-entropy loss, class weighting, cross-validation on `train_val/labels.csv`, and an ensemble of fold checkpoints for `test/subjects.csv` prediction.

## Installation

```bash
conda create -n ccac_maps python=3.10 -y
conda activate ccac_maps
pip install -r requirements.txt
```

## Run Baseline

```bash
PYTHONPATH=src python scripts/train_anxiety_baseline.py \
  --dataset-path datasets \
  --output-dir artifacts/baselines/anxiety_wavlm_dinov2_small
```

Useful options:

```bash
PYTHONPATH=src python scripts/train_anxiety_baseline.py \
  --dataset-path datasets \
  --output-dir artifacts/baselines/anxiety_wavlm_dinov2_small \
  --audio-feature-name audio_wavlm_base \
  --video-feature-name video_dinov2_small \
  --target-label-column t4_anxiety_level \
  --device cuda
```

If CUDA is unavailable, the code falls back to CPU.

## Outputs

The training script writes:

| Output | Description |
|---|---|
| `fold_metrics.csv` | Validation metrics for each fold |
| `oof_predictions.csv` | Out-of-fold predictions for `train_val` subjects |
| `test_predictions.csv` | Ensemble predictions for `test` subjects |
| `label_mapping.json` | Mapping from label string to class index |
| `baseline_config.json` | Training configuration |
| `summary.json` | Overall metrics and feature dimensions |
| `classification_report.txt` | Text classification report |
| `fold_*/best_model.pt` | Best checkpoint for each fold |

## Test

```bash
PYTHONPATH=src pytest -q tests
```
