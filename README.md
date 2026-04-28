# XyloMaMi-Bench

Code and benchmark metadata for **Protocol-aware macro-micro representation learning for wood identification with XyloMaMi-Bench**.

This repository contains the training and evaluation code used for the macro-micro wood identification experiments. It includes protocol-aware split metadata for P1, P2, and P3, single-modality baselines, staged exact-first macro-micro alignment, and P3 RGB-gray fusion experiments.

## Repository Layout

```text
configs/
  train/                 Shared training defaults
  experiment/            Reproducible experiment configurations
data/
  processed/manifests/   Protocol candidate manifests
  processed/splits/      Deterministic train/val/test split CSV files
  processed/reports/     Split and taxonomy summary files
  images/                Image root expected by the released CSV metadata
scripts/                 Multi-seed orchestration and summary utilities
src/
  datasets/              Manifest loading, taxonomy utilities, transforms
  engine/                Training and metric engines
  losses/                Contrastive and taxonomy-aware losses
  models/                Baseline, dual-encoder, and RGB-gray fusion models
  scripts/               Train/evaluate command-line entry points
```

The released metadata uses portable image paths under `data/images/macro` and `data/images/micro`. If the image files are stored elsewhere, pass `--macro_image_root` and `--micro_image_root` to the training or evaluation scripts.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The experiments use ImageNet-pretrained backbones from `timm`. The first run may download backbone weights unless they are already cached.

## Data Layout

Place image files so that each CSV row can be resolved as:

```text
data/images/macro/<image_rel_path>
data/images/micro/<image_rel_path>
```

For example, a macro row with `image_rel_path=Phase1/3248.Afzelia africana/1.jpg` should be available at:

```text
data/images/macro/Phase1/3248.Afzelia africana/1.jpg
```

The split CSV files are already provided in `data/processed/splits`. Rebuilding manifests or splits is optional and should only be done when the source image collection changes.

## Main P3 Experiments

Train RGB baselines needed for warm-starting:

```bash
python3 src/scripts/train_baseline.py --config configs/experiment/p3_baseline_macro.yaml --seed 42
python3 src/scripts/train_baseline.py --config configs/experiment/p3_baseline_micro.yaml --seed 42
```

Train grayscale baselines needed for the RGB-gray fusion experiment:

```bash
python3 src/scripts/train_baseline.py --config configs/experiment/p3_baseline_macro_grayscale.yaml --seed 42
python3 src/scripts/train_baseline.py --config configs/experiment/p3_baseline_micro_grayscale.yaml --seed 42
```

Train the main alignment models:

```bash
python3 src/scripts/train_align.py --config configs/experiment/p3_align_v2.yaml --seed 42
python3 src/scripts/train_align.py --config configs/experiment/p3_align_v2_grayscale.yaml --seed 42
python3 src/scripts/train_align.py --config configs/experiment/p3_align_v2_rgbgray_fusion.yaml --seed 42
python3 src/scripts/train_align.py --config configs/experiment/p3_align_v3_rgbgray_gated.yaml --seed 42
```

Run classification evaluation:

```bash
python3 src/scripts/eval_model.py \
  --config configs/experiment/p3_align_v3_rgbgray_gated.yaml \
  --checkpoint outputs/p3_align_v3_rgbgray_gated_seed42/checkpoints/best_macro.ckpt \
  --split test \
  --seed 42
```

Run embedding-space cross-scale retrieval:

```bash
python3 src/scripts/eval_align_retrieval.py \
  --run p3_rgb_gray_gated_v3::configs/experiment/p3_align_v3_rgbgray_gated.yaml::outputs/p3_align_v3_rgbgray_gated_seed42/checkpoints/best_macro.ckpt \
  --split test \
  --feature_source embedding \
  --seed 42
```

## Multi-Seed Robustness

The P3 multi-seed runner reproduces the robustness experiments for RGB alignment, grayscale alignment, and RGB-gray gated fusion:

```bash
python3 scripts/run_p3_multiseed.py \
  --models p3_rgb_align p3_gray_align p3_rgb_gray_gated_v3 \
  --seeds 42 2025 3407 \
  --output-root results/p3_multiseed \
  --resume
```

Summarize completed runs:

```bash
python3 scripts/summarize_p3_multiseed.py \
  --input results/p3_multiseed/per_seed_metrics.csv
```

## Reproducibility Notes

All released training and evaluation entry points accept `--seed`. The seed is propagated to Python, NumPy, PyTorch CPU, PyTorch CUDA, cuDNN deterministic mode, config snapshots, logs, metrics, and run directories.

Generated outputs are written to `outputs/` by default. Multi-seed outputs are written to `results/p3_multiseed/`.
