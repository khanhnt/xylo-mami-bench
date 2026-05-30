# Automated dual-scale wood species verification with XyloMaMi-Bench

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20350819.svg)](https://doi.org/10.5281/zenodo.20350819)

Code, split metadata, and reproducibility scripts for the companion paper:

> Nguyen-Trong, K., Le, L. T., & Nguyen Bao, N. (2026).
> *Automated dual-scale wood species verification for smart timber value chains:
> The XyloMaMi-Bench benchmark and overlap-aware deep learning framework.*

Repository: https://github.com/khanhnt/xylo-mami-bench

XyloMaMi-Bench contains protocol-aware benchmark metadata for macroscopic and
microscopic wood identification. The released metadata covers 36,742
macroscopic end-grain images from 100 tropical timber species, with P1/P2/P3
splits designed around the Congo microscopic reference collection.

## Dataset Summary

| Item | Value |
|---|---:|
| Macro images | 36,742 |
| Macro species | 100 species |
| Congo micro images | 1,219 |
| Congo micro species | 77 species |
| P3 protocol composition | 33 exact-overlap / 22 genus-only / 45 macro-only |
| Repeated-seed setting | 42, 2025, 3407 |
| Dataset DOI | 10.5281/zenodo.20350819 |

The full macroscopic image archive is distributed through Zenodo restricted
access. Split CSV files, protocol metadata, and code are kept in this repository
so experiments can be reproduced without committing raw images or model weights.

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

The released metadata uses portable image paths under `data/images/macro` and
`data/images/micro`. If the image files are stored elsewhere, pass
`--macro_image_root` and `--micro_image_root` to the training or evaluation
scripts.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The experiments use ImageNet-pretrained backbones from `timm`. The first run may
download backbone weights unless they are already cached.

## Main P3 RGB Alignment Experiment

Train RGB baselines needed for warm-starting:

```bash
python3 src/scripts/train_baseline.py --config configs/experiment/p3_baseline_macro.yaml --seed 42
python3 src/scripts/train_baseline.py --config configs/experiment/p3_baseline_micro.yaml --seed 42
```

Train the main P3 RGB alignment model:

```bash
python3 src/scripts/train_align.py --config configs/experiment/p3_align_v2.yaml --seed 42
```

Evaluate macro-branch classification:

```bash
python3 src/scripts/eval_model.py \
  --config configs/experiment/p3_align_v2.yaml \
  --checkpoint results/p3_multiseed/runs/p3_rgb_align_seed42/checkpoints/best.ckpt \
  --split test \
  --seed 42
```

Evaluate embedding-space cross-scale retrieval:

```bash
python3 src/scripts/eval_align_retrieval.py \
  --run p3_rgb_align::configs/experiment/p3_align_v2.yaml::results/p3_multiseed/runs/p3_rgb_align_seed42/checkpoints/best.ckpt \
  --split test \
  --feature_source embedding \
  --seed 42
```

## Multi-Seed Robustness

The P3 multi-seed runner reproduces the robustness experiments for RGB
alignment, grayscale alignment, and RGB-gray gated fusion:

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

## Additional Analyses

Calibration for the screening-confirmation rule:

```bash
python3 src/scripts/calibrate_threshold.py \
  --config configs/experiment/p3_align_v2.yaml \
  --checkpoint results/p3_multiseed/runs/p3_rgb_align_seed42/checkpoints/best.ckpt \
  --split data/processed/splits/p3_val.csv \
  --device cpu
```

The reported operating point is `tau = 0.85`, with 95.4% coverage and 98.12%
precision on the validation calibration split.

Bootstrap confidence intervals for cross-scale retrieval:

```bash
python3 src/scripts/bootstrap_retrieval_ci.py \
  --p3_embeddings outputs/retrieval/p3_embeddings.npz \
  --output_dir outputs/bootstrap_ci
```

Frozen ImageNet/CLIP baselines:

```bash
python3 src/scripts/frozen_baseline_retrieval.py \
  --p3_config configs/experiment/p3_align_v2.yaml \
  --macro_image_root data/images/macro \
  --micro_image_root data/images/micro
```

## Reported Results

| Result | Value |
|---|---:|
| P3 RGB align M->m R@1 | 79.49% |
| P3 RGB align m->M R@1 | 73.59% |
| iPhone 16 on-device latency | 954.65 ms |
| Xiaomi on-device latency | 1355.86 ms |
| Server-side RTX 3090 latency | 21.46 ms |
| Calibration threshold | tau = 0.85 |
| Calibration coverage | 95.4% |
| Calibration precision | 98.12% |

## Reproducibility Notes

All released training and evaluation entry points accept `--seed`. The seed is
propagated to Python, NumPy, PyTorch CPU, PyTorch CUDA, cuDNN deterministic
mode, config snapshots, logs, metrics, and run directories.

Generated outputs are written to `outputs/` by default. Multi-seed outputs are
written to `results/p3_multiseed/`. Both directories are ignored by Git.

## Citation

```bibtex
@dataset{nguyentrong2026xylomami_data,
  author    = {Nguyen-Trong, Khanh and Le, Loc Thi and Nguyen Bao, Ngoc},
  title     = {{XyloMaMi-Bench-v1.0}: A Dual-Scale Macroscopic--Microscopic
               Wood Species Benchmark},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20350819}
}
```
