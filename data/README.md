# Data Metadata

This directory contains the released benchmark metadata for XyloMaMi-Bench.

```text
processed/manifests/   Protocol candidate manifests for P1, P2, and P3
processed/splits/      Deterministic train/val/test splits used by the experiments
processed/reports/     Split and taxonomy summaries
images/macro/          Expected root for macroscopic wood images
images/micro/          Expected root for microscopic wood images
```

The CSV files contain portable paths. If images are stored outside the repository, keep the CSV files unchanged and pass image-root overrides at runtime:

```bash
python3 src/scripts/train_align.py \
  --config configs/experiment/p3_align_v2.yaml \
  --macro_image_root /path/to/macro/images \
  --micro_image_root /path/to/micro/images
```

The image binaries are intentionally separated from generated outputs, logs, checkpoints, and manuscript files.
