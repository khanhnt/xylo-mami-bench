"""Shared seed-control utilities for reproducible XyloMaMi-Bench experiments."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def set_global_seed(
    seed: int,
    *,
    cudnn_deterministic: bool = True,
    cudnn_benchmark: bool = False,
) -> dict[str, Any]:
    """Seed Python, NumPy, PyTorch, CUDA, and cuDNN runtime switches."""

    resolved_seed = int(seed)
    random.seed(resolved_seed)
    np.random.seed(resolved_seed % (2**32 - 1))
    torch.manual_seed(resolved_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(resolved_seed)
        torch.cuda.manual_seed_all(resolved_seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(cudnn_deterministic)
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    return {
        "seed": resolved_seed,
        "cudnn_deterministic": bool(getattr(torch.backends.cudnn, "deterministic", cudnn_deterministic)),
        "cudnn_benchmark": bool(getattr(torch.backends.cudnn, "benchmark", cudnn_benchmark)),
    }
