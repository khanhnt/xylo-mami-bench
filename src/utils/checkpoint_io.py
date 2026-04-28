"""Checkpoint I/O helpers with atomic replacement semantics."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import torch


def atomic_torch_save(payload: Any, path: str | Path, *, attempts: int = 2) -> None:
    """Save a PyTorch checkpoint without clobbering the previous file on write failure."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: BaseException | None = None

    for attempt_index in range(max(1, attempts)):
        tmp_path = output_path.with_name(
            f".{output_path.name}.tmp.{os.getpid()}.{attempt_index}"
        )
        try:
            with tmp_path.open("wb") as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, output_path)
            return
        except BaseException as exc:
            last_error = exc
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            if attempt_index + 1 < max(1, attempts):
                time.sleep(2.0)

    assert last_error is not None
    raise last_error
