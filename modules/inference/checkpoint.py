"""Versioned, task-aware checkpoint utilities for temporary inference models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.optim import Optimizer


CHECKPOINT_SCHEMA_VERSION = 1


def save_checkpoint(
    path: str | Path,
    *,
    task: str,
    model: nn.Module,
    model_config: Mapping[str, Any],
    epoch: int,
    global_step: int,
    optimizer: Optimizer | None = None,
    metrics: Mapping[str, float] | None = None,
) -> None:
    """Atomically save tensors and primitive metadata only."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "task": task,
        "model_class": type(model).__name__,
        "model_config": dict(model_config),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "metrics": dict(metrics or {}),
    }
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_checkpoint(
    path: str | Path,
    *,
    expected_task: str | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load and validate a repository checkpoint."""
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema version")
    if expected_task is not None and payload.get("task") != expected_task:
        raise ValueError(
            f"checkpoint task {payload.get('task')!r} does not match {expected_task!r}"
        )
    required = {"model_config", "model_state_dict", "epoch", "global_step"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"checkpoint is missing fields: {sorted(missing)}")
    return payload
