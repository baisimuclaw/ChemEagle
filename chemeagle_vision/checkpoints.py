"""Accuracy-preserving checkpoint loading helpers for vision workers."""

from __future__ import annotations

import warnings
from typing import Any, Mapping


def load_model_checkpoint(
    model: Any,
    state_dict: Mapping[str, Any],
    *,
    component: str,
) -> Any:
    """Load a checkpoint while refusing silently uninitialized model weights.

    Older PyTorch checkpoints may omit BatchNorm's bookkeeping-only
    ``num_batches_tracked`` buffers.  Every other missing key can change
    inference and is therefore fatal.  Unexpected checkpoint keys are safe to
    ignore but are surfaced for version-drift diagnosis.
    """

    normalized = {key.replace("model.", ""): value for key, value in state_dict.items()}
    incompatible = model.load_state_dict(normalized, strict=False)
    missing = [
        key
        for key in getattr(incompatible, "missing_keys", ())
        if not key.endswith("num_batches_tracked")
    ]
    if missing:
        preview = ", ".join(missing[:20])
        remainder = len(missing) - 20
        if remainder > 0:
            preview += f", ... (+{remainder} more)"
        raise RuntimeError(
            f"{component} checkpoint is missing model weights: {preview}"
        )

    unexpected = list(getattr(incompatible, "unexpected_keys", ()))
    if unexpected:
        preview = ", ".join(unexpected[:20])
        remainder = len(unexpected) - 20
        if remainder > 0:
            preview += f", ... (+{remainder} more)"
        warnings.warn(
            f"{component} checkpoint has unused keys: {preview}",
            RuntimeWarning,
            stacklevel=2,
        )
    return incompatible
