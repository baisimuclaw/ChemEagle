"""JSON-safe codec for images and NumPy values sent over the worker protocol."""

from __future__ import annotations

import base64
import io
import math
from pathlib import Path
from typing import Any

from .errors import VisionProtocolError


_TAG = "__chemeagle_wire_type__"


def encode_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise VisionProtocolError("Non-finite floats cannot be sent to a vision worker")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {_TAG: "bytes", "data": base64.b64encode(value).decode("ascii")}
    if isinstance(value, tuple):
        return {_TAG: "tuple", "items": [encode_value(item) for item in value]}
    if isinstance(value, list):
        return [encode_value(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise VisionProtocolError("Vision protocol dictionary keys must be strings")
        return {key: encode_value(item) for key, item in value.items()}

    try:
        from PIL import Image

        if isinstance(value, Image.Image):
            buffer = io.BytesIO()
            value.convert("RGB").save(buffer, format="PNG")
            return {
                _TAG: "pil-image",
                "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
            }
    except ImportError:
        pass

    try:
        import numpy as np

        if isinstance(value, np.generic):
            return encode_value(value.item())
        if isinstance(value, np.ndarray):
            buffer = io.BytesIO()
            np.save(buffer, value, allow_pickle=False)
            return {
                _TAG: "numpy",
                "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
            }
    except ImportError:
        pass

    try:
        import torch

        if isinstance(value, torch.Tensor):
            return encode_value(value.detach().cpu().numpy())
    except ImportError:
        pass

    raise VisionProtocolError(
        f"Unsupported vision protocol value: {type(value).__name__}"
    )


def decode_value(value: Any) -> Any:
    if isinstance(value, list):
        return [decode_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    kind = value.get(_TAG)
    if kind is None:
        return {key: decode_value(item) for key, item in value.items()}
    if kind == "tuple":
        items = value.get("items")
        if not isinstance(items, list):
            raise VisionProtocolError("Invalid tuple in vision protocol")
        return tuple(decode_value(item) for item in items)
    try:
        raw = base64.b64decode(value.get("data", ""), validate=True)
    except Exception as exc:
        raise VisionProtocolError("Invalid base64 in vision protocol") from exc
    if kind == "bytes":
        return raw
    if kind == "pil-image":
        try:
            from PIL import Image

            image = Image.open(io.BytesIO(raw))
            image.load()
            return image.convert("RGB")
        except Exception as exc:
            raise VisionProtocolError("Invalid image in vision protocol") from exc
    if kind == "numpy":
        try:
            import numpy as np

            return np.load(io.BytesIO(raw), allow_pickle=False)
        except Exception as exc:
            raise VisionProtocolError("Invalid NumPy array in vision protocol") from exc
    raise VisionProtocolError(f"Unknown vision protocol type: {kind!r}")

