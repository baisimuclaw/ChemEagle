"""Request-scoped retention helpers for chemical-vision predictions."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Iterable, MutableMapping
from typing import Any


MOLECULAR_TOOL_OMITTED_FIELDS = (
    "coords",
    "edges",
    "molfile",
    "atoms",
    "bonds",
    "category_id",
    "score",
    "corefs",
)


def compact_molecular_tool_result(
    raw_prediction: list,
    omitted_fields: Iterable[str],
) -> str:
    """Serialize an LLM projection without mutating the retained raw graph."""
    compact_prediction = copy.deepcopy(raw_prediction)
    for item in compact_prediction:
        for bbox in item.get("bboxes", []):
            for key in omitted_fields:
                bbox.pop(key, None)
    return json.dumps(compact_prediction)


def caching_molecular_tool(
    cache: MutableMapping[str, Any],
    predict: Callable[[str], list],
) -> Callable[[str], str]:
    """Bind a compact tool payload to one retained full vision prediction."""
    def invoke(image_path: str) -> str:
        if "raw_prediction" not in cache:
            cache["raw_prediction"] = predict(image_path)
        return compact_molecular_tool_result(
            cache["raw_prediction"],
            MOLECULAR_TOOL_OMITTED_FIELDS,
        )

    return invoke


def molecular_results_for_request(
    cache: MutableMapping[str, Any],
    predict: Callable[[str], list],
    image_path: str,
) -> list:
    """Return retained raw data, predicting once if the LLM skipped its tool."""
    if "raw_prediction" not in cache:
        cache["raw_prediction"] = predict(image_path)
    return cache["raw_prediction"]
