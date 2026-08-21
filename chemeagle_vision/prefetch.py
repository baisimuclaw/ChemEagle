"""Download all worker assets once on an internet-connected Linux machine."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable


WEIGHTS = (
    "molnextr.pth",
    "rxn.ckpt",
    "moldet.ckpt",
    "corefdet.ckpt",
    "ner.ckpt",
)


def prefetch(output: str, *, biobert_model: str) -> dict:
    root = Path(output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download, snapshot_download

    downloaded = {}
    for filename in WEIGHTS:
        downloaded[filename] = hf_hub_download(
            "CYF200127/ChemEAGLEModel",
            filename,
            local_dir=str(root),
        )
    downloaded["chemrxnextractor"] = snapshot_download(
        "amberwang/chemrxnextractor-training-modules",
        local_dir=str(root / "chemrxnextractor"),
    )

    biobert_dir = root / "biobert-large-cased"
    downloaded["biobert-large-cased"] = snapshot_download(
        biobert_model,
        local_dir=str(biobert_dir),
    )

    easyocr_dir = root / "easyocr"
    easyocr_dir.mkdir(exist_ok=True)
    import easyocr

    easyocr.Reader(
        ["en"],
        gpu=False,
        model_storage_directory=str(easyocr_dir),
        download_enabled=True,
    )
    downloaded["easyocr"] = str(easyocr_dir)

    manifest = {
        "format": 1,
        "assets": downloaded,
        "biobert_source": biobert_model,
    }
    with (root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prefetch offline ChemEAGLE worker assets")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--biobert-model",
        default=os.getenv("CHEMEAGLE_BIOBERT_MODEL", "dmis-lab/biobert-large-cased-v1.1"),
    )
    args = parser.parse_args(argv)
    manifest = prefetch(args.output, biobert_model=args.biobert_model)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
