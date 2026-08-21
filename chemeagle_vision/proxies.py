"""Small compatibility proxies used by the existing ChemEAGLE agents."""

from __future__ import annotations

from typing import Any, List

from .context import get_active_vision_backend


class RxnIMProxy:
    def predict_image_file(self, image_file: str, **kwargs: Any) -> Any:
        from PIL import Image

        with Image.open(image_file) as opened:
            image = opened.convert("RGB").copy()
        return get_active_vision_backend().call(
            "rxnim_predict_image",
            {"image": image, "options": kwargs},
        )


class MolNexTRProxy:
    def predict_images(self, images: List[Any], **kwargs: Any) -> Any:
        return get_active_vision_backend().call(
            "molnextr_predict_images", {"images": images, "options": kwargs}
        )

    def convert_graph_to_output(self, graphs: List[Any], images: List[Any]) -> Any:
        return get_active_vision_backend().call(
            "molnextr_convert_graph_to_output",
            {"graphs": graphs, "images": images},
        )


class ChemIEToolkitProxy:
    def __init__(self) -> None:
        self._molnextr = MolNexTRProxy()

    @property
    def molnextr(self) -> MolNexTRProxy:
        return self._molnextr

    def extract_molecule_corefs_from_figures(
        self, figures: List[Any], **kwargs: Any
    ) -> Any:
        return get_active_vision_backend().call(
            "extract_molecule_corefs",
            {"figures": figures, "options": kwargs},
        )


class ChemNERProxy:
    def predict_strings(self, strings: List[str], **kwargs: Any) -> Any:
        return get_active_vision_backend().call(
            "chemner_predict_strings", {"strings": strings, "options": kwargs}
        )


class ChemRxnExtractorProxy:
    def get_reactions(self, sentences: List[str]) -> Any:
        return get_active_vision_backend().call(
            "chemrxn_extract_sentences", {"sentences": sentences}
        )


vision_rxnim = RxnIMProxy()
vision_toolkit = ChemIEToolkitProxy()
vision_chemner = ChemNERProxy()
vision_chemrxnextractor = ChemRxnExtractorProxy()
