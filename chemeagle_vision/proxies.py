"""Small compatibility proxies used by the existing ChemEAGLE agents."""

from __future__ import annotations

from typing import Any, List

from .context import call_vision


class RxnIMProxy:
    def predict_image_file(self, image_file: str, **kwargs: Any) -> Any:
        from PIL import Image

        with Image.open(image_file) as opened:
            image = opened.convert("RGB").copy()
        return call_vision(
            "rxnim_predict_image",
            {"image": image, "options": kwargs},
        )


class MolNexTRProxy:
    def predict_images(self, images: List[Any], **kwargs: Any) -> Any:
        return call_vision(
            "molnextr_predict_images", {"images": images, "options": kwargs}
        )

    def convert_graph_to_output(self, graphs: List[Any], images: List[Any]) -> Any:
        """Apply upstream Graph2SMILES locally to the full remote graph.

        The upstream helpers call ``MolNexTR.convert_graph_to_output``, although
        the checked-in MolNexTR class does not define that method.  This
        compatibility implementation uses the exact upstream chemistry
        function without loading a neural model.  Keeping it in the
        orchestrator also preserves the upstream name-resolution fallbacks and
        the active local LLM context; only neural vision inference is remote.
        """
        if not graphs:
            return []
        if len(graphs) != len(images):
            raise ValueError("graphs and images must have the same length")

        from molnextr.chemistry import convert_graph_to_smiles

        coords = []
        symbols = []
        edges = []
        for graph in graphs:
            chartok = graph.get("chartok_coords") if isinstance(graph, dict) else None
            if not isinstance(chartok, dict):
                raise ValueError("each graph requires chartok_coords")
            coords.append(chartok["coords"])
            symbols.append(chartok["symbols"])
            edges.append(graph["edges"])

        smiles_values, molfile_values, _success = convert_graph_to_smiles(
            coords,
            symbols,
            edges,
            images=images,
            # Upstream MolNexTR defaults to one worker.  Besides matching that
            # default, in-process execution retains the active Codex context.
            num_workers=1,
        )
        return [
            {
                "smiles": smiles,
                "symbols": graph["chartok_coords"]["symbols"],
                "coords": graph["chartok_coords"]["coords"],
                "edges": graph["edges"],
                "molfile": molfile,
            }
            for graph, smiles, molfile in zip(
                graphs,
                smiles_values,
                molfile_values,
            )
        ]


class ChemIEToolkitProxy:
    def __init__(self) -> None:
        self._molnextr = MolNexTRProxy()

    @property
    def molnextr(self) -> MolNexTRProxy:
        return self._molnextr

    def extract_molecule_corefs_from_figures(
        self, figures: List[Any], **kwargs: Any
    ) -> Any:
        return call_vision(
            "extract_molecule_corefs",
            {"figures": figures, "options": kwargs},
        )


class ChemNERProxy:
    def predict_strings(self, strings: List[str], **kwargs: Any) -> Any:
        return call_vision(
            "chemner_predict_strings", {"strings": strings, "options": kwargs}
        )


class ChemRxnExtractorProxy:
    def get_reactions(self, sentences: List[str]) -> Any:
        return call_vision(
            "chemrxn_extract_sentences", {"sentences": sentences}
        )


vision_rxnim = RxnIMProxy()
vision_toolkit = ChemIEToolkitProxy()
vision_chemner = ChemNERProxy()
vision_chemrxnextractor = ChemRxnExtractorProxy()
