"""Lazy, shared model runtime used by local and remote vision backends."""

from __future__ import annotations

import os
import socket
import tempfile
import threading
from typing import Any, Callable, Dict

from .config import VisionConfig
from .errors import VisionConfigurationError


class VisionRuntime:
    """Load each expensive model once and expose a fixed method allowlist."""

    def __init__(self, config: VisionConfig):
        self.config = config
        self._toolkit: Any = None
        self._methods: Dict[str, Callable[[Dict[str, Any]], Any]] = {
            "health": self._health,
            "rxnim_predict_image": self._rxnim_predict_image,
            "extract_molecule_corefs": self._extract_molecule_corefs,
            "molnextr_predict_images": self._molnextr_predict_images,
            "chemner_predict_strings": self._chemner_predict_strings,
            "chemrxn_extract_sentences": self._chemrxn_extract_sentences,
        }
        if config.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if config.model_dir:
            root = os.path.abspath(os.path.expanduser(config.model_dir))
            os.environ.setdefault("CHEMEAGLE_VISION_MODEL_DIR", root)
            os.environ.setdefault("HF_HOME", os.path.join(root, "huggingface"))
            os.environ.setdefault("EASYOCR_MODULE_PATH", os.path.join(root, "easyocr"))

    def call(self, method: str, params: Dict[str, Any]) -> Any:
        handler = self._methods.get(method)
        if handler is None:
            raise VisionConfigurationError(
                f"Vision worker method {method!r} is not allowlisted"
            )
        if not isinstance(params, dict):
            raise VisionConfigurationError("Vision worker params must be a dictionary")
        return handler(params)

    @property
    def toolkit(self) -> Any:
        if self._toolkit is None:
            from chemietoolkit import ChemIEToolkit

            device = self.config.device
            if device == "auto":
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._toolkit = ChemIEToolkit(
                device=device,
                chemrxn_device=self.config.chemrxn_device,
                model_dir=self.config.model_dir,
                offline=self.config.offline,
            )
        return self._toolkit

    def _health(self, _params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            gpu_count = int(torch.cuda.device_count()) if cuda_available else 0
            gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
            torch_version = torch.__version__
        except Exception:
            cuda_available = False
            gpu_count = 0
            gpu_name = None
            torch_version = None
        requested = self.config.device
        if requested.startswith("cuda") and not cuda_available:
            status = "unavailable"
        else:
            status = "ok"
        return {
            "status": status,
            "host": socket.gethostname(),
            "device": requested,
            "cuda_available": cuda_available,
            "gpu_count": gpu_count,
            "gpu_name": gpu_name,
            "torch_version": torch_version,
            "offline": self.config.offline,
            "models_loaded": self._toolkit is not None,
        }

    def _rxnim_predict_image(self, params: Dict[str, Any]) -> Any:
        image = params.get("image")
        if image is None:
            raise VisionConfigurationError("rxnim_predict_image requires an image")
        options = params.get("options") or {}
        if not isinstance(options, dict):
            raise VisionConfigurationError("rxnim options must be a dictionary")
        handle = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        path = handle.name
        handle.close()
        try:
            image.convert("RGB").save(path, format="PNG")
            return self.toolkit.rxnim.predict_image_file(path, **options)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def _extract_molecule_corefs(self, params: Dict[str, Any]) -> Any:
        figures = params.get("figures")
        options = params.get("options") or {}
        if not isinstance(figures, list) or not isinstance(options, dict):
            raise VisionConfigurationError(
                "extract_molecule_corefs requires figures and dictionary options"
            )
        return self.toolkit.extract_molecule_corefs_from_figures(
            figures, **options
        )

    def _molnextr_predict_images(self, params: Dict[str, Any]) -> Any:
        images = params.get("images")
        options = params.get("options") or {}
        if not isinstance(images, list) or not isinstance(options, dict):
            raise VisionConfigurationError(
                "molnextr_predict_images requires images and dictionary options"
            )
        return self.toolkit.molnextr.predict_images(images, **options)

    def _chemner_predict_strings(self, params: Dict[str, Any]) -> Any:
        strings = params.get("strings")
        options = params.get("options") or {}
        if not isinstance(strings, list) or not all(
            isinstance(value, str) for value in strings
        ):
            raise VisionConfigurationError(
                "chemner_predict_strings requires a list of strings"
            )
        if not isinstance(options, dict):
            raise VisionConfigurationError("ChemNER options must be a dictionary")
        return self.toolkit.chemner.predict_strings(strings, **options)

    def _chemrxn_extract_sentences(self, params: Dict[str, Any]) -> Any:
        sentences = params.get("sentences")
        if not isinstance(sentences, list) or not all(
            isinstance(value, str) for value in sentences
        ):
            raise VisionConfigurationError(
                "chemrxn_extract_sentences requires a list of strings"
            )
        # Match get_text_agent.py upstream exactly: call the underlying
        # chemrxnextractor.RxnExtractor, whose return value is the raw list of
        # sentence predictions (not ChemIEToolkit's PDF/page wrapper).
        extractor = self.toolkit.chemrxnextractor.rxn_extractor
        try:
            return extractor.get_reactions(sentences)
        except AssertionError as exc:
            # Keep upstream's length-mismatch fallback on the worker side.
            # Transporting the AssertionError over JSONL would otherwise turn
            # it into VisionBackendError and prevent the caller's old fallback
            # from recognizing it.
            print(
                "Warning: batch processing failed, trying sentence-by-sentence "
                f"processing: {exc}"
            )
            reactions = []
            for sentence in sentences:
                try:
                    reactions.extend(extractor.get_reactions([sentence]))
                except Exception as sentence_exc:
                    print(
                        "Warning: skipping sentence (processing failed): "
                        f"{sentence[:50]}... Error: {sentence_exc}"
                    )
            return reactions


class LocalVisionBackend:
    provider_name = "local"

    def __init__(self, config: VisionConfig):
        self.config = config
        self.runtime = VisionRuntime(config)
        # The upstream model objects mutate lazy-load and decoder state.  One
        # local backend therefore preserves upstream's serial inference order
        # even if Codex requests several allowlisted tools concurrently.
        self._call_lock = threading.Lock()

    def call(self, method: str, params: Dict[str, Any]) -> Any:
        with self._call_lock:
            return self.runtime.call(method, params)

    def health(self) -> Dict[str, Any]:
        return self.runtime.call("health", {})

    def close(self) -> None:
        return None
