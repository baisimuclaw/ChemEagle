from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import main
import get_molecular_agent
import get_R_group_sub_agent
import get_reaction_agent
import get_text_agent
from chemeagle_llm import LLMResponse, backend_scope
from chemietoolkit import helper as chemistry_helper
from molnextr import chemistry as molnextr_chemistry
from chemeagle_vision.request_cache import (
    caching_molecular_tool,
    compact_vision_tool_value,
    molecular_results_for_request,
)


ROOT = Path(__file__).resolve().parents[1]


class _FakeBackend:
    def __init__(self, provider_name):
        self.provider_name = provider_name
        self.closed = False

    def generate(self, request):  # pragma: no cover - patched orchestration owns this
        raise AssertionError("unexpected direct generation")

    def run_tool_loop(self, request, tools, executor):  # pragma: no cover
        raise AssertionError("unexpected direct tool loop")

    def close(self):
        self.closed = True


class ImportAndOrchestrationTests(unittest.TestCase):
    def test_text_filter_is_opt_in_and_upstream_input_is_default(self):
        ocr = (
            "DBU (1.1 equiv) benzene reflux 2 h\n"
            "6a: Ar = Ph; 61%, E/Z = 90:10\n"
            "Scheme 1. The hydrophosphination reaction gives phosphorus ylides."
        )
        self.assertEqual(
            get_text_agent.filter_prose_sentences(ocr),
            ["The hydrophosphination reaction gives phosphorus ylides."],
        )
        self.assertEqual(
            get_text_agent.filter_prose_sentences("DBU (1.1 equiv)\n4 80%"),
            [],
        )
        with mock.patch.dict(
            os.environ, {"CHEMEAGLE_TEXT_PROSE_FILTER": "1"}, clear=False
        ):
            self.assertEqual(
                get_text_agent._text_model_sentences("DBU reagent."),
                [],
            )
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                get_text_agent._text_model_sentences("DBU reagent."),
                ["DBU reagent."],
            )

    def test_reaction_tool_reuses_raw_prediction_and_handles_no_detection(self):
        self.assertEqual(get_reaction_agent._reaction_summary_from_raw([]), {})
        raw = [
            {
                "reactants": [
                    {"smiles": "C", "bbox": [0, 0, 1, 1], "symbols": ["C"]}
                ],
                "conditions": [],
                "products": [],
            }
        ]
        cache = {}
        with mock.patch.object(
            get_reaction_agent.model1,
            "predict_image_file",
            return_value=raw,
        ) as predict:
            tool = get_reaction_agent._caching_reaction_tool(cache)
            summary = tool("input.png")
            cached_summary = tool("input.png")
        predict.assert_called_once()
        self.assertIs(cache["raw_prediction"], raw)
        self.assertEqual(summary["reactants"][0]["smiles"], "C")
        self.assertEqual(cached_summary, summary)

    def test_reaction_prediction_retries_two_empty_results_then_succeeds(self):
        raw = [{"reactants": [], "conditions": [], "products": []}]
        with mock.patch.object(
            get_reaction_agent.model1,
            "predict_image_file",
            side_effect=[[], None, raw],
        ) as predict:
            result = get_reaction_agent._predict_reaction_with_empty_retries(
                "input.png"
            )

        self.assertIs(result, raw)
        self.assertEqual(predict.call_count, 3)

    def test_reaction_tool_caches_final_empty_result_after_three_attempts(self):
        cache = {}
        with mock.patch.object(
            get_reaction_agent.model1,
            "predict_image_file",
            return_value=[],
        ) as predict:
            tool = get_reaction_agent._caching_reaction_tool(cache)
            self.assertEqual(tool("input.png"), {})
            self.assertEqual(tool("input.png"), {})

        self.assertEqual(predict.call_count, 3)
        self.assertEqual(cache["raw_prediction"], [])

    def test_reaction_prediction_does_not_hide_inference_errors(self):
        with mock.patch.object(
            get_reaction_agent.model1,
            "predict_image_file",
            side_effect=RuntimeError("GPU unavailable"),
        ) as predict:
            with self.assertRaisesRegex(RuntimeError, "GPU unavailable"):
                get_reaction_agent._predict_reaction_with_empty_retries(
                    "input.png"
                )

        predict.assert_called_once()

    def test_reaction_condition_ocr_configures_tesseract_before_use(self):
        opened_image = object()
        events = []

        with (
            mock.patch.object(
                get_text_agent,
                "configure_tesseract",
                side_effect=lambda: events.append("configure"),
            ) as configure,
            mock.patch.object(
                get_reaction_agent.Image,
                "open",
                side_effect=lambda _path: (
                    events.append("open"),
                    opened_image,
                )[1],
            ),
            mock.patch(
                "pytesseract.image_to_string",
                side_effect=lambda image: (
                    events.append("ocr"),
                    "Mn(CO)5Br, toluene, reflux",
                )[1],
            ) as image_to_string,
        ):
            result = get_reaction_agent._tesseract_ocr_image("scheme.png")

        self.assertEqual(result, "Mn(CO)5Br, toluene, reflux")
        self.assertEqual(events, ["configure", "open", "ocr"])
        configure.assert_called_once_with()
        image_to_string.assert_called_once_with(opened_image)

    def test_molecular_prediction_retries_empty_detections_then_succeeds(self):
        raw = [{"bboxes": [{"smiles": "C"}], "corefs": []}]
        converted_image = object()
        with (
            mock.patch.object(get_molecular_agent.Image, "open") as open_image,
            mock.patch.object(
                get_molecular_agent.model,
                "extract_molecule_corefs_from_figures",
                side_effect=[[], [{"bboxes": [], "corefs": []}], raw],
            ) as predict,
        ):
            open_image.return_value.convert.return_value = converted_image
            result = get_molecular_agent._predict_molecular("input.png")

        self.assertIs(result, raw)
        self.assertEqual(predict.call_count, 3)
        predict.assert_called_with([converted_image])

    def test_molecular_prediction_returns_safe_schema_after_three_empty_results(self):
        with (
            mock.patch.object(get_molecular_agent.Image, "open") as open_image,
            mock.patch.object(
                get_molecular_agent.model,
                "extract_molecule_corefs_from_figures",
                return_value=[],
            ) as predict,
        ):
            open_image.return_value.convert.return_value = object()
            result = get_molecular_agent._predict_molecular("input.png")

        self.assertEqual(predict.call_count, 3)
        self.assertEqual(result, [{"bboxes": [], "corefs": []}])

    def test_local_opsin_uses_an_isolated_temporary_input(self):
        observed = {}

        def fake_py2opsin(name, *, tmp_fpath):
            observed["name"] = name
            observed["path"] = Path(tmp_fpath)
            self.assertTrue(observed["path"].parent.is_dir())
            return "CC"

        fake_module = types.SimpleNamespace(py2opsin=fake_py2opsin)
        with mock.patch.dict(sys.modules, {"py2opsin": fake_module}):
            result = chemistry_helper._local_opsin_smiles("ethane")

        self.assertEqual(result, "CC")
        self.assertEqual(observed["name"], "ethane")
        self.assertFalse(observed["path"].parent.exists())

    def test_successful_llm_symbol_conversion_is_cached_per_request(self):
        class SymbolBackend:
            provider_name = "codex"
            config = types.SimpleNamespace(model="test-model")

            def __init__(self):
                self.calls = 0

            def generate(self, _request):
                self.calls += 1
                return LLMResponse(content='{"smiles": "[CH3]"}')

        backend = SymbolBackend()
        with backend_scope(backend):
            self.assertEqual(
                molnextr_chemistry._llm_symbol_to_smiles("unusual-group"),
                "[CH3]",
            )
            self.assertEqual(
                molnextr_chemistry._llm_symbol_to_smiles("unusual-group"),
                "[CH3]",
            )
        self.assertEqual(backend.calls, 1)

        # A new top-level request must not inherit chemistry from the old one.
        with backend_scope(backend):
            self.assertEqual(
                molnextr_chemistry._llm_symbol_to_smiles("unusual-group"),
                "[CH3]",
            )
        self.assertEqual(backend.calls, 2)

    def test_molecular_tool_reuses_raw_prediction_without_stripping_graph(self):
        raw = [
            {
                "bboxes": [
                    {
                        "category": "[Mol]",
                        "bbox": [0, 0, 1, 1],
                        "symbols": ["C"],
                        "coords": [[0.5, 0.5]],
                        "edges": [],
                        "atoms": [{"atom_symbol": "C"}],
                        "molfile": "molfile",
                        "score": 0.9,
                    }
                ],
                "corefs": [],
            }
        ]
        cache = {}
        predict = mock.Mock(return_value=raw)
        tool = caching_molecular_tool(cache, predict)
        compact = json.loads(tool("input.png"))
        cached_compact = json.loads(tool("input.png"))

        predict.assert_called_once_with("input.png")
        self.assertIs(cache["raw_prediction"], raw)
        self.assertEqual(cached_compact, compact)
        self.assertNotIn("coords", compact[0]["bboxes"][0])
        self.assertNotIn("edges", compact[0]["bboxes"][0])
        self.assertEqual(raw[0]["bboxes"][0]["coords"], [[0.5, 0.5]])
        self.assertEqual(raw[0]["bboxes"][0]["atoms"][0]["atom_symbol"], "C")

        fallback_cache = {}
        fallback_predict = mock.Mock(return_value=raw)
        fallback_raw = molecular_results_for_request(
            fallback_cache,
            fallback_predict,
            "input.png",
        )
        repeated_raw = molecular_results_for_request(
            fallback_cache,
            fallback_predict,
            "input.png",
        )
        fallback_predict.assert_called_once_with("input.png")
        self.assertIs(fallback_raw, raw)
        self.assertIs(repeated_raw, raw)

    def test_generic_vision_projection_keeps_chemistry_and_retained_raw_graph(self):
        raw = {
            "reaction_prediction": [
                {
                    "reactants": [
                        {
                            "smiles": "[1*]C#C[2*]",
                            "symbols": ["[R1]", "C", "C", "[R2]"],
                            "bbox": [0.1, 0.2, 0.3, 0.4],
                            "coords": [[0.1, 0.2]],
                            "edges": [[0]],
                            "molfile": "large molfile",
                            "atoms": [{"atom_symbol": "C"}],
                            "bonds": [],
                            "category_id": 1,
                            "score": 2011,
                        }
                    ],
                    "corefs": [[0, 1]],
                }
            ]
        }

        compact = compact_vision_tool_value(raw)

        molecule = compact["reaction_prediction"][0]["reactants"][0]
        self.assertEqual(molecule["smiles"], "[1*]C#C[2*]")
        self.assertEqual(molecule["symbols"], ["[R1]", "C", "C", "[R2]"])
        self.assertEqual(molecule["bbox"], [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(compact["reaction_prediction"][0]["corefs"], [[0, 1]])
        for field in (
            "coords",
            "edges",
            "molfile",
            "atoms",
            "bonds",
            "category_id",
            "score",
        ):
            self.assertNotIn(field, molecule)
        self.assertEqual(
            raw["reaction_prediction"][0]["reactants"][0]["coords"],
            [[0.1, 0.2]],
        )

    def test_final_synthesis_preserves_upstream_input_and_timeout(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        function_source = next(
            ast.get_source_segment(source, node)
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_chemeagle_cloud_impl"
        )

        self.assertIn(
            "agent_name: agent_result",
            function_source,
        )
        self.assertIn("'./prompt/prompt_final_simple_version.txt'", function_source)
        self.assertIn("'url': f'data:image/png;base64,{base64_image}'", function_source)
        self.assertIn('"synthesis_timeout"', function_source)
        self.assertGreaterEqual(
            function_source.count("graphical_input=main_area_result"),
            2,
        )

        r_group_source = (ROOT / "get_R_group_sub_agent.py").read_text(
            encoding="utf-8"
        )
        nested_function = next(
            ast.get_source_segment(r_group_source, node)
            for node in ast.parse(r_group_source).body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_run_image_tool_agent_with_results"
        )
        self.assertIn("llm_value = value", nested_function)
        self.assertNotIn("compact_vision_tool_value", nested_function)

    def test_all_molecular_agent_variants_use_request_scoped_raw_cache(self):
        source = (ROOT / "get_molecular_agent.py").read_text(encoding="utf-8")
        functions = {
            node.name: ast.get_source_segment(source, node)
            for node in ast.parse(source).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        variants = (
            "process_reaction_image_with_multiple_products_and_text",
            "process_reaction_image_with_multiple_products_and_text_correctR",
            "process_reaction_image_with_multiple_products_and_text_correctmultiR",
            "process_reaction_image_with_multiple_products_and_text_correctmultiR_OS",
        )
        for variant in variants:
            with self.subTest(variant=variant):
                function_source = functions[variant]
                self.assertIn("_caching_molecular_tool", function_source)
                self.assertIn("_molecular_results_for_request", function_source)
                self.assertNotIn(
                    "extract_molecule_corefs_from_figures",
                    function_source,
                )

    def test_reaction_template_tools_deliver_retried_rxnim_prediction(self):
        source = (ROOT / "get_R_group_sub_agent.py").read_text(encoding="utf-8")
        functions = {
            node.name: ast.get_source_segment(source, node)
            for node in ast.parse(source).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for variant in (
            "get_full_reaction_template",
            "get_full_reaction_template_OS",
        ):
            with self.subTest(variant=variant):
                function_source = functions[variant]
                self.assertIn(
                    "_predict_reaction_with_empty_retries",
                    function_source,
                )
                self.assertIn('"reaction_prediction": raw_prediction', function_source)
                self.assertNotIn("model1.predict_image_file", function_source)

    def test_product_variant_agent_handles_empty_reaction_prediction(self):
        molecular = [
            {
                "bboxes": [
                    {
                        "category": "[Mol]",
                        "smiles": "CC",
                        "bbox": [0, 0, 1, 1],
                        "coords": [[0.0, 0.0]],
                        "edges": [[0]],
                    }
                ],
                "corefs": [],
            }
        ]
        agent_output = {"CC": ["4", "no reaction", "reactant template"]}

        with (
            tempfile.NamedTemporaryFile(suffix=".png") as image_file,
            mock.patch.object(
                get_R_group_sub_agent,
                "get_active_backend",
                return_value=object(),
            ),
            mock.patch.object(
                get_R_group_sub_agent,
                "_run_image_tool_agent_with_results",
                return_value=(agent_output, [], {}),
            ),
            mock.patch.object(
                get_R_group_sub_agent,
                "_compensate_missing_molecules",
                side_effect=lambda value, *_args: value,
            ),
            mock.patch.object(
                get_R_group_sub_agent,
                "get_cached_multi_molecular",
                return_value=molecular,
            ),
            mock.patch.object(
                get_R_group_sub_agent,
                "get_cached_raw_results",
                return_value=[],
            ),
            mock.patch.object(get_R_group_sub_agent.Image, "open") as open_image,
            mock.patch.object(
                get_R_group_sub_agent.utils,
                "backout_without_coref",
                return_value=[],
            ),
            mock.patch.object(
                get_R_group_sub_agent,
                "normalize_product_variant_output",
                side_effect=lambda value: value,
            ),
        ):
            open_image.return_value.convert.return_value = object()
            result = (
                get_R_group_sub_agent
                .process_reaction_image_with_product_variant_R_group(
                    image_file.name
                )
            )

        self.assertEqual(
            result["reaction_template"],
            {"reactants": [], "products": []},
        )
        self.assertEqual(result["reactions"], {})
        self.assertEqual(result["original_molecule_list"], agent_output)

    def test_import_main_without_api_environment_or_heavy_models(self):
        env = dict(os.environ)
        for name in (
            "API_KEY",
            "AZURE_ENDPOINT",
            "API_VERSION",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "OPENAI_API_KEY",
        ):
            env.pop(name, None)
        completed = subprocess.run(
            [sys.executable, "-c", "import main; print(main.ChemEagle.__name__)"],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "ChemEagle")

    def test_three_providers_preserve_the_public_dictionary_schema(self):
        expected = {
            "reactions": [
                {
                    "reaction_id": "0_1",
                    "reactants": [],
                    "conditions": [],
                    "products": [],
                }
            ],
            "text_extraction": [],
        }
        for provider in ("azure", "codex", "local"):
            with self.subTest(provider=provider):
                backend = _FakeBackend(provider)
                with mock.patch.object(
                    main, "_chemeagle_cloud_impl", return_value=expected
                ) as implementation:
                    result = main.ChemEagle("unused.png", backend=backend)
                self.assertEqual(result, expected)
                implementation.assert_called_once()
                self.assertFalse(backend.closed, "caller-owned backends must be reusable")

    def test_business_modules_do_not_create_provider_sdk_clients(self):
        files = [
            "main.py",
            "get_observer.py",
            "get_molecular_agent.py",
            "get_reaction_agent.py",
            "get_R_group_sub_agent.py",
            "get_text_agent.py",
            "molnextr/chemistry.py",
        ]
        forbidden = ("AzureOpenAI(", "OpenAI(", ".chat.completions.create(", ".choices[0]")
        for relative in files:
            text = (ROOT / relative).read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, text, f"{relative} still contains {token}")


if __name__ == "__main__":
    unittest.main()
