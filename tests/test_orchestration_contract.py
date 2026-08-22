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

try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    _FULL_DEPENDENCIES = False
else:
    _FULL_DEPENDENCIES = True

if _FULL_DEPENDENCIES:
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


@unittest.skipUnless(
    _FULL_DEPENDENCIES,
    "full ChemEAGLE dependencies are not installed",
)
class ImportAndOrchestrationTests(unittest.TestCase):
    def test_chemner_receives_complete_upstream_ocr_paragraph_by_default(self):
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(
                get_text_agent,
                "_ocr_image_text",
                return_value="First sentence. trailing label R1",
            ),
            mock.patch.object(
                get_text_agent.model2,
                "predict_strings",
                return_value=[{"entities": []}],
            ) as predict,
        ):
            result = get_text_agent.NER_from_text_in_image("scheme.png")

        self.assertEqual(result, [{"entities": []}])
        predict.assert_called_once_with(["First sentence. trailing label R1"])

    def test_ocr_is_not_reused_across_independent_tool_calls(self):
        opened = mock.MagicMock()
        with (
            mock.patch.object(get_text_agent, "configure_tesseract"),
            mock.patch.object(get_text_agent.Image, "open", return_value=opened),
            mock.patch.object(
                get_text_agent.pytesseract,
                "image_to_string",
                side_effect=["first", "second"],
            ) as image_to_string,
        ):
            first = get_text_agent._ocr_image_text("scheme.png")
            second = get_text_agent._ocr_image_text("scheme.png")

        self.assertEqual((first, second), ("first", "second"))
        self.assertEqual(image_to_string.call_count, 2)

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

    def test_reaction_agents_never_replace_empty_rxnim_with_llm_guess(self):
        source = (ROOT / "get_reaction_agent.py").read_text(encoding="utf-8")
        functions = {
            node.name: ast.get_source_segment(source, node)
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef)
        }
        for name in (
            "get_reaction_withatoms",
            "get_reaction_withatoms_correctR",
            "get_reaction_withatoms_correctR_OS",
        ):
            with self.subTest(name=name):
                function_source = functions[name]
                self.assertIn("if not input2", function_source)
                self.assertIn("return []", function_source)
                self.assertNotIn(
                    "input2[0] if input2 else gpt_output",
                    function_source,
                )

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

    def test_llm_symbol_conversion_preserves_upstream_independent_calls(self):
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
        self.assertEqual(backend.calls, 2)

        # A new top-level request is likewise independent.
        with backend_scope(backend):
            self.assertEqual(
                molnextr_chemistry._llm_symbol_to_smiles("unusual-group"),
                "[CH3]",
            )
        self.assertEqual(backend.calls, 3)

    def test_llm_symbol_provider_failure_preserves_upstream_optional_fallback(self):
        backend = mock.Mock()
        backend.provider_name = "codex"
        backend.config = types.SimpleNamespace(model="test-model")
        backend.generate.side_effect = RuntimeError("provider unavailable")

        with backend_scope(backend):
            result = molnextr_chemistry._llm_symbol_to_smiles("UnresolvedGroup")

        self.assertIsNone(result)

    def test_llm_symbol_conversion_keeps_upstream_json_object_mode(self):
        backend = mock.Mock()
        backend.provider_name = "azure"
        backend.config = types.SimpleNamespace(model="test-model")
        backend.generate.return_value = LLMResponse(content='{"smiles": "[CH3]"}')

        with backend_scope(backend):
            result = molnextr_chemistry._llm_symbol_to_smiles("Me")

        self.assertEqual(result, "[CH3]")
        request = backend.generate.call_args.args[0]
        self.assertTrue(request.json_mode)
        self.assertIsNone(request.output_schema)

    def test_llm_symbol_conversion_keeps_upstream_embedded_json_fallback(self):
        backend = mock.Mock()
        backend.provider_name = "codex"
        backend.config = types.SimpleNamespace(model="test-model")
        backend.generate.return_value = LLMResponse(
            content='analysis {"smiles": "[CH3]"} complete'
        )

        with backend_scope(backend):
            result = molnextr_chemistry._llm_symbol_to_smiles("Me")

        self.assertEqual(result, "[CH3]")

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
        self.assertIn(
            'llm_value = {"image_path": image_path, name: value}',
            nested_function,
        )
        self.assertNotIn("compact_vision_tool_value", nested_function)

        table_function = next(
            ast.get_source_segment(r_group_source, node)
            for node in ast.parse(r_group_source).body
            if isinstance(node, ast.FunctionDef)
            and node.name == "process_reaction_image_with_table_R_group"
        )
        self.assertNotIn("original_reactant.get('symbols'", table_function)
        self.assertNotIn("original_reactant.get('smiles'", table_function)
        self.assertNotIn("original_product.get('symbols'", table_function)
        self.assertNotIn("original_product.get('smiles'", table_function)
        self.assertIn("reaction_id = reaction['reaction_id']", table_function)
        self.assertIn("for reaction in input2['reactions']", table_function)

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

    def test_upstream_agent_specific_tool_policies_are_preserved(self):
        molecular_source = (ROOT / "get_molecular_agent.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("followup_on_no_tool_call=True", molecular_source)

        reaction_source = (ROOT / "get_reaction_agent.py").read_text(
            encoding="utf-8"
        )
        self.assertGreaterEqual(
            reaction_source.count("followup_on_no_tool_call=True"),
            3,
        )

        r_group_source = (ROOT / "get_R_group_sub_agent.py").read_text(
            encoding="utf-8"
        )
        self.assertGreaterEqual(r_group_source.count('unknown_tool_policy="skip"'), 2)
        self.assertGreaterEqual(
            r_group_source.count('unknown_tool_policy="fallback"'),
            2,
        )

        text_source = (ROOT / "get_text_agent.py").read_text(encoding="utf-8")
        self.assertGreaterEqual(text_source.count('unknown_tool_policy="skip"'), 2)

    def test_reaction_and_molecular_tool_payloads_keep_upstream_envelope(self):
        class InvokingBackend:
            provider_name = "azure"
            config = types.SimpleNamespace(model=None)

            def __init__(self):
                self.tool_values = []

            def run_tool_loop(self, _request, _tools, executor):
                for name, handler in executor.items():
                    self.tool_values.append(handler(image_path="model-supplied.png"))
                    self.asserted_name = name
                return LLMResponse(content="{}")

        for module in (get_reaction_agent, get_molecular_agent):
            with self.subTest(module=module.__name__):
                backend = InvokingBackend()
                handler = mock.Mock(return_value={"raw": "prediction"})
                module._run_image_tool_agent(
                    backend,
                    "trusted.png",
                    [{"role": "user", "content": "test"}],
                    [
                        {
                            "type": "function",
                            "function": {
                                "name": "vision_tool",
                                "parameters": {"type": "object"},
                            },
                        }
                    ],
                    "test-model",
                    {"vision_tool": handler},
                )

                handler.assert_called_once_with("trusted.png")
                self.assertEqual(
                    backend.tool_values,
                    [
                        {
                            "image_path": "trusted.png",
                            "vision_tool": {"raw": "prediction"},
                        }
                    ],
                )

    def test_upstream_temperature_policy_is_preserved_per_variant(self):
        def functions(relative):
            source = (ROOT / relative).read_text(encoding="utf-8")
            return {
                node.name: ast.get_source_segment(source, node)
                for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef)
            }

        molecular = functions("get_molecular_agent.py")
        self.assertIn(
            "temperature=0",
            molecular["process_reaction_image_with_multiple_products_and_text"],
        )
        self.assertNotIn(
            "temperature=0",
            molecular["process_reaction_image_with_multiple_products_and_text_correctmultiR"],
        )
        self.assertIn(
            "temperature=0",
            molecular["process_reaction_image_with_multiple_products_and_text_correctmultiR_OS"],
        )

        reaction = functions("get_reaction_agent.py")
        self.assertNotIn("temperature=0", reaction["get_reaction_withatoms_correctR"])
        self.assertIn("temperature=0", reaction["get_reaction_withatoms_correctR_OS"])
        self.assertNotIn("temperature=0", reaction["get_reaction_con"])
        self.assertIn("temperature=0", reaction["get_reaction_con_OS"])

        r_group = functions("get_R_group_sub_agent.py")
        self.assertNotIn(
            "temperature=0",
            r_group["process_reaction_image_with_product_variant_R_group"],
        )
        self.assertIn(
            "temperature=0",
            r_group["process_reaction_image_with_product_variant_R_group_OS"],
        )
        self.assertNotIn(
            "temperature=0",
            r_group["process_reaction_image_with_table_R_group"],
        )
        self.assertIn(
            "temperature=0",
            r_group["process_reaction_image_with_table_R_group_OS"],
        )

    def test_table_r_group_no_tool_fallback_ignores_intermediate_content(self):
        backend = mock.Mock()
        backend.provider_name = "azure"
        backend.config = types.SimpleNamespace(model="test-model")
        backend.run_tool_loop.return_value = LLMResponse(content="not-json")

        result, messages, cache = (
            get_R_group_sub_agent._run_image_tool_agent_with_results(
                backend,
                "scheme.png",
                [{"role": "user", "content": "select tool"}],
                [{"type": "function", "function": {"name": "safe"}}],
                "test-model",
                {"safe": lambda _path: {"reaction": []}},
                fallback_on_no_tool_call=True,
            )
        )

        self.assertEqual(result, {})
        self.assertEqual(messages, [])
        self.assertEqual(cache, {})
        request = backend.run_tool_loop.call_args.args[0]
        self.assertTrue(request.defer_json_validation_for_tools)

    def test_reaction_template_tools_do_not_expose_raw_rxnim_prediction(self):
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
                self.assertNotIn(
                    '"reaction_prediction": raw_prediction',
                    function_source,
                )
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
                return_value=(
                    agent_output,
                    [{"role": "tool", "content": "{}"}],
                    {},
                ),
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
            mock.patch.object(
                get_R_group_sub_agent,
                "draw_mol_bboxes",
                return_value=get_R_group_sub_agent.np.zeros(
                    (2, 2, 3), dtype=get_R_group_sub_agent.np.uint8
                ),
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

    def test_local_provider_routes_to_upstream_os_agent_variants(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        expected_aliases = (
            "get_full_reaction_template_OS as get_full_reaction_template",
            "get_multi_molecular_full_OS as get_multi_molecular_full",
            "process_reaction_image_with_product_variant_R_group_OS as process_reaction_image_with_product_variant_R_group",
            "process_reaction_image_with_table_R_group_OS as process_reaction_image_with_table_R_group",
            "get_reaction_con_OS as get_reaction_con",
            "text_extraction_agent_OS as text_extraction_agent",
        )
        for alias in expected_aliases:
            self.assertIn(alias, source)
        self.assertIn(
            'local_mode = getattr(backend, "provider_name", None) == "local"',
            source,
        )

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
