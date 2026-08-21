from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import main
import get_reaction_agent
import get_text_agent


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
    def test_text_models_receive_only_sentence_like_prose(self):
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
            os.environ, {"CHEMEAGLE_TEXT_PROSE_FILTER": "0"}, clear=False
        ):
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
