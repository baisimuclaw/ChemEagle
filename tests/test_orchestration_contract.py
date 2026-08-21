from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import main


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
