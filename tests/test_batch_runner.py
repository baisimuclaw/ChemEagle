from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import run_batch


class BatchRunnerTests(unittest.TestCase):
    def test_manifest_paths_are_resolved_relative_to_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "figure.png"
            image.write_bytes(b"image")
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps({"items": [{"id": "paper-1", "image": "figure.png"}]}),
                encoding="utf-8",
            )

            manifest = run_batch.load_manifest(manifest_path)

            self.assertEqual(manifest["items"][0]["image"], str(image))

    def test_completed_results_are_skipped_on_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = root / "results" / "paper-1.json"
            run_batch.write_json(result, {"status": "succeeded"})
            manifest = {"items": [{"id": "paper-1", "image": "/figure.png"}]}

            status = run_batch.initial_status(manifest, root, 2)

            self.assertEqual(status["items"]["paper-1"]["state"], "succeeded")
            self.assertEqual(status["max_parallel"], 2)

    def test_config_snapshot_redacts_credentials(self):
        with mock.patch.dict(
            os.environ,
            {
                "CHEMEAGLE_VISION_PROVIDER": "slurm-ssh",
                "CHEMEAGLE_API_KEY": "secret",
                "CHEMEAGLE_TOKEN": "secret",
                "CODEX_MODEL": "gpt-test",
            },
            clear=True,
        ):
            selected = run_batch.selected_environment()

        self.assertEqual(selected["CHEMEAGLE_VISION_PROVIDER"], "slurm-ssh")
        self.assertEqual(selected["CODEX_MODEL"], "gpt-test")
        self.assertNotIn("CHEMEAGLE_API_KEY", selected)
        self.assertNotIn("CHEMEAGLE_TOKEN", selected)


if __name__ == "__main__":
    unittest.main()
