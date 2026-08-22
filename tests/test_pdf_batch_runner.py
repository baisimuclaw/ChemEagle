from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts import run_pdf_batch


class PdfBatchRunnerTests(unittest.TestCase):
    def test_manifest_resolves_pdf_relative_to_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf = root / "paper.pdf"
            pdf.write_bytes(b"pdf")
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps({"items": [{"id": "paper-1", "pdf": "paper.pdf"}]}),
                encoding="utf-8",
            )

            manifest = run_pdf_batch.load_manifest(manifest_path)

            self.assertEqual(manifest["items"][0]["pdf"], str(pdf))

    def test_extract_pdf_records_all_pngs_and_reuses_them(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf = root / "paper.pdf"
            pdf.write_bytes(b"pdf")
            item = {"id": "paper-1", "pdf": str(pdf)}
            calls = []

            def fake_extractor(*, pdf_dir, image_dir, model_size):
                calls.append((pdf_dir, image_dir, model_size))
                output = Path(image_dir)
                (output / "page_2.png").write_bytes(b"two")
                (output / "page_1.png").write_bytes(b"one")
                (output / "ignored.txt").write_text("ignored", encoding="utf-8")

            first = run_pdf_batch.extract_pdf(
                item, root, "large", extractor=fake_extractor
            )
            second = run_pdf_batch.extract_pdf(
                item, root, "large", extractor=fake_extractor
            )

            self.assertEqual(len(calls), 1)
            self.assertEqual(first["image_count"], 2)
            self.assertEqual(
                [image["id"] for image in first["images"]],
                ["page_1", "page_2"],
            )
            self.assertEqual(first, second)

    def test_extract_and_full_phases_have_independent_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {
                "items": [{"id": "paper-1", "pdf": "/input/paper.pdf"}]
            }
            run_pdf_batch.write_json(
                root / "extraction" / "paper-1.json", {"status": "succeeded"}
            )

            extract_status = run_pdf_batch.initial_status(manifest, root, 1, True)
            full_status = run_pdf_batch.initial_status(manifest, root, 1, False)

            self.assertEqual(extract_status["items"]["paper-1"]["state"], "succeeded")
            self.assertEqual(full_status["items"]["paper-1"]["state"], "pending")

    def test_paper_report_is_not_successful_when_an_image_failed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = run_pdf_batch.write_paper_report(
                {"id": "paper-1", "pdf": "/input/paper.pdf"},
                {"image_count": 2},
                [
                    {"id": "figure-1", "status": "succeeded"},
                    {"id": "figure-2", "status": "failed"},
                ],
                root,
            )

            self.assertEqual(report["status"], "completed_with_failures")
            self.assertEqual(report["succeeded_count"], 1)
            self.assertEqual(report["failed_images"], ["figure-2"])


if __name__ == "__main__":
    unittest.main()
