from __future__ import annotations

import unittest
from unittest import mock

import torch

import pdf_extraction
from pdfmodel import methods


class _MovableValue:
    def __init__(self):
        self.device = None

    def to(self, device):
        self.device = device
        return self


class _FakeProcessor:
    def __init__(self):
        self.input_ids = _MovableValue()
        self.pixel_values = _MovableValue()

    def __call__(self, **_kwargs):
        return {
            "input_ids": self.input_ids,
            "pixel_values": self.pixel_values,
        }

    def batch_decode(self, _generated_ids, skip_special_tokens=False):
        del skip_special_tokens
        return ["generated"]

    def post_process_generation(self, _text, task, image_size):
        self.post_process_call = (task, image_size)
        return {"<OD>": {"bboxes": [], "labels": []}}


class _FakeModel:
    def __init__(self):
        self.device = None
        self.eval_called = False
        self.generate_kwargs = None

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return [[1, 2, 3]]


class _FakeImage:
    width = 800
    height = 600


class VisualHeistDeviceTests(unittest.TestCase):
    def test_public_pdf_entrypoint_forwards_explicit_cuda_device(self):
        with mock.patch.object(
            pdf_extraction,
            "_pdf_to_figures_and_tables",
        ) as extract:
            pdf_extraction.run_pdf(
                pdf_dir="paper.pdf",
                image_dir="crops",
                model_size="large",
                device="cuda",
            )

        extract.assert_called_once_with(
            "paper.pdf",
            "crops",
            large_model=True,
            device="cuda",
        )

    def test_cpu_remains_the_default_device(self):
        self.assertEqual(methods._resolve_device(None), torch.device("cpu"))

    def test_auto_uses_cuda_when_available(self):
        with mock.patch.object(methods.torch.cuda, "is_available", return_value=True):
            self.assertEqual(methods._resolve_device("auto"), torch.device("cuda"))

    def test_explicit_cuda_fails_when_unavailable(self):
        with mock.patch.object(methods.torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                methods._resolve_device("cuda")

    def test_model_is_moved_to_requested_device_and_set_to_eval(self):
        fake_model = _FakeModel()
        fake_processor = object()
        with (
            mock.patch.object(methods.os.path, "exists", return_value=True),
            mock.patch.object(methods, "load_file", return_value={}),
            mock.patch.object(
                methods.AutoModelForCausalLM,
                "from_pretrained",
                return_value=fake_model,
            ),
            mock.patch.object(
                methods.AutoProcessor,
                "from_pretrained",
                return_value=fake_processor,
            ),
        ):
            model, processor = methods._create_model(
                "visualheist-test",
                "large",
                device="cpu",
            )

        self.assertIs(model, fake_model)
        self.assertIs(processor, fake_processor)
        self.assertEqual(model.device, torch.device("cpu"))
        self.assertTrue(model.eval_called)

    def test_detection_moves_inputs_to_requested_device(self):
        model = _FakeModel()
        processor = _FakeProcessor()

        annotation = methods._tf_id_detection(
            _FakeImage(),
            model,
            processor,
            device="cpu",
        )

        self.assertEqual(processor.input_ids.device, torch.device("cpu"))
        self.assertEqual(processor.pixel_values.device, torch.device("cpu"))
        self.assertIs(model.generate_kwargs["input_ids"], processor.input_ids)
        self.assertIs(model.generate_kwargs["pixel_values"], processor.pixel_values)
        self.assertEqual(model.generate_kwargs["num_beams"], 3)
        self.assertEqual(annotation, {"bboxes": [], "labels": []})


if __name__ == "__main__":
    unittest.main()
