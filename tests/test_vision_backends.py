from __future__ import annotations

import io
import json
import os
import tempfile
import types
import unittest
from unittest import mock

try:
    from PIL import Image
except ImportError:  # provider-only test environments intentionally stay lightweight
    Image = None

from chemeagle_vision.codec import decode_value, encode_value
from chemeagle_vision.config import VisionConfig
from chemeagle_vision.context import vision_scope
from chemeagle_vision.errors import VisionConfigurationError
from chemeagle_vision.proxies import vision_chemner, vision_rxnim, vision_toolkit
from chemeagle_vision.remote import RemoteVisionBackend
from chemeagle_vision.runtime import VisionRuntime
from chemeagle_vision.worker import serve_stdio


class VisionConfigTests(unittest.TestCase):
    def test_aliases_and_remote_defaults(self):
        config = VisionConfig.from_env(
            "cluster",
            env={
                "CHEMEAGLE_VISION_SSH_HOST": "cuhk",
                "CHEMEAGLE_VISION_REMOTE_DIR": "/shared/ChemEagle",
                "CHEMEAGLE_VISION_SLURM_ACCOUNT": "xlzhang",
            },
        )
        self.assertEqual(config.provider, "slurm-ssh")
        self.assertEqual(config.device, "cuda")
        self.assertTrue(config.offline)
        self.assertEqual(config.slurm_account, "xlzhang")

    def test_remote_requires_host_and_directory(self):
        config = VisionConfig(provider="ssh", ssh_host="worker")
        with self.assertRaisesRegex(ValueError, "REMOTE_DIR"):
            config.validate_remote()

    def test_slurm_command_is_fixed_and_uses_scheduler(self):
        config = VisionConfig(
            provider="slurm-ssh",
            device="cuda",
            offline=True,
            ssh_host="cuhk",
            ssh_config="/home/user/.ssh/config",
            remote_dir="/shared/ChemEagle",
            remote_python="/shared/env/bin/python",
            model_dir="/shared/models",
            slurm_account="xlzhang",
            slurm_qos="xlzhang",
            slurm_reservation="xlzhang_gpu",
            slurm_partition="chpc",
        )
        backend = RemoteVisionBackend(config)
        command = backend.ssh_command()
        self.assertEqual(command[0], "ssh")
        self.assertIn("BatchMode=yes", command)
        remote = command[-1]
        self.assertIn("srun --quiet --unbuffered", remote)
        self.assertIn("--account xlzhang", remote)
        self.assertIn("--gres=gpu:L40S:1", remote)
        self.assertIn("PYTHONNOUSERSITE=1", remote)
        self.assertIn("CHEMEAGLE_OFFLINE=1", remote)
        self.assertIn("chemeagle_vision.worker", remote)
        self.assertNotIn("OPENAI", remote)

    def test_ssh_config_expands_home_without_using_a_shell(self):
        config = VisionConfig(
            provider="ssh",
            ssh_host="worker",
            ssh_config="~/.ssh/config",
            remote_dir="/srv/ChemEagle",
        )
        command = RemoteVisionBackend(config).ssh_command()
        self.assertEqual(command[command.index("-F") + 1], os.path.expanduser("~/.ssh/config"))


class VisionCodecTests(unittest.TestCase):
    @unittest.skipIf(Image is None, "Pillow is part of the full vision environment")
    def test_image_numpy_tuple_round_trip(self):
        import numpy as np

        value = {
            "image": Image.new("RGB", (3, 2), "red"),
            "array": np.arange(6, dtype=np.float32).reshape(2, 3),
            "bbox": (0.1, 0.2, 0.3, 0.4),
        }
        decoded = decode_value(encode_value(value))
        self.assertEqual(decoded["image"].size, (3, 2))
        self.assertTrue(np.array_equal(decoded["array"], value["array"]))
        self.assertEqual(decoded["bbox"], value["bbox"])


class _FakeVisionBackend:
    provider_name = "fake"

    def __init__(self):
        self.calls = []
        self.close_calls = 0

    def call(self, method, params):
        self.calls.append((method, params))
        return {"method": method}

    def health(self):
        return {"status": "ok"}

    def close(self):
        self.close_calls += 1


class VisionProxyTests(unittest.TestCase):
    @unittest.skipIf(Image is None, "Pillow is part of the full vision environment")
    def test_existing_agent_compatibility_proxies_use_active_backend(self):
        backend = _FakeVisionBackend()
        with tempfile.NamedTemporaryFile(suffix=".png") as handle:
            Image.new("RGB", (4, 4), "white").save(handle.name)
            with vision_scope(backend):
                self.assertEqual(
                    vision_rxnim.predict_image_file(handle.name, molnextr=True),
                    {"method": "rxnim_predict_image"},
                )
                vision_toolkit.extract_molecule_corefs_from_figures(
                    [Image.new("RGB", (2, 2))]
                )
                vision_toolkit.molnextr.predict_images([])
                vision_chemner.predict_strings(["ethanol"])
        self.assertEqual(
            [call[0] for call in backend.calls],
            [
                "rxnim_predict_image",
                "extract_molecule_corefs",
                "molnextr_predict_images",
                "chemner_predict_strings",
            ],
        )


class WorkerProtocolTests(unittest.TestCase):
    def test_stdio_dispatch_and_unknown_method_failure(self):
        class FakeRuntime:
            def call(self, method, params):
                if method == "echo":
                    print("model diagnostic")
                    return params
                raise VisionConfigurationError("not allowlisted")

        requests = [
            {"id": 1, "method": "echo", "params": encode_value({"x": (1, 2)})},
            {"id": 2, "method": "shell", "params": {}},
            {"id": 3, "method": "shutdown", "params": {}},
        ]
        stdin = io.StringIO("".join(json.dumps(item) + "\n" for item in requests))
        stdout = io.StringIO()
        stderr = io.StringIO()
        code = serve_stdio(FakeRuntime(), stdin=stdin, stdout=stdout, stderr=stderr)
        self.assertEqual(code, 0)
        responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
        self.assertEqual(decode_value(responses[0]["result"]), {"x": (1, 2)})
        self.assertEqual(responses[1]["error"]["type"], "VisionConfigurationError")
        self.assertEqual(responses[2]["result"]["status"], "closing")
        self.assertIn("model diagnostic", stderr.getvalue())

    @unittest.skipIf(Image is None, "Pillow is part of the full vision environment")
    def test_runtime_reuses_one_toolkit_for_all_models(self):
        instances = []

        class FakeComponent:
            def predict_image_file(self, path, **options):
                return [{"path_exists": os.path.exists(path), "options": options}]

            def predict_images(self, values, **options):
                return [{"count": len(values), "options": options}]

            def convert_graph_to_output(self, graphs, images):
                return [{"graphs": len(graphs), "images": len(images)}]

            def predict_strings(self, strings, **options):
                return [[("chemical", [0, len(strings[0])])]]

        class FakeToolkit:
            def __init__(self, **kwargs):
                instances.append(self)
                self.rxnim = FakeComponent()
                self.molnextr = FakeComponent()
                self.chemner = FakeComponent()

            def extract_molecule_corefs_from_figures(self, figures, **options):
                return [{"count": len(figures), "options": options}]

        module = types.ModuleType("chemietoolkit")
        module.ChemIEToolkit = FakeToolkit
        runtime = VisionRuntime(VisionConfig(provider="local", device="cpu"))
        with mock.patch.dict("sys.modules", {"chemietoolkit": module}):
            image = Image.new("RGB", (2, 2), "white")
            runtime.call("rxnim_predict_image", {"image": image, "options": {}})
            runtime.call(
                "extract_molecule_corefs", {"figures": [image], "options": {}}
            )
            runtime.call("chemner_predict_strings", {"strings": ["EtOH"]})
        self.assertEqual(len(instances), 1)


class ApplicationVisionLifecycleTests(unittest.TestCase):
    def test_chemeagle_closes_factory_created_vision_backend(self):
        import main

        llm = mock.Mock()
        vision = _FakeVisionBackend()
        expected = {"reactions": []}
        with (
            mock.patch.object(main, "create_backend", return_value=llm),
            mock.patch.object(main, "create_vision_backend", return_value=vision),
            mock.patch.object(
                main, "_chemeagle_cloud_impl", return_value=expected
            ) as impl,
        ):
            result = main.ChemEagle("image.png", provider="codex", vision_provider="ssh")
        self.assertEqual(result, expected)
        impl.assert_called_once()
        llm.close.assert_called_once_with()
        self.assertEqual(vision.close_calls, 1)

    def test_chemeagle_preserves_caller_owned_backends(self):
        import main

        llm = mock.Mock()
        vision = mock.Mock(spec=_FakeVisionBackend())
        vision.provider_name = "fake"
        with mock.patch.object(
            main, "_chemeagle_cloud_impl", return_value={"ok": True}
        ):
            result = main.ChemEagle(
                "image.png",
                backend=llm,
                vision_backend=vision,
            )
        self.assertEqual(result, {"ok": True})
        llm.close.assert_not_called()
        vision.close.assert_not_called()


if __name__ == "__main__":
    unittest.main()
