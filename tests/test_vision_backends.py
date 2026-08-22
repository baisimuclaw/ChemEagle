from __future__ import annotations

import io
import json
import os
import tempfile
import threading
import types
import unittest
from contextvars import copy_context
from unittest import mock

try:
    from PIL import Image
except ImportError:  # provider-only test environments intentionally stay lightweight
    Image = None

from chemeagle_vision.codec import decode_value, encode_value
from chemeagle_vision.config import VisionConfig
from chemeagle_vision.context import call_vision, vision_scope
from chemeagle_vision.errors import VisionConfigurationError
from chemeagle_vision.proxies import (
    vision_chemner,
    vision_chemrxnextractor,
    vision_rxnim,
    vision_toolkit,
)
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
                "CHEMEAGLE_VISION_SLURM_SUBMIT_HOST": "sandbox",
            },
        )
        self.assertEqual(config.provider, "slurm-ssh")
        self.assertEqual(config.device, "cuda")
        self.assertEqual(config.chemrxn_device, "auto")
        self.assertTrue(config.offline)
        self.assertEqual(config.slurm_account, "xlzhang")
        self.assertEqual(config.slurm_submit_host, "sandbox")
        self.assertEqual(config.slurm_memory, "24G")

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
            slurm_submit_host="sandbox",
        )
        backend = RemoteVisionBackend(config)
        command = backend.ssh_command()
        self.assertEqual(command[0], "ssh")
        self.assertIn("BatchMode=yes", command)
        remote = command[-1]
        self.assertTrue(remote.startswith("exec ssh "))
        self.assertIn("sandbox", remote)
        self.assertIn("srun --quiet --unbuffered", remote)
        self.assertIn("--account xlzhang", remote)
        self.assertIn("--gres=gpu:L40S:1", remote)
        self.assertIn("PYTHONNOUSERSITE=1", remote)
        self.assertIn("CHEMEAGLE_OFFLINE=1", remote)
        self.assertIn("--chemrxn-device auto", remote)
        self.assertIn("chemeagle_vision.worker", remote)
        self.assertNotIn("OPENAI", remote)

    def test_slurm_without_submit_host_stays_on_login_host(self):
        config = VisionConfig(
            provider="slurm-ssh",
            ssh_host="cluster-login",
            remote_dir="/shared/ChemEagle",
        )
        remote = RemoteVisionBackend(config).ssh_command()[-1]
        self.assertTrue(remote.startswith("cd "))
        self.assertNotIn("exec ssh", remote)

    def test_ssh_config_expands_home_without_using_a_shell(self):
        config = VisionConfig(
            provider="ssh",
            ssh_host="worker",
            ssh_config="~/.ssh/config",
            remote_dir="/srv/ChemEagle",
        )
        command = RemoteVisionBackend(config).ssh_command()
        self.assertEqual(command[command.index("-F") + 1], os.path.expanduser("~/.ssh/config"))

    def test_remote_stderr_can_be_persisted_locally(self):
        with tempfile.TemporaryDirectory() as temporary:
            log_path = os.path.join(temporary, "remote.log")
            backend = RemoteVisionBackend(
                VisionConfig(
                    provider="ssh",
                    ssh_host="worker",
                    remote_dir="/srv/ChemEagle",
                    remote_log=log_path,
                )
            )

            backend._append_remote_log("model diagnostic")

            with open(log_path, "r", encoding="utf-8") as handle:
                value = handle.read()
            self.assertIn("model diagnostic", value)


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
                vision_chemrxnextractor.get_reactions(["ethanol reacted."])
        self.assertEqual(
            [call[0] for call in backend.calls],
            [
                "rxnim_predict_image",
                "extract_molecule_corefs",
                "molnextr_predict_images",
                "chemner_predict_strings",
                "chemrxn_extract_sentences",
            ],
        )

    def test_concurrent_identical_calls_share_request_cache(self):
        class SlowBackend(_FakeVisionBackend):
            def call(self, method, params):
                self.calls.append((method, params))
                threading.Event().wait(0.03)
                return {"method": method, "value": [1]}

        backend = SlowBackend()
        results = []
        caller_errors = []
        with vision_scope(backend):
            caller_context = copy_context()

            def invoke():
                try:
                    value = caller_context.copy().run(
                        call_vision, "same_method", {"value": 1}
                    )
                    results.append(value)
                except BaseException as exc:
                    caller_errors.append(exc)

            workers = [threading.Thread(target=invoke) for _ in range(2)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(timeout=2)

        self.assertEqual(caller_errors, [])
        self.assertEqual(
            results,
            [
                {"method": "same_method", "value": [1]},
                {"method": "same_method", "value": [1]},
            ],
        )
        self.assertEqual(len(backend.calls), 1)

    def test_graph_to_smiles_postprocessing_stays_local_and_complete(self):
        graph = {
            "chartok_coords": {
                "coords": [[0.1, 0.2]],
                "symbols": ["C"],
            },
            "edges": [[0]],
        }
        image = mock.sentinel.image
        backend = _FakeVisionBackend()
        with (
            vision_scope(backend),
            mock.patch(
                "molnextr.chemistry.convert_graph_to_smiles",
                return_value=(("C",), ("mol",), 1.0),
            ) as convert,
        ):
            result = vision_toolkit.molnextr.convert_graph_to_output(
                [graph], [image]
            )

        convert.assert_called_once_with(
            [[[0.1, 0.2]]],
            [["C"]],
            [[[0]]],
            images=[image],
            num_workers=1,
        )
        self.assertEqual(backend.calls, [])
        self.assertEqual(
            result,
            [
                {
                    "smiles": "C",
                    "symbols": ["C"],
                    "coords": [[0.1, 0.2]],
                    "edges": [[0]],
                    "molfile": "mol",
                }
            ],
        )

class RemoteVisionLifecycleTests(unittest.TestCase):
    def test_close_waits_for_graceful_slurm_chain_exit_after_shutdown_ack(self):
        class FakeProcess:
            def __init__(self):
                self.returncode = None
                self.wait_timeouts = []
                self.terminate_calls = 0
                self.kill_calls = 0

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                self.wait_timeouts.append(timeout)
                self.returncode = 0
                return 0

            def terminate(self):
                self.terminate_calls += 1

            def kill(self):
                self.kill_calls += 1

        process = FakeProcess()
        backend = RemoteVisionBackend(
            VisionConfig(
                provider="slurm-ssh",
                ssh_host="cluster-login",
                remote_dir="/shared/ChemEagle",
            ),
            process_factory=mock.Mock(),
        )
        backend._process = process
        backend._started = True
        with mock.patch.object(
            backend,
            "_request",
            return_value={"status": "closing"},
        ) as request:
            backend.close()

        request.assert_called_once_with("shutdown", {}, timeout=5)
        self.assertEqual(process.wait_timeouts, [30])
        self.assertEqual(process.terminate_calls, 0)
        self.assertEqual(process.kill_calls, 0)


class WorkerProtocolTests(unittest.TestCase):
    def test_local_model_offline_mode_does_not_disable_chemical_resolvers(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            VisionRuntime(VisionConfig(provider="local", offline=True))
            self.assertNotIn("CHEMEAGLE_OFFLINE", os.environ)
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")

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
                self.kwargs = kwargs
                self.rxnim = FakeComponent()
                self.molnextr = FakeComponent()
                self.chemner = FakeComponent()
                self.chemrxnextractor = types.SimpleNamespace(
                    rxn_extractor=types.SimpleNamespace(
                        get_reactions=lambda strings: [{"count": len(strings)}]
                    )
                )

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
            runtime.call(
                "chemrxn_extract_sentences", {"sentences": ["EtOH reacted."]}
            )
        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0].kwargs["chemrxn_device"], "auto")

    def test_chemrxn_batch_assertion_falls_back_sentence_by_sentence(self):
        class FakeExtractor:
            def get_reactions(self, strings):
                if len(strings) > 1:
                    raise AssertionError("length mismatch")
                if strings[0] == "bad":
                    raise RuntimeError("unreadable")
                return [{"sentence": strings[0]}]

        runtime = VisionRuntime(VisionConfig(provider="local", device="cpu"))
        runtime._toolkit = types.SimpleNamespace(
            chemrxnextractor=types.SimpleNamespace(
                rxn_extractor=FakeExtractor()
            )
        )

        result = runtime.call(
            "chemrxn_extract_sentences",
            {"sentences": ["first", "bad", "last"]},
        )

        self.assertEqual(
            result,
            [{"sentence": "first"}, {"sentence": "last"}],
        )


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
