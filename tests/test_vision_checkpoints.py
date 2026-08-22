from __future__ import annotations

from types import SimpleNamespace
import unittest
import warnings

from chemeagle_vision.checkpoints import load_model_checkpoint


class _FakeModel:
    def __init__(self, *, missing=(), unexpected=()):
        self.missing = list(missing)
        self.unexpected = list(unexpected)
        self.loaded = None
        self.strict = None

    def load_state_dict(self, state_dict, strict=True):
        self.loaded = state_dict
        self.strict = strict
        return SimpleNamespace(
            missing_keys=self.missing,
            unexpected_keys=self.unexpected,
        )


class VisionCheckpointTests(unittest.TestCase):
    def test_prefix_is_removed_and_complete_checkpoint_loads(self):
        model = _FakeModel()
        load_model_checkpoint(
            model,
            {"model.backbone.weight": object()},
            component="RxnIM",
        )
        self.assertEqual(list(model.loaded), ["backbone.weight"])
        self.assertFalse(model.strict)

    def test_missing_inference_weight_is_fatal(self):
        model = _FakeModel(missing=["backbone.body.layer1.weight"])
        with self.assertRaisesRegex(RuntimeError, "missing model weights"):
            load_model_checkpoint(model, {}, component="MolDetect")

    def test_batch_counter_may_be_absent_and_extra_keys_are_reported(self):
        model = _FakeModel(
            missing=["backbone.bn.num_batches_tracked"],
            unexpected=["legacy.head.weight"],
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            load_model_checkpoint(model, {}, component="RxnIM")
        self.assertEqual(len(caught), 1)
        self.assertIn("unused keys", str(caught[0].message))


if __name__ == "__main__":
    unittest.main()
