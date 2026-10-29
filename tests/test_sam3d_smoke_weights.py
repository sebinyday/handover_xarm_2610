import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.smoke_sam3d import check_stage1_weights


class SmokeWeightTests(unittest.TestCase):
    def test_only_reports_loaded_after_real_stage1_model_contract_is_met(self):
        engine = SimpleNamespace(
            pipeline=SimpleNamespace(
                models={
                    "ss_generator": object(),
                    "ss_decoder": object(),
                    "slat_generator": None,
                    "slat_decoder_mesh": None,
                }
            ),
            model_load_elapsed_s=2.5,
        )
        with patch(
            "tools.sam3d_generate.Sam3DStage1Engine", return_value=engine
        ) as constructor:
            result = check_stage1_weights({"settings": "test"}, Path("/models"))
        constructor.assert_called_once_with(
            {"settings": "test"}, sam3d_repo=Path("/models")
        )
        self.assertTrue(result["weights_loaded"])
        self.assertEqual(result["loaded_models"], ["ss_decoder", "ss_generator"])

    def test_load_failure_cannot_be_reported_as_passed(self):
        with patch(
            "tools.sam3d_generate.Sam3DStage1Engine",
            side_effect=RuntimeError("missing weights"),
        ):
            with self.assertRaisesRegex(RuntimeError, "missing weights"):
                check_stage1_weights({}, Path("/models"))

    def test_missing_stage1_or_unexpected_stage2_models_are_rejected(self):
        for models in (
            {"ss_generator": object()},
            {
                "ss_generator": object(),
                "ss_decoder": object(),
                "slat_generator": object(),
            },
        ):
            engine = SimpleNamespace(pipeline=SimpleNamespace(models=models))
            with (
                self.subTest(models=list(models)),
                patch("tools.sam3d_generate.Sam3DStage1Engine", return_value=engine),
            ):
                with self.assertRaisesRegex(RuntimeError, "Expected only stage1"):
                    check_stage1_weights({}, Path("/models"))
