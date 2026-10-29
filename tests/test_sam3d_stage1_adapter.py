import unittest
from unittest.mock import Mock, patch
from pathlib import Path
from tempfile import TemporaryDirectory
from dataclasses import dataclass, field
from typing import Callable
from perception.sam3d_stage1 import Stage1Initialization, load_stage1_pipeline


@dataclass
class CallablePreprocessor:
    transforms: list[Callable] = field(default_factory=list)


class Upstream:
    def run(self, *args, **kwargs):
        return kwargs


class Adapted(Stage1Initialization, Upstream):
    pass


class Stage1AdapterTests(unittest.TestCase):
    def test_unused_weight_loaders_do_not_open_stage2_assets(self):
        pipeline = Adapted()
        for name in [
            "init_slat_generator",
            "init_slat_condition_embedder",
            "init_slat_decoder_gs",
            "init_slat_decoder_mesh",
            "init_ss_encoder",
        ]:
            self.assertIsNone(
                getattr(pipeline, name)("/missing/config", "/missing/checkpoint")
            )

    def test_only_stage1_run_allowed(self):
        self.assertTrue(Adapted().run(None)["stage1_only"])
        with self.assertRaises(ValueError):
            Adapted().run(None, stage1_only=False)

    def test_root_uses_supported_upstream_kwargs(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "pipeline.yaml"
            path.write_text(
                "_target_: sam3d_objects.pipeline.inference_pipeline_pointmap.InferencePipelinePointMap\ndevice: cuda\n"
            )
            factory = Mock()
            with patch(
                "perception.sam3d_stage1.get_stage1_pipeline_type", return_value=factory
            ):
                load_stage1_pipeline(path)
            kwargs = factory.call_args.kwargs
            self.assertNotIn("stage1_only_init", kwargs)
            self.assertEqual(kwargs["workspace_dir"], directory)
            self.assertEqual(kwargs["rendering_engine"], "pytorch3d")

    def test_unexpected_pipeline_is_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "pipeline.yaml"
            path.write_text("_target_: arbitrary.Class\n")
            with self.assertRaises(ValueError):
                load_stage1_pipeline(path)

    def test_dataclass_callable_preprocessor_is_not_rewrapped(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "pipeline.yaml"
            path.write_text(
                "_target_: sam3d_objects.pipeline.inference_pipeline_pointmap.InferencePipelinePointMap\n"
                "ss_preprocessor:\n"
                f"  _target_: {__name__}.CallablePreprocessor\n"
                "  transforms:\n"
                "    - _target_: builtins.str\n"
                "      _partial_: true\n"
            )
            factory = Mock()
            with patch(
                "perception.sam3d_stage1.get_stage1_pipeline_type", return_value=factory
            ):
                load_stage1_pipeline(path)
            preprocessor = factory.call_args.kwargs["ss_preprocessor"]
            self.assertIsInstance(preprocessor, CallablePreprocessor)
            self.assertEqual(preprocessor.transforms[0](123), "123")
