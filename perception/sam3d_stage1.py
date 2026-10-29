"""Stage1-only initialization using the official SAM3D pipeline API.

Upstream Inference has no stage1_only_init argument. Keep the adaptation in
this project: omit unused SLAT/mesh/GS weights while retaining upstream
sparse-structure and pointmap inference unchanged.
"""

from pathlib import Path


class Stage1Initialization:
    def init_slat_generator(self, *args, **kwargs):
        return None

    def init_slat_condition_embedder(self, *args, **kwargs):
        return None

    def init_slat_decoder_gs(self, *args, **kwargs):
        return None

    def init_slat_decoder_mesh(self, *args, **kwargs):
        return None

    def init_ss_encoder(self, *args, **kwargs):
        return None

    def override_slat_generator_cfg_config(self, *args, **kwargs):
        return None

    def run(self, *args, stage1_only=True, **kwargs):
        if not stage1_only:
            raise ValueError("This pipeline loads only SAM3D stage1 weights")
        return super().run(*args, stage1_only=True, **kwargs)


def get_stage1_pipeline_type():
    from sam3d_objects.pipeline.inference_pipeline_pointmap import (
        InferencePipelinePointMap,
    )

    class Stage1PointMapPipeline(Stage1Initialization, InferencePipelinePointMap):
        pass

    return Stage1PointMapPipeline


def build_stage1_pipeline(**arguments):
    """Hydra target that preserves instantiated preprocessors as Python objects."""
    return get_stage1_pipeline_type()(**arguments)


def load_stage1_pipeline(config_file, *, compile=False):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    path = Path(config_file).resolve()
    config = OmegaConf.load(path)
    if (
        config.get("_target_")
        != "sam3d_objects.pipeline.inference_pipeline_pointmap.InferencePipelinePointMap"
    ):
        raise ValueError(
            "Unexpected SAM3D pipeline target; use the pinned pointmap configuration"
        )
    config.rendering_engine = "pytorch3d"
    config.compile_model = bool(compile)
    config.workspace_dir = str(path.parent)
    # Keep a root target so Hydra passes child dataclass/callable objects
    # directly to the constructor instead of rewrapping them in DictConfig.
    config["_target_"] = "perception.sam3d_stage1.build_stage1_pipeline"
    return instantiate(config)
