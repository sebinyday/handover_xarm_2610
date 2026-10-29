#!/usr/bin/env python3
"""Check SAM3D CUDA dependencies; --with-weights also loads stage1 models."""

from pathlib import Path
import argparse
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_stage1_weights(config, repo):
    """Report success only after the real stage1 models finish loading."""
    from tools.sam3d_generate import Sam3DStage1Engine

    engine = Sam3DStage1Engine(config, sam3d_repo=repo)
    loaded = sorted(
        name for name, model in engine.pipeline.models.items() if model is not None
    )
    if set(loaded) != {"ss_generator", "ss_decoder"}:
        raise RuntimeError(f"Expected only stage1 models, got {loaded}")
    return {
        "weights_loaded": True,
        "loaded_models": loaded,
        "model_load_elapsed_s": engine.model_load_elapsed_s,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/handover.yaml"))
    parser.add_argument(
        "--with-weights",
        action="store_true",
        help="Also load the configured stage1 weights onto the GPU (no inference or devices).",
    )
    args = parser.parse_args()
    import yaml

    config = yaml.safe_load(Path(args.config).read_text())
    repo = Path(config["runtime"]["sam3d_server"]["repo_path"]).expanduser()
    if not repo.is_absolute():
        repo = ROOT / repo
    sys.path.insert(0, str(repo.resolve()))
    os.environ["LIDRA_SKIP_INIT"] = "true"
    os.environ["ATTN_BACKEND"] = "sdpa"
    os.environ["SPARSE_ATTN_BACKEND"] = "sdpa"
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Run this smoke check on the host with CUDA access")
    from pytorch3d.ops import knn_points

    points = torch.rand((1, 8, 3), device="cuda")
    if knn_points(points, points, K=1).dists.max().item() > 1e-6:
        raise RuntimeError("PyTorch3D CUDA self-nearest-neighbor check failed")
    import kaolin

    quantized = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.int16, device="cuda")
    octree = kaolin.ops.spc.unbatched_points_to_octree(quantized, level=2, sorted=True)
    if not octree.is_cuda or not len(octree):
        raise RuntimeError("Kaolin CUDA octree check failed")
    import spconv.pytorch as spconv

    coords = torch.tensor(
        [[0, 0, 0, 0], [0, 1, 1, 1]], dtype=torch.int32, device="cuda"
    )
    features = torch.rand((2, 4), device="cuda")
    cloud = spconv.SparseConvTensor(features, coords, [4, 4, 4], 1)
    with torch.no_grad():
        result = spconv.SubMConv3d(4, 4, 1, bias=False).cuda()(cloud)
    if not torch.isfinite(result.features).all():
        raise RuntimeError("spconv CUDA check failed")
    from perception.sam3d_stage1 import get_stage1_pipeline_type

    pipeline_type = get_stage1_pipeline_type()
    weight_report = {"weights_loaded": False}
    if args.with_weights:
        weight_report = check_stage1_weights(config, repo.resolve())
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "python": sys.executable,
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0),
                "pytorch3d_cuda": True,
                "kaolin_cuda": True,
                "spconv_cuda": True,
                "sam3d_pipeline_import": pipeline_type.__name__,
                "check_scope": "dependencies_and_weights"
                if args.with_weights
                else "dependencies_only",
                **weight_report,
                "passed": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
