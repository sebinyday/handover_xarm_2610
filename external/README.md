# Shared handover runtime

SAM2, SAM3D, MediaPipe and robot control all use **`conda activate handover`**.
The SAM3D model server is a separate process using the same environment;
`runtime.sam3d_server.python_interpreter` points to that environment's Python.

Validated shared stack: Python 3.10, PyTorch 2.5.1+cu121,
torchvision 0.20.1+cu121, NumPy 1.26.4, MediaPipe 0.10.11,
Open3D 0.19.0, PyTorch3D 0.7.8, Kaolin 0.17.0 and spconv-cu121 2.3.8.

## Reinstalling the dependencies

The current `handover` environment is already configured. To reproduce it:

```bash
conda activate handover
python -m pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-handover.txt ninja
CUDA_HOME=/usr/local/cuda-12.1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=8.6 \
  MAX_JOBS=8 BUILD_NO_CUDA=1 python -m pip install --no-build-isolation \
  -r environment/sam3d-runtime-requirements.txt
```

The architecture `8.6` is the installed RTX A6000. PyTorch3D is built with
CUDA support; Kaolin uses its official Torch 2.5.1/CUDA 12.1 wheel.
`BUILD_NO_CUDA=1` skips the gsplat renderer extension, which stage1 template
extraction does not invoke. Both dense and sparse attention use upstream's
SDPA backend. Source commits and tested package versions are recorded in the
runtime requirements file.

The integration imports the official SAM3D checkout directly. Its stage1
runtime dependencies are listed explicitly; installing the upstream
training/development distribution is unnecessary for this pipeline.
`setuptools==80.9.0` preserves the `pkg_resources` API used by Lightning 2.3.3.
No global Hydra source patch is applied because the integration skips the
upstream training initialization through `LIDRA_SKIP_INIT=true`.

## Source checkouts

SAM2 source: https://github.com/facebookresearch/sam2

- Tested revision: `2b90b9f5ceec907a1c18123530e92e794ad901a4`.
- Existing checkout: `/home/sebin/hop/sam2`.
- Existing checkpoint: `/home/sebin/hop/sam2/checkpoints/sam2.1_hiera_small.pt`.
- Model config: `configs/sam2.1/sam2.1_hiera_s.yaml`.

SAM2 is installed in `handover`. Reinstall from the checkout with:

```bash
SAM2_BUILD_CUDA=0 python -m pip install --no-deps --no-build-isolation /home/sebin/hop/sam2
```

Its optional CUDA postprocessing extension is disabled. Base Torch GPU
tracking has passed the two-camera-state smoke test.

SAM3D source: https://github.com/facebookresearch/sam-3d-objects

- Checkout: `external/sam-3d-objects`.
- Pinned revision: `f91db411c50efee93d8db7aeb323885650f6f722`.
- The checkout remains unmodified.
- `perception/sam3d_stage1.py` adapts the official pointmap pipeline to skip
  SLAT, GS, mesh and unused encoder weight initialization. The original
  sparse-structure and pointmap inference implementations remain in use.
- The official notebook `Inference` has no `stage1_only_init` parameter;
  the adapter replaces that unsupported call in the original integration.

## Installed stage1 checkpoints

The user-provided archive [hf.zip](https://drive.google.com/file/d/1nDH7ghej5GwhsmqVsPww37ZuE2wyDWKv/view)
was downloaded (12,022,032,297 bytes). Only these stage1 files were extracted
into `external/sam-3d-objects/checkpoints/hf/`:

```text
pipeline.yaml
ss_generator.yaml
ss_generator.ckpt
ss_decoder.yaml
ss_decoder.ckpt
checkpoint_provenance.json
```

`checkpoint_provenance.json` records the source URL, byte counts and SHA-256
hashes of the imported files. Stage2 weights remain unextracted. The configured
checkpoint path already resolves to this directory. Public MoGe and DINO
weights needed by the stage1 conditioning path were fetched into the normal
Hugging Face/Torch caches during the first model load.

The real server/IPC generation path passed on the official example image
`shutterstock_stylish_kidsroom_1640806567`, mask 14, resized to 640×427:

- First model load: approximately 62.1 s, including auxiliary downloads.
- Stage1 inference: approximately 8.15 s on the RTX A6000.
- Template: 25,819 finite XYZ points; stage2 weights were not installed.
- Artifacts: `output/sam3d_checkpoint_validation/{image.png,mask.png,template.npy,metadata.yaml,validation.json}`.

The first real load exposed a Hydra/OmegaConf conversion error for callable
preprocessor dataclasses. The project adapter now retains a root Hydra factory
so those objects pass directly to the pipeline constructor. A regression test
covers the real configuration pattern; no upstream source or global Hydra
patch was required.

Official future downloads are available at
https://huggingface.co/facebook/sam-3d-objects after access approval. The
installed user-supplied checkpoint does not require another Hub download.

## Validation and startup

```bash
conda activate handover
python tools/smoke_sam2.py --frames 3
python tools/smoke_sam3d.py
python tools/smoke_sam3d.py --with-weights
python -m unittest discover -s tests -v
python tools/check_runtime.py --robot
```

The default SAM3D smoke test imports the real pipeline and runs CUDA operations
in PyTorch3D, Kaolin and spconv. Its `check_scope: dependencies_only` and
`weights_loaded: false` mean that weight loading was not requested, not that
the checkpoint is missing or broken. Add `--with-weights` to load the configured
stage1 checkpoint onto the GPU and verify that only `ss_generator` and
`ss_decoder` are initialized. Success reports `weights_loaded: true` and
`check_scope: dependencies_and_weights`. Neither mode runs template inference
or connects devices.
Run GPU checks in a normal host terminal; CUDA access is restricted inside
the execution sandbox. The latest readiness check passes the installed software, checkpoint and
teaching-data validation. It does not validate the physical camera/robot setup.

Start `bash run_sam3d_model_server.sh`
and `bash run_handover.sh` in two terminals with **the same `handover`
environment activated**. Model generation can also use `execution_mode:
oneshot` with the same interpreter.

## AnySkin

`anyskin==1.0.0` exposing `AnySkinProcess` is installed in `handover`.
The physical sensor and serial access remain necessary for default tactile
robot mode. Perception-only mode does not open that device.
