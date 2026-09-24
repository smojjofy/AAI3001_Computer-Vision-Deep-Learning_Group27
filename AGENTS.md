# Repository Guidelines

AAI3001 Computer Vision and Deep Learning (Group 27): a single-camera visual
reconstruction pipeline. Stack is Python (3.12/3.13) + PyTorch for inference
and C++20 for the Splat Constructor on Windows.

```text
RGB stream → segmentation + relative depth + optical flow
           → Gaussian splat reconstruction → Three.js viewer
```

## Project

- Pipeline: `stream_handler → frame_processor → inference → splat_constructor
  → temporal_cache → three_viewer`.
- `modules/inference/` contains temporary CNN baselines, and
  `modules/splat_constructor/` contains the native CPU reference constructor.
  Other modules remain scaffolding plus a `Tasklist.md`.
- Run everything as a module from the repo root, e.g.
  `python -m modules.inference.depth.train` — imports rely on the root being on
  `sys.path`.

## Commands

```bash
pip install -r modules/inference/requirements.txt   # numpy, Pillow, torch
python -m pytest                                    # full suite (7 tests)
```

Native Splat Constructor build and tests:

```powershell
./scripts/build_splat_constructor.ps1
./build/splat_constructor/splat_constructor_tests.exe
```

Dataset + baseline training (dataset is gitignored; PowerShell on Windows):

```powershell
./scripts/download_apc_training_sample.ps1
$data = "data/processed/apc_training_sample/training-sample"
python -m modules.inference.depth.train --data $data --epochs 5
python -m modules.inference.segmentation.train --data $data --epochs 5
python -m modules.inference.predict_baselines `
  --image path/to/image.jpg `
  --depth-checkpoint checkpoints/depth_baseline.pt `
  --segmentation-checkpoint checkpoints/segmentation_baseline.pt `
  --output outputs/baseline_prediction
```

Note: a broken `pyreadline` in the user site-packages can crash pytest startup on
this machine. Set `PYTHONNOUSERSITE=1` before `python -m pytest` if that happens.

## Architecture

- `modules/inference/depth/` — `TinyDepthNet` (U-Net-style CNN), `depth_loss`,
  `DepthEstimator` boundary, `train.py`.
- `modules/inference/segmentation/` — `TinySegmentationNet`, `segmentation_loss`,
  `SegmentationEstimator`, `train.py`.
- `modules/inference/apc_dataset.py` — `APCTrainingSample`, the aligned
  RGB/depth/mask adapter for Princeton's APC sample.
- `modules/inference/checkpoint.py` — versioned `save_checkpoint` /
  `load_checkpoint` (`CHECKPOINT_SCHEMA_VERSION`), task-tagged and atomic.
- `modules/inference/predict_baselines.py` — runs both checkpoints on one image.
- `shared/schemas/` — source of truth for module boundaries (packets still
  in-progress). `config/`, `docs/`, `scripts/`, `applications/`, `tests/` are
  supporting areas.

- `modules/splat_constructor/` contains the C++20 CPU library, CLI, native
  tests, and standard binary 3DGS PLY writer. CMake configuration is included;
  use the direct MinGW build script on this machine because CMake is absent.

## Conventions

- Public API per subpackage is re-exported from its `__init__.py`; import via
  that, and keep `from __future__ import annotations` + full type hints.
- Depth is unitless **relative inverse depth** normalized to `[0, 1]`, larger =
  nearer. Images are float `[0, 1]`; estimators accept `(3,H,W)` or `(B,3,H,W)`
  and always return `(B,1,H,W)`. Validate inputs and raise on misuse.
- Estimates are `@dataclass(frozen=True)`; models use `GroupNorm` (not BatchNorm)
  so batch size is irrelevant; run inference under `@torch.inference_mode()`.
- Checkpoints carry schema version, task, config, weights, optimizer, and metrics.
- Preserve timestamps, frame IDs, dimensions, alpha/depth semantics, and
  coordinate conventions across module boundaries. Keep dependencies inside the
  owning module and update the module's `Tasklist.md` + `shared/schemas/` first.
- Never commit credentials, camera URLs, raw recordings, or model weights
  (`data/`, `checkpoints/`, `outputs/`, `unreal_adapter/` are gitignored).

## Notes

- Multi-camera, pose estimation, and Unreal integration are deferred future scope.
- `tmp.md` and `test_data/` are local scratch and are not tracked.
