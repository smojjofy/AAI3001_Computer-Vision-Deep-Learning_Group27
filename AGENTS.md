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
python -m pytest                                    # Python suite (11 tests after native build)
```

Native Splat Constructor (C++20) build and tests:

```powershell
./scripts/build_splat_constructor.ps1
./build/splat_constructor/splat_constructor_tests.exe
# also emits build/splat_constructor/splat_constructor_bridge.dll

# Reference path: PNG/JPEG → PPM/PGM fixture → 3DGS PLY (see module README)
python ./scripts/prepare_splat_fixture.py --rgb <rgb> --depth <depth.png> `
  --mask <mask.png> --output outputs/splat_constructor_fixture
./build/splat_constructor/splat_constructor_cli.exe `
  --rgb outputs/splat_constructor_fixture/rgb.ppm `
  --depth outputs/splat_constructor_fixture/depth.pgm `
  --mask outputs/splat_constructor_fixture/mask.pgm `
  --output outputs/splat_constructor/traintest_initial.ply `
  --stride 2 --fov 50 --near 1 --far 4
python ./scripts/inspect_3dgs_ply.py outputs/splat_constructor/traintest_initial.ply

# Direct Python bridge + source-view reprojection validation
python -m scripts.validate_splat_source_view --rgb <rgb> --depth <depth.png> `
  --mask <mask.png> --frame-id 0001 --source-id 0001 --timestamp-ns 1 `
  --depth-semantics relative_inverse
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
- `modules/inference/constructor_handoff.py` — converts aligned estimator
  tensors and `ConstructionFrameMetadata` into `ConstructionFrame` v1.
- `modules/splat_constructor/` — dependency-free C++20 CPU library + CLI that
  turns aligned RGB + binary mask + pseudo inverse-depth into a standard binary
  3DGS PLY plus metadata JSON. `src/constructor.cpp` (RGB-D → Gaussians),
  `src/c_api.cpp` + `python_bridge.py` (NumPy/ctypes boundary),
  `src/image_io.cpp` (binary PPM/PGM reader), `src/ply_writer.cpp` (62-property
  3DGS PLY + metadata), `cli/main.cpp`, `tests/test_constructor.cpp`. CMake is
  included; use the MinGW build script on this machine (CMake absent).
- `shared/coordinate_system/README.md` — authoritative constructor convention
  (right-handed, Three.js-compatible: `+X` right, `+Y` up, camera forward `-Z`;
  `wxyz` quaternions; log scale; logit opacity).
- `shared/schemas/` — source of truth for module boundaries (packets still
  in-progress). Constructor boundaries are frozen as `ConstructionFrame` v1
  and `SplatState` v1; PLY/JSON are export artifacts rather than the Temporal
  Cache runtime contract. `config/`, `docs/`, `scripts/`, `applications/`,
  `tests/` are supporting areas.
- `modules/reconstruction_validation/` — point-center source-view rasterizer
  and geometry/appearance validation metrics; not a production renderer.

## Conventions

- Public API per subpackage is re-exported from its `__init__.py`; import via
  that, and keep `from __future__ import annotations` + full type hints.
- Relative inverse depth is unitless and normalized to `[0, 1]`, larger =
  nearer. Metric camera-Z depth uses meters. Images are float `[0, 1]` inside
  PyTorch estimators; estimators accept `(3,H,W)` or `(B,3,H,W)` and always
  return `(B,1,H,W)`. Validate inputs and raise on misuse.
- Estimates are `@dataclass(frozen=True)`; models use `GroupNorm` (not BatchNorm)
  so batch size is irrelevant; run inference under `@torch.inference_mode()`.
- Checkpoints carry schema version, task, config, weights, optimizer, and metrics.
- C++ (`modules/splat_constructor/`): C++20, `splat::` namespace, SoA
  `GaussianSet`, no third-party dependencies (the Windows API is used only for
  atomic file replacement). Validate inputs and throw
  `std::invalid_argument`/`runtime_error`; build with
  `-Wall -Wextra -Wpedantic -Werror`. Follow `shared/coordinate_system/README.md`
  for projection and the standard 62-property binary 3DGS PLY layout (normals
  and `f_rest_*` zero-filled, `wxyz` rotations, log scales, logit opacity).
- The in-memory boundary is `ConstructionFrame` v1 → `SplatState` v1. Python
  calls the native constructor through `modules.splat_constructor` using NumPy
  and `ctypes`; both input and returned output arrays are copied, so no native
  handle remains live after a call. Do not add a DLPack path without profiling.
- Inference handoff: the stream/frame owner supplies non-empty `frame_id`,
  positive `timestamp_ns`, `source_id`, intrinsics, and rigid `camera_to_world`.
  `constructor_handoff.py` rejects unaligned tensors, converts segmentation
  masks to binary `{0,1}`, maps depth confidence only to `depth_validity`, and
  never treats segmentation-confidence margin as calibrated quality.
- Source-view validation is renderer-agnostic and read-only. Its current
  point-center z-buffer reports reprojection RMSE, covered RGB/depth MAE, mask
  IoU, and foreground coverage. With `sample_stride=2`, about 25% foreground
  coverage is expected; it is not a full elliptical-Gaussian renderer.
- Preserve timestamps, frame IDs, dimensions, alpha/depth semantics, and
  coordinate conventions across module boundaries. Keep dependencies inside the
  owning module and update the module's `Tasklist.md` + `shared/schemas/` first.
- Never commit credentials, camera URLs, raw recordings, or model weights
  (`data/`, `checkpoints/`, `outputs/`, `unreal_adapter/` are gitignored).

## Notes

- Multi-camera, pose estimation, and Unreal integration are deferred future scope.
- Splat Constructor output is currently a mask-shaped "relief", not geometric
  reconstruction: the reference fixture feeds a segmentation probability as
  inverse depth (`depth_semantics: pseudo_inverse_from_segmentation`). Expect
  real geometry only once the handoff provides true depth.
- Gitignored: `tmp*`, `build/`, `outputs/`, `data/processed/`, `checkpoints/`,
  `*.ply`, and `modules/unreal_adapter/`. `test_data/` is only partially
  tracked — the `train/` scene plus README are committed; `drjohnson/`,
  `playroom/`, and `truck/` are ignored.
