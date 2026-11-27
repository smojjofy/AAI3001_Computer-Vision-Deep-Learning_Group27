# Repository Guidelines

## Workspace Purpose

AAI3001 Computer Vision and Deep Learning (Group 27): a single-camera visual
reconstruction pipeline. Stack is Python (3.11–3.13) + PyTorch for inference and
C++20 for the Splat Constructor on Windows.

```text
stream handler → optical flow → reranker → refresh through depth, pose, mask,
and Splat Constructor → active splat state → gsplat browser viewer
```

The current application is integrated. On reuse, the browser applies an
optical-flow view-space warp to the active state. It does not fuse persistent
splat geometry across refreshes. The optional environment compositor uses
Three.js. See `docs/integration_runbook.md` for current operation and
`docs/splat-representation-and-runtime.md` for the data path.

## Repository Structure

- `docs/` contains research, workflow diagrams, and interface documentation.
- `config/` contains example runtime, camera, and model configuration.
- `shared/` contains packet schemas, serialization helpers, and coordinate conventions.
- `modules/` contains independently developable pipeline stages:
  - `stream_handler/` receives and timestamps webcam/ZED/IP-camera frames.
  - `inference/` provides optical flow, depth, pose, and experimental segmentation.
  - `reranker/` compares each incoming frame with one retained significant frame and chooses reuse or refresh.
  - `splat_constructor/` converts inference output into Gaussian states (C++20 core, ctypes bridge, CLI, PLY export).
  - `reconstruction_validation/` projects constructor output back into the source view for metrics.
- `applications/` contains backend orchestration and GUI composition.
- `scripts/` contains repeatable development and validation utilities.
- Interface documentation and schemas describe each module's inputs and outputs.

Generated recordings, processed data, checkpoints, outputs, and the Unreal
adapter are ignored by Git. Do not commit credentials, camera URLs, raw
recordings, or model weights.

## Commands

Run everything as a module from the repository root; imports rely on the root
being on `sys.path`.

```bash
pip install -r modules/inference/requirements.txt        # numpy, Pillow, torch
python -m pytest                                          # Python suites
```

Native Splat Constructor build and tests (MinGW; CMake is not installed on this
machine):

```powershell
./scripts/build_splat_constructor.ps1
./build/splat_constructor/splat_constructor_tests.exe
# also emits build/splat_constructor/splat_constructor_bridge.dll
```

Device setup and live diagnostics:

```powershell
& .\scripts\setup_system.ps1                 # full setup: venv, modules, PyTorch, GUI, native
& .\scripts\run_gui_preview.py --source camera --device 0 --camera-backend dshow
```

Direct bridge validation and PLY export:

```powershell
python -m scripts.validate_splat_source_view --rgb <rgb> --depth <depth.png> `
  --mask <mask.png> --frame-id 0001 --source-id 0001 --timestamp-ns 1 `
  --depth-semantics relative_inverse
python ./scripts/prepare_splat_fixture.py --rgb <rgb> --depth <depth.png> `
  --mask <mask.png> --output outputs/splat_constructor_fixture
./build/splat_constructor/splat_constructor_cli.exe `
  --rgb outputs/splat_constructor_fixture/rgb.ppm `
  --depth outputs/splat_constructor_fixture/depth.pgm `
  --mask outputs/splat_constructor_fixture/mask.pgm `
  --output outputs/splat_constructor/traintest_initial.ply `
  --stride 2 --fov 50 --near 1 --far 4
```

## Architecture

- `modules/inference/depth/` — `YoloDepthEstimator` (Ultralytics YOLO26 depth, metric depth in metres, `NaN` for invalid), `PoseDepthPipeline` (pose + depth fusion), and the temporary `TinyDepthNet` baseline.
- `modules/inference/pose/` — MediaPipe pose (`PoseEstimator`, `VideoPoseEstimator`, `PoseFrameResult`).
- `modules/inference/optical_flow/` — OpenCV DIS (default, offline) and optional Torchvision RAFT Small.
- `modules/inference/segmentation/` — `TinySegmentationNet` research baseline. The live person mask uses pose and depth. A trained mask candidate remains outside the live path.
- `modules/splat_constructor/` — dependency-free C++20 CPU library + CLI + NumPy/ctypes bridge. `ConstructionFrame` v1 → `SplatState` v1 are the frozen in-memory contracts.
- `shared/coordinate_system/README.md` — authoritative constructor convention: right-handed, Three.js-compatible (`+X` right, `+Y` up, camera forward `-Z`); `wxyz` quaternions; log scale; logit opacity.
- `shared/schemas/` — source of truth for module boundaries.
- `applications/gui/` — Vite + TypeScript + `gsplat` viewer and optional Three.js environment.
- `applications/backend/` — snapshot and HTTP server.
- `applications/live_pipeline/` — integrated capture, selection, refresh, and reuse.

## Conventions

- Public API per subpackage is re-exported from its `__init__.py`; keep `from __future__ import annotations` + full type hints.
- Relative inverse depth is unitless and normalized to `[0, 1]`, larger = nearer. Metric camera-Z depth uses metres, larger = farther. Invalid depth is `NaN`, never 0.
- Images are float `[0, 1]` inside PyTorch estimators; estimators accept `(3,H,W)` or `(B,3,H,W)` and return `(B,1,H,W)`. Validate inputs and raise on misuse.
- Estimates are `@dataclass(frozen=True)`; models use `GroupNorm`; run inference under `@torch.inference_mode()`.
- Checkpoints carry schema version, task, config, weights, optimizer, and metrics.
- C++ (`modules/splat_constructor/`): C++20, `splat::` namespace, SoA `GaussianSet`, no third-party dependencies; build with `-Wall -Wextra -Wpedantic -Werror`.
- The stream/frame owner supplies non-empty `frame_id`, positive `timestamp_ns`, `source_id`, intrinsics, and rigid `camera_to_world`. `constructor_handoff.py` rejects unaligned tensors.
- Preserve timestamps, frame IDs, dimensions, alpha/depth semantics, and coordinate conventions across module boundaries.
- Do not add DLPack/zero-copy paths without profiling evidence.

## Module Contracts

Use `shared/schemas/` as the source of truth for module boundaries. Preserve
timestamps, frame IDs, image dimensions, alpha semantics, depth conventions, and
coordinate systems. Interface documentation defines each module's direct
dependency and output contract.

## Contribution Guidance

Work against recorded fixtures where possible so modules can develop
independently. Keep implementation-specific dependencies inside its owning
module. Before adding code, update the relevant tasklist and shared schema. Use
clear names and short imperative commits; include affected interfaces and visual
evidence in pull requests when applicable.

## Notes

- Multi-camera, camera tracking, and Unreal integration are deferred future scope.
- Splat Constructor output uses metric depth in the integrated application.
  The single camera still limits unseen geometry and orbit views.
- ZED SDK depth is available through the ZED source. The experimental stereo
  matcher has open quality checks. The webcam uses YOLO metric depth.
- `pytest` may crash at startup on this machine because of a broken `pyreadline` in user site-packages; set `PYTHONNOUSERSITE=1` first.
