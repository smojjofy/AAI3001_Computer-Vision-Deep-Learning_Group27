# AAI3001 Computer Vision and Deep Learning — Group 27

This repository contains the planning scaffold for a visual reconstruction system based on Gaussian representations, monocular depth estimation, segmentation, and temporal image correspondence.

## Intended Workspace

The initial project target is a single static webcam or IP camera:

```text
RGB stream
  → segmentation + relative depth + optical flow
  → timestamped Gaussian reconstruction
  → Three.js compositor
```

The first milestone is focused on a stable RGBA and depth-aware visual representation. Multi-camera calibration, pose estimation, multi-angle reconstruction, and improved relighting are future extensions.

## Navigating the Repository

- `docs/` — research notes, architecture diagrams, and interface documentation. Start with `docs/research.md` and `docs/single-camera-workflow.md`.
- `config/` — configuration templates for cameras, models, and runtime behavior.
- `shared/` — source of truth for packet schemas, serialization, and coordinate conventions.
- `modules/` — independently owned pipeline components. Each module has a `Tasklist.md` describing its purpose, inputs, outputs, and work items.
- `applications/` — future backend orchestration and GUI composition.
- `tests/` — integration tests and deterministic fixtures.
- `scripts/` — repeatable development and validation utilities.
- `Tasklist.md` — repository-level milestones and future scope.

The main dependency chain is:

```text
stream_handler → frame_processor → inference → splat_constructor
→ temporal_cache → three_viewer
```

The `modules/unreal_adapter/` directory is intentionally ignored by Git for local Unreal Engine experimentation. It should consume the same reconstruction state as the Three.js viewer rather than define a separate reconstruction pipeline.

## Current Status

Temporary Python/PyTorch depth and segmentation baselines provide aligned test
signals. `modules/splat_constructor/` now contains a C++20 CPU reference
constructor, native tests, a CLI, and standard 3DGS PLY output. See that
module's README for the build and fixed-fixture commands. Versioned logical
contracts for constructor input and runtime output live in `shared/schemas/`.
The same native core is available in Python through a NumPy/ctypes bridge after
running the native build script. `modules/reconstruction_validation/` provides
source-view reprojection metrics for constructor output.
Other modules remain planning scaffolds.

## Using the Splat Constructor

The implemented path is an aligned single-frame input:

```text
RGB + float depth + binary foreground mask + camera metadata
    → ConstructionFrame → native Splat Constructor → SplatState
```

Build the native core first. This produces the CLI, native tests, and the
Python-loadable bridge DLL:

```powershell
./scripts/build_splat_constructor.ps1
./build/splat_constructor/splat_constructor_tests.exe
```

### Recommended: direct Python handoff

Use the NumPy bridge for pipeline code. `rgb` is `uint8 (H,W,3)`, depth is
`float32 (H,W)`, and `foreground_mask` is binary `uint8 (H,W)`. The stream
owner must provide the identifiers, positive nanosecond timestamp, camera
intrinsics `(fx, fy, cx, cy)`, and a row-major rigid `camera_to_world` pose.

```python
import numpy as np
from modules.splat_constructor import (
    ConstructionFrame, ConstructionConfig,
    DepthSemantics, DepthUnits, construct_splat_state,
)

frame = ConstructionFrame(
    frame_id="0001",
    timestamp_ns=source_timestamp_ns,
    source_id="camera_01",
    rgb=rgb_u8,
    depth=relative_inverse_depth_f32,
    foreground_mask=foreground_mask_u8,
    intrinsics=(fx, fy, cx, cy),
    depth_semantics=DepthSemantics.RELATIVE_INVERSE,
    depth_units=DepthUnits.UNITLESS,
    depth_validity=depth_validity_f32,       # optional, values in [0,1]
    foreground_weight=foreground_weight_f32, # optional, values in [0,1]
)
state = construct_splat_state(frame, ConstructionConfig(sample_stride=2))
```

`state` contains copied NumPy arrays for positions, SH-DC color, opacity logits,
log scales, rotations, reconstruction weights, local IDs, and source pixels.
The constructor also supports `DepthSemantics.METRIC_CAMERA_Z` with
`DepthUnits.METERS`. The current bridge copies arrays at the Python/C++ boundary.

For estimator outputs, use `construction_frame_from_estimates()` from
`modules.inference`; it rejects misaligned outputs, preserves stream metadata,
uses depth confidence as `depth_validity`, and does not misuse segmentation
confidence as a generic quality value.

### Validate a constructed frame

Run the source-view acceptance command after building the bridge:

```powershell
python -m scripts.validate_splat_source_view `
  --rgb test_data/gaussian_splatting/FO_dataset/train/traintest_rgb.jpg `
  --depth outputs/traintest_inverted_050/relative_inverse_depth.png `
  --mask outputs/traintest_inverted_050/foreground_mask.png `
  --frame-id 0001 --source-id 0001 --timestamp-ns 1 `
  --depth-semantics pseudo_inverse_from_segmentation --stride 2
```

It reports center reprojection RMSE, RGB/depth error on covered pixels, mask
IoU, and foreground coverage. This is a point-center validation rasterizer;
with `sample_stride=2`, roughly 25% foreground coverage is expected. It is not
the future full Gaussian renderer.

### Legacy PLY export

The C++ CLI remains useful for viewer/export experiments. It accepts binary
PPM/PGM fixtures and writes a standard 62-property binary 3DGS PLY plus JSON
metadata. See [the module guide](modules/splat_constructor/README.md) for the
fixture preparation and export commands.

The supplied temporary fixture uses segmentation-derived pseudo-depth. Its PLY
is a pipeline smoke test and mask-shaped relief, not reliable 3D geometry;
replace it with metric or meaningful relative depth when the production
inference model is available.
