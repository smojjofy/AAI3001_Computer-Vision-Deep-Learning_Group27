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
Other modules remain planning scaffolds.
