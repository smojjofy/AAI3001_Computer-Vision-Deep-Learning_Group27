# Repository Guidelines

## Workspace Purpose

This repository is currently a planning and development scaffold for a single-camera visual reconstruction pipeline. Nothing is implemented yet. The intended first milestone is:

```text
stream handler → frame processor → inference → splat constructor
→ temporal cache → Three.js viewer
```

The initial visual signals are RGB, foreground segmentation, relative depth, and optical flow.

## Repository Structure

- `docs/` contains research, workflow diagrams, and interface documentation.
- `config/` contains example runtime, camera, and model configuration.
- `shared/` contains packet schemas, serialization helpers, and coordinate conventions.
- `modules/` contains independently developable pipeline stages:
  - `stream_handler/` receives and timestamps webcam/IP-camera frames.
  - `frame_processor/` resizes and normalizes frames.
  - `inference/` provides segmentation, depth, and optical flow.
  - `splat_constructor/` converts inference output into Gaussian states.
  - `temporal_cache/` stores, interpolates, and updates timestamped states.
  - `three_viewer/` renders reconstruction output in Three.js.
- `applications/` will contain backend orchestration and GUI composition.
- `tests/` contains cross-module fixtures and integration tests.
- `scripts/` contains repeatable development and validation utilities.
- `Tasklist.md` files describe each folder's purpose, inputs, outputs, and remaining work.

Generated recordings, processed data, checkpoints, outputs, and the Unreal adapter are ignored by Git. Do not commit credentials, camera URLs, raw recordings, or model weights.

## Module Contracts

Use `shared/schemas/` as the source of truth for module boundaries. Preserve timestamps, frame IDs, image dimensions, alpha semantics, depth conventions, and coordinate systems. A module's `Tasklist.md` defines its direct dependency and output contract.

## Contribution Guidance

Work against recorded fixtures where possible so modules can develop independently. Keep implementation-specific dependencies inside their owning module. Before adding code, update the relevant tasklist and shared schema. Use clear names and short imperative commits; include affected interfaces and visual evidence in pull requests when applicable.
