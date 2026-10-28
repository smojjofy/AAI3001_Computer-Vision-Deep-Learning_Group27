# Repository Guidelines

## Workspace Purpose

This repository is currently a planning and development scaffold for a single-camera visual reconstruction pipeline. The Stream Handler and Optical Flow slices are implemented; the intended first integrated milestone is:

```text
stream handler → optical flow → reranker → Frame Processor decision/dispatch
→ reuse through Temporal Cache or refresh through depth + motion + Splat Constructor
→ Splat Adjustor → Three.js viewer
```

The initial visual signals are RGB, foreground segmentation, relative depth, and optical flow.

## Repository Structure

- `docs/` contains research, workflow diagrams, and interface documentation.
- `config/` contains example runtime, camera, and model configuration.
- `shared/` contains packet schemas, serialization helpers, and coordinate conventions.
- `modules/` contains independently developable pipeline stages:
  - `stream_handler/` receives and timestamps webcam/IP-camera frames.
  - `frame_processor/` executes reranker-directed reuse or refresh and preprocesses refresh frames.
  - `inference/` provides direct-frame optical flow plus refresh-only segmentation, depth, and motion.
  - `reranker/` compares each incoming frame with one retained significant frame and chooses reuse or refresh.
  - `splat_constructor/` converts inference output into Gaussian states.
  - `temporal_cache/` stores, interpolates, and updates timestamped splat states downstream of reranking.
  - `splat_adjustor/` will merge reused cache state and new splat candidates.
  - `three_viewer/` renders reconstruction output in Three.js.
- `applications/` will contain backend orchestration and GUI composition.
- `scripts/` contains repeatable development and validation utilities.
- Interface documentation and schemas describe each module's inputs and outputs.

Generated recordings, processed data, checkpoints, outputs, and the Unreal adapter are ignored by Git. Do not commit credentials, camera URLs, raw recordings, or model weights.

## Module Contracts

Use `shared/schemas/` as the source of truth for module boundaries. Preserve timestamps, frame IDs, image dimensions, alpha semantics, depth conventions, and coordinate systems. Interface documentation defines each module's direct dependency and output contract.

## Contribution Guidance

Work against recorded fixtures where possible so modules can develop independently. Keep implementation-specific dependencies inside their owning module. Before adding code, update the relevant tasklist and shared schema. Use clear names and short imperative commits; include affected interfaces and visual evidence in pull requests when applicable.
