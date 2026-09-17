# Single-Camera Visual Reconstruction Workflow

This document describes the reconciled MVP and its extension path. The base system uses one static camera and learned visual signals. Skeletal animation, online optimization, and richer metadata are optional extensions rather than prerequisites for the first working demo.

## Base streaming pipeline

```mermaid
flowchart TD
    CAM[Static webcam or IP camera]
    STREAM[Stream Handler<br/>decode, timestamp, reconnect]
    PRE[Frame Processor<br/>resize, normalize, quality checks]
    BUFFER[Timestamped ring buffer<br/>bounded history and frame-drop handling]

    RGB[RGB frame]
    SEG[Foreground segmentation<br/>mask + confidence]
    DEPTH[Monocular depth<br/>relative depth + confidence]
    FLOW[Optical flow<br/>temporal correspondence]
    FRAME[ConstructionFrame<br/>RGB + mask + depth + flow<br/>optional camera/pose/rig metadata]

    SELECT[Keyframe selector<br/>motion, depth change, heartbeat]
    REUSE[Reuse and deform current state<br/>for non-keyframes]
    INIT[Depth-guided splat initializer<br/>keyframes only]
    FUSE[Merge, deduplicate, and prune]
    BIND[Optional skeletal binding<br/>when pose/rig data is available]
    CACHE[Temporal Gaussian cache<br/>timestamped states and confidence]
    RENDER[Streaming rasterizer<br/>fast render path]
    OUTPUT[RGBA + relative depth<br/>coverage + uncertainty]

    REPLAY[Replay buffer<br/>old keyframes]
    OPT[Optional background optimizer<br/>sliding-window refinement]

    ENV[Static Three.js environment<br/>background, floor, or 3D scene]
    VIEW[Three.js static compositor<br/>Gaussian renderer, alpha compositing,<br/>GUI and diagnostics]
    FINAL[Final Three.js view]

    CAM --> STREAM --> PRE --> BUFFER
    BUFFER --> RGB
    RGB --> SEG
    RGB --> DEPTH
    RGB --> FLOW
    RGB --> FRAME
    SEG --> FRAME
    DEPTH --> FRAME
    FLOW --> FRAME

    FRAME --> SELECT
    SELECT -->|No| REUSE
    SELECT -->|Yes| INIT --> FUSE
    FUSE --> BIND --> CACHE
    REUSE --> CACHE
    CACHE --> RENDER --> OUTPUT --> VIEW

    CACHE --> REPLAY --> OPT --> CACHE
    ENV --> VIEW --> FINAL

    classDef input fill:#dbeafe,stroke:#2563eb,color:#111827;
    classDef inference fill:#dcfce7,stroke:#16a34a,color:#111827;
    classDef construct fill:#fef3c7,stroke:#d97706,color:#111827;
    classDef render fill:#f3e8ff,stroke:#9333ea,color:#111827;
    classDef optional fill:#f3f4f6,stroke:#6b7280,color:#111827;

    class CAM,STREAM,PRE,BUFFER input;
    class RGB,SEG,DEPTH,FLOW,FRAME inference;
    class SELECT,REUSE,INIT,FUSE,BIND,CACHE construct;
    class RENDER,OUTPUT,ENV,VIEW,FINAL render;
    class REPLAY,OPT optional;
```

## MVP scope

The first implementation should include:

- One static webcam or IP camera.
- Live input with a bounded ring buffer and timestamp ordering.
- Frame preprocessing and quality checks.
- Foreground segmentation.
- Monocular relative-depth estimation.
- Optical flow for temporal correspondence.
- Keyframe selection with a configurable heartbeat.
- Depth-guided Gaussian initialization for keyframes.
- Merge, deduplication, confidence tracking, and pruning.
- Temporal reuse for non-keyframes.
- RGBA, relative depth, coverage, and uncertainty output.
- Three.js rendering over a static environment or background.

The initial viewer should remain close to the capture camera. Small virtual-camera movements are acceptable; large viewpoint changes and hidden-surface reconstruction are outside the reliable single-camera scope.

## Optional extension layers

The same `ConstructionFrame` can carry additional data when available:

```text
ConstructionFrame {
    rgba_or_rgb
    foreground_mask
    mask_confidence
    relative_depth
    depth_confidence
    optical_flow
    flow_valid_mask
    optional_camera_intrinsics
    optional_camera_extrinsics
    optional_skeletal_pose
    optional_bone_hierarchy
    optional_skinning_weights
}
```

When skeletal data is present, the Splat Constructor can add per-Gaussian binding and pose-aware deformation. This enables an animatable asset without changing the base RGB, mask, depth, and flow pipeline.

The replay buffer and background optimizer are quality extensions. They should refine persistent Gaussian state asynchronously and must not block the fast rendering path.

## Module ownership

The five-person work split maps to the diagram as follows:

- Teammate 1: Stream Handler and Frame Processor.
- Teammate 2: Foreground Segmentation.
- Teammate 3: Depth and Optical Flow.
- Teammate 4: Temporal Cache and Three.js Viewer.
- Project lead: Splat Constructor, shared integration, and evaluation.

## Future multi-camera extension

The `ConstructionFrame` can later be extended to a synchronized set of camera observations:

```text
Camera array
    → frame synchronization and calibration
    → per-camera RGB, segmentation, depth, and flow
    → multi-view correspondence and pose estimation
    → multi-angle Gaussian construction
```

The temporal cache, Gaussian state format, rasterizer outputs, and Three.js compositor should remain reusable for this extension.
