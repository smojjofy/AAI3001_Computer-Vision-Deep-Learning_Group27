# Original single-camera design record

This document preserves the early architecture and extension proposals. The
[milestone workflow](single-camera-workflow.md) describes the delivered
application. The diagram below includes stages that are outside the live path.

This document records the original design and its extension path. Its diagram
includes persistent fusion, trained segmentation, and a Three.js compositor;
these are not the base live path. The current application keeps one active
splat state, uses a pose-gradient person mask, and renders that state with
`gsplat`. It offers an optional Three.js environment compositor. Use the
[integration architecture](integration.md) for the current pipeline and the
[representation guide](splat-representation-and-runtime.md) for the live data
contract. The system uses one static camera. Skeletal animation and online
optimization remain optional extensions.

## Original streaming design

```mermaid
flowchart TD
    CAM[Static webcam or IP camera]
    STREAM[Stream Handler<br/>decode, timestamp, reconnect]
    PACKET[RGB8 FramePacket<br/>identity, timestamp, quality]
    FLOW[Optical flow<br/>retained reference to current FramePacket]
    SELECT[Reranker<br/>one retained significant frame]
    DISPATCH[Frame Processor<br/>execute reuse or refresh]
    PRE[Frame Processor<br/>prepare selected refresh frame]
    SEG[Foreground segmentation<br/>mask + confidence]
    DEPTH[Monocular depth<br/>relative depth + confidence]
    MOTION[Optional human motion or pose]
    FRAME[ConstructionFrame<br/>RGB + mask + depth + flow<br/>optional camera/pose/rig metadata]

    REUSE[Reuse observation<br/>frame + reliable flow]
    INIT[Depth-guided splat initializer<br/>keyframes only]
    FUSE[Merge, deduplicate, and prune]
    BIND[Optional skeletal binding<br/>when pose/rig data is available]
    ADJUST[Splat Adjustor<br/>merge reused state or splat candidate]
    CACHE[Temporal Gaussian cache<br/>persistent states and confidence]
    RENDER[Streaming rasterizer<br/>fast render path]
    OUTPUT[RGBA + relative depth<br/>coverage + uncertainty]

    REPLAY[Replay buffer<br/>old keyframes]
    OPT[Optional background optimizer<br/>sliding-window refinement]

    ENV[Static Three.js environment<br/>background, floor, or 3D scene]
    VIEW[Three.js static compositor<br/>Gaussian renderer, alpha compositing,<br/>GUI and diagnostics]
    FINAL[Final Three.js view]

    CAM --> STREAM --> PACKET --> FLOW --> SELECT
    SELECT -. retained significant frame .-> FLOW
    SELECT --> DISPATCH
    DISPATCH -->|refresh| PRE
    PRE --> SEG
    PRE --> DEPTH
    PRE --> MOTION
    PRE --> FRAME
    SEG --> FRAME
    DEPTH --> FRAME
    MOTION --> FRAME
    FLOW --> FRAME

    DISPATCH -->|reuse| REUSE
    FRAME --> INIT --> FUSE
    FUSE --> BIND --> ADJUST
    REUSE --> ADJUST
    CACHE --> ADJUST
    ADJUST --> CACHE
    CACHE --> RENDER --> OUTPUT --> VIEW

    CACHE --> REPLAY --> OPT --> CACHE
    ENV --> VIEW --> FINAL

    classDef input fill:#dbeafe,stroke:#2563eb,color:#111827;
    classDef inference fill:#dcfce7,stroke:#16a34a,color:#111827;
    classDef construct fill:#fef3c7,stroke:#d97706,color:#111827;
    classDef render fill:#f3e8ff,stroke:#9333ea,color:#111827;
    classDef optional fill:#f3f4f6,stroke:#6b7280,color:#111827;

    class CAM,STREAM,PACKET,PRE input;
    class SEG,DEPTH,MOTION,FLOW,FRAME inference;
    class SELECT,DISPATCH,REUSE,INIT,FUSE,BIND,ADJUST,CACHE construct;
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
- Reranking against one retained significant frame with capture-time reference expiry.
- A Frame Processor decision/dispatch step: reuse a valid cache state or prepare a refresh frame.
- Depth-guided Gaussian initialization for keyframes.
- Merge, deduplication, confidence tracking, and pruning.
- Temporal reuse for non-keyframes.
- Splat adjustment that merges a reused cache state or a new depth-guided splat candidate.
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

Each pipeline stage was built and owned by one team member. The table in the
[main README](../README.md#team-and-component-ownership) maps the five work
packages to the diagram above.

Two deviations from the original split are worth naming, because the diagram
still shows stages that integration replaced:

- **Foreground segmentation** was planned as a trained model. The live person
  mask uses pose and depth instead. A trained mask candidate exists outside
  the live path. See the [milestone reflection](Integration-Enhancements.md).
- **The Temporal Cache and its Three.js viewer** were generalised into the
  single display clock and the `gsplat` renderer described in
  [the integration architecture](integration.md).

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
