# Single-Camera Visual Reconstruction Workflow

```mermaid
flowchart TD
    CAM[Static webcam or IP camera]
    STREAM[Real-time video stream handler]
    BUFFER[Timestamped frame buffer<br/>ring buffer and frame-drop handling]
    PRE[Frame preprocessing<br/>decode, resize, color normalization]

    RGB[RGB frame]
    SEG[Foreground segmentation<br/>soft mask and confidence]
    DEPTH[Monocular depth estimation<br/>relative depth and confidence]
    FLOW[Optical flow / temporal correspondence]

    FEATURES[Temporal observation state<br/>RGB + mask + depth + flow]
    CACHE[Timestamped Gaussian cache<br/>stable, uncertain, and newly observed splats]
    UPDATE[Temporal splat update<br/>warp, smooth, interpolate, and neural correction]
    STATE[Current RGBA + depth reconstruction]

    ENV[Static Three.js environment<br/>background, floor, or 3D scene]
    COMP[Three.js static compositor<br/>virtual camera, Gaussian renderer,<br/>alpha compositing, and GUI overlays]
    VIEW[Final Three.js view<br/>composited environment + reconstruction]

    CAM --> STREAM --> BUFFER --> PRE
    PRE --> RGB
    RGB --> SEG
    RGB --> DEPTH
    RGB --> FLOW

    RGB --> FEATURES
    SEG --> FEATURES
    DEPTH --> FEATURES
    FLOW --> FEATURES

    FEATURES --> UPDATE
    CACHE --> UPDATE
    UPDATE --> CACHE
    CACHE --> STATE

    ENV --> COMP
    STATE --> COMP
    COMP --> VIEW

    classDef input fill:#dbeafe,stroke:#2563eb,color:#111827;
    classDef inference fill:#dcfce7,stroke:#16a34a,color:#111827;
    classDef state fill:#fef3c7,stroke:#d97706,color:#111827;
    classDef render fill:#f3e8ff,stroke:#9333ea,color:#111827;

    class CAM,STREAM,BUFFER,PRE input;
    class RGB,SEG,DEPTH,FLOW,FEATURES inference;
    class CACHE,UPDATE,STATE state;
    class ENV,COMP,VIEW render;
```

## MVP scope

- One static, initially uncalibrated webcam or IP camera.
- Real-time RGB frame ingestion.
- Foreground segmentation.
- Monocular relative-depth estimation.
- Optical-flow-based temporal correspondence.
- Timestamped Gaussian states for smoothing, interpolation, and dropped-frame recovery.
- RGBA and depth reconstruction output.
- A Three.js viewer that composites the reconstruction over a static 3D environment or background.

The initial viewer should remain close to the capture camera. Small virtual-camera movements can be supported, but large viewpoint changes are outside the reliable scope of the single-camera MVP.

## Future multi-camera extension

The single-camera observation state can later be extended with synchronized views from at least two cameras:

```text
RGB + segmentation + depth + optical flow
        + camera calibration
        + multi-view correspondence
        + pose estimation
        + multi-angle depth and segmentation
```

The Three.js compositor and timestamped Gaussian-cache interfaces should remain reusable when that extension is implemented.
