# Inference Tasklist

Input: `ProcessedFrame` from Frame Processor.

Output: aligned depth/segmentation estimates and a `ConstructionFrame` v1 adapter.
Optical flow remains a Temporal Cache input, not a Splat Constructor dependency.

- [ ] Orchestrate segmentation, depth, and optical-flow models.
- [x] Add a temporary aligned RGB/depth/mask training dataset adapter.
- [x] Add versioned checkpoint save/load handling.
- [x] Add a single-image baseline prediction command.
- [x] Convert aligned depth and segmentation estimates into `ConstructionFrame` v1.
- [x] Preserve frame ID, timestamp, source ID, intrinsics, dimensions, and depth semantics.
- [ ] Replace temporary CNNs with teammate-provided production models.
- [ ] Orchestrate the production models and stream-owned metadata in one live packet.
