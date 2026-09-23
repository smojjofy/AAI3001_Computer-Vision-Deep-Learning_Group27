# Inference Tasklist

Input: `ProcessedFrame` from Frame Processor.

Output: `InferencePacket` containing RGB, foreground mask, mask confidence, relative depth, depth confidence, optical flow, and valid-flow mask. This is the exact input contract for Splat Constructor.

- [ ] Orchestrate segmentation, depth, and optical-flow models.
- [x] Add a temporary aligned RGB/depth/mask training dataset adapter.
- [x] Add versioned checkpoint save/load handling.
- [x] Add a single-image baseline prediction command.
- [ ] Replace temporary CNNs with teammate-provided production models.
- [ ] Preserve timestamps and dimensions across every output.
