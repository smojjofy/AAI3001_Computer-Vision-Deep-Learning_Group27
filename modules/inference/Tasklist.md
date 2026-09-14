# Inference Tasklist

Input: `ProcessedFrame` from Frame Processor.

Output: `InferencePacket` containing RGB, foreground mask, mask confidence, relative depth, depth confidence, optical flow, and valid-flow mask. This is the exact input contract for Splat Constructor.

- [ ] Orchestrate segmentation, depth, and optical-flow models.
- [ ] Use pretrained models first.
- [ ] Preserve timestamps and dimensions across every output.
