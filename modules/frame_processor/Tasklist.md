# Frame Processor Tasklist

Input: `FramePacket` from Stream Handler.

Output: `ProcessedFrame` containing normalized RGB data, dimensions, timestamp, frame ID, source ID, and normalization metadata. This is the exact input contract for Inference.

- [ ] Implement resize, color conversion, and normalization.
- [ ] Preserve frame identity and timestamp exactly.
- [ ] Add quality checks without changing packet semantics.
