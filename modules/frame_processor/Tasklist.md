# Frame Processor Tasklist

Input: `FramePacket` selected by the reranker for estimator refresh.

Output: `ProcessedFrame` containing normalized RGB data, dimensions, timestamp,
frame ID, source ID, and normalization metadata for depth, motion, and foreground
estimators. Optical flow consumes Stream Handler `FramePacket`s directly before
reranking; it does not wait for this module.

- [ ] Implement resize, color conversion, and normalization.
- [ ] Preserve frame identity and timestamp exactly.
- [ ] Add quality checks without changing packet semantics.
