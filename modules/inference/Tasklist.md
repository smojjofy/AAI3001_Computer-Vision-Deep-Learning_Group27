# Inference Tasklist

Input: optical flow receives RGB8 `FramePacket` pairs directly from Stream
Handler. Depth and segmentation receive selected frames through Frame Processor
when requested by the reranker.

Output: separately timestamped flow results on processed frame pairs, and
segmentation/depth results only when requested by the scheduler. The estimator
results are joined by source frame identity before Splat Construction; the
combined packet contract still needs definition.

- [ ] Orchestrate selective segmentation/depth and per-pair optical flow.
- [ ] Use pretrained models first.
- [ ] Preserve timestamps and dimensions across every output.
