# Optical Flow Tasklist

Input: current and previous RGB8 `FramePacket`s directly from the Stream
Handler, with the same source, dimensions, image coordinates, frame identity,
and timestamp origin. The flow tracker retains one previous packet; it does not
depend on the Frame Processor. An explicitly compatible processed frame can also
be used if a future route requires it.

Output: `FlowResult` metadata plus forward `[height, width, 2]` float32 pixel
displacements and `[height, width]` boolean validity mask. Both arrays are
indexed at the **previous** frame's pixels: `(x, y) + flow[y, x]` locates the
corresponding point in the current frame. See `docs/interfaces/flow_result.md`.

- [x] Implement reduced-resolution OpenCV DIS baseline with configurable size.
- [x] Add a selectable Torchvision RAFT Small backend with optional local fine-tuned weights.
- [x] Define flow direction, pixel units, and metadata schema.
- [x] Handle missing frames, timestamp faults, and scene cuts.
- [x] Connect the Stream Handler's bounded `FramePacket` queue to the flow
      tracker and return each packet with its `FlowResult` for the reranker.
- [ ] Benchmark DIS against pretrained and fine-tuned RAFT Small on recorded project footage and target hardware.
- [ ] Fine-tune RAFT Small on a labeled flow dataset with paired image/flow augmentation and held-out scenes.
