# Optical Flow Source Tasklist

- [x] Implement bounded previous-frame lookup in `OpticalFlowTracker`.
- [x] Return flow, validity, and timestamp metadata.
- [x] Define previous-to-current flow for temporal cache warping.
- [x] Select DIS or RAFT Small through one estimator factory and result contract.
- [x] Consume Stream Handler `FramePacket`s directly from its latest-frame-wins
      queue without assuming consecutive frame IDs.
