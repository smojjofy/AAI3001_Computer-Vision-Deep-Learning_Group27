# Schema Tasklist

- [ ] Define `FramePacket` from Stream Handler to Frame Processor.
- [ ] Define `ProcessedFrame` from Frame Processor to Inference.
- [x] Define `ConstructionFrame` v1 from Inference to Splat Constructor.
- [x] Define `SplatState` v1 from Splat Constructor to Temporal Cache.
- [ ] Define `ReconstructionState` from Temporal Cache to renderers.
- [x] Add constructor-boundary validation rules.
- [x] Add an in-memory NumPy/ctypes transport for constructor boundaries.
- [ ] Add serialized examples once an inter-process transport is selected.
