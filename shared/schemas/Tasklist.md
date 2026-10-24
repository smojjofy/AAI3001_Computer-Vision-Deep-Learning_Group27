# Schema Tasklist

- [x] Define `FramePacket` from Stream Handler to Frame Processor in `frame_packet.schema.json`.
- [ ] Define `ProcessedFrame` from Frame Processor to Inference.
- [ ] Define `InferencePacket` from Inference to Splat Constructor.
- [ ] Define `SplatState` from Splat Constructor to Temporal Cache.
- [ ] Define `ReconstructionState` from Temporal Cache to renderers.
- [x] Add a serialisation-safe `FramePacket` example payload and validation rules.
- [x] Name the packet's categorical timestamp check `quality.timestamp_status` in schema version 1.0.
