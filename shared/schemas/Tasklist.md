# Schema Tasklist

- [x] Define `FramePacket` from Stream Handler to Frame Processor in `frame_packet.schema.json`.
- [x] Define `FlowResult` metadata in `flow_result.schema.json`; arrays are separate in-process or binary payloads.
- [ ] Define `ProcessedFrame` from Frame Processor to Inference.
- [ ] Define `InferencePacket` from Inference to Splat Constructor.
- [ ] Define `SplatState` from Splat Constructor to Temporal Cache.
- [ ] Define `ReconstructionState` from Temporal Cache to renderers.
- [x] Add a serialisation-safe `FramePacket` example payload and validation rules.
