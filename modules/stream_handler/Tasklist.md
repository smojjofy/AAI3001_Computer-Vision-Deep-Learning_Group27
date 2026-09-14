# Stream Handler Tasklist

Input: webcam or IP camera stream, plus camera configuration.

Output: `FramePacket` with decoded RGB data, frame ID, nanosecond timestamp, source ID, dimensions, and quality metrics. The output must match `shared/schemas/frame_packet.schema.json`.

- [ ] Support live input and recorded replay.
- [ ] Implement buffering, dropped-frame detection, and graceful reconnects.
- [ ] Add deterministic fixtures and stream-health metrics.
