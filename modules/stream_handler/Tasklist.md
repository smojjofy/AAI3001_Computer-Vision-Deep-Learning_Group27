# Stream Handler Tasklist

Input: webcam or IP camera stream, plus camera configuration.

Output: `FramePacket` with decoded RGB data, frame ID, nanosecond timestamp, source ID, dimensions, and quality metrics. The output must match `shared/schemas/frame_packet.schema.json`.

- [x] Support OpenCV live camera/OBS input, RTSP, and recorded-video replay.
- [x] Implement a bounded latest-frame queue, dropped-frame detection, and capped reconnects.
- [x] Add deterministic in-memory replay tests, a metadata fixture, and stream-health metrics.
