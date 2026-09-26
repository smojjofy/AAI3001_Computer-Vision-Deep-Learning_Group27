# Stream Handler Tasklist

Input: webcam or IP camera stream, plus camera configuration.

Output: `FramePacket` with decoded RGB data, frame ID, nanosecond timestamp, source ID, dimensions, and quality metrics. The output must match `shared/schemas/frame_packet.schema.json`.

- [x] Support OpenCV live camera/OBS input, RTSP, and recorded-video replay.
- [x] Implement a bounded latest-frame queue, dropped-frame detection, and reconnect backoff.
- [x] Replay recorded video at its presentation cadence by default, with an explicit fastest mode for batch use.
- [x] Report source health, no-frame stalls, capture rate, and decode-latency summaries.
- [x] Release a source before joining its worker during shutdown, so blocking reads can be interrupted.
- [x] Add a licence-cleared, test-generated recorded-video decoder fixture and OpenCV integration coverage.
- [x] Add deterministic live-source failure, reconnect, stall, and blocking-read shutdown coverage.
- [x] Add a dependency-light CLI argument-parser smoke check.
- [x] Add OpenCV-adapter smoke coverage once OpenCV is installed locally.
- [ ] Run the documented operator smoke test with an OBS virtual camera, physical camera, or RTSP source.
- [ ] Detect and report source-timestamp gaps as well as duplicate/backwards timestamps.
- [ ] Run a 10-minute live camera and recorded-video soak test for memory stability.
- [ ] Add optional per-source OpenCV capture settings (resolution/FPS, backend, read timeout, upstream buffer policy) if required by a chosen device.
