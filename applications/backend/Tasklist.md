# Backend Application

- Serve a bounded, read-only RGB preview derived from the Stream Handler's latest `FramePacket`.
- Keep the preview independent of the queue consumed by optical flow and reranking.
- Support the existing camera, OBS Virtual Camera, recorded-video, and RTSP source modes.
- Keep preview transport on localhost for development and avoid storing captured images.
