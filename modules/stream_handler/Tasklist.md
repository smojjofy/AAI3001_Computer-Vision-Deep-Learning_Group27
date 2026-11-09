# Stream Handler

- Purpose: capture, timestamp, normalize, and queue RGB8 camera or recorded-video frames.
- Inputs: camera/OBS device index, recorded video path, RTSP URL, or iterable fixture.
- Outputs: `FramePacket` under `shared/schemas/frame_packet.schema.json`.
- Camera capture can select the OpenCV default, DirectShow, or Media Foundation backend on Windows; backend selection does not change the packet schema.
- Remaining work: verify physical and virtual camera device indexes and supported modes on each deployment machine.
