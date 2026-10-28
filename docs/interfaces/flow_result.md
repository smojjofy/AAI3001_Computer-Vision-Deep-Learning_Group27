# FlowResult: Optical Flow to Reranker

`FlowResult` compares **two camera frames**, never a camera frame to splats.
The in-process object stores a float32 `[height, width, 2]` flow array and a
boolean `[height, width]` validity mask. Its serialisation-safe envelope is
`shared/schemas/flow_result.schema.json`; binary arrays travel separately.

For a valid pixel `(x, y)` in the **previous** frame, `flow[y, x] = (dx, dy)`
points to `(x + dx, y + dy)` in the current frame. Positive x is right;
positive y is down. Values use pixels at the reported output width and height,
even when DIS ran on smaller images. Pixels outside the current image or failing
forward/backward consistency are invalid. A valid mask means useful temporal
correspondence, not a calibrated probability or fresh depth estimate.

Both frames must have the same source, dimensions, and image coordinate system.
The intended live input is consecutive *available* RGB8 `FramePacket`s from the
Stream Handler; the Frame Processor is not on this optical-flow path. Callers
must not mix undistorted and raw images. IDs may skip, but must
increase; timestamps must increase within the same timestamp origin. A source
change starts a new tracker pair. `no_previous_frame`, `invalid_timestamps`, and
`scene_cut` return zero flow with an all-false mask so the scheduler can request
fresh estimation rather than reuse stale geometry. `elapsed_ns` records actual
time between the compared frames and is **not** inferred from frame IDs.

The reranker consumes the frame/flow observation and compares each new frame
with its own one-entry record of the last significant camera frame. A
significant difference replaces that record and yields `refresh`; otherwise it
yields `reuse`. The Frame Processor executes that decision. The Temporal Cache
is a downstream splat-format store and is not an input to the reranker. Flow
only reports frame correspondence and validity. Newly exposed surfaces require
fresh depth; flow cannot fill them in.

`OpticalFlowTracker.push(frame)` accepts the Stream Handler's RGB8
`FramePacket` directly. A compatible processed frame could also be passed if a
future route needs it, but the current routing does not require that. The tracker
keeps one previous frame. `create_estimator(FlowConfig(...))` selects DIS or
Torchvision RAFT Small while preserving this same result contract. Both return
vectors at input size. RAFT Small rounds its internal size to multiples of 8
and uses at least 128 pixels per axis; it runs forward and backward inference
for the validity check. PyTorch, Torchvision, and weights are optional and are
loaded only when RAFT Small is selected.

`StreamFlowConsumer.poll()` drains the existing Stream Handler queue and returns
`FlowObservation(packet, flow, processing_time_ns)` for the future reranker. It
pairs the current packet with the last packet **actually dequeued**, so skipped
frame IDs caused by the bounded queue are preserved and handled. Run
`scripts/run_optical_flow.py --source video --path path/to/video.mp4` for a
recorded diagnostic or choose `camera`/`obs` for live input. The command prints
JSON metrics and does not save raw frames.
