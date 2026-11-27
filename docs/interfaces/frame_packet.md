# FramePacket: Stream Handler to Optical Flow and Reranker

`FramePacket` represents one decoded input frame. It is the boundary between
capture and the downstream pipeline. It contains the left RGB image and can
also carry capture-hardware depth, intrinsics, or a rectified right RGB image
with stereo calibration. It does not contain estimated pose, segmentation,
optical flow, or splat data. The live webcam path supplies RGB only; the ZED
source can supply depth or a stereo pair. See the
[milestone workflow](../single-camera-workflow.md).

The GUI's read-only preview may encode the newest packet as a smaller JPEG
without consuming the pipeline queue. Its frame ID, source ID, and capture
timestamp still identify the original packet. See
`shared/schemas/preview_frame.schema.json` for the display transport.

The serialisation-safe metadata envelope is defined by
[`shared/schemas/frame_packet.schema.json`](../../shared/schemas/frame_packet.schema.json).
The in-process Python representation is `stream_handler.FramePacket`.

| Field | Type / units | Rule |
| --- | --- | --- |
| `frame_id` | unsigned monotonic integer | Unique for every emitted frame from one source. |
| `source_id` | non-empty string | Stable identifier such as `obs-virtual-camera` or `camera:0`. |
| `capture_timestamp_ns` | monotonic clock nanoseconds | Use source presentation time if reliably supplied; otherwise timestamp at decode completion. Never order with wall-clock time. |
| `timestamp_origin` | `source_presentation` or `decode_monotonic` | Explains how `capture_timestamp_ns` was produced. |
| `received_timestamp_ns` | optional monotonic nanoseconds | Diagnostic only; when the handler received/finished decoding the frame. |
| `rgb` | in-process `numpy.uint8[height, width, 3]` | RGB8, row-major. The JSON envelope carries its shape/byte count, not a fixed-size image field. |
| `depth`, `depth_units` | optional `float32[height, width]` and units | Capture depth in metres or millimetres; units are required with depth. Invalid samples can be `NaN`. The ZED source supplies metres. |
| `intrinsics` | optional four floats | `fx`, `fy`, `cx`, `cy` for this image size. |
| `right_rgb`, `stereo` | optional right RGB8 plane and calibration | Rectified pair for the selected-frame stereo experiment; both must be present together. |
| `image.stride_bytes` | bytes | The in-memory row stride. Do not assume a fixed resolution. |
| `quality` | metrics | Decode latency, frame age (only when comparable), queue depth, cumulative handler drops, timestamp status. |

`capture_timestamp_ns` should be non-decreasing per source. A duplicate or
backwards source timestamp is emitted with `quality.timestamp_status` set to
`duplicate` or `backwards` so consumers can reject it deliberately. The frame ID
remains monotonic even when a source timestamp is faulty.

`quality.frame_age_ns` is measurable only when capture time is from the same
monotonic clock as the handler (`decode_monotonic`). It is `0` and
`frame_age_available` is false for an independent source-presentation clock,
such as a replay file PTS; consumers must not mistake that value for real age.

The handler queue is bounded and latest-frame-wins: when full, it discards the
oldest unprocessed packet and increments `dropped_frame_count`. Consumers must
not infer a contiguous frame sequence.

When crossing a process or network boundary, send the JSON envelope and a
separate variable-length raw RGB8 binary plane (or a documented alternative
encoding). Do not impose a 512-bit packet size.
