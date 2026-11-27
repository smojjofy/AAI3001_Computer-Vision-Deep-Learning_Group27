# Depth and pose: milestone component result

The integrated application estimates depth and body pose for one selected
refresh frame. The webcam path uses YOLO26n depth. The ZED path can supply
SDK depth and skip YOLO. MediaPipe Pose Lite returns one person's 33 body
landmarks. The component combines those landmarks with depth in metres.
The [integration architecture](../../../docs/integration.md) shows the
component boundary. The [research record](RESEARCH_RECORD.md) preserves
component setup, model comparisons, detailed measurements, and alternatives.

## Selection made for the live path

YOLO26n depth provides a metric depth estimate for the webcam path. An
earlier Depth Anything V2 Small component comparison produced relative
depth and took 467 ms for one image on the tested CPU; YOLO26n took 411 ms
for a different model input size on the same photo. That one-photo test
supports the choice of metric output, not a general speed claim. A separate
20-frame CPU benchmark at 1280 x 720 measured a median 267 ms for the
PyTorch pose-depth pipeline and 196 ms for its OpenVINO version. These
component tests do not measure the integrated browser latency.

The live application uses a GPU YOLO path on this computer. It can skip
the depth confidence calculation when no live consumer reads it. The
returned confidence array keeps its shape and is zero-filled in that mode.
The [milestone reflection](../../../docs/Integration-Enhancements.md)
records the measured saving. The ZED SDK-depth path and the optional
left-right stereo matcher have separate behavior. The
[stereo result](../../../docs/STEREO_DEPTH_EXPERIMENT_PLAN.md) states the
matcher validation limit.

## Input and output contract

`PoseDepthPipeline.process_frame` takes an RGB `uint8` image with shape
`[H, W, 3]`, frame ID, source ID, capture timestamp, and optional external
depth. ZED BGRA input is converted to RGB before the call. Depth output
has shape `[H, W]`, uses `float32` metres, and marks invalid pixels with
`NaN`. The pipeline reports one pose, joint depth, body depth, depth
source, and stage timings. It preserves the selected frame identity.

| Result | Meaning |
| --- | --- |
| `out.pose.landmarks` | Up to 33 body landmarks with image position and visibility. |
| `out.depth.depth` | Metric camera depth in metres; `NaN` is invalid. |
| `out.depth.confidence` | Confidence plane in `[0,1]`, or a zero-filled plane when live confidence work is disabled. |
| `out.joint_depth` | Depth for each usable joint; `NaN` when no valid joint depth exists. |
| `out.body_depth` | Median depth from usable shoulder and hip joints. |
| `out.depth_source` | The method that supplied the depth plane. |
| `out.timings_ms` | Pose, depth, fusion, and total stage time. |

The browser reads the selected refresh preview from `/api/pose-depth`.
It returns HTTP 204 before a refresh result exists. The
[GUI guide](../../../applications/gui/README.md) describes its display.
The [construction-frame contract](../../../shared/schemas/construction-frame.md)
defines the depth units and validity rules at the constructor boundary.

## Limits

MediaPipe Lite selects one prominent person. Webcam depth is a model
estimate, so clothing, lighting, texture, and scene geometry can affect
it. The ZED SDK path and webcam path must be compared at matching image
size before a speed or quality claim. The optional stereo matcher has no
accepted live quality result. This component does not make a complete
3D scan from one camera view.

For the web application, use the [runbook](../../../docs/integration_runbook.md).
For component scripts and benchmark reproduction, use the
[research record](RESEARCH_RECORD.md).
