# Live splat representation and runtime

This document describes the integrated Gaussian-splat application. The
[runbook](integration_runbook.md) gives the run command. The
[technical record](integration-technical-history.md) keeps measurements.
The [construction contract](../shared/schemas/construction-frame.md),
[splat-state contract](../shared/schemas/splat-state.md), and
[coordinate convention](../shared/coordinate_system/README.md) define the
module boundary. The [original design record](single-camera-design-record.md) includes
designs that are not in the live application.

## One refresh

The selected RGB frame, depth, pose, and person mask refer to one camera frame.
The constructor samples usable mask pixels. It projects each pixel through the
camera intrinsics at its depth. It then applies `camera_to_world`. Positions
are in metres in a right-handed world: `+X` right, `+Y` up, and camera forward
`-Z`. The C++ constructor runs on the central processing unit (CPU). The
[Python bridge](../modules/splat_constructor/README.md) copies the result into
NumPy arrays. The live application keeps one active splat state. A reuse
decision moves displayed positions with optical flow; it does not fuse a new
persistent geometry state.

## Stored values and browser values

The constructor stores one set of arrays for each refresh. In the table,
`color` is an RGB channel value in `[0,1]`, and `SH_C0` is
`0.28209479177387814`. `SH` means spherical harmonics. The degree-zero term
holds the constant part of the color.

| Constructor value | Stored rule | Browser rule |
| --- | --- | --- |
| `positions` | Project a sampled depth pixel into camera space, then apply `camera_to_world`. | Rotate the point 180 degrees about X: `(x, -y, -z)` for the `gsplat` display world. |
| `sh_dc` | `(color - 0.5) / SH_C0` for each RGB channel. | Recover the channel with `0.5 + sh_dc * SH_C0`, then convert it to an 8-bit color. |
| `opacity_logits` | Apply `logit` to configured opacity times reconstruction weight. | Apply `sigmoid`, multiply by `display_alpha`, then convert to 8-bit alpha. |
| `log_scales` | Store the natural logarithm of each positive axis scale in metres. Pixel footprint, sample stride, and edge reduction set the scale. | Apply `exp` to each value to recover a linear scale. |
| `rotations` | Store a unit quaternion in `wxyz` order. A local depth gradient can orient its axes. | Apply the same display rotation: `(w, x, y, z)` becomes `(-x, w, -z, y)`. |
| `reconstruction_weights` and `source_pixels` | Keep the input weight and source pixel `(u,v)` for each splat. | Use the source pixel to sample optical flow for display reuse. |

The constructor makes these values in
[`constructor.cpp`](../modules/splat_constructor/src/constructor.cpp). The
browser converts them in
[`SplatOutputPanel.ts`](../applications/gui/src/components/SplatOutputPanel.ts).
The 180-degree rotation applies to the default splat panel. The optional
[environment compositor](../applications/gui/README.md#environment-compositor)
has its own camera and scene setup. Do not copy the panel conversion into a
different viewer without checking its camera convention.

## Transport and display order

The refresh worker builds a JSON header and packed array buffer. The server
publishes the splat, mask, and pose-depth preview as one snapshot under one
lock. `/api/splat` sends a little-endian 32-bit unsigned header length, the
UTF-8 JSON header, and the binary arrays. The header gives the type, count,
and byte offset of each array. The browser copies the binary part to an
aligned buffer before it makes typed-array views. See
[`adapters.py`](../applications/live_pipeline/adapters.py),
[`pipeline_server.py`](../applications/backend/pipeline_server.py), and
[`liveApi.ts`](../applications/gui/src/api/liveApi.ts).

The browser checks the active identity from `/api/diagnostics` before it shows
a new state. The `gsplat` worker must return a sorted index for each active
splat before the panel marks that state ready. This depth order lets the
renderer blend translucent splats in view order. The browser polls
diagnostics every 300 ms by default; `?poll=100` changes this interval. See
the [technical state-machine record](integration-technical-history.md#active-state-machine-phase-1-finalized)
for the rules that keep responses from different refreshes apart.

## Processor split and latency

MediaPipe pose runs on the CPU runtime. YOLO depth uses PyTorch on the graphics
processing unit (GPU) for the webcam path. The ZED source can supply SDK depth;
the optional stereo matcher is an experiment. The native splat constructor
runs on the CPU. The browser uses WebGL2 for `gsplat` rendering, and the
optional environment view uses Three.js. Optical flow defaults to OpenCV DIS.

These stages run at different times. A [measured warm-refresh run](enhancement-evidence.md#41-the-measured-starting-point)
reported mean GPU utilization near 5 percent. That aggregate reading does not
locate the delay. Capture, pose, mask selection,
CPU construction, transport, sorting, and browser display also take time.
Use the stage timings and the active-state latency in the
[technical record](integration-technical-history.md#earlier-measured-results-on-this-computer) before you
attribute a delay to one processor. An isolated warm refresh time is not a
live end-to-end latency measurement.

## Current limits

Each refresh replaces the active splat state. Reuse is a view-space motion
warp. The application does not assign persistent physical-point IDs or merge
splats across refreshes. The [splat-state schema](../shared/schemas/splat-state.md)
also describes a planned Temporal Cache consumer; that consumer is outside
the current live path. The single camera gives only one measured view of the
subject, so an orbit view can show stretched or incomplete geometry.
