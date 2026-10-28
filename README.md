# Live Stream Splat Reconstruction — Group 27

This repository is building a live, single-camera reconstruction prototype. The
first result is a **view-conditioned 2.5D coloured splat**: live RGB is
segmented, assigned relative depth, converted to splats, and temporally smoothed.

It is not a complete, metric 3D scan. One fixed camera cannot observe hidden
surfaces and monocular depth has no guaranteed real-world scale.

## Current implementation

The first vertical slice, `modules/stream_handler`, is implemented:

```text
OBS Virtual Camera / USB camera / recorded video / RTSP
                         |
                         v
                 Stream Handler
        decode → RGB8 → timestamp → bounded queue
                         |
                         v
                    FramePacket
```

It provides OpenCV camera, OBS, recorded-video, and RTSP adapters; deterministic
replay input for local validation; RGB8 normalisation; monotonic IDs; timestamp diagnostics;
a bounded latest-frame-wins queue; stream health; timestamp-paced recorded replay;
and exponential reconnect backoff. Live sources retry indefinitely by default;
applications may set a finite reconnect-attempt limit when that is safer.
It does not download models or create splats yet.

The capture contract is [`shared/schemas/frame_packet.schema.json`](shared/schemas/frame_packet.schema.json).
Read its accompanying explanation in [`docs/interfaces/frame_packet.md`](docs/interfaces/frame_packet.md).

## Prerequisites

- Windows 10/11 and Python **3.11+** (`py --version`).
- A physical camera, OBS Virtual Camera, local video, or RTSP source.
- OBS Studio only when using OBS as the source layer.

The capture install brings in `numpy` and `opencv-python`. MediaPipe, PyTorch,
Depth Anything, model weights, and the viewer are deliberately not dependencies
of this module yet.

## Install

Run these PowerShell commands from the repository root:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .\modules\stream_handler
```

If `py` is not configured on this Windows machine, create the environment with:

```powershell
& 'C:\Users\ziven\AppData\Local\Programs\Python\Python313\python.exe' -m venv .venv
```

## OBS live setup

OBS is the easiest local demo source: it can compose a webcam, phone camera,
video, crop, and colour correction, then expose the scene as a normal camera.

The current runnable reports captured frames in PowerShell; OBS Preview remains
the live visual display until the downstream Three.js viewer is implemented.

1. Create the desired OBS scene.
2. Set a stable initial output: **1280×720, 30 FPS**.
3. In OBS **Controls**, click **Start Virtual Camera**.
4. Start the capture diagnostic. Try device index `0` first; if it is the wrong
   device or fails, try `1`, then `2`.

```powershell
.\.venv\Scripts\splat-stream.exe --source obs --device 0 --duration 30
```

The diagnostic stores no imagery. It periodically reports frame number,
resolution, handler state, queue depth, cumulative drops, and timestamp status.
For a healthy source, frames keep arriving at the expected resolution and the
queue does not grow without bound.

OBS Virtual Camera is best for the local MVP. Avoid sending the same local source
through public RTMP: encoding and network delay make real-time timing harder to
diagnose. RTSP or NDI are later options when OBS runs on another machine.

## Other source modes

```powershell
# Direct webcam
.\.venv\Scripts\splat-stream.exe --source camera --device 0 --duration 30

# Recorded non-private video (replays at the recorded cadence)
.\.venv\Scripts\splat-stream.exe --source video --path .\path\to\demo.mp4 --duration 30

# Decode recorded video as fast as possible for batch processing
.\.venv\Scripts\splat-stream.exe --source video --path .\path\to\demo.mp4 --replay-mode fastest --duration 30

# RTSP; do not commit URLs or credentials
.\.venv\Scripts\splat-stream.exe --source rtsp --url "rtsp://camera-host:554/stream" --duration 30
```

Use environment variables or an untracked local launcher for authenticated RTSP
URLs so credentials do not enter Git or shell history. Do not commit recordings,
model weights, processed outputs, or camera details.

## `FramePacket` and real-time rules

| Field | Meaning |
| --- | --- |
| `frame_id` | Unique monotonic integer assigned by the handler. |
| `source_id` | Stable label such as `obs-virtual-camera`. |
| `capture_timestamp_ns` | Source presentation time when available, otherwise monotonic decode-completion time. |
| `timestamp_origin` | States how the timestamp was produced. |
| `rgb` | In-process `uint8[height, width, 3]` RGB8 image. |
| `quality` | Decode time, frame age (when its clock is comparable), queue depth, drops, and timestamp status. |

Images are variable-size data, not 512-bit packets. In one process the RGB array
is passed by reference. Across a process/network boundary, send the versioned JSON
metadata envelope plus a separate length-delimited binary image plane.

The default queue capacity is three frames. If full, it discards the oldest
unprocessed frame. This is intentional: live freshness matters more than finishing
a frame that is already stale. Timestamp duplicates/backwards jumps are flagged,
not silently hidden. A handler with no successful frame for its configurable
stall timeout reports `stalled`; its final metrics also include capture FPS and
mean/max decode latency.

## Planned pipeline: flow-guided reuse and selective estimation

Only the Stream Handler shown above is implemented. The remaining stages and
feedback path below are the intended design, not a working pipeline yet. The
Optical flow now operates immediately on the Stream Handler's available RGB8
`FramePacket` pairs. The reranker then selects the useful work. The Frame
Processor executes that selection: it sends a reuse update to the Temporal
Cache, or prepares the selected refresh frame for depth, motion/pose, and
foreground estimation. A refresh produces a splat candidate; a Splat Adjustor
combines that candidate, or the reused cache state, into the state shown by the
viewer.

```text
Stream Handler → FramePacket → Optical Flow → Reranker
                                                ↓
                              Frame Processor decision / dispatch
                               ├─ reuse → Temporal Cache ────────────┐
                               └─ refresh → ProcessedFrame            │
                                              → depth + segmentation   │
                                              → optional motion / pose  │
                                              → Splat Constructor ─────┤
                                                                       ↓
                                                           Splat Adjustor → Three.js viewer

```

Optical flow compares camera frames, not frames against splats. The reranker
uses its one retained significant camera frame as the reference for each new
frame. If the difference is significant, it flushes that reference, copies the
new frame into its one-entry cache, and sends the new frame to the Frame
Processor's refresh path. Otherwise the Frame Processor takes the reuse path.
The reranker does **not** judge splat quality. The Temporal Cache is a separate
splat-state store used downstream by the Splat Adjustor and viewer; its format
is not an input to the reranker.

OpenCV DIS is the implemented reduced-resolution baseline. Torchvision RAFT
Small is an implemented, optional backend when its PyTorch dependencies and
weights are available; it must be benchmarked against DIS on project footage.
SEA-RAFT remains a future candidate if selection or splat adjustment needs more
accurate flow. Depth Anything 3 Small/Base are candidates for depth; the depth
model and any human-pose estimator have not been selected or integrated.
Human body pose and camera pose are different outputs. Foreground segmentation
remains necessary for alpha. Estimator workers may finish at different times;
their results must be joined by original frame ID and timestamp, never callback
arrival order. Begin integration with model stubs and recorded fixtures.

The reranker selection contract is documented in
[`docs/interfaces/reranker.md`](docs/interfaces/reranker.md).

## Hardware target and troubleshooting

A useful initial target is 720p/30 capture, 16–32 GB RAM, a modern 6-core CPU,
and an RTX 3060-class 12 GB NVIDIA GPU or better. Actual performance depends on
resolution, runtime/precision, heat, and the simultaneous OBS workload. On weaker
hardware, reduce depth frequency rather than letting latency accumulate.

| Symptom | What to check |
| --- | --- |
| Cannot open source | Start OBS Virtual Camera, try another device index, and close other camera-using apps. |
| `opencv` import error | Re-run the editable install with the `.venv` Python shown above. |
| Drops constantly increase | Lower source resolution/FPS or speed up the consumer; never replace the bounded queue with an unbounded one. |
| Reconnects repeat | Check the physical/RTSP source and final `last_error`; keep credentials out of reports. |
| Reconstruction seems incomplete | Expected for a single view; hidden surfaces require multi-view or tracked camera movement later. |

## Layout

```text
shared/schemas/            source-of-truth packet definitions
modules/stream_handler/    implemented capture slice
modules/frame_processor/   next: resize, normalise, optional undistort
modules/inference/         next: flow, segmentation, depth; optional human pose
modules/splat_constructor/ later: depth/mask to Gaussian states
modules/temporal_cache/    later: state reuse, refresh feedback, smoothing
modules/three_viewer/      later: Three.js rendering
docs/interfaces/           packet boundary documentation
```

Preserve shared packet identity and document alpha/depth/coordinate decisions in
`shared/` rather than making incompatible private formats.
