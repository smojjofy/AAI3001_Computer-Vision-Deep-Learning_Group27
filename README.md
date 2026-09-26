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
replay input for tests; RGB8 normalisation; monotonic IDs; timestamp diagnostics;
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

Run tests (no camera, network connection, model download, or private recording
is needed):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s modules\stream_handler\tests -v
```

The tests check colour conversion, frame IDs, timestamp anomaly reporting,
bounded-queue dropping, invalid frames, replay end-of-stream, and the actual
OpenCV recorded-video adapter. That integration test generates a three-frame
primary-colour video at runtime, so no recording is committed to the repository.

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

## Next phase: frame processing and inference

The Frame Processor will accept a `FramePacket`, validate RGB, resize/normalise,
optionally undistort it, and emit a `ProcessedFrame` with the **same** frame ID,
source ID, and timestamp.

```text
FramePacket
  → Frame Processor → ProcessedFrame
  → parallel workers
       ├─ segmentation → foreground alpha mask
       ├─ MediaPipe Pose → body landmarks
       ├─ Depth Anything V2 Small → relative depth
       └─ optical flow → current/prior-frame motion
  → asynchronous join by original frame ID + timestamp
  → InferencePacket → Splat Constructor
```

Workers finish at different times. Their outputs must be joined by source frame
identity, never callback arrival order. Start with model stubs to prove that
joining works, then add segmentation, MediaPipe, and Depth Anything one by one.
The viewer can render at 30 FPS while depth updates at a sustainable lower rate.

MediaPipe supplies approximate single-camera pose landmarks, not full motion
capture. Depth Anything V2 Small supplies **relative**, not LiDAR-grade metric,
depth. A separate segmentation model is necessary for foreground alpha.

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
modules/inference/         next: segmentation, depth, pose, flow
modules/splat_constructor/ later: depth/mask to Gaussian states
modules/temporal_cache/    later: temporal smoothing/interpolation
modules/three_viewer/      later: Three.js rendering
docs/interfaces/           packet boundary documentation
```

Read the relevant `Tasklist.md` before extending a module. Preserve shared packet
identity and document alpha/depth/coordinate decisions in `shared/` rather than
making incompatible private formats.
