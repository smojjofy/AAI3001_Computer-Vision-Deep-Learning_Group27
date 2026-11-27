# Earlier README and component notes

This record preserves the component-era landing page. The [current README](../README.md) and [milestone workflow](single-camera-workflow.md) describe the delivered application.

# Live Stream Splat Reconstruction — Group 27

This project displays a live camera view as a depth-guided Gaussian splat
visualization. It combines timestamped RGB capture, optical flow, frame
selection, depth and pose estimation, person masking, splat construction,
and a browser view. A refresh replaces the active splat state. Reuse moves
displayed positions with optical flow.

The result is view-conditioned: a single camera does not reveal hidden
surfaces, and estimated depth does not recover unseen geometry. The
[milestone workflow](../docs/single-camera-workflow.md) describes the delivered
application. The [research conclusion](../docs/research.md) states what this
single-camera method can establish.

## Team and component ownership

Five students built the pipeline, one per work package. Each member owned the
stages listed here, and the integrated application is the join of all five.

| Member | Work package |
| --- | --- |
| Jaison | Pipeline architecture, Splat Constructor, and integration |
| ZeKai | Preliminary research and ZED SDK exploration |
| Ziven | Stream Handler, Optical Flow processor, and Reranker |
| Jaziel | DepthAnything and YOLO depth processor |
| Brejesh | MediaPipe pose processor |

## Integrated application (final milestone)

The `applications/` tree now runs the complete local pipeline:

```text
webcam / OBS frame
  -> 100 ms candidate window with optical flow
  -> refresh: YOLO depth + MediaPipe pose + pose-gradient person mask
              (depth-threshold mask when selection falls back)
              -> native Splat Constructor -> active splat state
  -> reuse:   screen-space flow warp of the displayed splats
  -> browser: video, depth, pose, Gaussian splats, diagnostics
```

Run it:

```powershell
& .\scripts\setup_system.ps1
& .\start_live_demo.ps1
```

Open `http://127.0.0.1:8765/`. The [runbook](../docs/integration_runbook.md)
contains the launch commands. The [integration architecture](../docs/integration.md)
explains the application.

The [technical record](../docs/integration-technical-history.md) records tests on this computer.
Its 51 ms warm refresh and about 150 FPS browser result came from earlier,
separate tests. They are not one end-to-end latency measurement. The current
live mask is pose-gradient selection with a depth-threshold fallback.

## Component contributions

The integrated application uses these component boundaries:

```text
camera / OBS / ZED -> Stream Handler -> RGB8 FramePacket -> Optical Flow -> Reranker
                                                          | reuse -> browser display warp
                                                          | refresh -> depth + pose -> person mask
                                                                       -> Splat Constructor
                                                                       -> active state -> browser
```

| Part | Milestone role | Interface |
| --- | --- | --- |
| Stream Handler | Captures and timestamps camera, OBS, ZED, video, or RTSP frames. | [`FramePacket`](../docs/interfaces/frame_packet.md) |
| Optical Flow | Estimates displacement against the retained reference. The integrated default is OpenCV DIS. | [`FlowResult`](../docs/interfaces/flow_result.md) |
| Reranker | Selects refresh or reuse from optical-flow magnitude. | [`SelectionDecision`](../docs/interfaces/reranker.md) |
| Depth and Pose | Estimate selected-frame depth and human joints. | [Depth component](../modules/inference/depth/README.md), [pose component](../modules/inference/pose/README.md) |
| Person mask and Splat Constructor | Select the subject, then build the active Gaussian state. | [Construction frame](../shared/schemas/construction-frame.md), [splat state](../shared/schemas/splat-state.md) |
| Backend and viewer | Publish one snapshot and render it with `gsplat`. | [Integration architecture](../docs/integration.md), [GUI guide](../applications/gui/README.md) |

The GUI preview is a display copy; processing keeps the original RGB8 pixels.
The separate live scheduler diagnostic simulates downstream refresh work. It
does not run the full reconstruction.

## Set up on a Windows development machine

- Windows 10/11 and Python **3.11+** (`py --version`).
- A physical camera, OBS Virtual Camera, local video, or RTSP source.
- OBS Studio only when using OBS as the source layer.

The integrated setup installs PyTorch and one OpenCV-Contrib package. The
application uses OpenCV DIS for optical flow by default. RAFT Small is an
optional optical-flow backend, but PyTorch also serves depth inference.

## Install

Run the setup script in PowerShell from the repository root. It creates
`.venv` and installs the integrated application's dependencies:

```powershell
& .\scripts\setup_system.ps1
```

The script selects a CPU or NVIDIA PyTorch wheel from the device information.
Override that choice when needed:

```powershell
& .\scripts\setup_system.ps1 -TorchPlatform cpu
```

The script does not configure AMD GPU acceleration. OpenCV DIS works on CPU and
remains a useful fallback on laptops. RAFT Small can run on CPU, but throughput
must be measured on that device. The first RAFT run may download pretrained
weights into ignored `outputs/torch_cache/`.

The ZED source is optional. Webcam and OBS runs do not require the ZED SDK.
See the [runbook](../docs/integration_runbook.md#select-another-source) for the
ZED launch command and the [technical record](../docs/integration-technical-history.md#capture-source-webcam-or-zed-2i)
for measured results.

For the GUI, also install Node.js/npm and run `npm ci` inside
`applications/gui/` once per checkout or whenever its lockfile changes.

### Inspect live capture, flow, decisions, and GUI input

Run this in **terminal 1** from the repository root. It uses RAFT Small, samples
at most once every 300 ms of capture time, and runs until Ctrl+C:

```powershell
.\.venv\Scripts\python.exe scripts\run_live_scheduler.py --source camera --device 0 --camera-backend dshow --config config\optical_flow_raft.example.json --sample-ms 300 --max-reference-age-ms 1000 --refresh-work-ms 1000 --duration-s 0 --gui-preview --summary-only
```

Run the GUI in **terminal 2**:

```powershell
cd applications\gui
npm ci
npm run dev
```

Open the local URL printed by Vite. Stop the Python command with Ctrl+C to see
its summary. `--summary-only` suppresses per-frame decisions; omit it when
inspecting individual `refresh` and `reuse` reasons.

For OBS, start **Virtual Camera** and use `--source obs` with the virtual
camera's device index. For a physical camera, close OBS or other apps that hold
that device. If Windows Media Foundation opens a camera but emits no frames,
try `--camera-backend dshow`; device indices vary by machine. Only one capture
process should open the physical camera at a time. The GUI itself does not open
the camera.

To inspect input without running flow or reranking, use
[`run_gui_preview.py`](../scripts/run_gui_preview.py). For a recording, replace the
camera arguments with `--source video --path <existing-video>`. Store private
recordings under ignored `data/recordings/`, and keep camera URLs, weights, and
generated outputs out of Git. The live scheduler accepts `video`, `camera`, and
`obs`; the standalone stream diagnostic also supports `rtsp`.

## Stream Handler: why frames can be skipped

Each `FramePacket` carries `source_id`, a monotonically increasing `frame_id`,
capture timestamp and its clock origin, RGB8 pixels, dimensions, and quality
diagnostics. The handler reports capture FPS, decode latency, dropped frames,
timestamp anomalies, reconnects, and health. Recorded video can replay at its
presentation cadence or decode as fast as possible. See the [packet
contract](../docs/interfaces/frame_packet.md) and
[`shared/schemas/frame_packet.schema.json`](../shared/schemas/frame_packet.schema.json).

Capture can run faster than inference. The default queue holds **three** frames
and drops old unprocessed frames when full. The live scheduler also takes the
newest queued packet and discards older ones. It then applies a **capture-time
sampling interval**, not a fixed frame-count buffer. For example, 300 ms means
at most about 3.3 flow/reranker comparisons per second whether the source is
30 or 60 FPS. Skipped frame IDs are expected; downstream stages must use
timestamps and IDs rather than assume adjacent frames.

The GUI reads the handler's latest packet without draining the inference queue.
Its JPEG preview preserves the packet's ID, source, and capture timestamp. The
preview does not save a recording and does not alter the original RGB8 frame.

## Optical Flow: model choice and measured results

Optical flow maps pixels from a **retained reference frame** to the current
camera frame. `flow[y, x] = (dx, dy)` points from `(x, y)` in the reference to
`(x + dx, y + dy)` in the current image. Both backends return flow at input
image size and a forward/backward consistency mask. RAFT Small performs a
forward and reverse inference for that mask. `valid_fraction` is the share of
pixels passing this check; it is **not** ground-truth accuracy.

- **RAFT Small** is a learned, iterative optical-flow model using all-pairs
  feature correlations. This repo uses Torchvision's pretrained small variant,
  12 update steps, and a configured 320×180 maximum processing size. [RAFT
  paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470392.pdf),
  [Torchvision model](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.optical_flow.raft_small.html).
- **DIS** searches for patch correspondences and refines a dense field. This
  repo uses OpenCV's `fast` preset at the same configured processing size.
  [DIS paper](https://www.cvlibs.net/projects/autonomous_vision_survey/literature/Kroeger2016ARXIV.pdf).

An earlier **300-frame `flow_eval.mp4` run**, with 10 warmup pairs skipped and
289 measured pairs per backend, produced these isolated flow timings:

| Backend | Settings | p50 | p95 | Mean valid fraction |
| --- | --- | ---: | ---: | ---: |
| OpenCV DIS | `fast`, 320×180 max | 83.051 ms | 89.161 ms | 0.969 |
| RAFT Small | 12 updates, 320×180 max | 200.564 ms | 230.591 ms | 0.911 |

On that machine and clip, RAFT's median comparison took about **2.4×** as long
as DIS. Visual review favored RAFT's highlighting of subject movement in
the inspected previews, including frame 250. That is a **qualitative
observation**, not a measured accuracy win: the recording has no labelled
ground-truth flow or endpoint-error score. The lower RAFT valid fraction does
not by itself mean less accurate motion. These timings exclude model loading,
the first 10 pairs, capture contention, depth, and splat work, so they do not
predict end-to-end broadcast latency.

**Integrated application:** OpenCV DIS is the default. RAFT Small is an optional
backend for isolated comparison. The optical-flow module still uses OpenCV for
image handling when RAFT is selected. These isolated timings do not establish
live application latency. See the [comparison procedure](../docs/optical-flow-device-verification.md).

To reproduce the isolated comparison on an existing local recording:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_optical_flow.py data\recordings\flow_eval.mp4 --config config\optical_flow.example.json --max-frames 300 --warmup-pairs 10
.\.venv\Scripts\python.exe scripts\benchmark_optical_flow.py data\recordings\flow_eval.mp4 --config config\optical_flow_raft.example.json --max-frames 300 --warmup-pairs 10
```

Run the two commands separately on the same footage and device. See
[`preview_optical_flow.py`](../scripts/preview_optical_flow.py) and the [device
verification guide](../docs/optical-flow-device-verification.md) for saved visual
comparisons. [The training plan](../docs/optical-flow-training.md) is optional
future work; pretrained RAFT Small is an optional backend.

## Reranker and live scheduling policy

The reranker retains **one significant RGB frame**. Optical flow compares that
reference with the next selected frame, rather than automatically comparing
adjacent captured frames. Its score is the median motion magnitude over valid
flow pixels. With the current defaults, a score of at least **2 px** requests
`refresh`; below that requests `reuse`, provided at least **25%** of pixels
pass validity. Bootstrap, unavailable or inconsistent flow, and an expired
reference also request `refresh`. The default reference age limit is **500 ms**;
the live demo command above sets it to **1000 ms**. These are initial operating
thresholds, not tuned reconstruction-quality guarantees.

`SelectionDecision` preserves the selected frame and reference identities,
capture times, action, reason, and difference score when available. The live
scheduler sends refresh decisions to one simulated slow worker with **one
replaceable pending request**. It never grows an unbounded refresh queue.
`reuse` still needs a completed matching Temporal Cache state before it can
affect rendered output. [Reranker interface and reasons](../docs/interfaces/reranker.md),
[`SelectionDecision` schema](../shared/schemas/selection_decision.schema.json).

A 15-second DirectShow webcam check previously captured **434 frames** at
roughly **28 FPS** and processed **47** selected frames: 13 `refresh`, 34
`reuse`, zero reconnects, and no capture error. The interval skipped 333 frames;
the bounded queue dropped 8 and skipped 43 older queued frames. This verifies
capture, flow, selection, and bounded scheduling on that test; it does **not**
measure reconstruction or prove the chosen thresholds preserve important
motion. Different cameras and machines need their own checks.

## Integration contract

Frame selection passes the selected original frame, optional matching flow,
and `SelectionDecision` to downstream processing. Depth, segmentation, pose,
and splat results must join by original frame ID and timestamp, not callback
order. The diagnostic command above substitutes a timed worker for downstream
processing, so its timing results cover frame selection rather than splat
reconstruction.

## FramePacket fields

| Document | Use now |
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

## Earlier design and diagnostic tools

The [original workflow](../docs/single-camera-workflow.md) records the proposed
Temporal Cache, Splat Adjustor, and Three.js-first viewer. The
[integration architecture](../docs/integration.md) states which parts reached the
live application. For isolated stream and reranker checks, use the
[optical-flow verification guide](../docs/optical-flow-device-verification.md).

## Splat Constructor (implemented)

`modules/splat_constructor/` contains a dependency-free C++20 CPU constructor,
native tests, a CLI that writes a standard 62-property 3DGS PLY plus metadata,
and a NumPy/ctypes bridge. The in-memory boundary is `ConstructionFrame` v1 to
`SplatState` v1, documented in `shared/schemas/` and
`modules/splat_constructor/README.md`.

Build the native core first; this emits the CLI, the test binary, and the
Python-loadable bridge DLL:

```powershell
./scripts/build_splat_constructor.ps1
./build/splat_constructor/splat_constructor_tests.exe
```

The bridge takes aligned `uint8 (H,W,3)` RGB, `float32 (H,W)` depth, and binary
`uint8 (H,W)` foreground mask, plus `frame_id`, positive `timestamp_ns`,
`source_id`, intrinsics `(fx, fy, cx, cy)`, and a row-major rigid
`camera_to_world` pose:

```python
from modules.splat_constructor import (
    ConstructionFrame, ConstructionConfig,
    DepthSemantics, DepthUnits, construct_splat_state,
)

frame = ConstructionFrame(
    frame_id="0001",
    timestamp_ns=source_timestamp_ns,
    source_id="camera_01",
    rgb=rgb_u8,
    depth=relative_inverse_depth_f32,
    foreground_mask=foreground_mask_u8,
    intrinsics=(fx, fy, cx, cy),
    depth_semantics=DepthSemantics.RELATIVE_INVERSE,
    depth_units=DepthUnits.UNITLESS,
)
state = construct_splat_state(frame, ConstructionConfig(sample_stride=2))
```

The constructor also accepts `DepthSemantics.METRIC_CAMERA_Z` with
`DepthUnits.METERS`. `construction_frame_from_estimates()` converts estimator
outputs and rejects misaligned tensors. The output is a mask-shaped relief until
a depth source with consistent scale feeds it; it is not a metric scan.

## Hardware target and troubleshooting

A useful initial target is 720p/30 capture, 16–32 GB RAM, a modern 6-core CPU,
and an RTX 3060-class 12 GB NVIDIA GPU or better. Actual performance depends on
resolution, runtime/precision, heat, and the simultaneous OBS workload. On weaker
hardware, reduce depth frequency rather than letting latency accumulate.

| Symptom | What to check |
| --- | --- |
| Cannot open source | Start OBS Virtual Camera, try another device index, and close other camera-using apps. |
| Windows MSMF camera starts but emits zero frames | Try `--camera-backend dshow` in the live scheduler diagnostic. If OBS owns the physical camera, start OBS Virtual Camera and use `--source obs` with its device index. |
| `opencv` import error | Re-run the editable install with the `.venv` Python shown above. |
| Drops constantly increase | Lower source resolution/FPS or speed up the consumer; never replace the bounded queue with an unbounded one. |
| Reconnects repeat | Check the physical/RTSP source and final `last_error`; keep credentials out of reports. |
| Reconstruction seems incomplete | Expected for a single view; hidden surfaces require multi-view or tracked camera movement later. |

## Layout

```text
shared/schemas/            source-of-truth packet definitions
modules/stream_handler/    implemented capture slice
modules/frame_processor/   preprocessing and decision support
modules/inference/         flow, depth, pose, and experimental segmentation
modules/splat_constructor/ implemented CPU Gaussian constructor and bridge
modules/temporal_cache/    separate module; not the active live-state owner
applications/gui/         live gsplat viewer and optional Three.js environment
docs/interfaces/           packet boundary documentation
```

Preserve shared packet identity and document alpha/depth/coordinate decisions in
`shared/` rather than making incompatible private formats.

## Documentation

| Document | Contents |
| --- | --- |
| [Integration architecture](../docs/integration.md) | Final pipeline, component boundaries, viewer decision, and limits. |
| [Runbook](../docs/integration_runbook.md) | Setup and launch commands for the web application. |
| [Splat representation and runtime](../docs/splat-representation-and-runtime.md) | Constructor values, browser conversion, processor split, and transport. |
| [Milestone reflection](../docs/Integration-Enhancements.md) | Final mask, latency, and learning conclusions with evidence limits. |
| [Integration evidence](../docs/integration-evidence.md), [technical history](../docs/integration-technical-history.md), [enhancement evidence](../docs/enhancement-evidence.md) | Phase measurements and repair details. |
| [Stereo result](../docs/STEREO_DEPTH_EXPERIMENT_PLAN.md), [stereo research record](../docs/stereo-research-record.md) | Implemented experimental path, measured device facts, and open live checks. |
| [Learning result](../docs/DEEP_LEARNING_PLAN.md), [learning research record](../docs/learning-research-record.md) | Trained mask candidate, checked test, and research method. |
| [Physical test checklist](../docs/physical_test_checklist.md) | P1 to P5, with commands and the evidence to return. |
| [Stream Handler packet](../docs/interfaces/frame_packet.md), [flow result](../docs/interfaces/flow_result.md), [reranker decision](../docs/interfaces/reranker.md) | Current module contracts and identity rules. |
| [Optical-flow device verification](../docs/optical-flow-device-verification.md) | Repeatable camera, replay, visual, and latency checks. |
| [Milestone workflow](../docs/single-camera-workflow.md), [original design](../docs/single-camera-design-record.md) | Delivered data flow and preserved early proposal. |
| [Optical-flow decision](../docs/optical-flow-training.md), [training design](../docs/optical-flow-training-record.md) | DIS default, RAFT comparison, and deferred fine-tuning method. |
| [Research conclusion](../docs/research.md), [technical survey](../docs/research-background.md) | Why the milestone uses depth-guided splats and what one camera cannot establish. |
| [GUI run guide](../applications/gui/README.md) | Preview commands and current GUI behavior. |
| [Pose estimator benchmark](../modules/inference/pose/README.md) | MediaPipe Lite versus Full on eleven scenarios. |
| [Depth estimator](../modules/inference/depth/README.md), [Splat Constructor](../modules/splat_constructor/README.md) | Component contracts and benchmarks. |
| [Shared contracts](../shared/schemas/construction-frame.md), [splat state](../shared/schemas/splat-state.md), [coordinates](../shared/coordinate_system/README.md) | Data shapes and coordinate rules. |
| [Reconstruction validation](../modules/reconstruction_validation/README.md) | Source-view checks for constructor output. |
