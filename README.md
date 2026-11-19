# Live Stream Splat Reconstruction — Group 27

This project reconstructs a live camera view as a **2.5D coloured Gaussian
splat** representation. It combines timestamped RGB capture, optical flow,
frame selection, depth and foreground estimation, splat construction, temporal
updates, and a viewer that can show the input beside the reconstruction.

The result is view-conditioned: a single camera does not reveal hidden surfaces,
and monocular depth alone does not establish real-world scale. The [pipeline
workflow](docs/single-camera-workflow.md) describes the intended architecture;
the [reconstruction research notes](docs/research.md) provide background, not
the current runtime specification.

## Pipeline overview

```text
camera / OBS / video / RTSP → Stream Handler → RGB8 FramePacket ─────────→ GUI input preview
                                                    ↓
                                           Optical Flow → Reranker
                                                           ├─ reuse ────→ Temporal Cache ───────┐
                                                           └─ refresh → depth + segmentation     │
                                                                       → Splat Constructor ───────┤
                                                                                                  ↓
                                                                               Splat Adjustor → Viewer
```

| Part | Role | Interface |
| --- | --- | --- |
| Stream Handler | Captures, decodes, timestamps, normalizes to RGB8, and queues frames. Supports OpenCV camera, OBS Virtual Camera, recorded video, and RTSP. | [`FramePacket`](docs/interfaces/frame_packet.md) |
| Optical Flow | Estimates previous-to-current pixel displacement with Torchvision RAFT Small or OpenCV DIS. | [`FlowResult`](docs/interfaces/flow_result.md) |
| Reranker | Compares the current frame with one retained significant frame and emits `refresh` or `reuse`. | [`SelectionDecision`](docs/interfaces/reranker.md) |
| Live Scheduler | Samples the newest queued frame by capture time and holds at most one active and one replaceable pending refresh job. | [`run_live_scheduler.py`](scripts/run_live_scheduler.py) |
| GUI input preview | Shows a JPEG preview derived from the **same** captured RGB frame. | [GUI run guide](applications/gui/README.md) |

The GUI preview is a display copy; processing keeps the original RGB8 pixels.
The live scheduler command below is a **capture-to-decision diagnostic**: it
simulates downstream refresh work and does not run the full reconstruction.

## Set up on a Windows development machine

From the repository root, use Python **3.11+** and create a separate `.venv` on
each machine. The setup script installs the Stream Handler, Optical Flow, and
Reranker packages.
With `-WithRaft`, it installs PyTorch and Torchvision using the device's detected
NVIDIA driver, or a CPU wheel when no compatible NVIDIA driver is detected:

```powershell
& .\scripts\setup_device.ps1 -WithRaft
```

The script does not configure AMD GPU acceleration. OpenCV DIS works on CPU and
remains a useful fallback on laptops. RAFT Small can run on CPU, but throughput
must be measured on that device. The first RAFT run may download pretrained
weights into ignored `outputs/torch_cache/`.

The current Stream Handler imports the ZED SDK Python package at module load.
Until the camera adapter update is merged, setup and diagnostics also require
that package on machines using an ordinary camera.

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
[`run_gui_preview.py`](scripts/run_gui_preview.py). For a recording, replace the
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
contract](docs/interfaces/frame_packet.md) and
[`shared/schemas/frame_packet.schema.json`](shared/schemas/frame_packet.schema.json).

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

**Demo choice:** use RAFT Small on the tested NVIDIA machine, keep DIS as a
portable CPU fallback and performance baseline. The optical-flow module still
uses OpenCV for image handling even when RAFT is selected. Re-measure with the
full downstream workload before promising a
live frame rate. [Detailed comparison procedure](docs/optical-flow-device-verification.md)

To reproduce the isolated comparison on an existing local recording:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_optical_flow.py data\recordings\flow_eval.mp4 --config config\optical_flow.example.json --max-frames 300 --warmup-pairs 10
.\.venv\Scripts\python.exe scripts\benchmark_optical_flow.py data\recordings\flow_eval.mp4 --config config\optical_flow_raft.example.json --max-frames 300 --warmup-pairs 10
```

Run the two commands separately on the same footage and device. See
[`preview_optical_flow.py`](scripts/preview_optical_flow.py) and the [device
verification guide](docs/optical-flow-device-verification.md) for saved visual
comparisons. [The training plan](docs/optical-flow-training.md) is optional
future work; pretrained RAFT Small is the current demo choice.

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
affect rendered output. [Reranker interface and reasons](docs/interfaces/reranker.md),
[`SelectionDecision` schema](shared/schemas/selection_decision.schema.json).

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

## Documentation

| Document | Use now |
| --- | --- |
| [Stream Handler packet](docs/interfaces/frame_packet.md), [flow result](docs/interfaces/flow_result.md), [reranker decision](docs/interfaces/reranker.md) | Current module contracts and identity rules. |
| [Optical-flow device verification](docs/optical-flow-device-verification.md) | Repeatable camera, replay, visual, and latency checks. |
| [Single-camera workflow](docs/single-camera-workflow.md) | Architecture and handoff design for the full pipeline. |
| [Optical-flow training plan](docs/optical-flow-training.md) | Optional future fine-tuning, not a prerequisite for the demo. |
| [NeRF and 3DGS research notes](docs/research.md) | General reconstruction background, not a current runtime specification. |
| [GUI run guide](applications/gui/README.md) | Preview commands and current GUI behavior. |
