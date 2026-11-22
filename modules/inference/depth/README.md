# Depth + Pose Estimator (YOLO26 depth + MediaPipe Pose Lite)

Part of the Group 27 single-camera reconstruction pipeline (AAI3001).

This module takes **one RGB frame** and returns:

- a **depth map in metres** (how far each pixel is from the camera), from **YOLO26n depth**,
  or from the **ZED camera's own depth** when the frame comes with it;
- the person's **33 body joints** from **MediaPipe Pose Landmarker (Lite)**;
- the **distance of every joint** and of the person as a whole, by reading the depth map at each joint.

```
camera / ZED ─► stream handler ─► reranker ─► refresh worker ─► PoseDepthPipeline (this module)
                                                                   ├─ MediaPipe Pose Lite → 33 joints (x, y)
                                                                   ├─ depth: ZED if given, else YOLO26n → metres
                                                                   └─ depth at each joint → distances
                                                                          │
                                                   preview server /api/pose-depth ─► GUI
```

Depth is **metric** (metres, bigger = farther, NaN = no valid depth), with a per-pixel
**confidence** (0–1, low at object edges) kept for splat reconstruction.
Full conventions: `DEPTH_CONVENTION.md`.

**Contents:** [Background](#background) · [Setup](#setup) · [Quick start](#quick-start) ·
[Usage](#usage) · [Output](#output) · [How it works](#how-it-works) · [Running tests](#running-tests) ·
[Performance](#performance) · [Design decisions](#design-decisions) · [Limitations](#limitations) ·
[Future work](#future-work) · [Troubleshooting](#troubleshooting) · [Acknowledgements](#acknowledgements)

---

## Background

**Monocular depth estimation** predicts, from a **single** RGB image, how far each pixel is from the
camera. A depth model learns this from cues such as object size, perspective, occlusion and texture.
It is cheaper than stereo (the ZED's two lenses) or multi-view methods, but it is an estimate, so it is
softest at object edges and can be off in absolute scale.

A depth map is one of two kinds:

| Kind | Values | Example | Used here? |
|---|---|---|---|
| **Metric (absolute) depth** | Real distances, e.g. metres | YOLO26 depth, ZED stereo depth | **Yes** |
| **Relative depth** | Only near/far *order*, scaled 0–1 per image; no units | Depth Anything V2 (relative model) | No (first version only) |

**Pose estimation** (MediaPipe Pose Landmarker) finds the person's 33 body joints as 2D image positions.
Reading the metric depth map at each joint gives every joint a distance from the camera. With the
camera's intrinsics (focal length, image centre), each joint can also be placed in 3D
(`joint_xyz`, optional).

**Where this fits in the project:** the splat constructor turns frames into Gaussians, and it needs
to know how far things are. This module provides that depth (plus a confidence map to drop unreliable
edge pixels) and the person's joints for human-specific steps such as pose binding. The reranker
decides, per sampled frame, whether to **reuse** the last result (little change) or **refresh**
it; this module runs on every refresh.

---

## Setup

Run everything on Windows PowerShell. Python 3.11 or newer (tested on 3.13).

**1. Virtual environment** (repo root, once):

```powershell
cd C:\path\to\AAI3001_Computer-Vision-Deep-Learning_Group27
python -m venv .venv
.venv\Scripts\Activate.ps1          # the prompt must now show (.venv)
```

**2. Packages** (from `modules\inference\depth`):

```powershell
cd modules\inference\depth
python -m pip install -r requirements.txt
python -m pip install openvino
```

**3. Export YOLO depth to OpenVINO** (once per PC, about 5 s):

```powershell
python -c "from ultralytics import YOLO; YOLO('models/yolo26n-depth.pt').export(format='openvino', imgsz=640)"
```

This creates `models\yolo26n-depth_openvino_model\`: the same model and the same 640 input,
about 25–40% faster on CPU (see [Performance](#performance)).

**4. Pose model:** download `pose_landmarker_lite.task` from the MediaPipe Pose Landmarker page and put it in
`modules\inference\pose\models\`.

> The OpenVINO folder and the `.task` file are **git-ignored**, so every person / PC does steps 3 and 4 once.
> Always use `python -m pip`, not plain `pip`.

---

## Quick start

```powershell
cd modules\inference\depth
python app.py
```

The browser opens. Upload a photo of a person in the **Pose + depth** tab to see the skeleton,
the distance per joint, and the time taken.

---

## Usage

### Gradio demo app

```powershell
python app.py          # from modules\inference\depth
```

| Tab | Shows |
|---|---|
| **YOLO depth** | Depth map in metres (click it to read a distance), confidence map, `.npy` download |
| **Pose + depth** | Skeleton with distances on the photo and on the depth map, plus a table of all 33 joints |

- Pick `yolo26n-depth_openvino_model` in the model box for the faster OpenVINO version.
- Keep the YOLO input size at **640** (see [Design decisions](#design-decisions)).
- The first photo after starting is slow (~3 s, model warm-up); later ones are normal.

### Command-line scripts

All from `modules\inference\depth`. Add `--no-show` to skip the pop-up window,
`--yolo-model yolo26n-depth_openvino_model` to use OpenVINO.

| Script | What it does | Example |
|---|---|---|
| `scripts\test_depth_image.py` | Depth for one image, prints the range in metres | `python scripts\test_depth_image.py --image photo.jpg` |
| `scripts\run_depth_folder.py` | Depth for every image in a folder (PNG + `.npy` each) | `python scripts\run_depth_folder.py --input photos --output photos_out` |
| `scripts\test_pose_depth_image.py` | Pose + depth for one image, prints every joint's distance, saves PNG + JSON | `python scripts\test_pose_depth_image.py --image photo.jpg` |
| same, with ZED depth | Uses a saved ZED depth map instead of YOLO | `... --image zed_left.png --zed-depth zed_depth.npy` (add `--depth-scale 0.001` if it is in mm) |

### In the live system

From the **repo root**, with the venv active:

```powershell
.\.venv\Scripts\python.exe scripts\run_live_scheduler.py --source camera --device 0 --camera-backend dshow --config config\optical_flow_raft.example.json --sample-ms 300 --max-reference-age-ms 1000 --duration-s 0 --gui-preview --pose-depth
```

- `--pose-depth` makes the scheduler's refresh worker run this pipeline on every refreshed frame
  (without it, the worker only simulates work). Each result prints one line in the terminal.
- With `--gui-preview`, the latest result is served at **`http://127.0.0.1:8765/api/pose-depth`**.
- The live script loads the model named in its `PoseDepthPipeline(...)` line. If that is
  `yolo26n-depth_openvino_model`, step 3 of Setup must have been done on that PC.

### From code

Create the pipeline **once**, then call `process_frame` for every frame
(creating it per frame would reload the models every time).

```python
import sys
sys.path.insert(0, "modules/inference/depth")       # path to this folder, from where your code runs

from src import PoseDepthPipeline, zed_to_rgb

pipe = PoseDepthPipeline(yolo_model="yolo26n-depth_openvino_model")   # once; add depth_scale=0.001 if ZED depth is in mm

out = pipe.process_frame(rgb, frame_id, source_id, timestamp_ns)                    # YOLO depth
out = pipe.process_frame(rgb, frame_id, source_id, timestamp_ns, depth=zed_depth)   # ZED depth (YOLO skipped)

pipe.close()                                          # when the app shuts down
```

`rgb` is an RGB `uint8` image `[H, W, 3]`. ZED images are BGRA: convert with `zed_to_rgb(image)` first.

---

## Output

### `out` (Python)

| Field | Meaning |
|---|---|
| `out.pose.landmarks` | 33 joints; `x`, `y` are 0–1 of the image width/height |
| `out.depth.depth` | Depth map `[H, W]`, float32 metres, NaN = invalid |
| `out.depth.confidence` | `[H, W]`, 0–1 (low at edges, 0 where invalid) |
| `out.joint_depth` | Metres per joint; NaN = joint off-screen, not visible enough, or no depth there |
| `out.body_depth` | How far the person is: median of shoulders + hips |
| `out.depth_source` | `"zed"` or `"yolo"` |
| `out.timings_ms` | `pose`, `depth`, `fuse`, `total` in ms |

### `GET /api/pose-depth` (JSON, for the GUI)

| Field | Meaning |
|---|---|
| `frame_id`, `source_id`, `capture_timestamp_ns` | Which camera frame this result belongs to |
| `depth_kind`, `depth_source` | `"metric_depth"`; `"zed"` or `"yolo"` |
| `near_m`, `far_m` | Depth range in the frame (2nd–98th percentile) |
| `person_detected`, `body_depth_m` | Whether a person was found, and how far away |
| `joints` | 33 × `{index, name, x, y, visibility, depth_m}`; `depth_m` is `null` when unknown |
| `depth_jpeg_base64` | Coloured depth map (0 m red → `far_m` blue), 640 px wide |
| `skeleton_jpeg_base64` | Camera frame with skeleton and distances |
| `timings_ms` | Same as above |

Returns **204 No Content** until the first result exists. Joints with `x`/`y` outside 0–1
are outside the picture (MediaPipe's guess) and always have `depth_m: null`.

---

## How it works

One call to `PoseDepthPipeline.process_frame(rgb, frame_id, source_id, timestamp_ns, depth=None)`:

1. **Pose.** MediaPipe Pose Landmarker (Lite) runs on the RGB frame and returns 33 joints
   (`x`, `y` in 0–1, plus a visibility score). Only the most prominent person is returned.
2. **Depth.**
   - **ZED depth given:** it is converted to metres (`depth_scale`), NaN / ±inf / ≤ 0 become
     invalid, and it is resized (nearest-neighbour, so no invented in-between depths) to the frame size.
     YOLO is not run.
   - **No depth given:** YOLO26n depth runs at a fixed 640 input and returns metres, resized back to the
     frame size. The YOLO model is loaded on first use.
3. **Shared depth convention** (`normalize.py`). Invalid values become NaN, a valid mask is built, and
   a **confidence** map is computed from how fast log-depth changes between neighbouring pixels:
   1 on smooth surfaces, near 0 at depth edges (where the person blends into the background).
4. **Depth per joint** (`pose_depth.py`). For each joint, the median valid depth in a small 5×5 window
   around it. Joints that are off-screen, or that MediaPipe is unsure about (visibility < 0.5), get NaN.
   The person's distance (`body_depth`) is the median of the shoulders and hips.
5. **Optional 3D joints.** With `CameraIntrinsics(fx, fy, cx, cy)`, each joint is back-projected to
   camera-space metres: `x = (u − cx)·z / fx`, `y = (v − cy)·z / fy`, `z` = joint depth.
6. **Identity check.** Pose and depth must come from the same frame (`frame_id`, `source_id`,
   timestamp); otherwise an error is raised rather than silently mixing frames.

| File (`src/`) | Role |
|---|---|
| `pipeline.py` | `PoseDepthPipeline`, ZED depth conversion, 3D joints |
| `yolo_depth.py` | YOLO26 depth model (`.pt` or OpenVINO export) |
| `normalize.py` | Shared depth convention, confidence map |
| `contracts.py` | `DepthEstimation` / `DepthFrameResult`, which check the convention on creation |
| `pose_depth.py` | Depth at each joint, drawing, joint table |
| `preview.py` | Builds the `/api/pose-depth` JSON for the GUI |
| `utils.py` | Colouring, depth sampling, device selection |

---

## Running tests

```powershell
cd modules\inference\depth
python -m pytest tests -q
```

Expect `58 passed`. They cover the depth conventions, invalid-depth handling, pose + depth fusion
and the pipeline (YOLO vs ZED depth, mm → m, ZED's NaN/inf values, 3D joints) using stand-in
models, plus real YOLO depth checks. The YOLO tests are skipped if `ultralytics` or
`models\yolo26n-depth.pt` is missing.

---

## Performance

### Depth Anything V2 vs YOLO26 depth (why YOLO)

| Model | Input size | Output | Time |
|---|---|---|---|
| Depth Anything V2 Small | 252 (its smallest) | relative 0–1, no metres | 467 ms |
| YOLO26n depth | 672 | **metres** | **411 ms** |

Same photo (shrunk to 640×336 from 1200×630), CPU, one run each in the Gradio app, so a rough
comparison. Depth Anything was slower **even at its smallest input size**, and only gives relative depth.

**Depth Anything V2 Small at input 252: 467 ms, relative depth**
<img width="1200" height="600" alt="Screenshot 2026-11-05 105010" src="https://github.com/user-attachments/assets/41150cfc-e691-4cf4-a2ef-044a545378f5" />

**YOLO26n depth at input 672: 411 ms, metric depth**
<img width="1200" height="600" alt="Screenshot 2026-11-05 105043" src="https://github.com/user-attachments/assets/77fd8f8c-01e1-437b-b875-3ca414f8d805" />

### OpenVINO vs PyTorch (YOLO26n depth, input 640, CPU)

Median of 20 frames after 5 warm-up frames:

| Test | PyTorch `.pt` | OpenVINO |
|---|---|---|
| 1280×720: YOLO model only | 163 ms | **117 ms** |
| 1280×720: full depth step (model + cleanup + confidence) | 268 ms | **149 ms** |
| **1280×720: whole pipeline (pose + depth)** | 267 ms (3.7 fps) | **196 ms (5.1 fps)** |
| 1920×1080: whole pipeline | 333 ms (3.0 fps) | **237 ms (4.2 fps)** |
| Live webcam, inside the scheduler with RAFT running | — | ≈220 ms per refresh (see below) |

With ZED depth instead of YOLO (YOLO skipped), the whole pipeline is about **100 ms** per frame (≈10 fps)
at 1280×720. Pose alone is about 20–40 ms.

### Live, inside the team's scheduler

`run_live_scheduler.py --pose-depth`, webcam, OpenVINO, RAFT optical flow running at the same time,
10 consecutive refreshes:

| | Median | Range |
|---|---|---|
| Pose | 26 ms | 14–55 ms |
| Depth | 192 ms | 172–202 ms |
| **Total per refresh** | **219 ms** | 206–252 ms |

The person read 0.48–0.85 m from the webcam while moving, and frames without a person were reported as
`person_detected: false`. The reranker reused the previous result for most sampled frames and
refreshed about once per second, so this module ran roughly once a second. On a busy or battery-powered
laptop, the same run took 2–3× longer, mainly because RAFT and this module share the CPU.

Reproduce:

```powershell
python scripts\benchmark_latency.py --images some_photos --resize 1280 720 --frames 20 --pipeline
python scripts\benchmark_latency.py --images some_photos --resize 1280 720 --frames 20 --pipeline --yolo-model yolo26n-depth_openvino_model
```

(`--images` takes a folder or one image; add `--pipeline-zed` to time the ZED path.)
Numbers vary by about ±30 ms on a laptop and are higher when other programs use the CPU.
Single clicks in the Gradio app are too noisy to compare models; use the benchmark.

---

## Design decisions

- **YOLO26n depth instead of Depth Anything V2.** The pipeline needs real distances per joint.
  Depth Anything V2 (used first) gives only relative depth, was slower on CPU, and needed its repo
  as a submodule plus a separate weights file. It is still in the git history of the
  `jaziel-depth-estimator` branch.
- **YOLO input size fixed at 640.** That is YOLO depth's training size. Other sizes change the
  metres themselves, not just sharpness: at 384 the same person read ≈60% farther (2.16 m vs 1.37 m).
  For speed, use OpenVINO, smaller camera frames, a GPU, or ZED depth instead.
- **Confidence map kept,** even though it costs time, because splat reconstruction can use it
  to drop "flying pixels" at object edges.
- **Depth per joint = median of a small window,** not one pixel, because joints often sit right on
  the person's edge. Joints MediaPipe is unsure about (visibility < 0.5) get no depth.
- **YOLO loads only when needed,** so a run where every frame has ZED depth never pays for it.

## Limitations

- One person per frame (MediaPipe Pose Lite returns a single pose).
- Webcam frames carry no depth, so YOLO is used; ZED depth plugs in once the stream handler's
  packet has a depth field (one line in the refresh worker).
- The GUI does not yet read `/api/pose-depth`.
- First frame after start-up takes ~3–4 s (model warm-up).
- The package is imported as `src`; if that clashes with another module after merging, rename it
  (e.g. to `depth_estimator`).

---

## Future work

- **Use ZED depth in the live system.** Once the stream handler's ZED packet carries depth, pass it as
  `depth=packet.<field>` in the refresh worker (one line). That skips YOLO (≈100 ms per frame instead of
  ≈200 ms) and gives stereo-measured metres.
- **Fill ZED's holes with YOLO depth.** ZED stereo leaves gaps on shiny, dark or textureless surfaces
  and very close up. Scaling YOLO depth to match ZED on the valid pixels (one scale + shift fit) and
  using it only inside the holes would give complete, metric depth.
- **Neural depth correction.** A learned correction step between depth and splat construction, in the
  spirit of SwiftNDC (monocular depth aligned by an affine fit to reliable sparse depth, then a light
  per-pixel neural correction and reprojection-based filtering). Mainly useful if the system must run
  without a ZED; with a ZED, its depth already serves as the reference.
- **Depth tolerance settings.** A `valid_range` (e.g. 0.3–8 m) to drop implausible depths, and a
  confidence threshold for which pixels count when reading joint depths.
- **Human body model.** Fitting a body model such as SMPL to the 3D joints, for human pose binding in
  the splat.
- **Speed.** A GPU (CUDA) build of PyTorch, computing confidence on a smaller copy of the depth map,
  and loading the models at start-up (`warmup`) so the first refresh isn't slow.
- **Multiple people.** MediaPipe can return several poses (`num_poses`), but the pose estimator's
  output currently holds one.

### Alternatives considered

| Approach | Why not (for now) |
|---|---|
| Depth Anything V2 (relative) | No metres; slower on CPU (see [Performance](#performance)) |
| Feed-forward geometry models (e.g. FastVGGT, SAIL-Recon) | Predict depth, camera pose and point maps in one pass. They could replace this module's depth source in a multi-frame setup, but need a GPU and are heavier than a single-frame model |
| Smoothing the depth map (bilateral / guided filters) | Tested on YOLO depth at 640: edges barely changed and metres were unchanged; a guided filter blurred edges where hair and background have similar colour. The confidence map is used instead to drop edge pixels |
| Smaller YOLO input (e.g. 384) for speed | Changes the metres (≈60% farther); OpenVINO gives the speed without that |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named cv2` / `ultralytics` | The venv isn't active: `.venv\Scripts\Activate.ps1` (prompt shows `(.venv)`) |
| `Fatal error in launcher` (pip) | Use `python -m pip ...` |
| `...yolo26n-depth_openvino_model' does not exist` | Run the OpenVINO export (Setup step 3), or use `yolo26n-depth.pt` |
| `Pose model missing` | Put `pose_landmarker_lite.task` in `modules\inference\pose\models\` |
| `cannot find empty port` | An old `app.py` is still running: close it |
| Browser shows an old version of the app | Close old terminals, rerun, press Ctrl+F5 |
| First run ~3–4 s, later ~0.2 s | Normal: model warm-up |
| Distances look too far / too near | Check the YOLO input size is 640 |
| Pylance can't find `pose_estimator` | Editor-only warning; select the `.venv` interpreter in VS Code |

---

## Authors

- **Jaziel**: depth estimation, pose + depth pipeline, OpenVINO, benchmarks
- **Brejesh**: MediaPipe pose estimator (`modules/inference/pose`)

## Acknowledgements

- [Ultralytics YOLO26](https://docs.ultralytics.com/) — YOLO26n depth model (AGPL-3.0)
- [MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) — pose estimation (Apache-2.0)
- [OpenVINO](https://docs.openvino.ai/) — CPU inference speed-up
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) — depth model used in the first version (Apache-2.0 for the Small model)
- [SwiftNDC](https://arxiv.org/abs/2602.22565) — neural depth correction, reference for future work
- [SMPL](https://smpl.is.tue.mpg.de/) — human body model, reference for future work
- Group 27 teammates: stream handler, reranker and scheduler, preview server and GUI
