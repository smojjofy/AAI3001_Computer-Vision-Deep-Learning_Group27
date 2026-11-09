# Optical flow: device verification and model benchmark

Run these commands in **PowerShell from the repository root**. The commands
use the Stream Handler's `FramePacket` output directly. DIS and RAFT Small use
the same `FlowResult` contract, so the same recording and frame IDs can be
compared. Generated previews and benchmark files belong in `outputs/`, which
is ignored by Git. Keep recordings in `data/recordings/`, also ignored by Git.

## 1. Prepare the environment on each device

If `.venv` does not exist on that device, create it with Python 3.11 or newer:

```powershell
py -3 -m venv .venv
```

Install the two local modules using the virtual environment's Python:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .\modules\stream_handler
.\.venv\Scripts\python.exe -m pip install -e .\modules\inference\optical_flow
```

For **RAFT Small**, install a matching PyTorch + Torchvision build. On an NVIDIA
device, choose Windows, Pip, and the device's supported CUDA option in the
[official PyTorch installer](https://pytorch.org/get-started/locally/). Run its
command through `.\.venv\Scripts\python.exe -m pip ...` so packages land in this
environment. On a CPU-only device, choose the CPU build. Check availability:

```powershell
.\.venv\Scripts\python.exe -c "import cv2; print('OpenCV', cv2.__version__, 'DIS', hasattr(cv2, 'DISOpticalFlow_create'))"
.\.venv\Scripts\python.exe -c "import torch, torchvision; print('torch', torch.__version__, 'vision', torchvision.__version__, 'CUDA', torch.cuda.is_available())"
```

`CUDA False` means RAFT Small will use the CPU with `raft_device: auto`. Do not
compare those times with another device's GPU times. The first pretrained RAFT
run downloads Torchvision's default weights to `outputs/torch_cache/`; later
runs use that cache. [Torchvision documents the RAFT Small weights and input
requirements](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.optical_flow.raft_small.html).

## 2. Check a physical camera or OBS Virtual Camera

Start with DIS. Replace device index `0` if the camera appears at a different
index. For OBS, start its Virtual Camera first and use `--source obs`.

```powershell
.\.venv\Scripts\python.exe scripts\run_optical_flow.py --source camera --device 0 --duration-s 20 --config config\optical_flow.example.json
.\.venv\Scripts\python.exe scripts\run_optical_flow.py --source obs --device 0 --duration-s 20 --config config\optical_flow.example.json
```

For an IP camera, use `--source rtsp --url '<your RTSP address>'`. The command
prints JSON diagnostics only; it does **not** record video or save images.

Check the output in order:

1. Frame IDs increase. A gap is allowed when the bounded capture queue drops
   frames; `queue_drops_total` reports that count.
2. The first result is `no_previous_frame`. Later valid pairs normally report
   `ok`; `previous_frame_id` is the last frame the flow consumer received.
3. `elapsed_ns` comes from capture timestamps. Duplicate or backward timestamps
   produce `invalid_timestamps`; a hard cut may produce `scene_cut`.
4. `valid_fraction` is the share of previous-frame pixels that passed the
   forward/backward check. It is **not** an accuracy score.
5. `median_motion_px` is the median over valid pixels. A moving person can be
   hidden by a mostly static background in this median; use the preview below
   to inspect local motion.

Run the same live check with RAFT Small:

```powershell
.\.venv\Scripts\python.exe scripts\run_optical_flow.py --source camera --device 0 --duration-s 20 --config config\optical_flow_raft.example.json
```

For a quieter throughput check, add `--summary-only`. Its final JSON reports
consumed and emitted frames, queue drops, drop fraction, p50/p95 flow time,
status counts, and mean valid fraction.

## 3. Record one shared evaluation clip

Record a clip using OBS or another camera recorder and place it at, for example,
`data/recordings/flow_eval.mp4`. Keep its original resolution and frame rate.
Include a still interval, a subject moving left/right and toward/away from the
camera, occlusion and newly exposed background, plus a short blur or lighting
change. Keep the camera fixed for the main comparison. Use **this exact file**
for every model and every device. The diagnostic script does not make the
recording for you.

Check the live-style playback first. `paced` replay follows the video's
presentation timestamps and exercises the bounded capture queue:

```powershell
.\.venv\Scripts\python.exe scripts\run_optical_flow.py --source video --path 'data\recordings\flow_eval.mp4' --replay-mode paced --duration-s 120 --summary-only --config config\optical_flow.example.json
.\.venv\Scripts\python.exe scripts\run_optical_flow.py --source video --path 'data\recordings\flow_eval.mp4' --replay-mode paced --duration-s 120 --summary-only --config config\optical_flow_raft.example.json
```

Set `--duration-s` longer than the clip. The command stops once a finite video
ends. Compare `frames_emitted`, `frames_consumed`, `queue_drops_total`, and
`flow_ms_p95`. Replayed RAFT can drop more frames because processing a pair may
take longer than the next capture interval. Use `fastest` only as an additional
stress test; it is not the capture-rate comparison.

## 4. Inspect the same frame pair visually

Choose frame IDs inside the clip for stillness, subject motion, and occlusion.
For each chosen ID, save the same pair with both backends:

```powershell
.\.venv\Scripts\python.exe scripts\preview_optical_flow.py 'data\recordings\flow_eval.mp4' --frame-id 60 --config config\optical_flow.example.json
.\.venv\Scripts\python.exe scripts\preview_optical_flow.py 'data\recordings\flow_eval.mp4' --frame-id 60 --config config\optical_flow_raft.example.json
```

Open the PNGs printed by the commands under `outputs/flow_previews/`. Each has
the previous frame with green motion arrows, the current frame, and a color
flow map. The arrows are enlarged **4× for display**; the reported flow itself
is in actual pixels. Both backends use the same fixed color scale by default.
Positive `dx` points right and positive `dy` points down, from the **previous**
frame to the **current** frame. Look for direction on the moving subject,
alignment at boundaries, and invalid/newly exposed regions. A smooth or high
validity map alone does not prove that motion is correct.

## 5. Measure isolated flow inference fairly

The benchmark script reads the same video sequentially through Stream Handler
and times flow calculation. It **does not** simulate the live capture queue and
does not include depth or rendering. Run each model separately, without other
GPU-heavy work, and repeat each command three times. Use the same
`--max-frames` and `--warmup-pairs`; warmup pairs are excluded from p50/p95.

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_optical_flow.py 'data\recordings\flow_eval.mp4' --config config\optical_flow.example.json --max-frames 900 --warmup-pairs 10
.\.venv\Scripts\python.exe scripts\benchmark_optical_flow.py 'data\recordings\flow_eval.mp4' --config config\optical_flow_raft.example.json --max-frames 900 --warmup-pairs 10
```

The 900-frame cap is an example; choose a value that covers the same part of
your clip in every run. Check `frames`, `measured_pairs`, and `statuses` match.
Record `flow_latency_ms_p50`, `flow_latency_ms_p95`, and
`valid_fraction_mean`. Model loading and the first warmup pairs are outside the
reported latency. A run with no measured pairs needs a longer clip or fewer
warmup pairs.

To try **RAFT Small with four update steps**, create a local config in the
ignored `outputs/` folder, then run the same replay and benchmark commands with
`--config outputs\optical_flow_raft4.json`:

```powershell
$raft4Config = Get-Content 'config\optical_flow_raft.example.json' -Raw | ConvertFrom-Json
$raft4Config.raft_iterations = 4
$raft4Config | ConvertTo-Json | Set-Content 'outputs\optical_flow_raft4.json'
```

Fewer updates may be faster, but compare visual alignment as well as timing.
Do not change resolution, crop, frame count, or hardware between model runs.

| Device / footage | Backend and settings | Isolated p50 / p95 ms | Paced queue drops / emitted | Visual notes |
| --- | --- | --- | --- | --- |
|  | DIS fast, 320×180 max |  |  |  |
|  | RAFT Small, 12 updates |  |  |  |
|  | RAFT Small, 4 updates |  |  |  |

## 6. Decide what the measurements establish

The same-clip comparison can show whether a backend tracks your subject and
whether it keeps up with capture. A 30 FPS source provides about 33 ms between
frames, but flow must share time and GPU resources with other stages. The final
backend choice therefore also needs a later run with depth, motion, splat
construction, and rendering active. On labeled optical-flow data, compute
endpoint error (EPE) separately; these scripts do not have ground-truth flow
and cannot report EPE. Keep the preview images and benchmark table as evidence
for the team decision.
