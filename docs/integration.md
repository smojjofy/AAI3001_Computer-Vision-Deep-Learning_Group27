# Integration guide

This folder is the final implementation of the live Gaussian-splat pipeline. It
combines the frozen teammate modules into one application on one computer.

## Pipeline

```text
webcam / OBS RGB frame
  -> 100 ms candidate window
  -> optical flow for each candidate against the retained refresh reference
  -> highest normalized flow score wins the window
       refresh -> YOLO metric depth + depth-threshold person mask + MediaPipe pose
                -> native Splat Constructor -> newest active splat state -> browser
       reuse   -> screen-space flow warp of the displayed splats -> browser
```

The retained reference advances only after a refresh produces a valid state.
A refresh without person pose or without a separable person mask keeps the last
valid state.

## Run

```powershell
# One-time environment (CUDA torch, mediapipe, ultralytics, one OpenCV wheel)
./scripts/setup_integration_env.ps1

# Build the native constructor
./scripts/build_splat_constructor.ps1

# Build the GUI once
cd applications/gui; npm ci; npm run build; cd ../..

# Run the application (webcam first, OBS or video also work)
.\.venv\Scripts\python.exe scripts\run_integration_app.py --source webcam --device 0
```

Open `http://127.0.0.1:8765/`. The server serves the built GUI and the API.

Options are in `--help`; the important ones are `--threshold` (fixed normalized
flow threshold), `--window-ms`, `--active-age-ms`, `--depth-band-m`,
`--max-mask-area`, `--sample-stride`, `--allow-missing-pose`.

## API

| Endpoint | Content |
| --- | --- |
| `GET /api/preview` | latest captured RGB frame (JPEG, identity preserved) |
| `GET /api/pose-depth` | depth map + skeleton JPEG and joint values for the latest refresh |
| `GET /api/mask` | depth-threshold person mask (PNG) |
| `GET /api/splat` | splat state: uint32 header length, JSON header, binary arrays |
| `GET /api/motion` | reuse flow preview and pose joints for the browser warp |
| `GET /api/diagnostics` | action, score, timings, cache age, counters, queue state |
| `GET /api/health` | stream health |

The `/api/splat` array order is `positions, sh_dc, opacity_logits, log_scales,
rotations, reconstruction_weights, source_pixels`; encodings are `wxyz`
rotations, natural-log scales, and logit opacity, all in the constructor
coordinate convention (right-handed, `+Y` up, camera forward `-Z`).

## Measured results on this computer

Computer: Windows, Python 3.13.7, NVIDIA GeForce RTX 4050 Laptop GPU (6 GB),
driver 591.74, torch 2.11.0+cu128, OpenCV 4.14 (contrib), mediapipe 1.2.3,
ultralytics 8.4.213.

| Test | Result |
| --- | --- |
| Refresh chain, 640x360 still, warm | 51 ms total: pose + YOLO depth 37 ms, mask 3 ms, constructor 6 ms |
| Refresh chain, cold start | 4.8 s (model load) |
| Depth-threshold mask, single person | 30.2% image area, covers the person, temporal IoU 1.0 on identical frames |
| Depth-threshold mask, group photo | selects the nearest person (19.4% area), other people at that depth can join the mask |
| Full-frame mask guard | a uniform scene is rejected (`max_area_fraction` 0.60) |
| Live webcam, 25 s | capture 13.5 FPS, 102 windows (4.05/s), 28 refresh states, 46 reuse decisions |
| Splat transport | ~68 bytes per splat; 11.3k splats ≈ 0.77 MB; full-frame 76.8k splats ≈ 5.2 MB |

## Known limits

- The depth-threshold mask selects a depth band, not a person. Objects at the
  same distance as the subject join the mask. The mask pass test must be
  repeated with the real demo scene.
- The webcam has no depth sensor, so YOLO depth supplies the depth and its
  scale is relative to the scene. Keep the camera and subject at a stable
  distance.
- MediaPipe pose runs on the CPU runtime. YOLO depth uses the GPU.
- Reuse warp is an image-space move of the displayed positions. It does not
  change stored 3D positions and a later refresh corrects drift.
- The GUI splat renderer uses `gsplat` (splat.js), a standalone WebGL2 Gaussian
  renderer. It is not part of a three.js scene. This is a recorded deviation
  from the earlier "Three.js Gaussian renderer" wording; `@mkkellogg/
  gaussian-splats-3d` was tested as the three.js-native alternative and also
  works.

## Code map

- `applications/live_pipeline/adapters.py` — metric depth conversion, depth-threshold mask, transport serialization.
- `applications/live_pipeline/pipeline.py` — one refresh: pose + depth -> mask -> native constructor.
- `applications/live_pipeline/window.py` — flow-only candidate window and refresh/reuse decision.
- `applications/live_pipeline/motion.py` — reuse flow preview for the browser.
- `applications/backend/pipeline_server.py` — HTTP API and built-GUI serving.
- `applications/gui/` — Vite + TypeScript GUI with the gsplat renderer.
- `scripts/run_integration_app.py` — application entry point.
