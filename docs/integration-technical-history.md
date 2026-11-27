# Integration technical history

This is a record of implementation decisions, tests, and repairs. It preserves
measurements with their original test scope. For commands to run the current
web application, use the [runbook](integration_runbook.md). For the current
data path, use the [representation guide](splat-representation-and-runtime.md).

This folder is the final implementation of the live Gaussian-splat pipeline. It
combines the frozen teammate modules into one application on one computer.

## Pipeline

```text
webcam / OBS RGB frame
  -> 100 ms candidate window
  -> optical flow for each candidate against the retained refresh reference
  -> highest normalized flow score wins the window
       refresh -> depth + MediaPipe pose + pose-gradient person mask
                  (depth-threshold mask on fallback)
                -> native Splat Constructor -> newest active splat state -> browser
       reuse   -> screen-space flow warp of the displayed splats -> browser
```

Webcam and OBS use YOLO metric depth. The ZED source uses SDK depth by default.
The experimental ZED stereo matcher uses the left and right images when
`--stereo-depth` is set. The retained reference advances only after a refresh
produces a valid state.
A refresh without person pose or without a separable person mask keeps the last
valid state.

## Run

The supported setup and launch commands are in the [runbook](integration_runbook.md):
`setup_system.ps1` and `start_live_demo.ps1`. The commands below record the
earlier manual path.

```powershell
# One-time environment (CUDA torch, mediapipe, ultralytics, one OpenCV wheel)
./scripts/setup_integration_env.ps1

# Build the native constructor
./scripts/build_splat_constructor.ps1

# Build the GUI once
cd applications/gui; npm ci; npm run build; cd ../..

# Run the application (webcam is the default; ZED, OBS, or video also work)
.\.venv\Scripts\python.exe scripts\run_integration_app.py --source webcam --device 0
```

Open `http://127.0.0.1:8765/`. The server serves the built GUI and the API.

Options are in `--help`; the important ones are `--threshold` (fixed normalized
flow threshold), `--window-ms`, `--active-age-ms`, `--depth-band-m`,
`--max-mask-area`, `--sample-stride`, `--allow-missing-pose`.

### Capture source: webcam or ZED 2i

`--source webcam` is the default. Both cameras remain available.

| `--source` | Behaviour |
| --- | --- |
| `webcam` | The default. Webcam by `--device`. The image size is the device default (640 x 480 on this machine). Depth is estimated by YOLO. |
| `auto` | Uses the webcam. Selects `zed` only when the ZED SDK imports and a ZED camera answers the device list. |
| `zed` | A StereoLabs ZED camera on USB. The ZED SDK supplies the depth map and the calibrated intrinsics. |
| `obs` / `video` | Unchanged. |

Check the ZED before a run:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_integration_app --list-zed
.\start_live_demo.ps1 -ListZed
```

Measured on this computer with a ZED 2i on USB, SDK 5.5.0, `--zed-resolution HD720`:

| Quantity | Value |
| --- | --- |
| Capture | 1280 x 720 at 30 fps, measured 29.2 fps |
| Intrinsics | fx = fy = 522.57, cx = 640.42, cy = 346.93 (from the SDK calibration) |
| Depth | The SDK depth map, converted to metres with `NaN` for every invalid pixel |
| Splats per refresh, p50 | 44,384 |
| Refresh latency p50 | 322 ms |

The 322 ms ZED refresh and 84 ms webcam refresh came from different camera
runs at 1280 x 720 and 640 x 480, respectively. They do not isolate the
effect of the camera or image size. Use `--sample-stride` to trade splat count
for construction time.

**Depth mode.** `--zed-depth-mode QUALITY` is the default and opens in about 2 s.
The `NEURAL` and `NEURAL_PLUS` modes build a TensorRT engine on first use. On
this computer that build showed an estimate of about 33 minutes before the first
frame. Use them only when you can wait for the build once.

**SDK install.** The `pyzed` Python package must match the interpreter. For
Python 3.13 and SDK 5.5:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps `
  https://download.stereolabs.com/zedsdk/5.5/whl/win_amd64/pyzed-5.5-cp313-cp313-win_amd64.whl
```

The source adds the SDK `bin` directory with `os.add_dll_directory`, so the
native library loads without a change to `PATH`.

## API

| Endpoint | Content |
| --- | --- |
| `GET /api/preview` | latest captured RGB frame (JPEG, identity preserved) |
| `GET /api/pose-depth` | depth map + skeleton JPEG and joint values for the latest refresh |
| `GET /api/mask` | selected person mask (PNG); diagnostics name its source and fallback reason |
| `GET /api/splat` | splat state: uint32 header length, JSON header, binary arrays |
| `GET /api/motion` | reuse flow preview and pose joints for the browser warp |
| `GET /api/diagnostics` | action, score, timings, cache age, counters, queue state |
| `GET /api/health` | stream health |

The `/api/splat` array order is `positions, sh_dc, opacity_logits, log_scales,
rotations, reconstruction_weights, source_pixels`; encodings are `wxyz`
rotations, natural-log scales, and logit opacity, all in the constructor
coordinate convention (right-handed, `+Y` up, camera forward `-Z`).
The [representation guide](splat-representation-and-runtime.md) explains the
constructor values, browser conversion, and processor split.

## Earlier measured results on this computer

The following tests were made before pose-gradient mask selection became the
live default. The still-image refresh and live webcam run have different
scopes. Do not add them or use them as one end-to-end result.

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

- The pose-gradient mask can fall back to a depth-threshold band. The band
  can select objects at the same distance as the subject. Check the mask
  source in diagnostics before judging a run.
- The webcam has no depth sensor, so YOLO depth supplies the depth and its
  scale is relative to the scene. Keep the camera and subject at a stable
  distance.
- MediaPipe pose runs on the CPU runtime. YOLO depth uses the GPU.
- Reuse warp is a **source-view** image-space move of the displayed positions.
  It is exact when the viewer camera looks along the capture view and gains
  parallax error as the orbit angle grows; only a refresh corrects that. It
  does not change stored 3D positions.
- The active-state cache holds one splat state. The browser displays the newest
  published state only; earlier states are not merged or interpolated.
- Reuse motion is applied only when its (source_id, generation,
  reference_frame_id) equals the displayed splat identity and its candidate
  frame is later than that reference. The backend rejects motion that does not
  match the published state.
- Hold and fade follow the **invalid-person** event, not the state age. The
  server latches the first invalid-person refresh and reports
  person_loss_age_ms; the browser runs the 500 ms hold and 500 ms fade on its
  own clock. A slow but valid refresh does not start a fade.
- The GUI splat renderer uses `gsplat` (splat.js), a standalone WebGL2 Gaussian
  renderer. It is not part of a three.js scene. This is a recorded deviation
  from the earlier "Three.js Gaussian renderer" wording; `@mkkellogg/
  gaussian-splats-3d` was tested as the three.js-native alternative and also
  works.

## Active-state machine (Phase 1 finalized)

- **Capture run generations.** A run generation starts at capture start and
  increases on a source reopen, a resolution change, a changed timestamp
  origin, or a backward capture time. A reconnect is detected from
  `StreamHandler.metrics.reconnect_count`, because a live OpenCV source has no
  source timestamp. A generation change clears the candidate window, the
  retained flow reference, the active state identity, and any motion.
- **Immutable job generation.** Each refresh job carries the generation it was
  created in. The worker rejects a job whose generation is no longer current
  before it publishes, and `WindowSelector.commit(packet, generation)` rejects a
  stale commit, so a slow refresh can never publish or become the reference of
  a newer run.
- **Consistent snapshot.** One publish operation writes the splat header and
  buffer, the depth and skeleton image, the mask, and the pose joints under one
  lock. A failed pose preview is published as `None`, so a poll cannot mix two
  refreshes or show an older depth image. `/api/mask` carries
  `X-Mask-Source-Id`, `X-Mask-Generation`, and `X-Mask-Frame-Id`.
- **Complete identity in the browser.** Every fetch key is
  `(source_id, generation, frame_id)`, not the frame id alone. If a fetch is in
  flight, the newest pending identity is fetched when it ends; a failure retries
  the same identity after 500 ms.
- **One display clock.** The server latches the first invalid-person event and
  reports `person_loss_age_ms`; it never forces alpha to zero. The browser
  anchors the age on `performance.now()`, holds 500 ms, fades over 500 ms, and
  releases the clock only after a state that is new for the loss generation is
  installed. A slow but valid refresh does not start a fade.
- **Motion counters.** `motion_published` and `motion_rejected` separate motion
  the backend accepted from motion it rejected. A reuse decision is not counted
  as a displayed warp.

- **Mask identity.** `/api/mask` returns the mask and the active
  state identity from one lock hold, so a snapshot cannot be published between
  the two values. The headers `X-Mask-Source-Id`, `X-Mask-Generation`, and
  `X-Mask-Frame-Id` describe that same read. The browser compares the header
  identity with the state it requested and retries a stale mask up to three
  times before accepting it with a warning.
- **Guarded publication.** One `RunGuard` lock owns the generation
  and authorizes publication or failure recording. A run change cannot occur
  between the generation check and the state change. A stale refresh failure
  never latches person loss in a later run.
- **Queue separation.** On a run change the app drains queued packets
  and drops the packet already in its hand, because it may predate the change.
  Counters `packets_discarded_run` and `refresh_discarded_stale` report this.

Measured on the final Phase 1 build: person -> empty scene latches the loss and
the viewer fades to 0.00 alpha after the hold; the following person segment
publishes a new state, clears the latch, and restores 1.00 alpha. In a 22 s
recorded run: 66 refreshes, 0 failures, 47 accepted motions, 0 rejected.

- **Mask key.** The mask request key is
  `(source_id, generation, active_identity.frame_id)`, never the refresh
  counter. The displayed mask records the identity from the response, and the
  retry timer waits the remaining deadline, so the approved 300 ms stale rule
  and the 500 ms failure rule both apply.
- **Packet baseline.** `PacketRunTracker` detects source, size,
  timestamp-origin, and backward-time changes. `advance_generation` resets the
  tracker, and the next packet becomes the new baseline; one permanent change
  cannot cause a run change on every following packet.
- **Run boundary clears the display.** `advance_generation` calls
  `PipelineState.clear_active()` and clears the local decision, outcome header,
  and error. The browser also clears its displayed splat, mask, depth, motion,
  and loss clock when diagnostics report no active state. A new run starts
  empty and repopulates only through a new refresh.
- **Boundary diagnostic.** `scripts/check_run_boundary.py`
  verifies a run boundary against a running app: the first new-generation
  diagnostics sample must have no active identity, no error, and zero new-run
  refreshes; `/api/splat` and `/api/mask` must return 204; the first published
  state must belong to the new generation. Run the app with
  `--force-run-change-after-s N` to exercise the boundary without a camera
  reconnect.
- **Response invalidation.** The viewer keeps a display epoch. A
  boundary clear increments it and clears pending keys and retry timers. Any
  splat, mask, or pose-depth response, or a whole diagnostics poll, that started
  before the clear is discarded, so an old response cannot restore old-run data.
- **Boundary diagnostics.** `clear_active()` also clears the stored
  diagnostics. `advance_generation()` publishes a boundary diagnostics snapshot
  immediately, so `/api/diagnostics` cannot serve the previous run's action,
  splat, or error.
- **Delayed responses across a clear.** The viewer keeps a display
  epoch. A clear runs when the display has data, a retry is pending, or a splat,
  mask, or pose-depth request is in flight. A response that started before the
  clear is discarded, including a pose-depth decode failure that would schedule
  a retry. A diagnostics poll that spans a clear is discarded as a whole, and a
  poll whose diagnostics have no active state skips its motion and pose-depth
  results. `src/api/displayGuard.ts` holds these decisions and
  `npm run check:identity` exercises the delayed-response cases.
- **Depth holds to the splat snapshot.** Depth is displayed only
  when its `splat_frame_id` equals the active diagnostics frame. A newer image
  is held in `poseDepthAhead` and shown when the active frame catches up; an
  older image, or one from another source or generation, is discarded. A
  displayed depth image is cleared when diagnostics move to a new active frame,
  and the identity is checked again when the JPEG decode completes.
- **Fetch epoch runner.** `src/api/fetchRunner.ts` runs one request
  with an epoch guard: a response or a failure that started before a clear is
  discarded and never applied or recorded. The splat, mask, and pose-depth
  paths use it.
- **Newest pending identity.** A failed request keeps a newer pending
  identity instead of replacing it (`pendingAfterFailure`); the `finally` drain
  then starts the newest request.
- **Delayed-response check.** `tools/check-display-paths.mjs` runs the
  real async runner with controlled delays and checks: an old-run clear discards
  a response and a failure, a same-run frame advance makes a decoded image old,
  a newer image is held until the splat catches up, mixed API order never pairs
  an old image with a new state, and a failure keeps the newest pending
  identity. `npm run check:identity` runs it after the pure checks.

- **Queued request drain.** Every splat, mask, and pose-depth
  completion drains the newest queued request in its `finally` block, including
  a response discarded for an old epoch. A new-run request queued behind an
  old in-flight request therefore still starts. The decisions are pure
  (`shouldDrainSplat`, `shouldDrainMask`, `shouldDrainPoseDepth`).
- **Auxiliary identity.** Motion and pose-depth are separate
  responses from diagnostics. Motion applies only when its source, generation,
  and `splat_frame_id` equal the active identity (`auxMatchesActive`). Pose-depth
  applies when it is in the same run and not older than the active frame
  (`auxIsCurrentOrNewer`). An old auxiliary response is cleared, not paired with
  the new splat.
- **Identity check failure gate.** `npm run check:identity` runs every
  check before it decides the exit code. `CHECK_IDENTITY_SELF_TEST=1` injects a
  deliberate failure and proves the command exits non-zero.

- **Screenshots use Firefox.** `applications/gui/tools/capture-screenshot.ps1`
  now calls `capture-firefox.py`, which runs headless Firefox through Selenium
  with an isolated temporary profile. The earlier Edge/DevTools helper stopped
  every running Microsoft Edge process, including the user's own browser, and
  is removed.
- **Mask display versus request.** The viewer stores the response
  identity that produced the displayed mask (`maskDisplayedKey`) separately
  from the requested identity it settled (`maskSatisfiedKey`).
- **Mask satisfaction.** The viewer records the requested identity as
  satisfied separately from the displayed identity. If the server keeps
  returning an older mask after three retries, the viewer accepts it once,
  warns, and does not request it again until the active identity changes.
- **Per-run counters.** Diagnostics report `RunCounters` for the
  current generation, reset on a run change. A stale success or failure cannot
  change the current run's `refresh_count` or `refresh_failures`.

## Phase 2 telemetry (2026-11-18)

- **Latency is the stage sum.** `pose_depth`, `mask`, `construct`, and
  `pose_preview` are measured, and `total` is their sum (`stage_total`), so the
  logged latency equals all processing time. The refresh log line carries
  `ms.total`, `ms.pose_preview`, `splats`, and `capture_age_ms`.
- **Capture age** is reported only for a live camera, because that source uses
  the same monotonic clock as the process. A recorded source reports `null`.
- **Grafana-style summary.** `--summary-json` now includes `latency_ms`,
  `latency_warm_ms` (excludes the model warm-up refresh), `capture_age_ms`, and
  `splat_count` summaries, plus a `cascade` block with `selected`, `submitted`,
  `started`, `replaced`, `published`, `failed`, and `reuse`.
- **Viewer latency.** The splat footer shows `Display: N ms`, computed as
  capture age at publish + server state age + browser parse and render time.
  For a recorded source this is publish-to-display latency.

## Phase 2 features (2026-11-18)

- **Depth-tab skeleton (RT Phase 2).** The Depth tab draws the pose skeleton over
  the depth image from the joints in the same pose-depth payload, using
  `src/api/skeleton.ts`. The overlay is drawn only for the displayed depth state,
  so it cannot show joints from another state. The footer shows the joint count.
- **Per-camera intrinsics.** `--camera-config <json>` loads pinhole intrinsics for
  one capture resolution. A different resolution falls back to the field-of-view
  estimate, and capture intrinsics still take precedence.

  ```json
  {"model": "pinhole", "width": 1280, "height": 720,
   "fx": 900.0, "fy": 900.0, "cx": 640.0, "cy": 360.0}
  ```

  The `horizontal_fov_deg` variant is accepted instead of `fx`/`fy`/`cx`/`cy`.
- **Setup checks.** The canonical setup script is `scripts/setup_system.ps1`.
  Splat Constructor and fails unless both exit 0. `-SkipNative` skips that step.
  It builds and tests the native
  Splat Constructor, checks the Python and native bridge, builds the GUI, and runs
  the Python test suite under the virtual-environment interpreter. Windows
  PowerShell 5.1 drops inner double quotes from a native `python -c` argument, so
  multi-part Python checks live in a script file. `scripts/setup_device.ps1` is now a
  thin wrapper that forwards to it for existing commands.
- **Depth validity.** The mask validity plane is passed to the construction frame
  (`depth_validity=mask_info.validity`) and covered by
  `test_construction_frame_replaces_nan_and_keeps_validity`; invalid depth is
  replaced with zero validity, never a non-finite value.

## Telemetry corrections

- **Display latency is an estimate.** The footer shows `Display (est): N ms`,
  written only after the renderer has applied that state, so the value covers
  every browser phase from the response arriving to the completed render.
  Phases: capture age at publish (shared monotonic clock), server state age, and
  the browser phase. The browser clock is separate from the capture clock, so
  the value is labelled an estimate. `src/api/latency.ts` holds the rule.
- **Live cascade is run-scoped.** Every diagnostics `cascade` field uses its
  offset at the start of the current capture run (`scope: "run"`,
  `run_generation`), so all fields describe the same run. The final summary
  keeps process-lifetime totals.
- **Capture age reaches diagnostics.** `splat.capture_age_ms` is copied into the
  diagnostics payload, so the Inference panel can show it for a live camera and
  show nothing for a recorded source.
- **One percentile definition.** `telemetry.percentile` is nearest-rank:
  rank `ceil(p/100 * n)` clamped to `[1, n]`. For `[1, 2, 3, 4]` the p50 is 2.
  Latency and count summaries both use it, and an even sample count is covered
  by tests.

## Latency and run-boundary corrections

- **Latency publishes after the first rendered frame of the state.** The panel
  records the phases with the new state, marks the state as pending
  presentation in `setSplatFrame`, and writes the footer value only after the
  first completed render of that state. This covers the first load, a same-count
  descriptor update, and a splat-count change.
- **Measured phases.** The estimate sums capture age at publish, server state
  age, browser transfer and decode (request start to decoded response), and the
  render wait (decoded response to the first render that shows the state). The
  browser clock differs from the capture clock, so network and decode cannot be
  separated further with the available clocks and the value stays labelled
  `Display (est)`.
- **Run-scoped starts.** At a run change the app now discards a pending refresh
  job before it takes the run offsets (`RefreshDispatcher.discard_pending`).
  An old-run pending job can no longer start inside the new run and inflate its
  `started` cascade. `run_offsets()` reads the lifetime counts under the
  dispatcher lock.
- **Browser check.** `applications/gui/tools/check-latency-browser.py` drives
  the real panel in Firefox and asserts a non-placeholder first value, a
  same-count update, and a splat-count-change update.

## Latency readiness and interval corrections

- **Disjoint latency intervals.** The estimate adds capture age (capture to
  publish), server state age (publish to server response), and the browser render
  wait (decoded response to the first render that shows the state). The browser
  request interval is not added, because a request that starts between publish
  and server response overlaps the server state age and would count part of the
  timeline twice. Transport transit and the browser-to-server clock offset
  cannot be measured with these clocks, so the value stays `Display (est)`.
- **Label waits for drawn data.** The panel keeps the state pending presentation
  until its own splat data is attached (`SplatData.detached` false), the
  descriptor is applied (no dirty flags), the data length matches the new state,
  and the sort worker has returned one index per active splat. Only then does
  it publish the label. It records `data-latency-frame`,
  `data-latency-count`, and `data-latency-sorted-count` for that state.
- **Browser check proves identity.** `check-latency-browser.py` now reads those
  data attributes and fails when a label is published without a rendered splat
  identity. The panel debug identity can lag the footer's newest frame, which is
  expected: the footer follows the header, the label follows the rendered data.

The viewer also waits for `gsplat` to return sorted indices for the active splat
before it publishes first-pack latency. This replaces the fixed 400 ms readiness
wait. The check uses the active object, its vertex count, and the sorted-index
count; a slow first pack keeps the label pending.

## Phase A.1 pose-gated background calibration

The first Phase A comparison showed the background comparison alone was too
noisy: 79 percent of calibrated refreshes fell back because depth noise made the
candidate region too large. The fix restricts the comparison to a spatial prior
from the pose:

- `pose_region_mask` builds a dilated bounding region around joints with
  visibility at least 0.3. The region margin is 15 percent of each image axis.
- The raw nearer-than-background pixels are intersected with that region, then
  the existing area guard runs. Depth noise outside the person cannot inflate the
  candidate, and the depth-threshold band is still the fallback.
- The pipeline derives the mask joints from the same pose estimate as the
  refresh, so the prior always belongs to the selected frame.
- Refresh log lines now record `mask_source` and the rejected `bg_area`, so a
  calibration that is not being used is visible in the log instead of silent.

Measured on a live camera, person swaying 30 s:

| Run | Refreshes | Mask source | Splat stdev | Splat min–max |
|---|---|---|---|---|
| P3 uncalibrated | 116 | depth band | 7,712 | 13,199–42,091 |
| Phase A (no pose gate) | 233 | 79% fallback | 9,857 | 3,873–46,180 |
| Phase A.1 (pose gate) | 191 post-calibration | **191 calibrated, 0 fallback** | **3,510** | **11,128–28,979** |

The calibrated run lowers splat-count variation by about 54 percent against the
uncalibrated reference and removes the fallback storms. The mean count is lower
because the calibrated mask covers the person body instead of the thicker depth
band, which is the intended behaviour.

## Phase A empty-scene background calibration

- **Manual capture.** `POST /api/calibrate` with an optional `seconds` body asks
  the refresh worker to capture empty-scene depth frames for that long, then
  builds one median background baseline and freezes it. The page has a
  **Calibrate background** button and the Inference panel shows
  `Background capturing N frames`, `calibrated · bg X m`, or the failure reason.
- **Freeze by construction.** The calibrator only collects during an explicit
  capture. It never learns while running, so a person cannot be absorbed into the
  baseline. A calibration frame that contains a detected person aborts the
  capture with `person detected during calibration; clear the scene and retry`.
- **Mask selection.** With a usable baseline (matching image size, source id, and
  capture generation) a person pixel is at least 0.30 m nearer than the
  background at that pixel, within a 1.5 m span limit and the existing area
  guard. If the comparison fails, the depth-threshold band runs as a fallback and
  the reason is recorded. A size, source, or generation mismatch rejects the
  baseline. Without a baseline the depth-threshold band runs exactly as before.
- **Identity.** `advance_generation` clears the baseline, so a new source or
  capture run cannot reuse the previous scene.
- **Diagnostics.** `/api/diagnostics` reports `calibration`: active, status,
  frames, baseline statistics (median, p05, p95, valid fraction), last error, and
  event count.
- **Tests.** 16 calibration and 8 mask-selection boundary tests cover empty
  capture, shape and size changes, low valid depth, implausible depth spread,
  person nearer and farther than the background, pixels without a baseline, the
  span limit, both area-guard sides, fallback reasons, and every identity
  mismatch.

## Pose-seeded depth-gradient mask

The mean-depth threshold is no longer the primary person mask. The mask is built
from the depth gradient, seeded by the pose:

1. Sobel gradient magnitude of the depth plane (invalid depth median-filled).
2. Watershed on that gradient, with visible pose joints as foreground markers and
   the area outside a 12 percent pose box as background.
3. Region kept inside the pose box, closed, and contour-filled.
4. Background removal: pixels whose depth is outside the visible-joint depth range
   (p10-p90 plus 0.35 m) are dropped, which removes background that leaks through
   weak edges between fingers or behind the head.
5. Fingertip discs and head/hand pose hulls are re-added inside that depth range,
   so a weak edge at a fist or the hairline does not chop the person.
6. Construction validity is limited to the same joint-depth range, so
   background-depth pixels inside the silhouette produce no splats. Measured on
   a live frame, the face box contained background depth up to 2.26 m while the
   pose joints reported about 0.75 m; that mismatch warped the face.
7. If the gradient mask fails (no pose, flat depth, or area guard), the
   depth-threshold band runs and `fallback_reason` records why.

Template matching, mean-threshold calibration and background baselines are no
longer on the mask path; the empty-scene calibration modules remain for the
record and can be revisited.

## Code map

- `applications/live_pipeline/adapters.py` — metric depth conversion, depth-threshold mask, transport serialization.
- `applications/live_pipeline/pipeline.py` — one refresh: pose + depth -> mask -> native constructor.
- `applications/live_pipeline/window.py` — flow-only candidate window and refresh/reuse decision.
- `applications/live_pipeline/motion.py` — reuse flow preview for the browser.
- `applications/backend/pipeline_server.py` — HTTP API and built-GUI serving.
- `applications/gui/` — Vite + TypeScript GUI with the gsplat renderer.
- `scripts/run_integration_app.py` — application entry point.
