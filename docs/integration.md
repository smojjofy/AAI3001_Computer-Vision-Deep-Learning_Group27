# Integration history: how the six components became one live pipeline

This document records the integration phase. For each step it states what we
decided, what we measured, what failed, and what remained open. Dates are the
dates on which the work happened, and the round order is the order of review and
repair.

The companion document `docs/ENHANCEMENTS.md` continues from the point where this
document stops.

Where this document and `docs/integration.md` overlap, `integration.md` is the
operating guide and this document explains why each rule exists. This document
does not change any measurement that `integration.md` records.

## 1. Starting point

The project had five components. Each one was a separate single-branch clone of
the same GitHub repository (AAI3001_Computer-Vision-Deep-Learning_Group27), and
each one was developed by one student. They did not compose. No person
segmentation existed in any component, and no viewer existed.

The components were:

| Component | What it provides |
| --- | --- |
| Stream Handler | capture, frame identity, timestamps, backend selection |
| Optical Flow | DIS flow for each frame |
| Reranker | compares a frame with one retained reference |
| Pose Estimator | MediaPipe 33-joint landmarks |
| Depth Estimator | YOLO metric depth, NaN where invalid |
| Splat Constructor | C++20 Gaussian construction from RGB, depth, and mask |

A read-only review found that the components did not fit together. We recorded
five class-A blockers. Each blocker was a contract defect in frozen code, not a
missing feature.

1. **Depth direction inverted.** The Depth Estimator emits relative depth where a
   larger value means farther. The Splat Constructor defaults to
   `DepthSemantics::RelativeInverse`, where a larger value means nearer, and maps
   distance as `near + (1 - depth) * (far - near)`. No producer of inverse depth
   existed outside the constructor's own model. Without an adapter, every splat
   would be at the wrong distance.
2. **NaN invalid pixels.** The Depth Estimator marks an invalid pixel as `NaN`.
   Both constructor validators reject a non-finite depth value. The adapter had to
   write a finite placeholder and carry the invalidity in a separate
   `depth_validity` plane with values in `[0, 1]`.
3. **Frame identity type.** The constructor takes `frame_id` as a string. Every
   upstream stage carries an integer frame ID. No conversion convention existed.
   The constructor also rejects `timestamp_ns` equal to 0.
4. **Pose timestamp order.** `detect_for_video` requires strictly increasing
   millisecond timestamps. The Stream Handler records duplicate and backward
   source timestamps and still emits those frames.
5. **Missing stages.** No person-mask inference, no 100 ms candidate window, no
   forced refresh, no temporal cache, and no browser viewer existed.

The same review found secondary defects that each cost us time later:

- A stream end never closed its queue, so a blocked consumer never woke.
- Flow resolution metadata was status dependent. A 640 x 480 frame reported
  320 x 180 or 240 x 180 depending on the code path that wrote it.
- Relative depth was normalised per frame, so the depth scale drifted between
  refreshes while the subject stood still.
- The native library was loaded from disk on every frame.
- The reranker held a byte-identical fork of the optical-flow stream modules, so
  every flow repair had to be applied twice.

The working estimate to reach integration readiness was about five to six
engineering days, and person segmentation was the largest single gap.

## 2. Consolidation

The user approved a sixth component, `live-splat-pipeline/`, on a new branch
`integration`. It was created from `origin/main` and merged three branches:
`origin/brejesh/gui`, `origin/jaziel/merge-depth-pose`, and
`origin/jaison/splat-constructor`. The constructor work had never been merged
anywhere, and it was the only source of the Gaussian construction stage.

Three merge decisions matter for later work:

- **`pyzed.sl` at module level.** The stream source imported the ZED SDK at import
  time, so *every* source failed on a computer without the SDK. We made the
  import lazy and raised `SourceReadError` only for the ZED source.
- **Test collection.** A root `pytest.ini` scopes collection to real test
  suites, because `modules/inference/*/scripts` contain diagnostic `test_*.py`
  entry points that import optional SDKs at module level.
- **Model weights.** The merged branches already contain model weights, which
  violate the workspace rule on committed weights. We kept them because they were
  already present on those branches, and we added no new weights.

The consolidation build passed 65 tests with 7 skipped. The skips needed
`ultralytics`.

## 3. Binding scope decisions

We fixed the scope with the user before implementation started. These decisions
shaped everything that followed, so I record them with their reasons.

- **One computer, not two.** A two-device layout (capture on one machine,
  construction and viewer on another) was the earlier plan. The user set the
  current milestone to one computer and deferred the two-device layout. This
  removed all network transport from the acceptance gates.
- **Flow-only selection.** Optical flow runs for every candidate frame. The
  reranker selects one frame in each 100 ms window by flow magnitude alone. Depth
  runs only for the selected frame, and a depth threshold then makes the person
  mask. We withdrew the earlier three-term score (flow, RGB change, mask shape)
  because it needed a mask for every candidate, which needs depth for every
  candidate.
- **Background motion is accepted.** A window can be won by background motion
  rather than by the person. The user accepted this trade-off. The 500 ms timed
  refresh covers a scene with little motion. The timer *requests* a refresh and
  does not make an invalid result active.
- **One binary mask** feeds both the constructor and opacity.
- **The retained reference advances only after a successful construction.** A
  failed refresh must not authorise a later reuse decision.
- **The final demonstration is the browser application.** The user chose a
  Vite GUI with a Gaussian renderer, and this GUI is the deliverable.

## 4. The renderer decision and its recorded deviation

The plan required a test of one real splat state in a browser renderer before any
GUI work. We tested three candidates with a synthetic 5000-splat person cloud:

| Candidate | Result |
| --- | --- |
| `gsplat` 1.2.9 | Takes the constructor's raw arrays directly: `SplatData(count, positions, rotations, scales, colors)` with `wxyz` rotation order, linear-metre scales, and RGBA bytes. Updates in place at 5 Hz with zero skipped updates. The WebAssembly module is embedded, so Vite needs no configuration. |
| `@mkkellogg/gaussian-splats-3d` 0.4.7 | Renders inside a three.js scene from an in-memory `.splat` buffer, but it is JavaScript only and rebuilds the buffer on each update. |
| Custom instanced-quad shader | Renders after we fixed a spike defect, but it needs the most code and its own sort management. |

We chose `gsplat`, because it needed the least code and it mapped onto the
constructor convention with a single camera rotation of 180 degrees about X.

This decision carries a deviation that I state in every document that describes
the renderer: **`gsplat` is a standalone WebGL2 renderer and is not part of a
three.js scene.** The three.js-native alternative works and remains the fallback
if the requirement is read strictly. `docs/integration.md` records this deviation
in its known limits, and this document does not soften it.

## 5. The delivered pipeline

```text
webcam / OBS RGB frame
  -> 100 ms candidate window
  -> optical flow for each candidate against the retained refresh reference
  -> highest normalized flow score wins the window
       refresh -> YOLO metric depth + person mask + MediaPipe pose
                -> native Splat Constructor -> newest active splat state -> browser
       reuse   -> source-view flow warp of the displayed splats -> browser
```

The stages, with the module that owns each one:

| Stage | Module | Output |
| --- | --- | --- |
| Capture and identity | `modules/stream_handler/` | `FramePacket` with `source_id`, integer `frame_id`, `capture_timestamp_ns` |
| Candidate window | `applications/live_pipeline/window.py` | one selection decision each 100 ms |
| Optical flow | `modules/inference/optical_flow/` | normalised flow score for each candidate |
| Refresh: pose and depth | `modules/inference/pose/`, `modules/inference/depth/` | joints with visibility, metric depth in metres |
| Refresh: mask | `applications/live_pipeline/mask_selection.py` | binary person mask, `mask_source`, `fallback_reason` |
| Refresh: construction | `modules/splat_constructor/` | Gaussian state arrays |
| Refresh: preview | `modules/inference/depth/src/preview.py` | depth image and pose preview for the browser |
| Publish | `applications/backend/pipeline_server.py` | one consistent snapshot |
| Display | `applications/gui/` | video, mask, depth with skeleton, diagnostics, splat panel |
| Reuse motion | `applications/live_pipeline/motion.py` | 40 x 30 flow field for the browser warp |

The operator path is one command. `scripts/setup_system.ps1` is the canonical
setup script: it creates the virtual environment, installs PyTorch from the CUDA
wheel index, enforces one OpenCV wheel, fetches the model assets, builds and tests
the native bridge, builds the GUI, and runs the Python test suite under the
virtual-environment interpreter. `scripts/setup_device.ps1` is a thin wrapper
that forwards to it. `start_live_demo.ps1` in the component root starts the
application, waits for `/api/health` and the first published state, and opens the
viewer.

`start_live_demo.ps1` immediately exposed two shutdown defects:

- The application raised `TimeoutError: refresh worker did not stop within
  timeout` at exit. The cause was `RefreshDispatcher.close(drain=True)`, which
  waited for in-flight model work after a finite video source reached its end.
  Shutdown now closes without drain and prints a warning.
- When the browser closed a connection, `_send_json` printed a full
  `ConnectionAbortedError` traceback. The binary handler already ignored this
  exception; the JSON handler now ignores it too.

The launcher then exposed a defect in itself. Its first version joined the
application arguments into one command string, so a path that contains spaces
lost its quotes and Python received a split path (`can't open file
'D:\SiT\3001'`). It now passes each argument separately.

## 6. First delivery, then nine rounds of repair

The first delivery declared the integration phase complete with 90 passing tests.
An independent review found four defects that prevented the demonstration within
one day. This was the correct outcome of a review, and it set the pattern for the
rest of the phase: a delivered slice, an independent review, a repair commit with
a focused test, and a re-review.

I list the repair rounds as a table. Each row states the stimulus, the root cause,
and the test that now covers it. The rounds are in the order that they happened.
This table is the evidence that the integration was verified rather than assumed.

| Round | Defect | Root cause | Repair and covering test |
| --- | --- | --- | --- |
| 1 | A small mask region that touched the image corner became a full-frame mask. We reproduced the case: reported `area_fraction` 0.061875, actual mask area 1.0. | Hole filling flooded from pixel (0, 0). When the largest component touched that pixel, the flood inverted. The area guard measured the pre-fill area, so it could not see the result. | Pad by one background pixel, flood from the guaranteed background, and keep only the holes that the flood did not reach. Apply the area guard to the final mask. Regression test: a corner-touching region keeps its exact original size. |
| 1 | The browser recorded a new frame ID before the in-flight gate, so a new state could be skipped. | The fetch gate returned early after the ID had already been recorded as requested. | Keep a pending identity and start the fetch when the running one finishes. Same rule for the mask and pose-depth paths. |
| 1 | The splat alpha did not advance after the first response, so a lost person never faded. | Alpha was computed only inside the `/api/splat` response, and the browser fetches only on a new frame ID. | The server latches the loss event. The browser runs the hold and fade on its own clock. |
| 1 | Reuse motion was applied to a splat that it did not belong to. The saved screenshot shows splat frame 403 with motion reference 400. | Motion had no identity check. | Motion applies only when `(source_id, generation, reference_frame_id)` equals the displayed splat identity and the candidate frame is later than that reference. |
| 2 | My first repair of the corner-touching mask was also broken. `np.maximum(padded, 1 - padded)` is 1 at every pixel of a binary mask, so the repair would have turned every mask into the full frame. | The corrected algorithm mixed the original component into the flood result. | Keep the original component separate and add only the unreachable holes. The review found this before the commit, because the review ran the tests instead of reading the diff. |
| 2 | The proposed "camera-view warp with a pluggable camera model" is algebraically the same operation as the current source-view warp. | The proposed projection and the existing offset both reduce to `delta_X = flow_u * depth / fx` and `delta_Y = -flow_v * depth / fy`. | We recomputed both forms on a worked example: identical to 1e-9. The source-view warp stayed, and its orbit limit is documented. The user then confirmed this choice. |
| 3 | A refresh that started in one generation could publish in the next. We reproduced an old frame committed under generation 1 after generation 0 was reset. | The pipeline read a mutable generation value when the job finished. | Each job carries an immutable generation. The worker rejects a stale job before processing it and again before publishing. `WindowSelector.commit(packet, generation)` rejects a stale commit. |
| 3 | The candidate window survived a generation reset, so an old candidate could become the new bootstrap selection. | The reset cleared the reference but not the window. | A run change clears the window and its timing state. The flush drops candidates from another generation. |
| 3 | A camera reopen was not detected. | A live OpenCV source supplies no source timestamp, so the timestamp checks saw nothing. | Watch `StreamHandler.metrics.reconnect_count` in addition to source ID, dimensions, timestamp origin, and backward time. |
| 4 | A mask from frame 1 could be paired with the identity of frame 2. We reproduced it. | The mask and the active identity came from two separate lock holds. | One locked read returns the mask and its identity. `/api/mask` writes `X-Mask-Source-Id`, `X-Mask-Generation`, and `X-Mask-Frame-Id` from that same read. The mask key is the published frame ID and not `refresh_count`. |
| 4 | A stale job failure could start person loss in the new run. | The `RefreshError` path had no generation check. | The failure path runs through `RunGuard.record_failure_if_current`. |
| 5 | A run change could occur between the generation check and the state change that it authorised. | The check and the publish did not share one lock. | New `applications/live_pipeline/runstate.py`. One reentrant `RunGuard` lock owns the generation and authorises publication and failure recording. The first non-reentrant version deadlocked, and the twelve-thread test found it. |
| 5 | A queued packet from before a reconnect could be processed after it. | The queue survived a run change. | A run change drains the queue and drops the packet already in hand. Counters `packets_discarded_run` and `refresh_discarded_stale` report the discard. |
| 6 | The mask key still used `refresh_count`, which increments before publication, so an old mask could pass the identity comparison. | The earlier identity repair kept the original key. | The key is `(source_id, generation, active_identity.frame_id)`. The displayed mask records the identity from the response. |
| 6 | A permanent source or size change discarded every later frame. | The old packet identity survived the drop, so each new packet caused another run change. | New `PacketRunTracker`. `advance_generation` resets it, and the next accepted packet becomes the new baseline. |
| 6 | A stale job changed the new run's counters. | `failure_count` and `refresh_count` were updated outside the generation guard. | `RunCounters` resets on a run change and checks the generation on every update. |
| 6 | The stale-mask retry waited 500 ms although the rule is 300 ms. | The deadline and the timer used different values. | Each retry waits `max(0, deadline - now)`. |
| 7 | A run change kept the old splat, mask, and error on screen beside the new generation. | The boundary cleared the selector but not the published state. | `PipelineState.clear_active()` clears the snapshot, mask, depth, joints, motion, and the loss latch under one lock. The browser clears its display when diagnostics report no active state. |
| 7 | A mask that stayed stale caused repeated fetch cycles. | The viewer recorded the request key as satisfied and then re-requested it on the next poll. | The viewer records the requested identity as satisfied separately from the displayed identity. The documented identity command also gained `--ignoreConfig`, because the documented command failed with `TS5112`. |
| 8 | An in-flight response could restore old-run data after a boundary clear. | The clear did not invalidate requests that were already running. | A display epoch. A clear increments it, clears pending keys and timers, and every response that started before the clear is discarded. A whole poll that spans a clear is discarded. |
| 8 | Diagnostics stayed old at the boundary. | `clear_active()` cleared the snapshot but not the stored diagnostics. | `clear_active()` clears the stored diagnostics and `advance_generation()` publishes a boundary diagnostics snapshot immediately. `scripts/check_run_boundary.py` verifies this against a running application. |
| 9 | A clear that happened while only a request was in flight did not increment the epoch. | The clear was skipped when no data was displayed. | The clear condition includes the splat, mask, and pose-depth in-flight flags. |
| 9 | A poll could apply old motion and depth after the clear in the same poll. | Motion and depth results were handled after the clear inside the same poll. | A poll that spans a clear is discarded as a whole. A poll whose diagnostics have no active state skips its auxiliary results. A failed depth decode checks the epoch before it stores or retries. |
| 10 | A discarded old response stranded the queued new-run request. | The early return skipped the pending-request drain. | The drain runs in the `finally` block of every completion, including a discarded response. |
| 10 | A poll could pair old motion or depth with new diagnostics. | Motion and depth are separate responses from diagnostics. | `auxMatchesActive` for motion and `auxIsCurrentOrNewer` for depth. An old auxiliary response is cleared, not paired. |
| 10 | The identity check could not fail for checks added after the failure gate. | The failure gate and the success message were written before later checks. | The gate moved after every check. `CHECK_IDENTITY_SELF_TEST=1` injects a failure and proves that the command exits non-zero. |
| 11 | A depth image newer than the active splat was displayed beside an older splat. | Identity was compared as "newer than", not as "equal to". | Depth is displayed only when its `splat_frame_id` equals the active frame. A newer image is held in `poseDepthAhead` until the active frame catches up. |
| 11 | An old JPEG decode could update the depth panel after the active frame advanced. | The decode completion checked only the display epoch. | The decode completion checks source, generation, and splat frame again. |
| 11 | A failed request replaced a newer pending identity. | The failure path assigned the failed request to the pending slot. | `pendingAfterFailure` keeps the newer pending identity, and the drain starts the newest request. The reported 46 pure checks plus 20 delayed-response checks cover these paths. |
| 12 | The display latency value could appear before the new splat was drawn. | The label was computed from the response, and the panel recorded `receivedAt` after transfer and decode. | The panel publishes the value only after the renderer applies that state, and the label is `Display (est): N ms`. |
| 12 | Live cascade counters mixed capture-run totals with process totals. | Some cascade fields reset at a run change and others did not. | Every diagnostics cascade field uses its offset at the start of the current run, with `scope: "run"`. The final summary keeps process-lifetime totals. |
| 12 | The Inference panel could not show capture age. | `splat.capture_age_ms` was not copied into diagnostics. | The field is copied, and a recorded source correctly shows nothing. |
| 12 | The reported p50 was wrong for an even sample count. `summarize_ms([1,2,3,4])` returned 3.0. | `round(0.5 * (n - 1))` is not a nearest-rank percentile. | `percentile()` uses rank `ceil(p/100 * n)` clamped to `[1, n]`, so p50 of four samples is 2. Tests cover even counts, odd counts, p0, p100, and unsorted input. |
| 13 | The latency label was not updated for the first splat or for a changed splat count. | The publish condition required a dirty flag to change, which is false when both dirty flags were already false. | `setSplatFrame` marks the state as pending presentation, and `renderFrame` publishes after the first completed render of that state. |
| 13 | A stale pending job could enter the new run's `started` count. | The offset was read before the dispatcher dropped the pending job. | `RefreshDispatcher.discard_pending()` drops the queued job before the offsets are read. New `test_scheduler_run_boundary.py` reproduces the exact order. |
| 14 | The estimate counted part of the request twice. | `stateAge` and the browser request interval overlapped. | The estimate adds three disjoint intervals only: capture age, server state age, and browser render wait. The browser request interval is excluded on purpose. |
| 14 | The label could publish before the new splat data was drawn. | A same-count descriptor update can return with buffers detached, and the first pack for a new object is asynchronous. | `stateDataReady()` requires attached data, applied descriptors, a matching buffer length, and the renderer's sorted indices. The panel writes `data-latency-frame`, `data-latency-count`, and `data-latency-sorted-count` only for the drawn state, and the browser check reads them. |

Two defects in this phase came from me and were recorded in the same round as the
repair, because a repair record that hides its own errors is not evidence:

- The pose import path in `_import_pose_estimator` was repaired, and I first
  attributed two test failures to module caching. The real cause was that bare
  `python` resolved to the system interpreter without `mediapipe`. The lesson is
  in the workspace habit file (`_agent-habit.md`): compare `sys.prefix` with
  `sys.base_prefix` before diagnosing a missing module.
- The first `RunGuard` was not reentrant and deadlocked, because the guarded
  publish callback read the generation. The concurrency test found it before the
  commit.

The screenshot helper also changed in this phase. The earlier helper controlled
Microsoft Edge through the DevTools protocol and terminated every running Edge
process, including the user's own browser. It was replaced by
`applications/gui/tools/capture-firefox.py`, which runs headless Firefox through
Selenium with an isolated temporary profile and never touches the user's browser.

## 7. Phase 2: measurement and operator support

After the repair rounds closed, Phase 2 added measurement and operator support.
The user gave two instructions for this phase: use the habit file, and trace and
test the boundaries that each change affects instead of only the contract it was
written for.

- **Latency is the sum of all stages.** `pose_depth`, `mask`, `construct`, and
  `pose_preview` are measured, and `total` is their sum. The earlier total left
  `pose_preview` out. On a recorded run, 27 published lines showed 0 differences
  between `total` and the stage sum.
- **Capture age** is reported only for a live camera, because only a live camera
  shares the process monotonic clock. A recorded source reports `null`.
- **Cascade counters** report `selected`, `submitted`, `started`, `replaced`,
  `published`, `failed`, and `reuse`. A reuse decision is never counted as a
  displayed warp.
- **Depth-tab skeleton.** The Depth tab draws the pose skeleton over the depth
  image from the joints in the same pose-depth payload. Executable checks cover
  the 0.3 visibility floor, missing and non-finite coordinates, unconnected
  joints, and a full 33-joint pose.
- **Per-camera intrinsics.** `--camera-config <json>` loads pinhole intrinsics for
  one resolution. A different resolution falls back to the field-of-view estimate.
  Capture intrinsics take precedence.
- **`depth_validity` pass-through** was already present and is covered by an
  adapter test.
- **Setup checks.** A failed native build or test now fails the setup script.

Phase 2 then went through its own review and repair rounds before we accepted it,
and that is where the last rows of the table in section 6 come from.

## 8. Two contracts that everything later depends on

Every enhancement in the companion document relies on the two contracts below. I
state them separately because a reader who understands only these two contracts
can follow most of the code.

### Contract A: identity is `(source_id, generation, frame_id)`

A frame ID alone is not an identity. A frame ID is unique only inside one capture
run. A generation increases on each of these events:

- capture start;
- a source reopen, detected from `StreamHandler.metrics.reconnect_count`, because
  a live OpenCV source supplies no source timestamp;
- a resolution change;
- a changed timestamp origin;
- a backward capture time.

A generation change clears, in one operation, the candidate window, the retained
flow reference, the active state identity, any motion, and any pending refresh
job. It also drains the queue.

The reason for this strictness is that the browser keeps state between frames. A
splat, a mask, a depth image, and a flow field are four separate responses that the
browser receives at different times. If any one of them can outlive its run, the
display can show a person from one run beside a mask from another. Every repair in
section 6 that concerns identity comes from one instance of this problem.

### Contract B: one display clock

The server never tells the browser what alpha to show. The server latches the
first invalid-person refresh and reports `person_loss_age_ms`, an age, not an
instant. The browser anchors that age to `performance.now()` and runs the 500 ms
hold and the 500 ms fade in its own animation loop. It releases the clock only
when a state that is new for the loss generation is installed.

An absolute monotonic instant never crosses the boundary, because the Python
monotonic clock and the browser `performance.now()` clock do not share an epoch,
even on one computer. Only an age crosses.

A slow but valid refresh does not start a fade. Only an invalid-person result
starts it. Other inference failures are separate and never set the latch.

## 9. Verification: what each command proves, and what it does not

| Command | Proves | Does not prove |
| --- | --- | --- |
| `.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider` | Unit and boundary contracts for the stages, the state holders, the scheduler, the run guard, the telemetry, and the replay tools. | That the live constructor signature accepts the options that the live pipeline passes; a separate live test covers that (see below). |
| `applications/live_pipeline/tests/test_live_construction.py` | A real `RefreshPipeline` builds a real `PoseDepthPipeline` and processes one refresh. `depth_confidence` reaches both objects. | Mask quality, or that a camera is present. The test skips when the model files are absent. |
| `npx tsc --noEmit` and `npm run build` | The GUI type checks and builds. | That the page behaves correctly at runtime. |
| `npm run check:identity` | 46 pure decisions and 20 delayed-response paths for the identity and display rules. | That a browser rendered the result. `CHECK_IDENTITY_SELF_TEST=1` proves that the command can fail. |
| `applications/gui/tools/check-latency-browser.py` | The real panel publishes the latency label only for the state that was drawn. | That the value equals the true capture-to-photon delay. The clocks do not allow that measurement. |
| `scripts/check_run_boundary.py` | A run boundary clears the active identity, the diagnostics, and returns 204 from `/api/splat` and `/api/mask`. | That a real camera reopen was performed. The boundary is forced with `--force-run-change-after-s`. |
| `docs/physical_test_checklist.md` P1 to P5 | The behaviour with a real person in front of a real camera. | Nothing else. This is the only acceptance evidence for person behaviour. |

The physical tests are the only acceptance gate that a computer cannot satisfy
alone. We recorded no physical test while the user was away from the machine, and
we produced `docs/physical_test_checklist.md` instead. We must keep that
discipline, because a claim of person behaviour that nobody observed is the one
claim this project cannot defend.

At the end of the integration phase the reported suite result was between 145 and
152 tests depending on which optional packages were installed in the environment
that ran it. The difference is a collection difference and not a defect. Later
work raised the suite to 264 tests.

## 10. What remained open when this phase ended

- **Physical P1 to P5 with a real person.** Open at the end of integration, then
  run on 2026-11-19 (see the companion document, which continues from there).
- **Threshold and mask calibration from real footage.** Open.
- **The first-pack readiness risk.** We recorded a code-supported risk: the
  installed `gsplat` can defer the first pack until its WebAssembly module is
  ready, and the panel could then publish a latency label after a render call that
  had no new splat pixels. Later work replaced the fixed 400 ms wait with a
  requirement for the renderer's sorted indices.
- **ZED depth.** The ZED 2i and its SDK were unavailable for the whole phase.
  Every ZED path is code-supported and unverified against hardware, and the ZED
  stereo experiment remained a planned document rather than an implementation.
- **The three.js requirement.** `gsplat` is not part of a three.js scene. This is
  a recorded deviation with a working alternative.

## References

- `docs/integration.md`: the operating guide, the API, the state machine, and the
  measured results.
- `docs/physical_test_checklist.md`: P1 to P5, with commands and the evidence to
  return.
- `docs/ENHANCEMENTS.md`: the work that followed this phase.
