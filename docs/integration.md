# Milestone integration architecture

The milestone delivers one live Gaussian-splat web application on one
computer. A static camera supplies frames. The application selects refresh
frames, estimates human pose and depth, builds a Gaussian splat state, and
shows the newest valid state in a browser. The [runbook](integration_runbook.md)
contains the setup and launch commands. The [technical record](integration-technical-history.md)
and [integration evidence](integration-evidence.md) retain measurements and
repair details.

## Scope and pipeline

```text
webcam / OBS / ZED frame
  -> Stream Handler and 100 ms candidate window
  -> OpenCV DIS optical flow and flow-only reranking
       refresh -> depth + MediaPipe pose -> pose-gradient person mask
               -> native CPU Splat Constructor -> active state -> browser
       reuse   -> flow warp of displayed splats -> browser
```

The webcam is the default source and uses YOLO metric depth. The ZED source
can supply SDK depth. An optional stereo experiment estimates depth from the
left and right ZED images on a selected refresh. Optical flow and reranking
use the left image. The person mask is made after selection, so depth and
pose run only for a selected refresh. Background motion can win a candidate
window. A timed refresh also requests a new state when motion is low. A
request does not publish an invalid state.

The live application keeps one active splat state. A successful refresh
replaces it. A reuse decision changes only the browser display positions;
it does not merge splats across frames. The [original design record](single-camera-design-record.md)
contains a persistent Temporal Cache and Splat Adjustor design that is
outside this milestone path.

## Component boundaries

| Boundary | Owner | Preserved data |
| --- | --- | --- |
| Camera to candidate window | `modules/stream_handler/` and `applications/live_pipeline/window.py` | `source_id`, capture generation, frame ID, capture time, dimensions, RGB image |
| Candidate to decision | `modules/inference/optical_flow/` and `modules/reranker/` | retained reference identity, flow score, selected frame |
| Selected frame to refresh | `applications/live_pipeline/pipeline.py` | selected RGB frame, aligned depth, pose joints, mask, intrinsics |
| Refresh to Gaussian state | `modules/splat_constructor/` and its Python bridge | `ConstructionFrame` v1 to `SplatState` v1 |
| State to browser | `applications/backend/pipeline_server.py` | one published snapshot and active identity |
| Browser display | `applications/gui/` | `gsplat` state, previews, motion, and diagnostics |

The [construction contract](../shared/schemas/construction-frame.md),
[splat-state contract](../shared/schemas/splat-state.md), and
[representation guide](splat-representation-and-runtime.md) define the
arrays, units, and browser conversion. The constructor uses CPU C++20. The
web browser uses WebGL2 for the main `gsplat` view.

## Identity and publication

A frame ID is unique only inside one capture run. The application uses
`(source_id, generation, frame_id)` for a live state. A new capture, source
reopen, resolution change, timestamp-origin change, or backward capture
time advances the generation. The application then clears the candidate
window, retained flow reference, active state, motion, and pending refresh.
It rejects results from the earlier generation.

The server publishes the splat state, mask, and pose-depth preview as one
snapshot under one lock. The browser reads diagnostics for the active
identity and accepts auxiliary responses only when their identity matches.
This rule prevents a new mask or motion field from appearing with an older
splat. `/api/splat` sends a 32-bit header length, a JSON header, and packed
arrays. The [technical record](integration-technical-history.md#active-state-machine-phase-1-finalized)
retains the detailed boundary cases.

## Viewer decision

The main viewer uses `gsplat`, a standalone WebGL2 Gaussian renderer. It is
not a Three.js scene. It accepts the constructor arrays after one display
rotation and conversion of color, scale, and opacity. The optional
[environment compositor](../applications/gui/README.md#environment-compositor)
uses Three.js and an imported `.glb` scene. It keeps the old splat display
visible until the next splat buffer is sorted and ready. The
[representation guide](splat-representation-and-runtime.md) gives the exact
conversion.

## Measured result and limits

The integrated web application ran with a live camera and a person on this
computer. Tests covered the capture path, frame identity, refresh and reuse,
person loss, and browser display. The [technical record](integration-technical-history.md)
keeps hardware, input resolution, and the scope of each timing. The
[milestone reflection](Integration-Enhancements.md) relates those timings to
mask quality and the trained mask candidate.

The camera provides one view. The state is a depth-guided visualization,
not a complete scan of hidden geometry. An orbit view can show stretched or
missing surfaces. The trained mask candidate stays outside the live path.
The optional stereo matcher needs further live quality checks. The
two-device transport design and multi-camera reconstruction are outside
this milestone.
