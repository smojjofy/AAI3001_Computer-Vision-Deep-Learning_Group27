# Live Stream Splat Reconstruction — Group 27

This milestone delivers a live, single-camera Gaussian splat visualization
on one computer. A webcam is the default source. OBS Virtual Camera and the
ZED 2i are also supported. The application selects refresh frames with
optical flow, estimates depth and human pose, constructs one splat state,
and displays it in a browser. Reuse moves displayed splats with optical
flow until a valid refresh replaces the state.

The project goal was a near-real-time 2.5D display: depth-guided relief
from one camera view.
The team did not set a numeric refresh-rate or latency acceptance limit.
The reported times describe measured runs; they are not pass/fail gates.

The source camera view is the strongest view. One camera cannot reveal
hidden surfaces, and an orbit view can show incomplete geometry. The
[research conclusion](docs/research.md) explains this limit.

## Run the web application

In PowerShell, from the repository root:

```powershell
& .\scripts\setup_system.ps1
& .\start_live_demo.ps1
```

The first setup needs internet access for packages and the MediaPipe Lite
model. The setup script creates `.venv`, installs the required packages,
builds the native constructor and web interface, and runs the checks. The launcher
opens the browser. The first webcam refresh can download YOLO depth weights
if they are absent. The [runbook](docs/integration_runbook.md) gives the
commands for OBS, ZED, and recorded video.

## Delivered workflow

```text
camera frame -> Stream Handler -> OpenCV DIS flow -> Reranker
  refresh -> selected-frame depth + MediaPipe pose -> person mask
          -> Splat Constructor -> active state -> browser
  reuse   -> view-space flow warp of displayed splats -> browser
```

The browser shows input, mask, depth and pose previews, diagnostics, and
Gaussian splats. Its main renderer is `gsplat`. An optional Three.js
compositor imports a `.glb` environment. The
[workflow](docs/single-camera-workflow.md) and
[integration architecture](docs/integration.md) describe the delivered
boundaries. The [representation guide](docs/splat-representation-and-runtime.md)
defines the constructor values and browser conversion.

The live mask uses a pose-seeded depth gradient with a depth-threshold
fallback. A trained mask candidate improved difficult checked frames but
did not meet the planned gain on time-spread frames, so it is outside the
live path. The ZED source can use SDK depth. An optional left-right stereo
matcher is implemented but lacks an accepted live quality result. See the
[milestone reflection](docs/Integration-Enhancements.md),
[learning result](docs/DEEP_LEARNING_PLAN.md), and
[stereo result](docs/STEREO_DEPTH_EXPERIMENT_PLAN.md).

## Team and component ownership

Five students contributed to the integrated application. The work packages
record responsibility for each component.

| Member | Work package |
| --- | --- |
| Jaison | Pipeline architecture, Splat Constructor, and integration |
| ZeKai | Preliminary research and ZED SDK exploration |
| Ziven | Stream Handler, Optical Flow processor, and Reranker |
| Jaziel | DepthAnything and YOLO depth processor |
| Brejesh | MediaPipe pose processor |

## Documentation

| Document | Purpose |
| --- | --- |
| [Runbook](docs/integration_runbook.md) | Setup and launch commands. |
| [Integration architecture](docs/integration.md), [milestone workflow](docs/single-camera-workflow.md) | Current components, data flow, and scope. |
| [Person mask](docs/person-mask.md) | How the live mask selects the subject, and its fallback. |
| [Splat representation](docs/splat-representation-and-runtime.md), [coordinate convention](shared/coordinate_system/README.md) | Constructor values, browser conversion, and coordinate rules. |
| [Milestone reflection](docs/Integration-Enhancements.md), [research conclusion](docs/research.md) | Quality, latency, method choice, and evidence limits. |
| [Optical-flow decision](docs/optical-flow-training.md), [device verification](docs/optical-flow-device-verification.md) | DIS default, RAFT comparison, and repeatable checks. |
| [Learning result](docs/DEEP_LEARNING_PLAN.md), [stereo result](docs/STEREO_DEPTH_EXPERIMENT_PLAN.md) | Experimental results and remaining validation limits. |
| [GUI guide](applications/gui/README.md), [pose benchmark](modules/inference/pose/README.md), [depth component](modules/inference/depth/README.md), [Splat Constructor](modules/splat_constructor/README.md) | Component behavior and measurements. |
| [Inference package](modules/inference/README.md) | The live inference stages, and the research baselines outside the live path. |
| [Frame packet](docs/interfaces/frame_packet.md), [flow result](docs/interfaces/flow_result.md), [reranker decision](docs/interfaces/reranker.md), [construction frame](shared/schemas/construction-frame.md), [splat state](shared/schemas/splat-state.md) | Data contracts. |
| [Physical test protocol](docs/physical_test_checklist.md), [reconstruction validation](modules/reconstruction_validation/README.md) | Repeatable physical and source-view checks. |
| [Evidence and design records](docs/milestone-records.md) | Earlier experiments, repair history, and original proposals. |
