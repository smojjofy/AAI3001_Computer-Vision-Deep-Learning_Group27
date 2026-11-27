# Research conclusion for the single-camera milestone

The project uses depth-guided Gaussian splats to display a live subject
from one static camera. It does not train a conventional 3D Gaussian
Splatting (3DGS) scene from many posed views. It does not train a Neural
Radiance Field (NeRF). The result is a view-conditioned visualization:
the source camera view is the reliable view, and unseen surfaces remain
unknown. The [technical survey](research-background.md) explains NeRF,
multi-view 3DGS, camera geometry, and their source literature.

## Why this representation fits the milestone

A conventional NeRF or 3DGS reconstruction uses several camera views,
calibrated camera poses, and scene-specific optimization. One static live
camera cannot supply parallax around the person. Waiting for multi-view
optimization would also change the live-stream objective. The team instead
uses one selected RGB frame, estimated depth, and a person mask to create
one Gaussian per sampled usable pixel. The native constructor sets each
Gaussian's position, color, opacity, scale, and orientation. The browser
renders the newest state and can move its displayed positions with optical
flow between refreshes.

This choice makes the process visible. Face edges, hands, and missing back
surfaces can remain imperfect. A limited orbit can show how one-camera
depth projection differs from a full 3D scan. The project does not claim
an industry-ready virtual-production asset.

## Evidence from the implementation

| Research question | Milestone result |
| --- | --- |
| Can a single view make a live splat visualization? | The integrated application constructs and displays a Gaussian state from webcam or ZED input. See the [integration architecture](integration.md). |
| Can optical flow reduce reconstruction work? | Flow and the reranker choose refresh frames; reuse moves only the browser display. See the [workflow](single-camera-workflow.md). |
| Can predicted depth give accurate hidden geometry? | No such claim is supported. Depth sets a source-view relief; unseen surfaces have no measurement. |
| Can the viewer place a subject in a virtual scene? | The optional Three.js compositor can import a `.glb` environment and display published splat frames. See the [GUI guide](../applications/gui/README.md). |

## Limits of the evidence

The application has no multi-view ground truth for geometry or novel-view
image quality. The optical-flow clip has no ground-truth displacement.
The trained mask candidate was checked against human masks, but it remains
outside the live path. The experimental stereo matcher still needs live
quality validation. The [milestone reflection](Integration-Enhancements.md)
states the measured mask and latency results. The
[representation guide](splat-representation-and-runtime.md) gives the
exact Gaussian values used by the implementation.
