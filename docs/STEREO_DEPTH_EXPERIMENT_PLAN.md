# Milestone result: ZED stereo depth experiment

The ZED 2i works as a capture source for the integrated application. The
default webcam path remains available. The standard ZED path uses SDK depth.
An optional experiment uses synchronized left and right ZED images and
OpenCV StereoSGBM to estimate depth only for a reranker-selected refresh.
The [research record](stereo-research-record.md) preserves the detailed
capture contract, implementation steps, and acceptance tests. The
[runbook](integration_runbook.md) gives the ZED launch command.

## Data path

The Stream Handler attaches a rectified right image and stereo calibration
to the left frame. Optical flow and reranking use the left image. On a
selected refresh, StereoSGBM matches the pair. Positive disparity gives
camera-axis depth in metres from focal length and camera baseline.
Invalid disparity becomes invalid depth. Pose, mask selection, and splat
construction use the left image and depth aligned to it. A failed stereo
estimate can fall back to whole-frame YOLO depth. The published depth source
records which method supplied the selected frame.

The experiment keeps frame identity, capture time, intrinsics, depth units,
and source generation with the selected pair. It does not merge two splat
views. The right image supplies depth information; it does not make a
second color view.

## Verified scope

The optional frame-pair contract, selected-pair matcher, whole-frame YOLO
fallback, person-region check, source-aware background calibration, and
stereo decision report are implemented and tested. The device was tested
with SDK 5.5.0 at 1280 x 720 and 30 frames per second. Its left intrinsics
were `fx = fy = 522.57`, `cx = 640.42`, and `cy = 346.93` in that test.
These values describe that device run; they are not a general calibration.

The five-scene comparison against YOLO and the live capture-to-display
delay gate have no accepted result. The [research record](stereo-research-record.md)
also reports a native access violation during a live ZED stereo run.
Software contract tests do not prove a stable or higher-quality live
stereo splat. SDK-depth measurements do not accept or reject the separate
StereoSGBM method.

## Milestone boundary

Stereo depth is an implemented option with open live validation. The
working webcam and ZED SDK-depth paths remain the demonstrated sources.
The experiment does not add a moving camera, two-view splat fusion, or
online learning. Its [research record](stereo-research-record.md) contains
the fixed-scene quality and delay gates for later promotion.
