# ZED stereo research record

This file preserves the experiment design, implementation steps, and
acceptance gates. The [milestone stereo summary](STEREO_DEPTH_EXPERIMENT_PLAN.md)
states the end-of-milestone result.

## Status and objective

This is an implemented experiment with open quality checks. **The ZED 2i is now connected and working** as a
capture source: `--source zed` in the integrated application, SDK 5.5.0, tested
at HD720 and 30 frames per second. The camera arrival is no longer a blocker.

The current webcam remains the default live source. The ZED path uses the SDK
QUALITY depth map on each capture. That is a **benchmark path, not the
left-right pair estimate** described below, so a ZED SDK runtime result must not
be used to accept or reject the StereoSGBM path. The
[runbook](integration_runbook.md) gives launch commands. The
[technical record](integration-technical-history.md) keeps the measured behaviour.

Use synchronized left and right ZED images to estimate depth only for a
reranker-selected refresh frame. The optical-flow processor and reranker use
the left image. Pose, person masking, and splat construction use the left image
and a depth map aligned to it. The right image supplies depth information; it
does not supply another set of splat colours or another viewer viewpoint.

The first experiment uses OpenCV StereoSGBM. A trained stereo model is a later
decision, based on the first experiment's results. This experiment does not
replace the separate [learning research record](learning-research-record.md).

## Capture and frame contract

**Progress: steps 1 to 4 are implemented and tested.** `FramePacket` carries an
optional rectified right view (`right_rgb`) and a frozen `StereoCalibration`
record, the ZED source emits the pair under `DEPTH_MODE.NONE`, and
`frame_packet.schema.json` is version 1.1. Step 5, adding `zed` to the
application source choices, was already present in the frozen demo.

1. Open the ZED SDK with `DEPTH_MODE.NONE`. Do not request an SDK depth map in
   the live path. Start at HD720 and 30 frames per second. From one successful
   `grab()`, retrieve rectified `LEFT` and `RIGHT` images and the image capture
   timestamp. Copy both images before the SDK can reuse its buffers.
2. Read the rectified left-camera intrinsics and stereo baseline from the SDK
   calibration for the active resolution. Convert the baseline to metres.
   Reject missing, non-finite, or non-positive calibration. Do not use an
   estimated field of view for stereo depth.
3. Extend the in-process `SourceFrame` and `FramePacket` with an optional right
   RGB image and an immutable stereo calibration record. Update the shared
   frame-packet schema for these optional fields. Keep existing one-image
   sources valid. Validate equal left and right dimensions, RGB8 format, and
   matching calibration resolution. One packet owns one left-right pair, one
   frame ID, one source ID, and one capture timestamp. Do not put the two views
   in separate queues.
4. Keep optical flow, reranking, and reuse on the left image. The selected
   packet carries its matching right image into the refresh worker. A source
   restart, resolution change, or backward timestamp starts a new capture run
   under the existing run-identity rule.
5. Add `zed` to the integrated application source choices. Keep the ZED SDK
   optional for webcam and OBS use. Check for `pyzed` at ZED startup and give a
   clear setup error when it is unavailable.

The ZED SDK provides rectified left and right views, and `DEPTH_MODE.NONE`
disables its depth computation. Its rectified calibration describes those
views. See [ZED image views](https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/camera-controls),
[depth modes](https://docs.stereolabs.com/docs/development/zed-sdk/modules/depth-sensing/depth-modes),
and [camera calibration](https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/camera-calibration).

## Selected-pair depth and refresh

**Software status:** The selected-pair matcher, whole-frame YOLO fallback,
person-region check, source-aware background calibration, and stereo decision
report are implemented. Live stereo remains experimental. The five-scene quality
comparison and the capture-to-display delay gate have no accepted result. The
ZED live application still has an unresolved native access violation.

1. Run StereoSGBM only for a selected refresh pair. Convert both rectified
   images to grayscale. Divide OpenCV's fixed-point disparity by 16. For each
   valid positive disparity, calculate camera-axis depth in metres as
   `left_fx * baseline_m / disparity_px`. Set invalid depth to `NaN` and its
   validity to zero. Keep the output at left-image coordinates. See the
   [OpenCV StereoSGBM reference](https://docs.opencv.org/3.4.9/d2/d85/classcv_1_1StereoSGBM.html).
2. First measure matching at native HD720. Also measure a half-resolution
   matching mode. Scale intrinsics and disparity with the images, then align
   the output depth and validity with the original left image. Select the
   native mode if it meets the quality and delay gates. Otherwise select the
   half-resolution mode only if it meets both gates. Do not activate either
   mode if neither passes.
3. Estimate pose once for the selected left image. Apply the current active
   person-mask policy to the selected depth map. If stereo depth is unusable
   or its person mask fails validation, estimate YOLO depth for that whole
   selected frame. Record the fallback reason. Do not combine stereo and YOLO
   depth pixels in this first experiment. Missing pose keeps the current
   invalid-person behavior; YOLO depth cannot repair missing pose.
   A manual background baseline must also match the depth source that made
   it. Never apply a YOLO background depth map to StereoSGBM depth, or the
   reverse. Reject a baseline for a frame from another depth source and use
   the depth-band mask for that frame. Keep the baseline for its own source.
   An intentional change of the active depth mode requires a new empty-scene
   calibration before the background mask can operate in that mode.
4. Pass metric depth, its validity, the left RGB image, left intrinsics, and
   the selected frame identity to the existing Splat Constructor. Label the
   published depth source `stereo_sgbm` or `yolo`. The current external-depth
   path labels every supplied depth map `zed`; correct that label before
   publishing stereo results.
5. Keep the current live output as the default. Run the stereo path against
   recorded pairs first. Permit live activation only after the gates below
   pass. Keep a runtime switch to return to YOLO depth.

## Verification and acceptance

Use a simulated SDK source and synthetic rectified pairs with known disparity.
Check left-right pairing, RGB order, image copies, frame identity, calibration
units, depth values, invalid pixels, whole-frame fallback, run changes, and
unchanged webcam behavior. These tests establish the software contract. They do
not establish ZED hardware performance.

The camera is now available, so these contract checks can also run against real
ZED frames. The measured real-device facts are: SDK 5.5.0, 1280 x 720 at
30 frames per second, left-camera intrinsics fx = fy = 522.57 and
cx = 640.42, cy = 346.93, and a raw depth plane that uses `inf` and `NaN` for
no-measurement pixels. The capture source converts millimetres to metres and
maps every non-finite, zero, or negative value to `NaN`.

Record ZED stereo footage with SDK depth disabled. Keep the
camera serial number, SDK version, resolution, frame rate, calibration, and
timestamps with each session. Replay identical selected pairs through the
StereoSGBM and YOLO paths. Generate ZED SDK depth from the recorded footage
**offline only** for comparison. The SDK can compute depth during replay even
if depth was disabled during recording. See [ZED recording behavior](https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/recording).

Compare five fixed scenes: a still person, side-to-side motion, a partial
turn, leaving and returning, and a low-texture scene. Keep the same source
frames and display settings for both paths. Accept stereo for live use only
when visual review prefers its splats in at least four scenes, its valid
refresh rate is no lower than YOLO's, and still-person mask-area and splat-count
variation are no worse. Run the initial comparison without a manual background
baseline on either path. Then compare each path again after its own empty-scene
calibration. Record the result for each scene, including failures.

The 95th-percentile capture-to-display delay must be 500 ms or less on this
computer. Record capture, matching, pose, mask, construction, and YOLO-fallback
times separately. The current display estimate does not include capture age
for a ZED source timestamp. Validate the SDK timestamp against the host clock
and make an independent live delay measurement before claiming this gate.

## Boundaries

This experiment uses one static ZED 2i on one computer. It adds stereo depth
for the left source view. It does not add two-view splat fusion, moving-camera
tracking, online training, or live ZED SDK depth. **The camera is present and
working.** The measured device facts are recorded under verification below, and
the capture contract is implemented; parameter calibration and the quality and
delay gates remain unverified until the recorded and live checks run.
