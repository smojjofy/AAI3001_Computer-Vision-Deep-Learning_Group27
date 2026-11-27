# Milestone reflection: quality, latency, and learning

The live application receives a camera stream, selects refresh frames with
optical flow, estimates depth and human pose, constructs one active Gaussian
splat state, and shows it in a browser. The [runbook](integration_runbook.md)
starts the application. The [representation guide](splat-representation-and-runtime.md)
defines the constructor-to-browser data path. This document states the
end-of-milestone conclusions and the evidence limits. The
[experiment record](enhancement-evidence.md) keeps the detailed measurements,
failed attempts, and repair history.

## Person selection and splat quality

The live path uses a pose-seeded depth-gradient mask. It uses visible pose
joints to select the subject in the depth plane. It limits construction
validity to the joint depth range. A depth-threshold band remains the fallback
when the gradient mask fails. The selected mask source and fallback reason
appear in diagnostics.

This choice addressed two visible faults. A fixed depth band changed its body
coverage as the person swayed. Background-depth pixels inside the face area
also put splats too far from the face. The pose-gradient mask and joint-depth
validity gate reduced these errors in the observed source view. A live paired
frame put 21 of 23 visible joints inside the selected mask, with no failed
refresh in that session. That observation does not give a measured mask
accuracy for every pose or scene. Hands and face edges can still be wrong.

An empty-scene background baseline was tested and rejected for the webcam
path. Under live YOLO depth variation, the calibrated mask selected room
areas and removed body areas. Lower splat-count variation over that wrong
region was not a quality gain. The calibration code remains separate from
the accepted live mask path.

## Latency and display

The tested live refresh consists of CPU MediaPipe pose, GPU YOLO depth, mask
selection, CPU splat construction, transport, and browser rendering. The
measured 127-refresh webcam profile had a 180.8 ms median and 216.3 ms
95th-percentile warm refresh. These are refresh-stage times, not camera-to-
display latency. A busy-machine session had a 577 ms median and 2,177 ms
95th percentile on the same code. Hardware utilization alone did not identify
the delay.

The live path stopped computing a depth confidence plane that it did not
read. A paired, synchronized measurement saved 13.22 ms per refresh in pose
and depth processing. A sampled gradient percentile reduced mask work by
about 3.9 ms in a small test, but it changed some mask pixels. The sampled
rule therefore needs checked-mask evidence before a broad quality claim.

The browser keeps one active splat state. A reuse decision applies optical
flow to displayed positions. It does not merge geometry from earlier
refreshes. The browser uses `gsplat` for the main view. The optional
Three.js environment compositor can place the splat view with an imported
`.glb` scene. The single camera leaves unseen geometry and orbit views
incomplete. See the [GUI guide](../applications/gui/README.md) for the
viewer behavior.

## Learning result

The replay tool records selected frames, depth, pose, flow, accepted masks,
and frame identity. A saved accepted mask is a baseline output; it is not a
checked label. Human-checked masks supplied the later training, validation,
and untouched test comparisons. The [learning result](DEEP_LEARNING_PLAN.md)
defines the dataset and evaluation method.

The frozen trained mask candidate improved mean source-size intersection over
union (IoU) from 0.7354 to 0.7997 on 18 checked test frames. It won on 11
frames and lost on 7. Four difficult frames contributed much of the gain. On
the 14 time-spread frames, the mean gain was 0.0272, below the planned 0.03
margin, with seven wins and seven losses. The clip was short and did not
measure live latency or a clear exit-and-return case. The candidate remains
outside the live mask path. The motion head is not trained.

## ZED and stereo scope

The integrated application supports the ZED 2i as a capture source. Its
standard ZED path uses SDK depth. The [stereo result](STEREO_DEPTH_EXPERIMENT_PLAN.md)
adds an optional path that uses the synchronized left and right images to
estimate depth only for a selected refresh frame. That experiment does not
replace the webcam default. Its live quality checks remain open; SDK-depth
measurements do not validate the separate stereo matcher.

## Evidence and limits

| Claim | Evidence | Limit |
| --- | --- | --- |
| The integrated web application runs on this computer. | Live camera and browser tests; see the [technical record](integration-technical-history.md). | Frame rate and latency change with source, resolution, and system load. |
| The pose-gradient mask improved the observed subject view. | Paired live frame, joint coverage, and visual inspection. | The paired observation is not a pixel-checked accuracy score. |
| The trained candidate can improve difficult masks. | Untouched 18-frame checked test. | The time-spread mean gain did not reach the planned margin; live cost is not established. |
| The ZED path captures aligned images and depth. | ZED device test in the [stereo result](STEREO_DEPTH_EXPERIMENT_PLAN.md). | The optional stereo matcher still needs its live quality gate. |

No current result proves accurate geometry from an unseen camera viewpoint.
The project demonstrates a live single-camera splat visualization and records
the conditions under which its quality or latency changes.
