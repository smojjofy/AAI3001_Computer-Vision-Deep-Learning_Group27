# Milestone learning result: person mask candidate

The project trained and tested a person-mask correction candidate. The live
application still uses the pose-gradient mask with a depth-band fallback.
The trained candidate is outside the published mask path. The proposed
motion-correction head and online weight updates were not trained or used.
The [research record](learning-research-record.md) keeps the input contract,
training method, commands, and interim results. The
[milestone reflection](Integration-Enhancements.md) relates this work to
the whole application.

## Baseline and candidate

The baseline uses MediaPipe pose and estimated depth to select the person
for each refresh. OpenCV DIS optical flow supplies the browser reuse warp.
The candidate learns a mask correction from aligned reference and selected
frames, depth, pose, flow, and the accepted baseline mask. A baseline mask is
a model output, not a checked label. Human-checked masks provided the
training and evaluation labels.

The research path keeps frame identity, capture generation, image size,
and active reference identity with each input. It drops late results from
an earlier run. A model failure must leave the baseline available. The
live implementation does not depend on the research checkpoint.

## Checked result

The candidate checkpoint was frozen before a separate 27.6-second test
clip was scored. The user checked 18 frames. At the source size of
1280 x 720, mean mask intersection over union (IoU) was 0.7997 for the
candidate and 0.7354 for the accepted baseline. The candidate won on 11
frames and lost on 7.

The 14 frames spread across time had a mean gain of 0.0272 and an even
seven-to-seven split. Four added difficult frames had a larger mean gain
and strongly affected the 18-frame mean. The time-spread gain was below
the planned 0.03 margin. The clip did not show a clear exit and return.
The test did not measure live capture-to-display delay. These limits
prevent promotion of the candidate to the live mask path on this evidence.

A warm, model-only CUDA check took 1.874 ms median and 2.548 ms at the
95th percentile for one 320 x 180 input. This excludes depth, pose,
transfer, and browser work. It is not an end-to-end latency result.

## Milestone boundary

The result supports a narrow claim: a trained mask can improve selected
difficult frames on this dataset. It does not establish better geometry,
reliable gain across a full live clip, or faster display. The fixed
baseline remains the demonstration path. The [research record](learning-research-record.md)
contains the unchanged checkpoint hash, detailed comparisons, and gates
for any later live trial.
