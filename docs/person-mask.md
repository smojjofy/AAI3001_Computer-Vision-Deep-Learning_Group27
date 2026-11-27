# The live person mask

The live application builds one binary person mask for each selected refresh
frame. The mask selects the pixels that the Splat Constructor turns into
Gaussians. A pixel inside the mask can become a splat. A pixel outside the mask
cannot.

The live method is the **pose-seeded depth-gradient mask**. A depth-threshold
band remains as a fallback. This document states the current behaviour. The
[milestone reflection](Integration-Enhancements.md) gives the measured quality
and its limits. The [enhancement evidence](enhancement-evidence.md) records the
rejected alternatives.

## Why the mask is not a learned model

A trained mask candidate exists and is measured, but it is outside the live
path. See the [learning result](DEEP_LEARNING_PLAN.md). The live mask uses only
the estimated depth plane and the MediaPipe pose joints, so it adds no model
weight, no checkpoint, and no inference cost to a refresh.

## Where it runs

The mask is made after the reranker selects a refresh frame, and after depth and
pose are estimated for that frame. It runs on the central processing unit (CPU).
A reuse decision makes no mask.

```text
selected refresh frame
  -> MediaPipe pose (33 joints) + depth plane (metres, NaN invalid)
  -> pose-seeded depth-gradient mask
       success -> mask_source "pose_gradient"
       failure -> depth-threshold band, mask_source "depth_threshold_fallback"
  -> Splat Constructor
```

The pipeline reports `mask_source` and, on a fallback, `fallback_reason` in its
refresh log line and in diagnostics. A run that falls back is visible, not
silent.

## The gradient mask, step by step

The implementation is `applications/live_pipeline/pose_gradient_mask.py`.

1. **Fill invalid depth.** Invalid depth samples are median-filled so that a
   `NaN` hole does not become a false edge.
2. **Compute the depth gradient.** A 3x3 Sobel gradient magnitude is computed on
   the filled depth plane. The person's silhouette is a depth discontinuity, so
   it appears as a ridge in this gradient.
3. **Check that a silhouette exists.** If the depth inside the pose box has too
   small a spread, the mask returns `flat_depth`. A flat plane would otherwise
   be split arbitrarily by the watershed.
4. **Run a watershed.** The gradient is inverted and labelled by watershed.
   Visible pose joints are foreground markers. The area outside a pose box, with
   a margin of 12 percent of each image axis, is the background marker.
5. **Keep the pose region.** The region inside the pose box is kept, closed with
   a 9x9 kernel, and its external contours are filled.
6. **Remove background depth.** A pixel is dropped when its depth falls outside
   the visible-joint depth range. That range is the 10th to 90th percentile of
   the visible joint depths, plus and minus a margin of 0.35 m. This step
   removes background that leaks through weak edges between fingers or behind
   the head.
7. **Re-add the pose extremities.** Fingertip discs and the head and hand hulls
   are re-added, but only inside that same depth range. This step prevents a
   weak edge at a fist or at the hairline from cutting the person.
8. **Apply the area guard.** A mask smaller than 0.005 of the image returns
   `too_small`. A mask larger than 0.60 returns `too_large`. Both are treated as
   a failure and fall back.

The gradient percentile is estimated at 97 percent. The live pipeline
subsamples the gradient image at a stride of 4 to estimate that percentile.

## Joint selection constants

These values are declared in the module:

| Constant | Value | Use |
| --- | --- | --- |
| `MIN_JOINT_VISIBILITY` | `0.15` | The visibility floor for a joint to count as a marker. |
| `HEAD_JOINTS` | joints `0` to `12` | The head hull re-add. |
| `LEFT_HAND_JOINTS` | `15, 17, 19, 21` | The left hand hull re-add. |
| `RIGHT_HAND_JOINTS` | `16, 18, 20, 22` | The right hand hull re-add. |

## The fallback

The depth-threshold band is the fallback. It selects the pixels within a fixed
distance band above the nearest valid depth. The band can select objects at the
same distance as the subject, so its result is weaker than the gradient mask.

The fallback runs when the gradient mask returns one of these reasons, which
also become the recorded `fallback_reason`:

| Reason | Cause |
| --- | --- |
| `no_pose` | The pose stage found no usable person. |
| `no_region` | The watershed found no separable region, or the region was empty. |
| `flat_depth` | The depth inside the pose box had too small a spread. |
| `too_small` | The mask covered less than 0.005 of the image. |
| `too_large` | The mask covered more than 0.60 of the image. |

A missing pose is an ordinary refresh outcome, not a fault. The milestone treats
a failed refresh as a reason to keep the last valid splat state.

## Construction validity

The mask decides which pixels may become splats. A second gate, the joint-depth
range from step 6, decides which of those pixels are constructible. The two
gates together prevent background-depth pixels that sit inside the silhouette
from becoming splats.

This gate exists because of a measured defect. On a live frame, the face box
contained background depth up to 2.26 m while the pose joints reported about
0.75 m. Those background-depth pixels became splats inside the face, and the
face appeared warped in the viewer. With construction limited to the joint-depth
range, those pixels produce no splats.

## Measured quality, and its limit

On a paired live frame, 21 of 23 visible joints were inside the selected mask,
with 0 failed refreshes in that session. The body mask was correct in the
viewer. The mask source was `pose_gradient` in that session.

That observation is not a pixel-checked accuracy score. The known limits are:

- Hands are partially covered at rest.
- The face remains imperfect.
- Quality is measured by diagnostic proxies on red-green-blue (RGB) overlays,
  not against hand-drawn masks.

The checked-mask comparison exists only for the trained candidate, not for the
live mask. See the [learning result](DEEP_LEARNING_PLAN.md) for those
measurements and their scope.

A stride-4 gradient subsample was compared against a full-image percentile on
689 recorded frames. The two masks agreed at a median union of 0.9984, but the
worst frame reached 0.567, where the subsampled gradient made the mask include a
door and part of a desk. The live path keeps stride 4 until checked-mask
evidence supports a change. Two masks that agree on most pixels can still differ
where it matters.

## Rejected alternatives

These are recorded so that a later reader does not repeat them. The
[enhancement evidence](enhancement-evidence.md) holds the full record.

- **A fixed depth band.** The band changed its body coverage as the person
  swayed. The splat count ranged from 13,199 to 42,091 in one 30-second run.
- **An empty-scene background baseline.** The calibrated mask selected room
  areas and removed body areas under live depth variation. Lower splat-count
  variation over the wrong region was not a quality gain.

## Related documents

- [Milestone reflection](Integration-Enhancements.md): the accepted quality
  claim and the latency result.
- [Milestone workflow](single-camera-workflow.md): the refresh and reuse path
  around the mask.
- [Depth component](../modules/inference/depth/README.md): the depth and pose
  estimates that the mask consumes.
- [Pose benchmark](../modules/inference/pose/README.md): the MediaPipe Lite and
  Full comparison.
