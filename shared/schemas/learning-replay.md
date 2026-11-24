# Learning replay record, version 1

One `rows.jsonl` line describes one frame selected by `WindowSelector` from a
recorded clip. The output directory is local processed data and is not committed.

- `session_id`, `clip_id`, and `split` identify a recording session. All clips
  in one session have one split (`train`, `validation`, or `test`).
- `source_id`, `generation`, `selected_frame_id`, `selected_timestamp_ns`, and
  `timestamp_origin` identify the selected RGB frame. `reference_frame_id` and
  `reference_timestamp_ns` identify the retained successful refresh, if any.
- `action`, `reason`, `flow_score`, and `candidates` record the window decision.
  `status` is `ok`, `person_missing`, or `error`. A failed refresh does not
  advance the retained reference.
- `selected_rgb`, `reference_rgb`, `flow`, `selected_depth`, `reference_depth`,
  `selected_pose`, `reference_pose`, `mask_pose`, and `baseline_mask` are
  relative paths or null. RGB and mask files are lossless PNG. Depth files are
  NPZ with `depth_m` (float32 metres, NaN for invalid) and `validity` (float32,
  zero or one). Flow files are NPZ with `flow` (float32 [H,W,2] in pixels at flow
  resolution) and `valid_mask` (uint8 [H,W]); the direction is reference to
  selected.
- `selected_pose` holds the display joints. The preview encoder rounds x and y
  to four decimals and visibility to two, so it is not exact for measurement.
  `mask_pose` is a separate JSON file with the unrounded joints and joint depths
  that mask selection received for this frame: `schema_version`, `frame_id`,
  `timestamp_ns`, `source_id`, and `joints` (one record for each landmark with
  `index`, `x`, `y`, `visibility`, and `depth_m`). It is null when mask selection
  received no joint record, and it is not written for a row whose refresh failed.
- `mask_source` is `pose_gradient` or `depth_threshold_fallback` when a mask
  exists. `fallback_reason`, `mask_area_fraction`, `splat_count`, and
  `timings_ms` record the accepted baseline result. A baseline mask is a model
  output. It is never a checked label.
- `models` and `resize` record estimator names, versions or model paths, the
  fixed 320 x 180 research size, and the rule to scale flow by each image axis.

`manifest.json` records the clip hash, window settings, model settings, and
timestamp rule. Presentation timestamps are used only when they strictly
increase for the complete clip. Otherwise the tool derives time from frame
index and the clip frame rate, with one timestamp origin for the complete clip.
Put manually checked masks and body-point matches in a separate
label set. Do not write them into `baseline_mask`.

## Paired research training input

The diagnostic trainer reads the `training-filter` eligible list. It checks
that the replay session has the `train` split. It uses the saved reference and
selected RGB, depth, exact mask-input pose, DIS flow, and selected baseline
mask. It reads the checked selected mask from the separate label store.
The trainer resizes all images to 320 x 180. The channel order is reference
RGB (3), selected RGB (3), reference depth (1), selected depth (1), reference
depth validity (1), selected depth validity (1), reference-to-selected DIS
flow (2), reference pose map (1), selected pose map (1), and selected baseline
mask (1). RGB and mask values are in [0, 1]. Valid finite depth in metres is
clipped to [0, 10] and divided by 10. An invalid depth is zero in the depth
channel and zero in its validity channel. Each flow component is divided by
the source image width or height. The pose map contains visible joint disks.

The model returns one mask-logit channel and two residual-flow channels.
The first diagnostic run has no motion target. It trains the mask head only,
and its checkpoint metadata marks the motion head as untrained. A checkpoint
from one session has training metrics only. It is not a validation or live
deployment result.

A validation or test replay uses the same 15-channel transform and its own
checked mask store. The fixed-checkpoint evaluator rejects a replay whose
session ID or clip hash matches the checkpoint's training data. It reports
mask intersection over union, person coverage, and background leakage for
the model and the saved baseline mask at 320 x 180 and at the source image
size. At source size, the evaluator upsamples model logits with bilinear
interpolation and uses the fixed zero-logit threshold. It compares against
the original checked mask and the original saved baseline mask. It does not
score motion.

The optional bounded mask correction uses the saved baseline mask in input
channel 15. It starts from that mask and permits a learned correction only
inside the mask or within a fixed pixel radius of it at 320 x 180. Outside
that region, the model output is background. Training augmentation applies
one spatial transform to both RGB frames, depths, validity maps, flow, pose
maps, baseline mask, and checked mask. A horizontal flip changes the sign of
horizontal flow. A resize scales both flow components. This transform is used
only on training samples; validation and test samples keep their recorded
geometry.

## Live shadow record, version 1

A live shadow run computes the candidate mask beside the live refresh and never
changes the published splat. It writes one JSON object for each poll to a JSONL
file. The run uses one depth frame, depth validity plane, pose joint record, and
RGB frame from the accepted refresh outcome, so the shadow pair matches the
frame that the live pipeline built.

- `schema_version`, `kind`, and `session_id` identify the record and the run.
- `generation`, `frame_id`, `source_id`, and `capture_timestamp_ns` identify the
  selected frame. `reference_frame_id` and `reference_timestamp_ns` identify the
  retained reference of the rolling window. The window is not an empty-scene
  calibration; it is the older end of a short frame window.
- `shadow_status` is `ok`, `skipped`, or `failed`. `skip_reason` names why a
  skipped frame produced no pair, for example `warming_up`. `error` and
  `error_type` hold a short text for a failed frame. A failed shadow poll never
  raises inside the live path.
- `timings_ms` holds `data`, `model`, and `total`. These values are the shadow
  cost. They are not live latency, because the shadow run is off the refresh
  path.
- `accepted` describes the live mask of the same frame: `source`,
  `fallback_reason`, `pixels`, and `area_fraction`.
- `candidate` describes the shadow mask: `mask_mode`, `threshold` (fixed 0.5),
  `source_upsample_rule` (bilinear logits, `align_corners=False`, threshold
  zero), `pixels`, `area_fraction`, and `baseline_restricted`.
- `compare` compares the candidate with the accepted mask: `intersection`,
  `union`, `iou`, `added_pixels`, `removed_pixels`, `removed_fraction_of_accepted`,
  and `added_outside_accepted_frac`.

The accepted mask in `accepted` is rebuilt from the saved depth and the exact
`mask_pose` joints. It is an input, never a checked label. A shadow record says
nothing about accuracy; only a checked mask gives accuracy.
