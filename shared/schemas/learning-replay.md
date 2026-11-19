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
  `selected_pose`, `reference_pose`, and `baseline_mask` are relative paths or
  null. RGB and mask files are lossless PNG. Depth files are NPZ with `depth_m`
  (float32 metres, NaN for invalid) and `validity` (float32, zero or one).
  Flow files are NPZ with `flow` (float32 [H,W,2] in pixels at flow resolution)
  and `valid_mask` (uint8 [H,W]); the direction is reference to selected.
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
