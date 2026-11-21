# Person recording checklist for learning replay

This first run checks the replay data and the current person mask. It is not a
test of a trained model. Record one continuous, unmodified colour video from
the same camera view that the live splat pipeline uses. The replay command runs
after recording. You do not need to operate the viewer during the recording.

## Set up

- [ ] Fix the camera in place. Keep its position, zoom, focus, and resolution
  constant for the complete recording.
- [ ] Frame the head, torso, both hands, and as much of the legs as the room
  permits. Leave space above the head for raised hands and at both sides for
  arm movement. Do not cut off the face or hands during the test gestures.
- [ ] Use steady room light. Wear clothing that is distinct from the wall.
- [ ] Record the raw camera image without the splat, mask, pose, or diagnostic
  overlay. Note the camera source, resolution, frame rate, and any OBS settings.
- [ ] Give this recording a new session ID. Use `train` for this first pilot.
  Reserve a different recording session for the untouched `test` split.

## One 90-second pilot recording

Keep your body in view except during the exit-and-return step. Move at a normal
speed unless a step says to move quickly. Say the step number aloud or note its
time so that we can find the same frames later.

| Time | Action | What it checks |
| --- | --- | --- |
| 0-15 s | Face the camera. Stand still with hands at your sides. | Initial refresh, timed refresh, and reuse during stillness. |
| 15-30 s | Sway slowly left and right. Keep facing the camera. | Flow selection and mask position during body movement. |
| 30-45 s | Raise both arms to shoulder height, then above your head. Spread your fingers. Hold each pose for about 3 s. | Hand, arm, and face-edge mask coverage. |
| 45-55 s | Move one hand in front of your chest, then near your face. Keep the other hand visible. | Partial occlusion and false background pixels near the face. |
| 55-65 s | Turn about 45 degrees to each side, then face front again. | Pose and mask stability when part of the body is less visible. |
| 65-75 s | Make two quick arm movements, then stand still. | Fast flow and recovery to a stable state. |
| 75-85 s | Step fully out of frame for about 5 s, then return to the same place. | Missing-person records and a new valid refresh after return. |
| 85-90 s | Stand still with both hands visible. | Final reference and reuse pair. |

If 90 seconds is inconvenient, keep the same order and shorten each step. A
clip with only stillness or only raised hands will miss important cases.

## Run the replay command

From `live-splat-pipeline/`, use the project Python environment. Replace the
clip path and session ID with your own values. Keep the original video in
`data/recordings/` or another local folder that Git ignores.

```powershell
& .\.venv\Scripts\python.exe -m scripts.build_learning_replay `
  --clip "data/recordings/person_pilot.mp4" `
  --session-id person_pilot_01 --clip-id movement_01 --split train
```

The command writes `manifest.json`, `rows.jsonl`, and frame files under
`data/processed/learning_replay/person_pilot_01/movement_01/`. It refuses to
overwrite an earlier extraction with the same session and clip IDs. It also
refuses to give one session two different splits.

Add `--staged-output` to write into a sibling staging directory and publish it
only after the extraction and the manifest succeed. A failed run then leaves only
a staging directory, and the session split stays reserved. This flag cannot
repair a partial extraction that a run without the flag left in place.

## Inspect the masks

Run these commands from `live-splat-pipeline/` after a successful extraction.
Both write their report to a separate folder next to the replay data, so the
replay data stays unchanged.

```powershell
& .\.venv\Scripts\python.exe -m scripts.inspect_replay_masks `
  --output-dir data/processed/learning_replay/person_pilot_01/movement_01

& .\.venv\Scripts\python.exe -m scripts.compare_mask_strides `
  --output-dir data/processed/learning_replay/person_pilot_01/movement_01
```

The first command writes `index.html`, `report.json`, and one overlay for each
usable row in `movement_01.mask_report/`. The second command writes the same
three kinds of file in `movement_01.stride_compare/`. Open the HTML pages in a
browser. The face and hand numbers are a diagnostic proxy: a joint-derived box
includes background, so review the paired images.

## Check masks by hand

A saved `baseline_mask` is model output. It is not a checked label. Three
commands turn a review set into checked masks and a scored comparison:

```powershell
# 1. Write the review set: low-agreement frames, a spread of high-agreement
#    frames, every no-pose fallback row, and missing-person samples.
& .\.venv\Scripts\python.exe -m scripts.learning_data review-set `
  --output-dir data/processed/learning_replay/person_pilot_01/movement_01

# 2. Draw the true person mask for each review frame. The window uses the
#    standard library and OpenCV only.
& .\.venv\Scripts\python.exe -m scripts.label_review_masks `
  --review-dir data/processed/learning_replay/person_pilot_01/movement_01.review `
  --labels-root data/processed/learning_replay/labels --dataset movement_01

# 3. Score the live mask and the stride variants against the checked masks.
& .\.venv\Scripts\python.exe -m scripts.learning_data evaluate `
  --output-dir data/processed/learning_replay/person_pilot_01/movement_01 `
  --labels-root data/processed/learning_replay/labels --dataset movement_01
```

In the label window: left drag paints, right drag erases, `d` clears the frame,
`[` and `]` change the brush, `g` saves, `s` skips, `p` goes back, `q` quits.
Press `d` twice on an empty frame to record "no person". The checked masks go to
`labels/movement_01/masks/` and never into the replay data.

The evaluation writes `movement_01.eval/index.html`. Person coverage is the part
of the checked person that a mask keeps. Background leakage is the mask area
outside the checked person, and the largest leakage component shows a solid
false region such as a door or a desk.

## Record the training set

```powershell
& .\.venv\Scripts\python.exe -m scripts.learning_data training-filter `
  --output-dir data/processed/learning_replay/person_pilot_01/movement_01 `
  --labels-root data/processed/learning_replay/labels --dataset movement_01
```

The command writes `movement_01.training_filter.json` with the eligible rows and
the count for each exclusion reason: a failed row, a row with no reference, a
no-pose fallback mask, a row without `mask_pose`, a row without selected depth,
and a row without a checked mask. A pair enters the first 15-channel set only
with a checked person-mask label.

## Check the output

- [ ] The command completes. The summary reports `refresh_ok` greater than
  zero. It should report some `reuse` rows during the still segments. Record
  all four counts: `windows`, `refresh_ok`, `refresh_failed`, and `reuse`.
- [ ] In `rows.jsonl`, successful rows have a `selected_rgb`,
  `selected_depth`, `selected_pose`, and `baseline_mask` file. Paired rows
  also have a reference RGB, reference depth, reference pose, and flow file.
  A bootstrap row has no reference. A missing-person row can have no depth or
  mask file; it must not become the next reference.
- [ ] Successful rows also have a `mask_pose` file. It holds the unrounded
  joints that mask selection used, plus the frame identity. A null `mask_pose`
  is an older row and its joint-region numbers are approximate.
- [ ] Look at at least one selected RGB image and its matching
  `baseline_mask` from stillness, raised hands, hand-near-face, side turn, and
  return. Check the same frame ID in both file names. Note missing face or
  hand pixels and background pixels inside the person region. Do not judge
  the mask only by splat count or mask area.
- [ ] Note each `mask_source` that appears. Record any
  `depth_threshold_fallback` and its `fallback_reason`; a fallback is evidence
  to inspect, not an automatic failure of the recording.
- [ ] Record any `person_missing` row while you are clearly in frame, with its
  selected frame ID and the step time. Missing-person rows while you are out
  of frame are expected.

The replay tool runs offline and in order. Its processing times do not measure
live capture-to-visible-splat delay. Check that delay in a separate live run.

## Send back

Send the session ID, clip name, resolution, frame rate, the four summary counts,
and notes for the five mask views above. Include a few paired RGB and mask
images if you want us to inspect them together. Keep the original recording
local unless you choose to share it. Do not commit the recording or extracted
frame files.
