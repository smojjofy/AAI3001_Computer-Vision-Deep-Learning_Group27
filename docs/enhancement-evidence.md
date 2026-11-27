# Enhancement evidence and chronology

This document preserves the detailed experiments, measurements, and rejected
claims from the integration work. Its phase-specific status statements are
historical. The [milestone reflection](Integration-Enhancements.md) gives the
end-of-milestone conclusions.

This document continues [the integration evidence](integration-evidence.md). The integration phase produced a
working live application, a verified identity and display contract, and a prompt to
run the physical tests. This document records what was attempted after that point:
four lines of work, their measured results, and what each attempt proved or failed
to prove.

This document is a failure record as well as a result record, because three of the
four lines produced a rejected or withdrawn claim. A reader who wants only the
final state can read section 1 and section 7. A reader who wants to know why the
current mask is a depth-gradient watershed and why the poll interval is still
300 ms needs the whole document.

## Status after this record

This document records an earlier phase. Its open items and measurements state
what was known at that time. Checked masks and a trained mask candidate now
exist. The candidate passed an untouched 18-frame mask test, but it is not in
the live path. See the [learning research record](learning-research-record.md) for its test
results and limits. The ZED source and an experimental stereo matcher also
exist. See the [stereo research record](stereo-research-record.md) for the
current code status and open quality checks. The [integration architecture](integration.md)
describes the live application.

Measurements are the values recorded when the work happened. Where a later attempt
withdrew an earlier measurement, both are shown, and the document states which one
stands.

## 1. Overview and verdicts

| Line of work | Question | Verdict |
| --- | --- | --- |
| B1. Background calibration | Can a frozen empty-scene depth baseline make the person mask stable while the person sways? | **Rejected.** Phase A was implemented and then rejected on real footage: the mask covered the room and left the person as a hole. The reported stability improvement was measured on the wrong region. |
| B2. Pose-seeded depth-gradient mask | Can the depth gradient, seeded by the pose, select the person without a background model? | **Adopted.** It is the current live mask. It fixed the face warping as a side effect. Its quality is measured only by diagnostic proxies, not by checked pixels. |
| B3. Latency | Where does the 180 ms refresh go, and what can we remove from it? | **One real win.** Removing an unused 13.22 ms plane per refresh. Two claims were withdrawn: a poll-interval comparison and a "launch-bound" description of the YOLO call. One defect that the suite could not catch was found and closed. |
| B4. Replay harness and learning data | Can we extract a reproducible training record from recorded footage? | **Built and measured on real footage.** The extractor, the report tools, and the checked-mask workflow exist and are verified. The numerical mask comparison and the training set are blocked on hand-drawn masks. |

The one-sentence version: the live pipeline is finished and verified, the mask is
good enough to demonstrate and not yet measured against truth, and the learning
milestone cannot start until somebody draws checked masks.

## 2. Starting state: what the physical rehearsal told us

The physical rehearsal P1 to P5 ran on 2026-11-19 with a real person in front of a
live camera. The camera ran at 640 x 480 through DirectShow with a 100 ms window.
All five tests passed. The result that shaped all later mask work is P3, the
swaying person:

| P3 measurement | Value |
| --- | --- |
| Reuse decisions in 30 s | 2 of 140 |
| Published refreshes | 116 |
| Refresh failures | 0 |
| Warm latency p50 / p95 | 100.5 ms / 109.5 ms |
| **Splat count range** | **13,199 to 42,091, standard deviation 7,712** |
| Mean frame-to-frame splat change | 5,936 |
| **Mask area range** | 0.172 to 0.548 |
| Refreshes below 20,000 splats | 13 |

Pose dropout was not the cause. The cause was the design of the mask: a fixed
0.75 m depth band above the nearest depth, rebuilt from scratch on every refresh,
with no temporal smoothing. As the person swayed, the band captured more or less
of the body, and the splat state grew and shrank with it.

Other results from the same rehearsal that we still rely on:

- **P1, empty scene, 20 s.** 0 published, 68 failed, all with `pose not detected in
  the selected refresh frame`. No person state is published when no person is
  present. Note the order of checks: all 68 failures came from the pose
  requirement, so the mask-area guard did not run in this test. The mask-area
  rejection has separate evidence from an earlier run with mask area 1.000.
- **P2, person still, 30 s.** 40 published, 0 failed, 77 reuse. Splat count 31,380
  with standard deviation 940. Warm latency p50 98.4 ms, p95 110.0 ms.
- **P4, leave and return, 45 s.** Person loss latched, `person_loss_age_ms` rose to
  about 6.9 s, the splat state stayed frozen with no publishes, and publication
  resumed on return. One transient 19 ms loss showed that brief pose dropouts
  happen.
- **P5, depth tab.** Eight consecutive samples had the depth `splat_frame_id`
  equal to the active diagnostics frame, so the depth-hold contract holds with a
  live camera.

The P3 result is the problem statement for section 3.

## 3. Enhancements Tried and Tested

### 3.1: Background Calibration

#### 3.1.1 The Idea

Initializing the Depth+Pose Pipeline to an initial frame with no Pose and only the Background to allow the Pipeline to calibrate first to the background.

#### 3.1.2 Initial Implementation

- `POST /api/calibrate` starts a bounded capture of empty-scene depth frames. The
  worker runs pose and depth, collects the frames, builds one median background
  baseline, and freezes it.
- A person pixel is at least 0.30 m nearer than the background at that pixel,
  inside a 1.5 m span limit, and the existing 0.005 to 0.60 area guard applies.
- A calibration frame that contains a detected person aborts the capture.
- A size, source, or generation mismatch rejects the baseline. A generation
  change clears it.

**Freeze by construction.** The calibrator collects only during an explicit
capture and never learns while running. A person therefore cannot be absorbed into
the baseline, which was the overfitting failure mode identified before any code was
written. This property was correct and it stayed correct. The feature failed for a
different reason.

#### 3.1.3 The first comparison failed

With a live person swaying for 30 s, 233 refreshes:

| Result | Value |
| --- | --- |
| Fell back to the depth band | 183 of 233 |
| Fallback reason `too_large` | candidate area mean 0.637, maximum 0.824 |
| Fallback reason `too_small` | area near 0.001 |
| Refreshes that used the calibrated mask | 49 |
| Splat standard deviation | 9,857 |
| Splat range | 3,873 to 46,180 |

This was **worse** than the uncalibrated reference. The cause: a global per-pixel
background comparison marks large room areas as nearer than the baseline under
live YOLO depth noise, and a fixed 0.30 m margin also misses body pixels whose
depth is close to the background.

### 3.2 B2: the pose-seeded depth-gradient mask (adopted)

#### 3.2.1 The Idea

Select the person by the depth discontinuity between the body and its
surroundings, and use the pose only to say where to look. Do not model the
background, and do not threshold depth by a band.

#### 3.2.2 Implementation

`applications/live_pipeline/pose_gradient_mask.py`:

1. Median-fill invalid depth, then compute the Sobel gradient magnitude of the
   depth plane.
2. Run watershed on that gradient. Visible pose joints are foreground markers. The
   area outside a 12 percent pose box is background marker.
3. Keep the region inside the pose box, close it, and fill contours.
4. Remove background: drop a pixel whose depth is outside the visible-joint depth
   range, which is the p10 to p90 range of the visible joints plus 0.35 m. This
   removes background that leaks through weak edges between fingers or behind the
   head.
5. Re-add fingertip discs and the head and hand hulls *inside* that depth range,
   so a weak edge at a fist or at the hairline does not chop the person.
6. If the gradient mask fails, run the depth-threshold band and record the reason.

The joint constants are declared in the module: `MIN_JOINT_VISIBILITY = 0.15`,
`HEAD_JOINTS` are joints 0 to 12, `LEFT_HAND_JOINTS` are 15, 17, 19, and 21, and
`RIGHT_HAND_JOINTS` are 16, 18, 20, and 22.

#### 3.2.3 Significant Improvement

On a paired live frame, the face box contained background depth up to 2.26 m while
the pose joints reported about 0.75 m. Those stray background-depth pixels became
splats inside the face, which is the warping that was observed in the viewer.
Construction validity is now limited to the joint-depth range, so those pixels
produce no splats.

#### 3.2.4 Limitations

Measured on the same live session: mask source `pose_gradient` with 0 refresh
failures, 21 of 23 visible joints inside the mask on a paired frame, and we
confirmed in the viewer that the body mask is correct. 

The honest limits, stated then and not removed since:

- Hands are partially covered at rest.
- The face is still imperfect.
- Quality is measured by diagnostic proxies on RGB overlays, not by checked
  pixels. Nobody has yet drawn a true person mask and compared it. That is the
  blocking item in section 7.

## 4. Performance and Latency

### 4.1 The measured starting point

We measured a warm refresh on a live camera with a person present (127 samples):

| Stage | p50 | p95 |
| --- | --- | --- |
| pose + depth | 119.6 ms | 145.5 ms |
| mask | 30.8 ms | 37.2 ms |
| construct | 15.8 ms | 19.4 ms |
| pose preview | 15.6 ms | 19.0 ms |
| **total** | **180.8 ms** | **216.3 ms** |

Facts that stayed valid: the GPU sits at about 5 percent mean utilisation because
MediaPipe runs on the CPU and the YOLO depth model is small; the first inference
after model load takes 4.3 to 5 s because of the CUDA context and Ultralytics
setup; and a busy machine roughly triples observed latency (one 3-minute session at
577 ms p50 and 2,177 ms p95 on the same code).

### 4.2 Refresh and Reuse

The same interleaved profile split `normalize_prediction` at 25.29 ms for metric
depth. Almost all of it was `edge_confidence`, at 8.37 ms measured directly and up
to 13 ms inside a live call, which builds a log-depth image, two `np.gradient`
passes, a hypot, a NaN mask write, and a 3x3 dilate on every refresh.

`DepthEstimation.confidence` was **never read by the live pipeline**.
`PoseDepthPipeline.fuse_pose_depth` uses `depth_result.depth` only.

We added `normalize_prediction(..., compute_confidence=False)` and
`YoloDepthEstimator(compute_confidence=...)`, and the live path defaults to
`depth_confidence=False`. Side-by-side on the same frame with CUDA
synchronisation: pose + depth was 48.64 ms with confidence on and 35.42 ms with it
off, a **13.22 ms saving per refresh**, about 7 percent of the warm total. The
depth and validity arrays are bit-identical between the two paths. Only the unused
plane differs, and it is zero-filled to keep the contract shape. Callers that read
confidence keep the previous behaviour by default.

### 4.3 Micro-Optimizations

The gradient scale used `np.percentile` over the whole gradient image, 3.7 ms per
refresh, to pick a 97th-percentile scale. The stride-4 subsample (`gradient[::4,
::4]`) gives an estimate of that statistic at a fraction of the cost: mask stage
25.2 ms to 21.3 ms, a 3.9 ms saving.

On three real frames the sampled and full masks agreed at intersection over union
0.9893, 0.9884, and 0.9977, with pixel differences at most 652 pixels at the edges
and no fallback change. The code comment was corrected to say that the subsample
*estimates* the statistic with about 1 percent mask difference, instead of
describing it as identical.

That 3.9 ms optimisation became a much larger question on real footage, and section
6.5 records the result. This is a good example of a small performance change with a
quality consequence that needs its own investigation.

## 5. B4: the replay harness and learning data

### 6.1 Why a replay harness

A trained mask or motion head needs labels, and a label needs a reproducible
record: the exact input that produced the current baseline output, the baseline
output itself, and a way to compare against it later. The live application cannot
provide that, because it holds one state and discards everything else.

We approved the harness with a label rule that we have kept without exception:
**the saved accepted mask is `baseline_mask`, an input, and never ground truth.**
Manually checked masks are a separate label store.

### 6.2 What the harness does

`scripts/build_learning_replay.py` and `applications/live_pipeline/replay.py`:

- Read a recorded video in decode order with one timestamp origin for the whole
  clip.
- Apply the live `WindowSelector` to 100 ms windows, so the selection behaviour is
  the same as live.
- For each window, write one JSONL row plus the selected RGB frame, a compressed
  depth file with a validity plane, pose joints, a `baseline_mask` PNG, and the
  flow field when one exists.
- Record `mask_source` and `fallback_reason` for every row.
- Record every row and do not advance the reference on a failed refresh.
- Run fully in serial with one `RefreshPipeline`, no queue, and no browser.

The harness does not add latency to the live application. The live application
does not import it.

### 6.3 The review of the harness: three real defects and findings of mine that were wrong

A review of the harness ran before 37 minutes per clip were spent on it, and it
found things that changed the run cost:

- **The reuse path repeats full inference.** Every reuse window ran MediaPipe pose
  and YOLO depth again: 59.2 s per 100 input frames, about 0.6 s per frame.
- **Output size.** 34 windows produced 83.6 MB, about 2.5 MB per window.
- **A failed run blocks the next run.** `split.json` was written before the replay
  started, so a second run on the same clip raised `FileExistsError`.

Three other findings from that review were then **wrong**, and the corrections are
recorded:

- The claim that the reuse path did not apply the live validity gate and could
  disagree with live mask pixels was wrong, for the reason in section 4.5.
- The proposal to reuse the retained reference estimates for reuse rows to reduce
  cost would break the 15-channel input contract, because the input needs
  estimates that belong to the *selected* frame. Repeated inference stays.
- The "9-fold disk reduction" was a projection and not a measurement of file
  categories.

The decisions that followed are worth restating because they are counter-intuitive:

- **Keep the full output and the repeated inference.** Disk is not a constraint,
  and the input contract is more important than speed.
- **Serial replay time is offline cost only.** It is not capture-to-visible-splat
  evidence and must never be quoted as live latency.
- **Do not defer the session split.** A split is a session assignment even when an
  extraction fails, because two clips from one session must never start with
  different splits.
- **`--staged-output`, default off.** A unique staging name, an exclusive split
  reservation before any work, all files closed inside staging before publication,
  publication by one directory rename, a stale staging directory reported and never
  deleted, and a help text that says the flag cannot repair a partial output left
  by the default path.
- **Reports live outside the replay data**, and a report directory inside it is
  refused.

The timing estimate was also corrected: 3 minutes at 30 frames/s is 5,400 frames,
and 5,400 / 100 x 59.2 s is 53.3 minutes, not the 30 minutes first written.

### 6.4 Measured on the real clip

The clip `data/recordings/replay_1.mp4` was recorded for this work: 1280 x 720,
30.07 frames/s, 3,481 frames, 115.8 s. The first replay run was stopped at 5.2
minutes. It had written 165 rows (frames 1 to 778, clip time 0 to 25.8 s) and no
`manifest.json`, so it was incomplete. It was kept as
`movement_01.partial-20261121T1437` and **not** deleted. The full extraction with
`--staged-output` then completed in **37.1 minutes** and published
`data/processed/learning_replay/person_pilot_01/movement_01` with
`staged_output: true` and no staging directory left.

| Measure | Value |
| --- | --- |
| Windows | 735 |
| `refresh_ok` | 183 |
| `refresh_failed` | 41 |
| `reuse` | 511 |
| Output size | 7.36 GB, about 10 MB per row, 9.6 MB of it the depth file |
| Projected time for this clip | about 35 minutes; measured 37.1 minutes, so the projection held |

The per-row size is about four times the earlier 2.5 MB estimate, because the clip
is 720p rather than 640 x 480. Disk free was 865 GB, so this remains a non-issue,
but the estimate was wrong and the document says so.

**The strongest result in the whole replay line.** We added an additive replay
field `mask_pose` that stores the unrounded joints that mask selection actually
received, from the same pose result and with no second inference. On the real clip,
689 of 689 rows with a pose have a `mask_pose` file, and for every one of those
rows the stride-4 mask recomputed from the saved depth and the saved exact joints
has union **1.0** with the saved `baseline_mask` (p05, p50, and p95 all 1.0). The
saved record therefore reproduces the accepted live mask bit-exactly. That property
is what makes every later comparison in section 6.5 meaningful.

**Inspection of the 735 rows** (`scripts/inspect_replay_masks.py`):

| Measure | Value |
| --- | --- |
| Rows `ok` | 694 |
| Rows `person_missing` | 41 |
| `pose_gradient` masks | 689 |
| `depth_threshold_fallback` masks, reason `no_pose` | 5 |
| Rows with no mask | 41 |
| `pose_detected` values | true 689, false 5, null 41 |
| Missing or unreadable files | 0 |
| RGB and mask size mismatches | 0 |

All 40 `pose not detected` rows lie between 95.5 s and 102.8 s, which matches the
exit-and-return step. One row (frame 2631, 87.5 s) failed during fast arm motion
because the fallback band covered 0.83 of the image.

Region coverage, which uses the 0.15 visibility floor, at least three visible
joints per region, and a 2-pixel box pad, is a head box coverage p05/p50/p95 of
0.605/0.752/1.0, left hand 0.94/1.0/1.0, and right hand 0.93/1.0/1.0. **These are
box occupancy values and not accuracy.** The head box includes wall around the
face, and the paired images show the face and hair inside the mask.

### 6.5 The stride question, and why it is still open

The 3.9 ms saving in section 5.6 has a quality tail on real footage:

| Measure (689 real frames) | p05 | p50 | p95 |
| --- | --- | --- | --- |
| Mask union, stride 4 against stride 1 | 0.914 | 0.9984 | 0.9999 |
| Different-pixel fraction | 0.0 | 0.0003 | 0.0205 |
| Area delta | | 0.0002 | |

No fallback changed. Head and hand region coverage delta p50 is 0.0. The earlier
claim of 0.988 to 0.998 came from three frames and is **optimistic for the tail**.

The worst row is frame 3308 (arms raised, fingers spread) at union **0.567**. There
stride 4 adds the door and the lower-left desk while stride 1 stays on the person.
The five lowest rows are all between 0.567 and 0.613. Frame 337 shows the opposite:
stride 1 adds unwanted lower-body and desk pixels that stride 4 excludes.

A 32-row spot check of stride 1, 2, and 4 found stride 2 close to stride 1 on frame
3308 but close to stride 4 on frame 3319 and frame 337, so **stride 2 is not a cure
for the tail by itself**. On frame 3308 the 97th-percentile gradient scale differs
by less than 1 percent across strides, yet the watershed mask changes greatly. Small
scale changes can cause large region changes after 8-bit conversion and watershed.

Our decision stands: **keep stride 4 as the current live baseline** until
checked-mask evidence supports a change. Do not promote stride 1, stride 2, or a
downscaled-full percentile on agreement scores alone. A downscaled-full percentile
keeps the cost but not the sample bias; it is an experiment, not an assumed fix.

The rule this produced: **two masks that agree on 99.8 percent of pixels can still
be different where it matters** — at the boundary of a hand and a door. Agreement
statistics are a triage tool. Only a checked mask decides accuracy.

### 6.6 A data-quality finding that would have poisoned a training set

The five `depth_threshold_fallback` rows with `no_pose` occur during the exit
(95.9 s to 101.6 s). Their masks are pure background (wardrobe, desk) at about
0.24 area, and their `status` is `ok`. A 15-channel pair built from them would use
a background mask as the person label.

The rule this produced: **`status=ok` is not a training-eligibility rule.** A row
needs a non-fallback mask source, an exact `mask_pose`, a saved selected depth, a
flow file when it is a refresh row, and a checked label.

### 6.7 The tools we built on top of the replay

| Tool | What it produces |
| --- | --- |
| `scripts/inspect_replay_masks.py` | Overlay, `index.html`, and `report.json` per clip in `<clip>.mask_report/`. Separate denominators for `no_mask`, `missing_file`, `size_mismatch`, `no_joint_record`, and `insufficient_visible_joints`. Counts `pose_detected` true, false, and null separately. |
| `scripts/compare_mask_strides.py` | Runs the unpatched selector at stride 4 and then stride 1 on the same saved depth and exact joints, and reports union, different-pixel fraction, area delta, region deltas, and paired overlays in `<clip>.stride_compare/`. |
| `scripts/learning_data review-set` | 45 review frames in `<clip>.review/`: 22 low-agreement, 12 high-agreement, 5 no-pose fallback, and 6 missing-person. |
| `scripts/label_review_masks.py` | An OpenCV label window. Left drag paints, right drag erases, `d` clears, `[` and `]` change the brush, `g` saves, `s` skips, `p` goes back, `q` quits. |
| `scripts/learning_data evaluate` | Person coverage, background leakage, largest leakage component, and hand and head kept fraction, for the saved baseline and for strides 1, 2, and 4 on the same saved depth and joints, with per-method timing. |
| `scripts/learning_data training-filter` | Eligible pairs and a count for each exclusion reason. |

Report defects found and fixed along the way: the `pose_detected_false_rows` and
`pose_detected_null_rows` counters existed but were never incremented, so the report
showed zero for both in every group. We found this during review, repaired it, and
added a test with false, null, and true values across three groups. The regenerated
report shows true 689, false 5, null 41.

### 6.8 Where the training set stands

`training-filter` on the real clip reports **0 eligible of 735 rows**, with these
exclusion reasons: `no_label_store` 735, `no_exact_mask_pose` 46,
`selected_status_person_missing` 41, `no_selected_depth` 41,
`no_pose_fallback_mask` 5, `no_reference` 1, and `refresh_row_has_no_flow` 1. A row
can carry more than one reason, so the counts do not sum to 735.

The bootstrap row is the only row with a null reference ID.

## 7. What is open, and what we are blocked on

### 7.1 Blocked on hand-drawn masks

This was the state at this stage of the work. Checked masks were later made.
The [learning result](DEEP_LEARNING_PLAN.md) gives the later test result.

Every number that could decide a mask question needs a **checked person mask** that
a person drew while looking at the RGB image. We cannot generate that label. The
workflow is ready and recorded in
`docs/LEARNING_REPLAY_PERSON_TEST_CHECKLIST.md`:

```powershell
# 1. Review set (45 frames, already written)
& .\.venv\Scripts\python.exe -m scripts.learning_data review-set `
  --output-dir data/processed/learning_replay/person_pilot_01/movement_01

# 2. Hand-drawn checked masks
& .\.venv\Scripts\python.exe -m scripts.label_review_masks `
  --review-dir data/processed/learning_replay/person_pilot_01/movement_01.review `
  --labels-root data/processed/learning_replay/labels --dataset movement_01

# 3. Score the baseline and the stride variants
& .\.venv\Scripts\python.exe -m scripts.learning_data evaluate `
  --output-dir data/processed/learning_replay/person_pilot_01/movement_01 `
  --labels-root data/processed/learning_replay/labels --dataset movement_01
```

About 20 to 30 of the 45 frames are enough for a first answer. The 22 low-agreement
frames decide the stride question, so they come first. The full set is 45 frames.

### 7.2 Also open

- **A proposed rule that needs a decision.** A refresh row without a `flow` file is
  treated as ineligible for the 15-channel set, because the input needs flow. That
  rule is proposed and not yet confirmed or rejected.
- **A separate recording session for the `test` split.** The pilot clip is used for
  research and review. A trained model that is accepted needs an untouched session.
- **The physical P1 to P5 gate, again.** The rehearsal in section 2 passed on the
  **depth-band** mask. That mask was then replaced by the pose-gradient mask. No
  physical session has repeated P1 to P5 on the current code. The physical gate is
  therefore still open for the code that the demonstration runs today.
- **Step timings for the 2-minute clip.** The exit window can be identified from the
  data, but not the other steps. Approximate action times for the clip would let the
  segments be labelled. Image review can proceed without them.
- **The GPU budget for a research model.** The learning plan requires a measured
  budget before the model resolution is fixed: pose + depth plus one model forward,
  with median and p95 time, `torch.cuda.max_memory_allocated()`, and the effect on
  refresh time. This is not measured yet.

### 7.3 Not attempted, and why

- **Learned depth or pose replacement.** It is the only change with a large latency
  effect and the highest risk, because it touches the signal the whole pipeline
  depends on. We put it after the joint mask and motion milestone.
- **Online weight updates.** See section 5.7.
- **ZED stereo depth at this stage.** The hardware was not yet available, and
  no stereo code existed. The later [stereo result](STEREO_DEPTH_EXPERIMENT_PLAN.md)
  records the implemented experiment and its open quality checks.

## 8. Claim discipline

These rules came out of the failures in this document. They are short on purpose,
and each one has a specific failure behind it.

1. **A saved `baseline_mask` is model output, never a label.**
2. **A lower variance over the wrong region is not improvement.** Check the mask
   against the RGB image before you report any mask metric.
3. **A synthetic depth map tests your assumption about the sensor, not the
   sensor.**
4. **`pose_detected=null` is unknown, never false.** A `person_missing` status can
   also come from a rejected mask, and it is not proof that no person was in the
   frame.
5. **`status=ok` is not training eligibility.**
6. **Agreement between two masks is not accuracy.** Only a checked mask decides.
7. **Serial replay time is offline cost, never live latency.**
8. **A passing unit suite does not prove the live wiring.**
9. **Utilisation is not a cost model.**
10. **A measured claim must name the event it measures.** "The label changed" is not
    "the splat is visible".
11. **Do not delete an interrupted or rejected artefact.** We kept the partial
    extraction, the stale staging directory, and the unused calibration modules, and
    each one is labelled.
12. **No physical claim without a physical run.** The physical P1 to P5 gate is
    stated as passing on the mask that existed then, and as open for the mask that
    exists now.

## References

- [Integration evidence](integration-evidence.md): the phase that this document continues.
- [Integration technical history](integration-technical-history.md): the API and state machine.
- `docs/physical_test_checklist.md`: P1 to P5.
- `docs/LEARNING_REPLAY_PERSON_TEST_CHECKLIST.md`: recording, replay, inspection,
  labelling, evaluation, and the training filter, in order.
- [Learning research record](learning-research-record.md): the planned learning milestone and its gates.
- `shared/schemas/learning-replay.md`: the replay row contract, including the
  `mask_pose` field and the rounded `selected_pose` field.
