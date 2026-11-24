# Joint Learning Plan for Person Masks and Splat Reuse

## Status and objective

This is the next research milestone for `live-splat-pipeline/`. The
demonstration pipeline is implemented through Phase 2, with the display and
diagnostics repairs from RT-0049. The current acceptance gate is the physical
person rehearsal P1-P5 in `docs/physical_test_checklist.md` and the real-person
threshold and mask calibration; recordings for this plan can be collected in
parallel with that gate. The first trained model must improve the person mask
used by splat construction, the image-space motion used by splat reuse, or both.
It must run on this computer during a live stream. Run it in shadow mode first.
Do not change the default demonstration result.

The current path uses a pose-seeded depth-gradient mask on refresh, with a
depth-band fallback when the gradient mask fails. It uses OpenCV DIS optical
flow on reuse. `RefreshPipeline` owns pose, depth, mask, and construction.
`MotionPreview` sends a 40 x 30 flow field to the browser. The browser moves
displayed splats in the source view. It does not change their stored three-
dimensional (3D) positions. Keep these baselines for every comparison.

The first model is a **joint residual model**. It has one shared image encoder,
one person-mask head, and one motion-correction head. The motion head predicts
a correction to DIS flow. It does not predict a new Gaussian state or metric
depth. Train the model offline. Online weight updates are a later experiment.

## Design rules and interfaces

Use these design patterns:

1. **Ports and adapters.** Keep PyTorch and model preprocessing inside a new
   research inference package under `modules/inference/`. Expose a typed
   predictor interface to `applications/live_pipeline/`. Keep the native Splat
   Constructor and the browser independent of PyTorch.
2. **Immutable frame data.** Use frozen, typed request and result records.
   Each record carries `source_id`, `generation`, `frame_id`,
   `capture_timestamp_ns`, width, height, and the active splat reference ID.
   The reference record also carries its own frame and capture identity.
   Copy arrays when an asynchronous job takes ownership of a camera frame.
3. **Single inference owner.** Share one `PoseDepthPipeline` instance. Serialize
   calls to it. Give refresh jobs priority over research reuse jobs. Do not
   load a second YOLO depth model for shadow inference on the 6 GB graphics
   processing unit (GPU). Inject this owner into `RefreshPipeline` instead of
   making a second owner inside the research path. Permit a fake owner in tests.
4. **Latest-only work.** Permit one running research job and one replaceable
   pending research job. Capture, optical flow, ranking, baseline reuse, and
   refresh publication must not wait for a research reuse result. Drop a result
   when its run, selected frame, or active splat reference is no longer current.
5. **Strategy with a feature switch.** Support `off`, `shadow`, `mask`,
   `motion`, and `mask_motion` modes. Default to `off`. `shadow` computes and
   records model outputs but publishes the baseline. `mask` changes only
   refresh masks. `motion` changes only reuse motion. `mask_motion` changes
   both. A missing checkpoint, invalid model output, or model failure selects
   the baseline for that frame.

Build one aligned input from the retained refresh frame and one selected frame.
Use matching red-green-blue (RGB) images, DIS flow, current and reference depth
maps, depth-validity maps, and current and reference pose maps. Generate pose
maps from joint positions and visible limb lines in image coordinates. Supply
the current accepted pose-gradient mask (or its recorded depth-band fallback)
as an input. Each depth and pose estimate must
belong to its RGB frame. Do not use the cached pose or depth as a substitute
for the selected frame's estimate. The retained reference can use the pose and
depth saved with its successful refresh.

Use one fixed training resolution, initially 320 x 180 pixels. Record the
resize transform and apply it to RGB, depth, validity, masks, joints, and flow.
Scale horizontal and vertical flow vectors with the corresponding image axes.
Use this 15-channel input order: reference RGB (3), current RGB (3), reference
depth (1), current depth (1), reference depth validity (1), current depth
validity (1), DIS horizontal and vertical flow (2), reference pose map (1),
current pose map (1), and current accepted baseline mask (1). Record its
`mask_source` as `pose_gradient` or `depth_threshold_fallback`. Scale finite depth
in metres to `[0, 1]` by clipping it to `[0, 10]` and dividing by 10. Set an
invalid model depth value to zero and mark it in the separate validity map.
Divide each flow component by its image width or height. Do not change the
original depth units and invalid values at the constructor boundary.

The predictor returns a person probability map and a two-channel residual
flow map in normalized image units at model resolution. Train the motion head
against teacher flow minus DIS flow, after both are normalized by image width
and height. Upsample both outputs to source resolution. Multiply the residual
components by the source width and height to recover pixel units. The adapter adds the
residual to DIS flow, applies the existing validity and maximum-offset rules,
and then uses `build_motion_preview` to produce the existing browser payload.
The model must not change `MotionPreview` wire fields or stored splat positions.

For a refresh, create the accepted baseline mask and the model mask after fresh
pose and depth are ready. Compare the model with the pose-gradient result and
the existing pose-keypoint-hull baseline on the same held-out frames. Keep the
depth-band mask as a recorded fallback. A model mask may be used only when it
has finite values, the correct dimensions, a valid connected person region,
and an allowed area. Intersect it with valid-depth pixels before construction.
Keep the current pose requirement. A valid model mask may rescue a failed
  baseline mask, but it cannot rescue missing pose or depth. If the model
mask fails validation, use the accepted baseline mask and its existing failure
rule. Publish the mask with the same identity as the splat state.

## Data and training method

1. Record static-camera RGB clips for one demo subject. Include stillness,
   gestures, fast motion, partial occlusion, exit and return, clothing changes,
   and lighting changes. Keep camera settings, frame dimensions, frame rate,
   and capture timestamps in a local manifest. Do not commit recordings or
   model weights.
2. Split by complete recording session before extracting frame pairs. Keep an
   untouched test session. Do not put adjacent frames from one session into
   different splits. Select pairs at the same time gaps that the 100 ms window
   and research refresh mode will produce.
3. Manually check person masks for selected frames. Mark matching body points
   in selected frame pairs. Keep some masks and point pairs in the untouched
   test set. Do not treat the current accepted mask or DIS flow as ground
   truth.
4. Generate pose, depth, DIS flow, and optional high-quality flow teacher
   outputs offline. Keep model versions and input transforms in the manifest.
   Record the actual `mask_source` for each sample. Use teacher flow only as a
   training target where it is valid. The held-out
   hand-marked points remain the independent motion test.
5. Train a small encoder-decoder with `GroupNorm` and two output heads. Start
   with the existing `TinySegmentationNet` block style, but use paired-frame
   inputs and separate heads. Train the mask head with binary cross-entropy
   plus Dice loss on checked masks. Train the motion head with a robust,
   validity-masked residual-flow loss against teacher flow. Use a fixed weight
   for each loss and report each loss separately. Do not train against a target
   when its label is absent. Use paired crop, resize, and flip transforms;
   transform flow vectors and point labels with the images.
6. Use a batch size that fits the measured 6 GB GPU memory. Select the best
   checkpoint with validation data. Save the model configuration, input
   channel order, resize rule, target units, training-data manifest ID, and
   measured validation results with the existing versioned checkpoint helper.
   Run inference with `torch.inference_mode()` and a loaded, fixed checkpoint.

## Runtime sequence

1. Build an offline replay tool that reads recorded clips, reuses the current
   frame identity and window-selection rules, and writes baseline data for the
   same selected frames. Keep session and clip IDs, source and generation,
   selected and reference frame IDs and timestamps, RGB locations, flow,
   model versions, resize rules, and `mask_source` in each row. Store the
   accepted mask as `baseline_mask`; it is not a checked label. Store checked
   masks and body-point matches separately. Split by recording session before
   extracting pairs. Add model metrics after a checkpoint exists.
2. Add the predictor adapter and the shared inference owner. On refresh,
   calculate pose and depth once, then calculate the baseline and model masks.
   Save an immutable research reference with the active refresh RGB, depth,
   pose, and full frame identity. Replace it only when that refresh becomes
   the active splat state. Clear it on a run change.
   On reuse, publish the baseline flow immediately. Submit the selected RGB
   frame for fresh pose and depth. Run the joint model only after those
   estimates arrive. A late model result is for diagnostics only if the active
   reference has changed.
3. Add shadow diagnostics: model mode, model version, input and active-reference
   identities, mask comparison, motion comparison, queue drops, stale-result
   drops, per-stage time, capture-to-result time, and peak GPU memory. Keep
   the existing diagnostic fields valid.
4. Add an explicit research preset with a longer timed-refresh interval. Keep
   the default 500 ms rule in normal mode. Record the chosen interval in each
   result. Compare each model run with a baseline run at the same interval.
5. Permit `mask`, `motion`, or `mask_motion` only after the matching acceptance
   checks below pass.
   Make each mode an explicit launch option. Do not promote a checkpoint
   automatically.

The first replay extractor is `python -m scripts.build_learning_replay --clip
data/recordings/<clip>.mp4 --session-id <session> --split train`. Use the
project `.venv` when the system Python lacks the pose and depth packages. Give
all clips from one recording session the same session ID and split. The tool
writes ignored files under `data/processed/learning_replay/`. Its
`baseline_mask` is the accepted live mask, not a checked label. The current
extractor has no trained-model score and no live capture-to-display measure.

## Tests and acceptance

### First mask-training diagnostic (2026-11-22)

`modules.inference.paired_research.train` now loads checked replay pairs with
the fixed 15-channel input. It has a shared encoder, a mask head, and a motion
head. The first run trained the mask head only. The motion head has no motion
target and remains untrained. Use the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m modules.inference.paired_research.train `
  --replay-dir data/processed/learning_replay/person_pilot_01/movement_01 `
  --labels-root data/processed/learning_replay/labels `
  --training-filter data/processed/learning_replay/person_pilot_01/movement_01.training_filter.json `
  --output checkpoints/paired_research/movement_01_mask_diagnostic.pt `
  --report outputs/paired_research/movement_01_mask_diagnostic.json `
  --epochs 10 --batch-size 2 --base-channels 12 --device cuda
```

The run used 32 checked pairs from one `train` session. Its final mask
intersection over union was 0.895 on those same training pairs. The saved
baseline mask measured 0.693 on those pairs at 320 x 180. These values show
that the model can fit this training set. They do not measure performance on
new footage. The checkpoint records `train_only_no_validation`, and it must
not be used as the live mask or motion output. Keep its ignored checkpoint
and report outside Git. Next, collect distinct validation and untouched test
sessions with checked masks. Use independent motion targets before training
or scoring the motion head.

### Validation recording and fixed-checkpoint comparison

Record a new raw camera clip under a new session ID and the `validation`
split. A different clothing colour in the same room checks an appearance
change. Keep the camera and light fixed for a direct comparison. Include
stillness, raised arms and open hands, partial occlusion, turns, fast arm
movement, and a brief exit and return. A 60-second clip is sufficient for the
first validation pass. Keep the recording outside Git.

After the recording is available, extract it with `scripts.build_learning_replay`
and `--staged-output`. Build a review set, check selected person masks, and run
`scripts.learning_data training-filter` with that session's label dataset.
The filter lists eligible paired frames in each split, although its command
name says `training-filter`. Then run the fixed-checkpoint evaluator:

```powershell
.\.venv\Scripts\python.exe -m modules.inference.paired_research.evaluate `
  --checkpoint checkpoints/paired_research/movement_01_mask_diagnostic.pt `
  --replay-dir data/processed/learning_replay/person_validation_01/clothes_01 `
  --labels-root data/processed/learning_replay/labels `
  --training-filter data/processed/learning_replay/person_validation_01/clothes_01.training_filter.json `
  --split validation --report outputs/paired_research/clothes_01_validation.json
```

This evaluator uses a fixed 0.5 mask threshold and the same 320 x 180 resize
for both methods. It rejects the training session and a duplicate training
clip. It does not retrain or score the motion head. Keep a later test session
untouched until model and threshold decisions are final.

### First validation result (2026-11-22)

The `person_validation_01/clothes_01` session contains a different shirt and
different camera framing in the same room. Its replay has 386 selected rows,
102 successful refreshes, and no failed refreshes. A time-spread review set
with 12 low-agreement and 12 other complete pairs received user-checked masks.
All 24 pairs passed the fixed-checkpoint evaluation. The accepted baseline
mask had mean intersection over union (IoU) 0.7333. The model mask had mean IoU
0.6830. The model won on 6 frames, lost on 17, and tied within 0.001 on 1.
Its mean person coverage was 0.8870 versus 0.9286 for the baseline. Its mean
background leakage was 0.3080 versus 0.2849 for the baseline. Do not enable
the checkpoint on the live mask path.

Visual inspection showed disconnected model regions on room objects and lost
body regions. An exploratory largest-connected-region cleanup raised model
mean IoU to 0.7513 on these 24 frames. This cleanup was selected after the
validation result was seen. It is diagnostic evidence, not an independent
acceptance result. It also remains below the planned +0.03 IoU margin over
the baseline. The training-only score of 0.895 did not transfer to this new
session. The small training set, clothing change, and framing change are
possible causes; this experiment does not isolate them.

This result led to the bounded correction and augmentation comparison below.
The validation session remains separate from training. A broader training
session with varied subject scale, clothing, and pose remains useful. The
motion head still needs independent motion targets and a separate check.

### Bounded correction and augmentation experiment (2026-11-22)

The model now has an optional `bounded_residual` mask mode. It starts with
the accepted baseline mask. It can change the mask only within that mask or
12 model pixels from it. The new `paired` augmentation applies one flip,
scale, and translation to all 15 input channels and the checked mask. It
changes the horizontal flow sign on a flip and scales both flow components
on a resize. It also changes the RGB brightness, contrast, and colour. The
motion head remains untrained. All variants used the same 32 training pairs,
12 base channels, seed 7, and a 0.5 mask threshold. The three new variants
ran for 30 epochs. The original absolute model ran for 10 epochs, so the
table is a development comparison, not a controlled claim about each factor.

| Mask mode | Augmentation | Mean IoU on 24 validation frames | Mean coverage | Mean background leakage |
| --- | --- | ---: | ---: | ---: |
| Accepted baseline | None | 0.7333 | 0.9286 | 0.2849 |
| Absolute model, original | None | 0.6830 | 0.8870 | 0.3080 |
| Absolute model | Paired | 0.7635 | 0.9494 | 0.2557 |
| Bounded correction | None | 0.6363 | 0.7443 | 0.1870 |
| Bounded correction | Paired | 0.8052 | 0.9147 | 0.1412 |

The bounded, augmented model won against the baseline on 21 frames and lost
on 3. Its mean IoU gain was 0.0719. Visual inspection showed that it did not
make the distant room-object regions seen with the first model. It still
removed too much of the person on two frames where the baseline was already
accurate. A 12-pixel correction gate cannot reach a mean of 0.0063 of checked
person pixels in the validation set; the worst frame had 0.0778 outside the
gate. These values are design limits, not model scores.

The selected weights are copied without change to the ignored
`checkpoints/paired_research/person_mask_candidate_v1.pt`. Its SHA-256 is
`d821ffcc66557047b50625c4c7345634aacf72b55b0181d0d34d7e487d927606`.
The ignored `outputs/paired_research/candidate_v1_manifest.json` records the
four reports and the fixed threshold. A model-only CUDA measurement on this
computer gave 1.874 ms median and 2.548 ms 95th percentile for one 320 x 180
input, after warm-up. This does not include depth, pose, data transfer, or
browser display time.

The validation session was used to select this variant. Do not train on its
checked masks. Keep the candidate out of the live path. The next quality
gate is one untouched test session with checked masks, followed by visual
review and a live shadow-mode timing test. The planned +0.03 IoU margin is
assessed on that untouched test. A passing test would support mask promotion
only; it would not validate the untrained motion head.

### Frozen candidate test result (2026-11-22)

The separate `person_test_01/test_01` clip is 27.6 seconds at about 15 frames
per second. Its replay has 138 selected rows, 46 successful refreshes, and no
failed refreshes. Before model inference, 14 complete pairs were selected
across time and four extra pairs were selected from the lowest agreement
between the two baseline stride settings. The user checked all 18 masks.
The candidate checkpoint hash matched its frozen manifest before and after
the test. No model weight or threshold changed.

At 320 x 180, mean mask IoU was 0.7917 for the candidate and 0.7339 for the
accepted baseline. At the source size of 1280 x 720, with bilinear upsampling
of model logits and the fixed zero-logit threshold, mean IoU was 0.7997 for
the candidate and 0.7354 for the accepted baseline. The source-size mean
gain was 0.0643. The candidate won on 11 frames and lost on 7. Visual review
showed losses on close or easy frames where the baseline was already good.

The 14 time-spread frames had a mean gain of only 0.0272 and an even split
of seven wins and seven losses. The four extra difficult frames had a mean
gain of 0.1941. The difficult frames strongly affect the overall score, so
do not treat the 0.0643 gain as a representative clip-wide gain. The
time-spread result is below the planned 0.03 margin. The separate
pose-keypoint-hull baseline named in this plan is not implemented as an
independent method, so that comparison remains open. The short clip does not
show a clear exit and return and cannot measure live latency.

The ignored `outputs/paired_research/candidate_v1_test_decision.json`
records the test result and limits. Keep this candidate out of the live mask
path. It can enter shadow-mode timing and visual inspection without changing
the baseline result. If the team changes the model or threshold after seeing
this test, use a new untouched recording session for the next final test.

Use unit tests for paired resize and flip transforms, exact frame identity,
invalid depth, missing pose, malformed model output, model failure, run change,
and late-result rejection. Use a controlled slow predictor to prove that
baseline capture and reuse continue while research inference runs. Check that
the `off` mode gives the same output as the current pipeline.

Evaluate the untouched recordings with these measurements:

- Person-mask intersection over union and foreground leakage against checked
  masks. Report results for the model, the pose-gradient baseline, and the
  pose-keypoint-hull baseline. Require at least +0.03 absolute intersection
  over union against the stronger baseline before enabling the mask mode.
- Body-point endpoint error in source-image pixels. Report results for the
  corrected flow and DIS flow on the same marked pairs. Require at least 15
  percent lower error before enabling the motion mode. Show failure examples.
- Visible temporal jitter, missing body parts, and background splats in paired
  viewer recordings. Review the source view and an oblique view.
- Median and 95th-percentile capture-to-display delay, completed model updates,
  stale drops, refresh count, and peak GPU memory during live replay.

For this milestone, success means: the team trains and loads its own checkpoint;
the joint model runs in shadow mode beside a live stream; its mask and motion
outputs are measured on held-out data; and the working baseline stays usable.
Enable a model output only if it improves its matching quality measure on the
untouched test set, passes visual review, and does not cause an unacceptable
delay or GPU memory failure in a live run. Report the result even if neither
output passes. A trained model is a valid research result; an unmeasured claim
of better 3D reconstruction is not.

## Limits and later work

The recorded RGB clips have no measured depth ground truth. This milestone
cannot establish more accurate 3D geometry or newly visible surfaces. The
model can improve person selection and source-view motion. A refresh still
corrects reuse drift. After the offline and live comparison, consider bounded
online weight updates as a separate experiment. Give that experiment its own
rollback rule and comparison against the fixed checkpoint.
