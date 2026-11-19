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
