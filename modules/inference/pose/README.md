# Pose Estimator: MediaPipe Pose Landmarker

This module estimates 33 body landmarks from one RGB frame. It is the input to the
person mask and to the depth-based validity gate before splat construction.
The milestone uses MediaPipe Lite. The measured comparison below explains why
it was selected and what the eleven scenarios do not establish.

## Model choice

Two MediaPipe Pose Landmarker variants were compared: **Lite** and **Full**.

**Lite is the runtime model.** Both variants detected the same scenarios, but Lite
runs at lower latency and the two were comparable on pose quality for the
operating conditions this project uses. Full detected one additional extreme
motion-blur case, which was not enough to justify its cost in a live pipeline.

The local copy of the runtime model is `models/pose_landmarker_lite.task`. That
directory is ignored by Git, so the model is fetched during setup rather than
committed.

## Measured comparison

Eleven scenarios, each run through both models. The full data is in
[`overall_comparison_outputs/comparison_results_expanded_with_occlusion.csv`](overall_comparison_outputs/comparison_results_expanded_with_occlusion.csv).

| Scenario | Lite detected | Lite ms | Full detected | Full ms |
| --- | --- | ---: | --- | ---: |
| front_facing_runner | yes | 18.78 | yes | 22.21 |
| sideways_runner | yes | 18.55 | yes | 22.73 |
| lower_body_cutoff | yes | 54.03 | yes | 56.65 |
| multiple_people | yes | 54.41 | yes | 57.62 |
| multiple_people_varying_distances | yes | 42.57 | yes | 45.65 |
| low_light_visible_body | yes | 18.72 | yes | 22.23 |
| extreme_low_light_silhouette | **no** | 11.70 | **no** | 11.25 |
| moderate_motion_blur | yes | 18.58 | yes | 22.48 |
| **extreme_motion_blur** | **no** | 18.42 | **yes** | 22.42 |
| face_mask_occlusion | yes | 19.01 | yes | 22.43 |
| object_occlusion_full_body | yes | 19.99 | yes | 22.24 |

Lite infers in **18 to 19 ms** on the typical cases; the two multi-person and
cropped-body cases take longer because MediaPipe runs more detection passes.
Full costs roughly **3.5 ms more** per frame on the same cases.

**Both models fail extreme low-light silhouettes.** That case is a genuine limit of
the detector, not a model-choice difference, and it is the reason the live pipeline
treats a missing pose as an ordinary refresh outcome rather than a fault.

## Reading the results honestly

Three limits apply to this comparison:

- **Detection and timing are measured; pose quality is not.** The CSV records
  whether a pose was found, how many landmarks, mean visibility, and inference
  time. It does not contain a ground-truth joint comparison, so "comparable pose
  quality" is a visual judgement from the overlays, not a numeric accuracy result.
- **Timings are per-image inference on this computer**, not end-to-end pipeline
  latency. The live refresh adds depth, mask selection, and construction.
- **`average_visibility` is MediaPipe's own confidence**, not a measure of error
  against truth. A high value means the model is confident, not that it is right.

The face-mask-occlusion row shows this clearly: Lite reports mean visibility 0.7137
with lower-body visibility 0.0940, meaning the model found the upper body and
largely gave up below the mask. That is plausible behaviour, not a proven correct
answer.

## Image evidence

Each scenario has a **Lite** and a **Full** overlay, named
`lite_<scenario>.jpg` and `full_<scenario>.jpg` in
[`overall_comparison_outputs/`](overall_comparison_outputs/).

**Three scenarios have no overlay in this repository:** `multiple_people`,
`multiple_people_varying_distances`, and `lower_body_cutoff`. The files committed
for those three under `overall_comparison_outputs/` are full-resolution source
images rather than skeleton overlays, so they were moved to external storage with
the other source photos. The CSV still records the measured result for all three.

**The source photos are large** (9 files, 113 MB) and are hosted externally rather
than committed to the repository:

| Source photo | Link |
| --- | --- |
| multiple_people.jpg | [Google Drive](https://drive.google.com/file/d/1BY6HyaCnBa4DkILfauIiFrSR0_Esv2bX/view) |
| multiple_people_varying_distances.jpg | [Google Drive](https://drive.google.com/file/d/1nm8q08GlCSAXDPzgog4JpsKV9uMkFwA2/view) |
| Runner_with_lower_body_cutoff.jpg | [Google Drive](https://drive.google.com/file/d/1vRE2UmguXe_93DscEimvKho1XTpyOomR/view) |
| full_multiple_people.jpg | [Google Drive](https://drive.google.com/file/d/15-huu1Xw2GIfH3Kp8FhgYo-OLDL2YWpa/view) |
| lite_multiple_people.jpg | [Google Drive](https://drive.google.com/file/d/1sZQI0eH1K3CofbyRp26W4EL8AuBcQiG0/view) |
| full_multiple_people_varying_distances.jpg | [Google Drive](https://drive.google.com/file/d/1mJRJQqQ-9J4Q3F95eILREG1qmAqelidP/view) |
| lite_multiple_people_varying_distances.jpg | [Google Drive](https://drive.google.com/file/d/1exY2pZHmIuF7_iUgTBuZtZlky02o3e_1/view) |
| full_lower_body_cutoff.jpg | [Google Drive](https://drive.google.com/file/d/1jWhOwhE5awZFtg87i3OQry2Yy2x7wFHQ/view) |
| lite_lower_body_cutoff.jpg | [Google Drive](https://drive.google.com/file/d/1EB06YeVeGSLLKqO8swNCyt7x-Ar0kNyZ/view) |

If a link is unavailable, any of these scenarios can be regenerated with
`modules/inference/pose/scripts/compare_models.py`, which writes the same overlay pair for every image in
`test_images/`.

## Reproducing the comparison

```powershell
& .\.venv\Scripts\python.exe -m modules.inference.pose.scripts.compare_models
```

The script writes the overlays and the CSV into `overall_comparison_outputs/`. It
requires `models/pose_landmarker_lite.task` and the Full variant in place.

## Where this module sits in the pipeline

`PoseDepthPipeline` runs this estimator on the selected refresh frame and fuses
each landmark with the depth plane at that pixel. The fused joints drive two
downstream stages:

- the pose-seeded depth-gradient mask, which uses joints as watershed markers
- the splat validity gate, which limits construction to the joint depth range

See the [integration architecture](../../../docs/integration.md) for the identity
contract and the [enhancement record](../../../docs/Integration-Enhancements.md)
for the mask work that consumes these joints.
