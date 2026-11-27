# Inference package: live models and temporary baselines

The live application uses YOLO depth, MediaPipe pose, and a pose-seeded depth
mask. These are documented in the subpackage READMEs:

| Live stage | Subpackage and document |
| --- | --- |
| Depth | [`depth/`](depth/README.md) — YOLO26n metric depth in metres, `NaN` invalid. |
| Pose | [`pose/`](pose/README.md) — MediaPipe Pose Landmarker Lite, 33 joints. |
| Optical flow | `optical_flow/` — OpenCV DIS by default, optional Torchvision RAFT Small. See the [optical-flow decision](../../docs/optical-flow-training.md). |
| Person mask | [`../../docs/person-mask.md`](../../docs/person-mask.md) — pose-seeded depth gradient with a depth-band fallback. |

The compact CNNs described below were research baselines used during early
Splat Constructor work. **They are outside the live path.** No live stage loads
them, and no live code depends on a checkpoint from this page. They are retained
for the research record.

## Training data

The baseline uses Princeton's 93.5 MB APC 2016 training sample. It provides 99
aligned RGB, 16-bit depth, and binary foreground-mask triplets. The source is
available for research use from <https://apc.cs.princeton.edu/>. Cite Zeng et al.,
“Multi-view Self-supervised Deep Learning for 6D Pose Estimation in the Amazon
Picking Challenge,” ICRA 2017, when using the dataset.

Extract the sample so the dataset root ends in `training-sample/`. Data and
generated checkpoints remain ignored by Git.

## Commands

```powershell
./scripts/download_apc_training_sample.ps1
$data = "data/processed/apc_training_sample/training-sample"
python -m modules.inference.depth.train --data $data --epochs 5
python -m modules.inference.segmentation.train --data $data --epochs 5
```

Run both temporary checkpoints on one image:

```powershell
python -m modules.inference.predict_baselines `
  --image path/to/image.jpg `
  --depth-checkpoint checkpoints/depth_baseline.pt `
  --segmentation-checkpoint checkpoints/segmentation_baseline.pt `
  --output outputs/baseline_prediction
```

For experiments where a temporary model learns reversed mask polarity, add
`--invert-foreground-probability`. The optional `--use-foreground-as-depth`
flag exports that heuristic signal as inverse depth; it is not geometric depth.

Depth learns unitless relative inverse depth, where larger values are nearer.
Segmentation learns a binary foreground mask. Checkpoints contain a schema
version, task name, model configuration, weights, optimizer state, epoch, step,
and metrics. Load them with `DepthEstimator.from_checkpoint(...)` or
`SegmentationEstimator.from_checkpoint(...)`.

## Constructor handoff

`constructor_handoff.py` converts one aligned RGB image, `DepthEstimate`,
`SegmentationEstimate`, and stream-owned `ConstructionFrameMetadata` into the
Splat Constructor's `ConstructionFrame` v1. It preserves frame/source IDs,
timestamp, intrinsics, and camera pose; rejects resolution mismatches; and
uses depth confidence only as `depth_validity`. It does not reinterpret the
temporary segmentation-confidence margin as a calibrated quality weight.
