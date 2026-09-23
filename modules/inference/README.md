# Temporary Inference Baselines

These compact CNNs exist to unblock the Splat Constructor while the production
depth, pose, and mask models are developed. They are replaceable implementations
of the shared inference contract, not target-quality models.

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
