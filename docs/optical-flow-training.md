# Optical-flow CNN training plan

## First model and comparison

Use Torchvision RAFT Small with its pretrained `DEFAULT` weights as the first CNN.
The optical-flow module also retains OpenCV DIS. Evaluate both on the same
recorded project footage, processed resolution, and machine before choosing the
live backend. Compare p50/p95 flow latency, GPU memory, valid fraction, motion
alignment, estimator calls saved, viewer frame age, and visible stale regions.
The module's RAFT backend supports a local fine-tuned checkpoint through
`raft_checkpoint`; model weights belong in ignored `checkpoints/`.

## Supervised dataset

Start with a labeled optical-flow dataset such as Sintel's training split. Its
samples include paired images and a ground-truth flow field, and Torchvision
exposes a loader for them. Use clean and final passes for different appearance
conditions. Hold out whole scenes for validation so near-adjacent pairs do not
leak across training and evaluation. FlyingThings3D is another labeled source,
but the default RAFT Small weights were already trained on FlyingChairs and
FlyingThings3D; fine-tuning on a different domain is the more informative first
experiment. Keep dataset archives and extracted frames out of Git.

Recorded webcam footage is useful for live latency and visual evaluation, but
ordinary frames alone do not provide supervised flow labels. To train on the
project's camera domain, obtain ground-truth motion (for example from a
rendered scene with known motion) or label a small validation set. Pseudo-labels
from a stronger model are possible but must be checked against independent
held-out footage; they are not ground truth.

## Paired augmentation

Apply geometric changes to both images and the ground-truth vectors together:

| Change | Required label change |
| --- | --- |
| Shared crop | Crop both images, flow, and validity at the same coordinates. |
| Resize | Resize flow and multiply `dx` by the x scale and `dy` by the y scale; resize validity with nearest neighbor. |
| Horizontal flip | Flip all planes and negate `dx`. |
| Vertical flip | Flip all planes and negate `dy`. |

Use modest color, exposure, noise, blur, and compression changes to represent
capture conditions. These may differ slightly between the two images, but
geometry must remain paired. Mask invalid or occluded labels in the supervised
loss. Do not reverse time unless reverse-direction ground truth is available.

## Training and acceptance

Fine-tune pretrained RAFT Small with a masked sequence loss over its iterative
predictions. Start with a small crop and batch on the available 12 GB GPU, then
increase only after observing memory and throughput. Track validation endpoint
error (EPE), outlier rate, and visual examples; retain the best validation
checkpoint. Run the final model through the same replay benchmark as DIS and
test it concurrently with depth inference and rendering. A lower EPE alone
does not justify a slower live path if frame age or cache alignment worsens.

## Source references

- [Torchvision RAFT Small weights and model](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.optical_flow.raft_small.html)
- [Torchvision Sintel dataset and flow/validity transform contract](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.Sintel.html)
- [Original RAFT training code](https://github.com/princeton-vl/RAFT)
