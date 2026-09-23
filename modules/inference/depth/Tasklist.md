# Depth Tasklist

Input: normalized `ProcessedFrame`.

Output: relative depth map and per-pixel depth confidence aligned to the input dimensions.

- [x] Draft a compact encoder-decoder CNN and inference boundary.
- [x] Define normalized inverse-depth and confidence outputs.
- [x] Document that output is not metric depth in the single-camera MVP.
- [x] Add temporary APC sample training data and a training command.
- [x] Add masked depth, gradient, and confidence losses.
- [x] Add versioned checkpoint save/load support.
- [ ] Train or distill weights on a broader depth dataset.
- [ ] Calibrate confidence and invalid-depth behavior after training.
- [ ] Benchmark against a pretrained monocular-depth teacher model.
