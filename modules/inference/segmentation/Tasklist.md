# Segmentation Tasklist

Input: normalized `ProcessedFrame`.

Output: foreground mask and per-pixel mask confidence aligned to the input dimensions.

- [x] Draft a compact encoder-decoder CNN and inference boundary.
- [x] Define thresholding, foreground probability, and confidence behavior.
- [x] Add temporary APC sample training data and a training command.
- [x] Add BCE and Dice training losses.
- [x] Add versioned checkpoint save/load support.
- [ ] Train or distill weights on a broader foreground dataset.
- [ ] Compare the draft against a pretrained segmentation teacher model.
- [ ] Benchmark quality and latency.
