# Reconstruction Validation

This module validates a `SplatState` against the `ConstructionFrame` that
produced it. It is intentionally separate from the Splat Constructor and does
not change its state.

`validate_source_view` in `source_view.py` inverts the frame's rigid camera
pose, projects Gaussian centers to the source camera, and z-buffers the nearest
center at each pixel. It reports reprojection RMSE, RGB/depth error on covered
pixels, mask IoU, and foreground coverage.

This is a deterministic **point-center** rasterizer for validation. It does not
replace a full elliptical 3D Gaussian renderer; stride-two construction is
therefore expected to cover roughly one quarter of foreground pixels.

Run the current aligned fixture after building the native bridge:

```powershell
python -m scripts.validate_splat_source_view `
  --rgb test_data/gaussian_splatting/FO_dataset/train/traintest_rgb.jpg `
  --depth outputs/traintest_inverted_050/relative_inverse_depth.png `
  --mask outputs/traintest_inverted_050/foreground_mask.png `
  --frame-id 0001 --source-id 0001 --timestamp-ns 1 `
  --depth-semantics pseudo_inverse_from_segmentation --stride 2
```
