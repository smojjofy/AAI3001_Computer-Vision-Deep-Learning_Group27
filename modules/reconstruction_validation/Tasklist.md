# Reconstruction Validation Tasklist

Input: `ConstructionFrame` plus `SplatState` from the Splat Constructor.

Output: source-view point raster and quantitative reprojection report. This
module must not modify reconstruction state.

- [x] Project constructed splat centers back to their source camera.
- [x] Rasterize nearest visible centers into RGB, alpha, and depth buffers.
- [x] Report pixel reprojection, RGB, depth, mask, and coverage metrics.
- [ ] Add full elliptical-Gaussian rasterization and visual-regression fixtures.
- [ ] Add novel-view validation when calibrated multi-view data exists.
