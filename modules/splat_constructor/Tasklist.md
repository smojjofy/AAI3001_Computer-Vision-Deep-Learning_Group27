# Splat Constructor Tasklist

Input: `ConstructionFrame` v1 from Inference.

Output: `SplatState` v1 containing Gaussian attributes and source provenance.
PLY + JSON remain export artifacts, not the Temporal Cache boundary.

- [x] Convert RGB, mask, and pseudo-depth into initial camera-space Gaussians.
- [x] Assign fixed initial opacity; defer confidence weighting to the shared contract.
- [x] Define preservation of frame ID, timestamp, and source ID in the runtime contract.
- [x] Export a standard binary 3DGS PLY plus construction metadata.
- [x] Add the language-neutral `ConstructionFrame` and `SplatState` contracts.
- [x] Add deterministic frame-local Gaussian IDs and reconstruction weights.
- [x] Wire float32 relative-inverse and metric camera-Z depth into construction.
- [x] Propagate validity/foreground weights and camera-to-world pose.
- [x] Initialize depth-gradient orientation and reduce scale at mask/depth edges.
- [x] Add a NumPy/ctypes in-memory binding to the C++ `ConstructionFrame` entry point.
- [ ] Add a PyTorch/DLPack zero-copy path only if profiling shows the copied NumPy boundary is limiting.
