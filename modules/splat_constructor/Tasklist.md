# Splat Constructor Tasklist

Input: `InferencePacket` from Inference.

Output: `SplatState` containing current Gaussian attributes, RGBA, relative depth, and confidence. The output must match the Temporal Cache input contract exactly.

- [x] Convert RGB, mask, and pseudo-depth into initial camera-space Gaussians.
- [x] Assign fixed initial opacity; defer confidence weighting to the shared contract.
- [ ] Preserve frame ID and timestamp.
- [x] Export a standard binary 3DGS PLY plus construction metadata.
- [ ] Add the language-neutral `ConstructionFrame` handshake after the fixed fixture.
