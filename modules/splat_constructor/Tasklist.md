# Splat Constructor Tasklist

Input: `InferencePacket` from Inference.

Output: `SplatState` containing current Gaussian attributes, RGBA, relative depth, and confidence. The output must match the Temporal Cache input contract exactly.

- [ ] Convert RGB, mask, and depth into initial camera-space Gaussians.
- [ ] Assign opacity and confidence from mask/depth confidence.
- [ ] Preserve frame ID and timestamp.
