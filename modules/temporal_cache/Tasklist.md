# Temporal Cache Tasklist

Input: `SplatState` from Splat Constructor, plus prior cached states.

Output: `ReconstructionState` for Three.js and future renderers.

- [ ] Maintain timestamped states and bounded history.
- [ ] Warp, smooth, interpolate, and confidence-weight splats using optical flow.
- [ ] Handle dropped frames and stale regions.
- [ ] Support full-state and delta updates.
