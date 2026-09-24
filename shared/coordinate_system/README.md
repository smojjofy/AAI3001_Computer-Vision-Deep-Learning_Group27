# Coordinate Convention

The Splat Constructor uses a right-handed, Three.js-compatible camera/world
frame for the single-camera MVP:

- Image origin is top-left; pixel `u` increases right and `v` increases down.
- World `+X` points right, `+Y` points up, and camera forward is world `-Z`.
- Input depth is a positive distance along the camera viewing direction.
- Camera space equals world space until camera poses are introduced.

Back-projection is therefore:

```text
X =  (u - cx) * depth / fx
Y = -(v - cy) * depth / fy
Z = -depth
```

Camera matrices are row-major, right-handed rigid `camera_to_world` transforms. For intrinsics
derived from horizontal field of view, the constructor assumes square pixels
(`fy = fx`) and places the principal point at `((W-1)/2, (H-1)/2)`. Consumers
must preserve this half-pixel convention when reprojecting constructor output.

Gaussian rotations are unit quaternions stored in `wxyz` order. Scale values in
standard 3DGS PLY files are natural logarithms; opacity values are logits.
