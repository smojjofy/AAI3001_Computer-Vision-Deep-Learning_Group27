# SplatState v1

`SplatState` is the in-memory boundary from Splat Constructor to Temporal
Cache. PLY and metadata JSON are diagnostic/export artifacts, not the runtime
contract.

The state preserves `frame_id`, `timestamp_ns`, `source_id`, source dimensions,
camera intrinsics, row-major `camera_to_world`, and source depth semantics and
units. It owns one structure-of-arrays Gaussian collection:

| Attribute | Shape/type | Meaning |
|---|---|---|
| `positions` | `M x 3` float32 | World-space XYZ |
| `sh_dc` | `M x 3` float32 | Degree-zero RGB spherical-harmonic coefficients |
| `opacity_logits` | `M` float32 | Opacity before sigmoid |
| `log_scales` | `M x 3` float32 | Natural-log anisotropic scales |
| `rotations` | `M x 4` float32 | Unit quaternions in `wxyz` order |
| `reconstruction_weights` | `M` float32 | Validity/foreground-derived weight in `[0,1]` |
| `local_ids` | `M` uint64 | Deterministic ID within one frame; currently source pixel index |
| `source_pixels` | `M x 2` uint32 | Source image `(u,v)` association |

The identity of a frame-local candidate is the tuple
`(source_id, frame_id, local_id)`. Persistent cross-frame IDs are assigned by
the Temporal Cache after matching/fusion; the constructor does not claim that
the same pixel identifies the same physical point over time.

Constructor positions are transformed into world space by `camera_to_world`.
The initial opacity incorporates `reconstruction_weights`; the explicit weight
is retained so downstream fusion/cleaning does not need to infer it from opacity.
