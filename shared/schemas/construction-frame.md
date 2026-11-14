# ConstructionFrame v1

`ConstructionFrame` is the language-neutral logical contract consumed by the
Splat Constructor. An in-memory binding may choose its own tensor/container
representation, but it must preserve these fields and validation rules.

| Field | Type | Rule |
|---|---|---|
| `schema_version` | unsigned integer | Exactly `1` |
| `frame_id` | string | Non-empty; formatting such as `0001` is preserved |
| `timestamp_ns` | unsigned 64-bit integer | Positive, monotonic source timestamp in nanoseconds |
| `source_id` | string | Non-empty camera/stream identifier |
| `rgb` | `H x W x 3` uint8 | RGB channel order |
| `depth` | `H x W` float32 | Finite; constraints depend on semantics |
| `foreground_mask` | `H x W` uint8 | Binary `{0,1}`; `1` means constructible foreground |
| `depth_validity` | optional `H x W` float32 | Validity weight in `[0,1]`; not calibrated confidence |
| `foreground_weight` | optional `H x W` float32 | Soft foreground weight in `[0,1]` |
| `intrinsics` | four float32 values | `fx`, `fy` positive; `cx`, `cy` in pixel coordinates |
| `depth_semantics` | enum | `relative_inverse`, `metric_camera_z`, or `pseudo_inverse_from_segmentation` |
| `depth_units` | enum | `unitless` for relative depth; `meters` for metric camera-Z |
| `camera_to_world` | 16 float32 values | Row-major, right-handed rigid 4x4 transform; identity for the static-camera MVP |

Relative inverse depth is normalized to `[0,1]`, where larger values are
nearer. Metric camera-Z is non-negative distance along camera forward. The
temporary PPM/PGM CLI is an adapter and is not this contract.

Construction rejects masked metric-depth samples at zero distance and samples
whose combined `depth_validity * foreground_weight` does not exceed the
configured minimum reconstruction weight.
