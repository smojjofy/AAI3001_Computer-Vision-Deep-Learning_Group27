# Splat Constructor

The first implementation is a dependency-free C++20 CPU constructor. It turns
aligned RGB, binary foreground mask, and unitless pseudo inverse-depth images
into a standard binary 3DGS PLY. The implementation uses the coordinate system
documented in `shared/coordinate_system/README.md`.

The runtime boundaries are `ConstructionFrame` v1 and `SplatState` v1,
documented under `shared/schemas/`. The current CLI is a fixture adapter: it
still reads 8-bit PPM/PGM and exports PLY + JSON, while the future in-memory
binding will call the implemented float32 entry point directly. That entry
point supports relative-inverse and metric camera-Z depth, source validity and
foreground weights, rigid camera-to-world poses, depth-gradient orientation,
and edge-aware scale reduction.

## Python bridge

The build script also produces `splat_constructor_bridge.dll`. The Python API
uses `ctypes` and NumPy, so it adds no package beyond the repository's existing
NumPy dependency. It copies input arrays into C++ and copies the completed
`SplatState` back into NumPy; this makes returned arrays independent of native
handle lifetime.

```python
import numpy as np
from modules.splat_constructor import (
    ConstructionFrame, DepthSemantics, DepthUnits, construct_splat_state,
)

frame = ConstructionFrame(
    frame_id="0001", timestamp_ns=1, source_id="0001",
    rgb=np.zeros((480, 640, 3), dtype=np.uint8),
    depth=np.ones((480, 640), dtype=np.float32),
    foreground_mask=np.ones((480, 640), dtype=np.uint8),
    intrinsics=(600.0, 600.0, 319.5, 239.5),
    depth_semantics=DepthSemantics.RELATIVE_INVERSE,
    depth_units=DepthUnits.UNITLESS,
)
state = construct_splat_state(frame)
```

`foreground_mask` must be binary `{0,1}`. Optional `depth_validity` and
`foreground_weight` are float32 `H x W` arrays in `[0,1]`. The legacy CLI
continues to accept 8-bit PPM/PGM fixtures.

## Build and test

The repository includes CMake configuration. On the current Windows workspace,
where CMake is not installed, use the direct MinGW build script:

```powershell
./scripts/build_splat_constructor.ps1
./build/splat_constructor/splat_constructor_tests.exe
```

## Fixed first fixture

Prepare the aligned `traintest_rgb.jpg` fixture from the temporary inference
outputs, then construct the PLY:

```powershell
python ./scripts/prepare_splat_fixture.py `
  --rgb test_data/gaussian_splatting/FO_dataset/train/traintest_rgb.jpg `
  --depth outputs/traintest_inverted_050/relative_inverse_depth.png `
  --mask outputs/traintest_inverted_050/foreground_mask.png `
  --output outputs/splat_constructor_fixture

./build/splat_constructor/splat_constructor_cli.exe `
  --rgb outputs/splat_constructor_fixture/rgb.ppm `
  --depth outputs/splat_constructor_fixture/depth.pgm `
  --mask outputs/splat_constructor_fixture/mask.pgm `
  --output outputs/splat_constructor/traintest_initial.ply `
  --stride 2 --fov 50 --near 1 --far 4
```

PPM/PGM keeps the native constructor independent of image-decoding libraries.
PLY and metadata JSON are debug/export formats, not the Temporal Cache runtime
contract. Writes use same-directory temporary files and atomic replacement so
an interrupted run does not leave a partially written artifact.
