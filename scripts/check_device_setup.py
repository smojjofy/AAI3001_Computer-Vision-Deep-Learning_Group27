"""Setup checks used by scripts/setup_device.ps1.

The calls live in this file because Windows PowerShell 5.1 removes the inner
double quotes of a native ``python -c`` argument. A script path has no quoting
problem, so the setup checks are reliable.
"""

from __future__ import annotations

import argparse
import sys


def check_python() -> int:
    if sys.version_info < (3, 11):
        print(f"Python 3.11 or newer is required; found {sys.version.split()[0]}")
        return 1
    print(f"Python {sys.version.split()[0]} is supported")
    return 0


def check_imports() -> int:
    import cv2
    import optical_flow  # noqa: F401
    import reranker  # noqa: F401
    import stream_handler  # noqa: F401

    print(f"Setup OK; OpenCV {cv2.__version__}")
    return 0


def check_torch() -> int:
    try:
        import torch
        import torchvision
    except ImportError as error:
        print(f"PyTorch or Torchvision is not installed: {error}")
        return 1
    print(
        "PyTorch",
        torch.__version__,
        "Torchvision",
        torchvision.__version__,
        "CUDA available",
        torch.cuda.is_available(),
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("python", "imports", "torch"))
    args = parser.parse_args()
    if args.mode == "python":
        return check_python()
    if args.mode == "imports":
        return check_imports()
    return check_torch()


if __name__ == "__main__":
    raise SystemExit(main())
