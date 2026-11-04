from pathlib import Path

import cv2
import numpy as np

from .contracts import DepthEstimation, DepthFrameResult
from .estimator import MODELS_DIR, resolve_device
from .normalize import normalize_prediction

DEFAULT_YOLO_DEPTH = "yolo26n-depth.pt"     # n = fastest; s / m / l are bigger and slower
DEFAULT_YOLO_IMGSZ = 640


class YoloDepthEstimator:
    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_YOLO_DEPTH,
        device: str | None = "auto",
        imgsz: int = DEFAULT_YOLO_IMGSZ,
        output_size: tuple[int, int] | None = None,
        half_precision: bool = True,
        valid_range: tuple[float, float] | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("YOLO depth needs Ultralytics: pip install ultralytics") from e

        if not isinstance(imgsz, int) or imgsz < 32:
            raise ValueError(f"imgsz must be an integer >= 32, got {imgsz!r}")

        if output_size is not None:
            if len(output_size) != 2 or min(output_size) < 1:
                raise ValueError("output_size must be (height, width) with both >= 1, or None")
            output_size = (int(output_size[0]), int(output_size[1]))

        path = Path(checkpoint)
        if not path.is_file():
            path = MODELS_DIR / path.name           # download into models/ instead of the cwd

        self._model = YOLO(str(path))
        if self._model.task != "depth":
            raise ValueError(
                f"{path.name} is a '{self._model.task}' model, not a depth model "
                f"(use a *-depth.pt checkpoint, e.g. {DEFAULT_YOLO_DEPTH})"
            )

        self.device = resolve_device(device)
        self.model_name = str(path)
        self.imgsz = imgsz
        self.output_size = output_size
        self.valid_range = valid_range
        self.kind = "metric_depth"
        self.half = bool(half_precision and self.device.startswith("cuda"))

        # Ultralytics wants "cpu", "mps", or a GPU index like "0".
        if self.device in ("cpu", "mps"):
            ultra_device = self.device
        else:
            ultra_device = self.device.split(":")[1] if ":" in self.device else "0"

        self._predict_kwargs = {"imgsz": self.imgsz, "device": ultra_device, "verbose": False}
        if self.half:
            self._predict_kwargs["half"] = True   # only passed when on: newer versions warn about it

    @property
    def settings(self) -> dict:
        return {
            "checkpoint": self.model_name,
            "kind": self.kind,
            "device": self.device,
            "dtype": "float16" if self.half else "float32",
            "imgsz": self.imgsz,
            "output_size": self.output_size,
            "valid_range": self.valid_range,
        }

    def predict_raw(self, rgb_frame: np.ndarray) -> np.ndarray:
        """Metres per pixel at the output resolution, before the shared convention."""

        if rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
            raise ValueError("rgb_frame must have shape [height, width, 3]")
        if rgb_frame.dtype != np.uint8:
            raise ValueError("rgb_frame must use dtype uint8")
        if self._model is None:
            raise RuntimeError("this YoloDepthEstimator was closed")

        # Ultralytics treats numpy images as BGR (like cv2.imread).
        bgr = np.ascontiguousarray(rgb_frame[:, :, ::-1])
        result = self._model.predict(bgr, **self._predict_kwargs)[0]
        depth = result.depth.data.float().cpu().numpy().astype(np.float32)

        if self.output_size is not None:
            height, width = self.output_size
            depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)
        return depth

    def estimate(self, rgb_frame: np.ndarray) -> DepthEstimation:
        """Estimate metric depth from one RGB uint8 frame, in the shared depth convention."""

        return normalize_prediction(
            self.predict_raw(rgb_frame),
            kind="metric_depth",
            model_name=self.model_name,
            valid_range=self.valid_range,
        )

    def estimate_frame(
        self,
        rgb_frame: np.ndarray,
        frame_id: int,
        source_id: str,
        capture_timestamp_ns: int,
    ) -> DepthFrameResult:
        """Estimate depth while preserving source-frame identity (same checks as DepthEstimator)."""

        if frame_id < 0:
            raise ValueError("frame_id must be non-negative")
        if not source_id:
            raise ValueError("source_id must not be empty")
        if capture_timestamp_ns < 0:
            raise ValueError("capture_timestamp_ns must be non-negative")

        return DepthFrameResult(
            frame_id=frame_id,
            source_id=source_id,
            capture_timestamp_ns=capture_timestamp_ns,
            depth=self.estimate(rgb_frame),
        )

    def close(self) -> None:
        self._model = None
        if self.device.startswith("cuda"):
            import torch
            torch.cuda.empty_cache()

    def __enter__(self) -> "YoloDepthEstimator":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
