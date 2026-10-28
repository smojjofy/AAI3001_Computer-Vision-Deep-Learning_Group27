import os
import sys
from pathlib import Path

import cv2
import numpy as np

from .contracts import (
    DepthEstimation,
    DepthFrameResult,
)
from .normalize import normalize_prediction


DEPTH_ROOT = Path(__file__).resolve().parents[1]          # .../modules/inference/depth
MODELS_DIR = DEPTH_ROOT / "models"

# Where the official repo (github.com/DepthAnything/Depth-Anything-V2) is cloned.
# find_repo() also checks other common places. Override with the DEPTH_ANYTHING_REPO env var.
DEFAULT_REPO_PATH = DEPTH_ROOT / "third_party" / "Depth-Anything-V2"

# Architecture settings for the official checkpoints (from the repo's run.py).
ENCODER_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}

# Backbone width -> encoder name, for checkpoints whose filename doesn't say.
EMBED_DIM_TO_ENCODER = {384: "vits", 768: "vitb", 1024: "vitl", 1536: "vitg"}

# The model works on images whose sides are multiples of the ViT patch size.
PATCH_SIZE = 14
DEFAULT_INPUT_SIZE = 518


def _has_code(folder: Path, metric: bool) -> bool:
    root = folder / "metric_depth" if metric else folder
    return (root / "depth_anything_v2" / "dpt.py").is_file()


def find_repo(metric: bool = False, repo_path: str | Path | None = None) -> tuple[Path | None, list[Path]]:
    """Locate a clone of github.com/DepthAnything/Depth-Anything-V2.

    Checks, in order: the explicit repo_path, the DEPTH_ANYTHING_REPO env var,
    third_party/, and then a folder named Depth-Anything-V2 (or -main, from a
    zip download) next to this module or in any parent folder up to the drive
    root, plus the current working directory. Returns (found, places_checked).
    """

    candidates: list[Path] = []
    for explicit in (repo_path, os.environ.get("DEPTH_ANYTHING_REPO")):
        if explicit:
            candidates.append(Path(explicit))
    candidates.append(DEFAULT_REPO_PATH)

    for base in [DEPTH_ROOT, *DEPTH_ROOT.parents, Path.cwd(), *Path.cwd().parents]:
        for sub in ("", "third_party"):
            parent = base / sub if sub else base
            candidates += [parent / "Depth-Anything-V2", parent / "Depth-Anything-V2-main"]

    checked = []
    for folder in dict.fromkeys(c.resolve() for c in candidates):   # de-duplicate, keep order
        checked.append(folder)
        if _has_code(folder, metric):
            return folder, checked
    return None, checked


def find_default_model() -> str:
    """First .pth in models/ if there is one, otherwise "" (DepthEstimator then explains what's missing)."""

    checkpoints = sorted(MODELS_DIR.glob("*.pth")) if MODELS_DIR.is_dir() else []
    return str(checkpoints[0]) if checkpoints else ""


# Kept for code that imports it; DepthEstimator looks up the default at creation time instead,
# so a .pth added while the app is running is still found.
DEFAULT_MODEL = find_default_model()


def resolve_device(device: str | None) -> str:
    """'auto'/None -> best available. Otherwise checks the requested device exists."""

    import torch

    if device in (None, "", "auto"):
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    device = str(device).lower()

    if device == "cpu":
        return device

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError(
                "device='cuda' requested but PyTorch can't see a CUDA GPU. Install the CUDA build "
                "of torch from pytorch.org, or use device='cpu'."
            )
        index = int(device.split(":")[1]) if ":" in device else 0
        if index >= torch.cuda.device_count():
            raise ValueError(f"device='{device}' requested but only {torch.cuda.device_count()} GPU(s) found")
        return device

    if device == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise ValueError("device='mps' requested but Apple Metal isn't available on this machine")
        return device

    raise ValueError(f"unknown device '{device}' (use 'auto', 'cpu', 'cuda', 'cuda:N' or 'mps')")


class DepthEstimator:
    """Depth Anything V2 estimator for single RGB frames.

    checkpoint:      official .pth from the Depth-Anything-V2 repo; default = first .pth in models/.
                     e.g. depth_anything_v2_vits.pth (relative) or
                     depth_anything_v2_metric_hypersim_vits.pth (metric). Needs the repo cloned.
    device:          "auto" (GPU if available), "cpu", "cuda", "cuda:1", "mps".
    input_size:      resolution the model runs at (short side, rounded to a multiple of 14).
                     Lower = faster, less detail. 518 is what the model was trained at.
    output_size:     (height, width) of the returned depth; None = same as the input frame.
    half_precision:  use fp16 on GPU (about 2x faster). Ignored on CPU.
    metric:          force metric (True) / relative (False). None = decide from the filename.
    max_depth:       metric models' cap in metres (default 20 indoor, 80 for vkitti checkpoints).
    valid_range:     optional (min_m, max_m) for metric depth; outside -> invalid.

    Output always follows the shared depth convention (DEPTH_CONVENTION.md).
    Depth Anything has no tracking / video mode (unlike MediaPipe), so one
    class handles both single images and sequential video frames.
    """

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        device: str | None = "auto",
        input_size: int = DEFAULT_INPUT_SIZE,
        output_size: tuple[int, int] | None = None,
        half_precision: bool = True,
        repo_path: str | Path | None = None,
        metric: bool | None = None,
        max_depth: float | None = None,
        valid_range: tuple[float, float] | None = None,
    ) -> None:
        # Imported here so the contracts can be used without torch installed.
        import torch

        self._torch = torch

        checkpoint = Path(checkpoint) if checkpoint else Path(find_default_model() or MODELS_DIR)

        if checkpoint.suffix != ".pth" or not checkpoint.is_file():
            raise FileNotFoundError(
                f"No Depth Anything checkpoint at: {checkpoint}\n"
                f"Put a .pth (e.g. depth_anything_v2_vits.pth) in: {MODELS_DIR}"
            )

        if not isinstance(input_size, int) or input_size < PATCH_SIZE:
            raise ValueError(f"input_size must be an integer >= {PATCH_SIZE}, got {input_size!r}")

        if output_size is not None:
            if len(output_size) != 2 or min(output_size) < 1:
                raise ValueError("output_size must be (height, width) with both >= 1, or None")
            output_size = (int(output_size[0]), int(output_size[1]))

        self.device = resolve_device(device)
        self.model_name = str(checkpoint)
        self.input_size = input_size
        self.output_size = output_size
        self.valid_range = valid_range

        # fp16 roughly doubles speed on GPU; CPU stays fp32.
        self._dtype = (
            torch.float16
            if half_precision and self.device.startswith("cuda")
            else torch.float32
        )

        self._load_official(checkpoint, repo_path, metric, max_depth)

        self._model = self._model.to(device=self.device, dtype=self._dtype).eval()

    def _load_official(
        self,
        checkpoint: Path,
        repo_path: str | Path | None,
        metric: bool | None,
        max_depth: float | None,
    ) -> None:
        torch = self._torch

        name = checkpoint.name.lower()
        is_metric = ("metric" in name) if metric is None else bool(metric)

        repo, checked = find_repo(is_metric, repo_path)

        if repo is None:
            shown = "\n".join(f"  {p}" for p in checked[:8])
            raise FileNotFoundError(
                "Official Depth Anything V2 code not found. Looked in:\n"
                f"{shown}\n  ...\n"
                "Fix: set DEPTH_ANYTHING_REPO to the folder you cloned (the one that "
                "contains depth_anything_v2\\dpt.py), or clone it with:\n"
                f"  git clone https://github.com/DepthAnything/Depth-Anything-V2 \"{DEFAULT_REPO_PATH}\""
            )

        # Relative and metric models live in different folders of the repo,
        # both as a package called `depth_anything_v2`.
        package_root = repo / "metric_depth" if is_metric else repo
        self.repo_path = repo

        # Both folders use the same package name, so forget any earlier import first.
        for loaded in [m for m in sys.modules if m == "depth_anything_v2" or m.startswith("depth_anything_v2.")]:
            del sys.modules[loaded]

        sys.path.insert(0, str(package_root))
        try:
            from depth_anything_v2.dpt import DepthAnythingV2
            from depth_anything_v2.util.transform import NormalizeImage, PrepareForNet, Resize
        finally:
            sys.path.remove(str(package_root))

        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)

        encoder = next((e for e in ENCODER_CONFIGS if e in name), None)
        if encoder is None:
            embed_dim = state_dict["pretrained.cls_token"].shape[-1]
            encoder = EMBED_DIM_TO_ENCODER[embed_dim]

        config = dict(ENCODER_CONFIGS[encoder])

        self.max_depth = None
        if is_metric:
            # Hypersim = indoor (20 m), Virtual KITTI = outdoor (80 m), per the repo.
            self.max_depth = max_depth or (80.0 if "vkitti" in name else 20.0)
            config["max_depth"] = self.max_depth

        model = DepthAnythingV2(**config)
        model.load_state_dict(state_dict)

        # Same preprocessing as the repo's image2tensor(), built once, without its
        # habit of moving the image to the GPU even when we asked for the CPU.
        self._resize = Resize(
            width=self.input_size,
            height=self.input_size,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=PATCH_SIZE,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        )
        self._normalize = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self._prepare = PrepareForNet()

        self.encoder = encoder
        self._model = model
        self.kind = "metric_depth" if is_metric else "relative_depth"

    # ------------------------------------------------------------ settings

    @property
    def settings(self) -> dict:
        """Everything that affects the output, e.g. for logging or showing in a GUI."""
        return {
            "checkpoint": self.model_name,
            "encoder": self.encoder,
            "kind": self.kind,
            "device": self.device,
            "dtype": str(self._dtype).replace("torch.", ""),
            "input_size": self.input_size,
            "output_size": self.output_size,
            "max_depth": self.max_depth,
            "valid_range": self.valid_range,
            "repo_path": str(self.repo_path),
        }

    # ------------------------------------------------------------ inference

    def predict_raw(self, rgb_frame: np.ndarray) -> np.ndarray:
        """Raw model output at the output resolution, before the shared convention.

        Relative models: disparity (bigger = closer). Metric models: metres.
        Most callers want estimate() instead.
        """

        if rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
            raise ValueError(
                "rgb_frame must have shape [height, width, 3]"
            )

        if rgb_frame.dtype != np.uint8:
            raise ValueError(
                "rgb_frame must use dtype uint8"
            )

        if self._model is None:
            raise RuntimeError("this DepthEstimator was closed")

        torch = self._torch
        height, width = self.output_size or rgb_frame.shape[:2]

        sample = {"image": np.ascontiguousarray(rgb_frame).astype(np.float32) / 255.0}
        sample = self._prepare(self._normalize(self._resize(sample)))
        pixel_values = torch.from_numpy(sample["image"]).unsqueeze(0)
        pixel_values = pixel_values.to(device=self.device, dtype=self._dtype)

        with torch.inference_mode():
            predicted = self._model(pixel_values)

            # Model runs at ~input_size px; resize to the output resolution,
            # the same way the repo's infer_image() does.
            predicted = torch.nn.functional.interpolate(
                predicted.unsqueeze(1).float(),
                size=(height, width),
                mode="bilinear",
                align_corners=True,
            )

        return predicted[0, 0].cpu().numpy().astype(np.float32)

    def estimate(self, rgb_frame: np.ndarray) -> DepthEstimation:
        """Estimate dense depth from one RGB uint8 frame, in the shared depth convention."""

        return normalize_prediction(
            self.predict_raw(rgb_frame),
            kind=self.kind,
            model_name=self.model_name,
            max_depth=self.max_depth,
            valid_range=self.valid_range,
        )

    def estimate_frame(
        self,
        rgb_frame: np.ndarray,
        frame_id: int,
        source_id: str,
        capture_timestamp_ns: int,
    ) -> DepthFrameResult:
        """Estimate depth while preserving source-frame identity."""

        if frame_id < 0:
            raise ValueError(
                "frame_id must be non-negative"
            )

        if not source_id:
            raise ValueError(
                "source_id must not be empty"
            )

        if capture_timestamp_ns < 0:
            raise ValueError(
                "capture_timestamp_ns must be non-negative"
            )

        return DepthFrameResult(
            frame_id=frame_id,
            source_id=source_id,
            capture_timestamp_ns=capture_timestamp_ns,
            depth=self.estimate(rgb_frame),
        )
# Closes the estimator and frees up resources. If the estimator is using a GPU, it also clears the CUDA cache to free up GPU memory. This is important for managing resources, especially when working with large models or processing multiple images in sequence. The close method should be called when the DepthEstimator is no longer needed to ensure that resources are properly released.
    def close(self) -> None:
        self._model = None
# Empty cache because the model may have been using GPU memory, and we want to free that up when the estimator is closed. This is especially important in scenarios where multiple models or large datasets are being processed, as it helps prevent out-of-memory errors and ensures that GPU resources are available for other tasks.
        if self.device.startswith("cuda"):
            self._torch.cuda.empty_cache()
# The __enter__ method is part of the context management protocol in Python. It allows the DepthEstimator to be used in a with statement, which ensures that resources are automatically cleaned up when the block of code is exited. When the with block is entered, the __enter__ method is called, and it returns the DepthEstimator instance itself, allowing it to be used within the with block.
    def __enter__(self) -> "DepthEstimator":
        return self
# The __exit__ method is part of the context management protocol in Python. It allows the DepthEstimator to be used in a with statement, ensuring that resources are automatically cleaned up when the block of code is exited. When the with block is exited, whether normally or due to an exception, the __exit__ method is called, which in turn calls the close method to release resources.
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
