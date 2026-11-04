# import libraries
import sys
import tempfile
from pathlib import Path
from time import perf_counter
 
import cv2
import gradio as gr
import numpy as np
 
# set up path and insert it to sys.path for module imports
DEPTH_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(DEPTH_ROOT))
 
from depth_estimator import DepthEstimator, find_default_model
from depth_estimator.estimator import DEFAULT_INPUT_SIZE
from depth_estimator.utils import colorize, colorize_confidence
 
# The comparison tab reuses the drawing code from scripts/compare_depth_yolo.py
sys.path.insert(0, str(DEPTH_ROOT / "scripts"))
from scripts.compare_depth_yolo import compare_panels
from depth_estimator.yolo_depth import DEFAULT_YOLO_IMGSZ
 
# list the image types that can be used for depth estimation
MODELS_DIR = DEPTH_ROOT / "models"
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
 
_estimators: dict[tuple, DepthEstimator] = {}   # loaded once per (model, device, input size), then reused
_runs: dict[str, int] = {}
 
# Find all available models in directory then sort them, if none found return the default model path
 
def available_models() -> list[str]:
    found = sorted(str(p.relative_to(DEPTH_ROOT)) for p in MODELS_DIR.glob("*.pth")) if MODELS_DIR.is_dir() else []
    return found or [find_default_model()]
 
# Devices offered in the dropdown: "auto" plus whatever this machine actually has.
 
def available_devices() -> list[str]:
    devices = ["auto", "cpu"]
    try:
        import torch
        devices += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            devices.append("mps")
    except ImportError:
        pass
    return devices
 
# Get DepthEstimator for a model + settings, then load it if not already loaded.
 
def get_estimator(model: str, device: str, input_size: int) -> DepthEstimator:
    path = DEPTH_ROOT / model
    checkpoint = str(path) if path.exists() else model
    key = (checkpoint, device, int(input_size))
    if key not in _estimators:
        _estimators[key] = DepthEstimator(checkpoint, device=device, input_size=int(input_size))
    return _estimators[key]
 
# Shrink big photos before processing. The models shrink the image anyway, but everything after the model (depth convention, confidence, colouring, sending images to the browser) runs at the photo's full size: a 12 MP phone photo is ~4x slower than 1280 px for no visible gain.
DEFAULT_MAX_SIDE = 1280
 
 
def limit_size(image: np.ndarray, max_side) -> tuple[np.ndarray, str]:
    h, w = image.shape[:2]
    max_side = int(max_side or 0)
    if max_side <= 0 or max(h, w) <= max_side:
        return image, ""
    scale = max_side / max(h, w)
    small = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return small, f" (shrunk from {w}×{h})"
 
# Convert BGR to RGB and return results as numpy array
def to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
 
# Run depth estimation using specific model. Check if image is None, then load the estimator and estimate the depth. Return the results as a tuple of (image, view), view, info, npy_path, and state.
 
def run(image, model, device, input_size, max_side=DEFAULT_MAX_SIDE):
    if image is None:
        return None, None, None, "Upload an image first.", None, None
    image, shrunk = limit_size(image, max_side)
 
    try:
        estimator = get_estimator(model, device, input_size)
    except (FileNotFoundError, ValueError) as e:
        return None, None, None, f"**Couldn't load the model.**\n\n```\n{e}\n```", None, None
 
# measure the time with perf_counter() to estimate the depth and store the result in a variable. Calculate the time taken in milliseconds and update the _runs dictionary with the number of runs for that model.
    start = perf_counter()
    result = estimator.estimate(image)
    ms = (perf_counter() - start) * 1000
    run_key = (model, device, int(input_size))
    _runs[run_key] = _runs.get(run_key, 0) + 1
 
    view = to_rgb(colorize(result))
    confidence_view = to_rgb(colorize_confidence(result))
 
    npy_path = Path(tempfile.mkdtemp()) / "depth.npy"
    np.save(npy_path, result.depth)
 
    d = result.depth
    if not result.valid.any():
        meaning = "**no valid depth** in this image (e.g. a blank or black frame)"
        rng = "–"
    elif result.is_metric:
        meaning = "**metres** from the camera (bigger = farther)"
        rng = f"{np.nanmin(d):.2f} m – {np.nanmax(d):.2f} m (median {np.nanmedian(d):.2f} m)"
    else:
        meaning = ("**relative** 0–1 within this image (0 = nearest, 1 = farthest). "
                   "Good for shape and ordering; not real distances. Use a metric checkpoint for metres.")
        rng = f"{np.nanmin(d):.2f} – {np.nanmax(d):.2f} (median {np.nanmedian(d):.2f})"
# info string with model name, kind, device, time taken, values meaning and range
    info = (
        f"**Model:** `{Path(model).name}` · {result.kind} · on **{estimator.device}** "
        f"at input size **{estimator.input_size}**  \n"
        f"{'**Repo code:** `' + str(estimator.repo_path) + '`  ' + chr(10) if getattr(estimator, 'repo_path', None) else ''}"
        f"**Time:** {ms:.0f} ms for {image.shape[1]}×{image.shape[0]} px{shrunk}"
        f"{'  (first run includes warm-up)' if _runs[run_key] == 1 else ''}  \n"
        f"**Values:** {meaning}  \n"
        f"**Range:** {rng}  \n"
        f"**Valid pixels:** {result.valid_fraction * 100:.1f}% · "
        f"**mean confidence:** {result.confidence.mean():.2f}"
    )
    state = {"depth": d, "confidence": result.confidence, "view": view, "metric": result.is_metric}
    return (image, view), view, confidence_view, info, str(npy_path), state
 
# Read a point from the depth map when user clicks, retuns a marked image and text with value of that point
 
def read_point(state, evt: gr.SelectData):
    if not state:
        return None, "Run the estimator first."
    x, y = evt.index
    d = state["depth"]
    y = int(np.clip(y, 0, d.shape[0] - 1))
    x = int(np.clip(x, 0, d.shape[1] - 1))
    window = d[max(y - 2, 0):y + 3, max(x - 2, 0):x + 3]
    window = window[np.isfinite(window)]
    confidence = float(state["confidence"][y, x])
 
    marked = state["view"].copy()
    r = max(4, min(d.shape) // 80)
    cv2.circle(marked, (x, y), r + 3, (255, 255, 255), 2)
    cv2.circle(marked, (x, y), r, (0, 0, 0), -1)
 
# If the depth values are in metric units, format the text to show the value in meters. Otherwise, format it as a relative value with three decimal places. Return the marked image and the formatted text.
    if window.size == 0:
        text = f"**({x}, {y}) → no valid depth here**"
    elif state["metric"]:
        text = f"**({x}, {y}) → {float(np.median(window)):.2f} m** from the camera · confidence {confidence:.2f}"
    else:
        text = (f"**({x}, {y}) → {float(np.median(window)):.3f}** (relative 0–1; bigger = farther) "
                f"· confidence {confidence:.2f}")
    return marked, text
 
# Depth Anything VS YOLO depth comparison. The YOLO depth models are listed in the YOLO_DEPTH_MODELS variable, and the get_yolo_depth function loads a YOLO depth model once and caches it for reuse. The run_compare function runs both Depth Anything and YOLO depth on the same image, compares their outputs, and returns the results along with performance statistics.
 
YOLO_DEPTH_MODELS = ["yolo26n-depth.pt", "yolo26s-depth.pt", "yolo26m-depth.pt"]   # n = fastest
_yolo_depths: dict[tuple, object] = {}   # loaded once per (model, device, imgsz), then reused
 
 
# Load a YOLO depth model once. Weights download into models/ on first use (git-ignored).
def get_yolo_depth(name: str, device: str, imgsz: int):
    from depth_estimator import YoloDepthEstimator
    key = (name, device, int(imgsz))
    if key not in _yolo_depths:
        _yolo_depths[key] = YoloDepthEstimator(name, device=device, imgsz=int(imgsz))
    return _yolo_depths[key]
 
 
# Explain an import failure honestly: either ultralytics itself is missing, or the depth_estimator files are out of date.
def import_error_message(e: ImportError) -> str:
    return (f"**Couldn't import YOLO depth.**\n\n```\n{e}\n```\n\n"
            "If this says `ultralytics` is missing: `pip install ultralytics`. If it mentions "
            "`YoloDepthEstimator` or `yolo_depth`: the updated `depth_estimator` files aren't in this folder.")
 
 
# Run YOLO depth ONLY (no Depth Anything), so its time is all you wait for. Same outputs as the first tab: slider, depth map (click to read metres), confidence, .npy, info.
def run_yolo(image, yolo_name, device, yolo_imgsz, max_side=DEFAULT_MAX_SIDE):
    if image is None:
        return None, None, None, "Upload an image first.", None, None
    image, shrunk = limit_size(image, max_side)
 
    try:
        estimator = get_yolo_depth(yolo_name, device, yolo_imgsz)
    except ImportError as e:
        return None, None, None, import_error_message(e), None, None
    except Exception as e:  # not a depth model, failed download, unknown device, ...
        return None, None, None, f"**Couldn't load YOLO depth.**\n\n```\n{e}\n```", None, None
 
    start = perf_counter()
    result = estimator.estimate(image)
    ms = (perf_counter() - start) * 1000
    run_key = ("yolo", yolo_name, device, int(yolo_imgsz))
    _runs[run_key] = _runs.get(run_key, 0) + 1
 
    view = to_rgb(colorize(result))
    confidence_view = to_rgb(colorize_confidence(result))
    npy_path = Path(tempfile.mkdtemp()) / "yolo_depth.npy"
    np.save(npy_path, result.depth)
 
    d = result.depth
    rng = (f"{np.nanmin(d):.2f} m – {np.nanmax(d):.2f} m (median {np.nanmedian(d):.2f} m)"
           if result.valid.any() else "–")
    info = (
        f"**Model:** `{Path(yolo_name).name}` · metric_depth · on **{estimator.device}** "
        f"at input size **{estimator.imgsz}**  \n"
        f"**Time:** {ms:.0f} ms for {image.shape[1]}×{image.shape[0]} px{shrunk}"
        f"{'  (first run includes warm-up)' if _runs[run_key] == 1 else ''}  \n"
        f"**Values:** **metres** from the camera (bigger = farther). Click the depth map to read a distance.  \n"
        f"**Range:** {rng}  \n"
        f"**Valid pixels:** {result.valid_fraction * 100:.1f}% · "
        f"**mean confidence:** {result.confidence.mean():.2f}"
    )
    state = {"depth": d, "confidence": result.confidence, "view": view, "metric": True}
    return (image, view), view, confidence_view, info, str(npy_path), state
 
 
# Run Depth Anything and YOLO depth on the same image. Both depth maps are shown on the same colour scale (metres; Depth Anything's relative output is scaled to YOLO's metres for display), plus where they disagree and how fast each was.
def run_compare(image, model, device, input_size, yolo_name, yolo_imgsz, max_side=DEFAULT_MAX_SIDE):
    nothing = (None, None, None, None)
    if image is None:
        return *nothing, "Upload an image first."
    image, shrunk = limit_size(image, max_side)
 
    try:
        depth_anything = get_estimator(model, device, input_size)
    except (FileNotFoundError, ValueError) as e:
        return *nothing, f"**Couldn't load Depth Anything.**\n\n```\n{e}\n```"
 
    try:
        yolo = get_yolo_depth(yolo_name, depth_anything.device, yolo_imgsz)
    except ImportError as e:
        return *nothing, import_error_message(e)
    except Exception as e:  # not a depth model, failed download, ...
        return *nothing, f"**Couldn't load YOLO depth.**\n\n```\n{e}\n```"
 
    import torch
    sync = torch.cuda.synchronize if depth_anything.device.startswith("cuda") else (lambda: None)
    panels, stats = compare_panels(image, depth_anything, yolo, sync)
 
    da_ms, yolo_ms = stats["depth_anything_ms"], stats["yolo_ms"]
    faster, ratio = ("YOLO depth", da_ms / yolo_ms) if yolo_ms < da_ms else ("Depth Anything", yolo_ms / da_ms)
    near, far = stats["colour_range_m"]
    lo, hi = stats["yolo_range_m"]
    scaled = ("" if stats["depth_anything_kind"] == "metric_depth" else
              "  \nDepth Anything's output is relative (no units), so for display its overall scale was "
              "fitted to YOLO's metres. Its *shape* is unchanged; the ordering number needs no fitting.")
    info = (
        f"**Depth Anything V2:** {da_ms:.0f} ms · **YOLO depth ({Path(yolo_name).stem}):** {yolo_ms:.0f} ms "
        f"· on **{depth_anything.device}** → **{faster} is {ratio:.1f}× faster** "
        f"· {image.shape[1]}×{image.shape[0]} px{shrunk}  \n"
        f"**Near/far ordering agrees:** {stats['ordering_agreement'] * 100:.0f}% of pixel pairs  \n"
        f"**Median disagreement** (after matching scale): {stats['median_disagreement'] * 100:.0f}%  \n"
        f"**YOLO depth range:** {lo:.1f}–{hi:.1f} m · colours: red {near:.1f} m → blue {far:.1f} m"
        + scaled
    )
    slider = (to_rgb(panels["depth_anything"]), to_rgb(panels["yolo"]))
    return (slider, to_rgb(panels["depth_anything"]), to_rgb(panels["yolo"]),
            to_rgb(panels["difference"]), info)
 
 
# Create a custom theme with white background for the Gradio interface
 
theme = gr.themes.Default(
    primary_hue=gr.themes.colors.green,
).set(
    body_background_fill="#ffffff",
    block_background_fill="#ffffff",
)
# Demonstration interface using gradio via Blocks similar to div in HTML, with a title and description, and a row with two columns: one for input and controls, and one for output and results.
 
# Gradio 6 moved `theme` from gr.Blocks(...) to demo.launch(...); support both.
GRADIO_6 = int(gr.__version__.split(".")[0]) >= 6
theme_in_blocks = {} if GRADIO_6 else {"theme": theme}
theme_in_launch = {"theme": theme} if GRADIO_6 else {}
 
with gr.Blocks(title="Depth Estimator", **theme_in_blocks) as demo:
    gr.Markdown(
        "# Depth Anything V2: depth estimator\n"
        "Upload a photo in .jpg or .png format. "
    )
    max_side = gr.Slider(0, 4096, value=DEFAULT_MAX_SIDE, step=64,
                         label="Max image side (px): bigger photos are shrunk to this first. "
                               "Lower = faster; 0 = never shrink")
    with gr.Tabs():
        with gr.Tab("Depth estimator"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_in = gr.Image(label="Input image", type="numpy", image_mode="RGB",
                                        sources=["upload", "clipboard", "webcam"])
                    models = available_models()
                    model = gr.Dropdown(models, value=models[0], label="Model (.pth files in models/)",
                                        allow_custom_value=True)
                    with gr.Row():
                        device = gr.Dropdown(available_devices(), value="auto", label="Device")
                        input_size = gr.Slider(252, 1022, value=DEFAULT_INPUT_SIZE, step=14,
                                               label="Model input size (lower = faster)")
                    go = gr.Button("Estimate depth", variant="primary")
                    info = gr.Markdown()
                    npy = gr.File(label="Depth (.npy, float32, NaN = invalid)")
                with gr.Column(scale=2):
                    compare = gr.ImageSlider(label="Drag: image ↔ depth", max_height=520)
                    with gr.Row():
                        depth_view = gr.Image(label="Depth map (click to read a value)", interactive=False)
                        confidence_view = gr.Image(label="Confidence (white = trusted, dark = edges/invalid)",
                                                   interactive=False)
                    point = gr.Markdown()
        # Set up state, outputs, and event handlers for the Gradio interface. The state is used to store the depth map and other relevant information, while the outputs are the results of the depth estimation process. The event handlers are set up to trigger the run function when the user clicks the "Estimate depth" button, uploads an image
 
            state = gr.State()
            inputs = [image_in, model, device, input_size, max_side]
            outputs = [compare, depth_view, confidence_view, info, npy, state]
            go.click(run, inputs, outputs)
            image_in.upload(run, inputs, outputs)
            depth_view.select(read_point, state, [depth_view, point])
 
        with gr.Tab("YOLO depth"):
            with gr.Row():
                with gr.Column(scale=1):
                    yo_image = gr.Image(label="Input image", type="numpy", image_mode="RGB",
                                        sources=["upload", "clipboard", "webcam"])
                    yo_name = gr.Dropdown(YOLO_DEPTH_MODELS, value=YOLO_DEPTH_MODELS[0], allow_custom_value=True,
                                          label="YOLO depth model (n = fastest, m = most accurate)")
                    with gr.Row():
                        yo_device = gr.Dropdown(available_devices(), value="auto", label="Device")
                        yo_imgsz = gr.Slider(256, 1024, value=DEFAULT_YOLO_IMGSZ, step=32,
                                             label="YOLO input size (lower = faster)")
                    yo_go = gr.Button("Estimate depth", variant="primary")
                    yo_info = gr.Markdown()
                    yo_npy = gr.File(label="Depth (.npy, float32, metres, NaN = invalid)")
                with gr.Column(scale=2):
                    yo_compare = gr.ImageSlider(label="Drag: image ↔ depth", max_height=520)
                    with gr.Row():
                        yo_depth_view = gr.Image(label="Depth map (click to read metres)", interactive=False)
                        yo_confidence_view = gr.Image(label="Confidence (white = trusted, dark = edges/invalid)",
                                                      interactive=False)
                    yo_point = gr.Markdown()
 
            yo_state = gr.State()
            yo_inputs = [yo_image, yo_name, yo_device, yo_imgsz, max_side]
            yo_outputs = [yo_compare, yo_depth_view, yo_confidence_view, yo_info, yo_npy, yo_state]
            yo_go.click(run_yolo, yo_inputs, yo_outputs)
            yo_image.upload(run_yolo, yo_inputs, yo_outputs)
            yo_depth_view.select(read_point, yo_state, [yo_depth_view, yo_point])
 
if __name__ == "__main__":
    demo.launch(server_port=7861, inbrowser=True, **theme_in_launch)
 
