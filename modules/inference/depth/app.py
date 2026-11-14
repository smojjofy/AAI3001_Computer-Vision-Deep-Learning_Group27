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
POSE_ROOT = DEPTH_ROOT.parent / "pose"
sys.path.insert(0, str(DEPTH_ROOT))                  # so "from src import ..." works from any folder
sys.path.insert(0, str(POSE_ROOT / "src"))           # pose_estimator package (teammate's MediaPipe module)
 
from src.pose_depth import draw_pose_depth, fuse_pose_depth, joint_table
from src.utils import colorize, colorize_confidence
from src.yolo_depth import DEFAULT_YOLO_IMGSZ
 
_runs: dict[tuple, int] = {}   # how many times each model has run (first run includes warm-up)
 
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
 
# Shrink big photos before processing. The model shrinks the image anyway, but everything after the model (depth convention, confidence, colouring, sending images to the browser) runs at the photo's full size: a 12 MP phone photo is ~4x slower than 1280 px for no visible gain.
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
 
# Read a point from the depth map when user clicks, returns a marked image and text with value of that point
 
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
 
    if window.size == 0:
        text = f"**({x}, {y}) → no valid depth here**"
    else:
        text = f"**({x}, {y}) → {float(np.median(window)):.2f} m** from the camera · confidence {confidence:.2f}"
    return marked, text
 
# YOLO depth. Only n is in models/; s and m download there when first picked. get_yolo_depth loads a model once and caches it for reuse.
 
YOLO_DEPTH_MODELS = ["yolo26n-depth_openvino_model","yolo26n-depth.pt", "yolo26s-depth.pt", "yolo26m-depth.pt"]   # n = fastest
_yolo_depths: dict[tuple, object] = {}   # loaded once per (model, device, imgsz), then reused
 
 
def get_yolo_depth(name: str, device: str, imgsz: int):
    from src import YoloDepthEstimator
    key = (name, device, int(imgsz))
    if key not in _yolo_depths:
        _yolo_depths[key] = YoloDepthEstimator(name, device=device, imgsz=int(imgsz))
    return _yolo_depths[key]
 
 
# Explain an import failure honestly: either ultralytics itself is missing, or the src files are out of date.
def import_error_message(e: ImportError) -> str:
    return (f"**Couldn't import YOLO depth.**\n\n```\n{e}\n```\n\n"
            "If this says `ultralytics` is missing: `pip install ultralytics`. If it mentions "
            "`YoloDepthEstimator` or `yolo_depth`: the updated `src` files aren't in this folder.")
 
 
# Run YOLO depth: slider, depth map (click to read metres), confidence, .npy, info.
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
    state = {"depth": d, "confidence": result.confidence, "view": view}
    return (image, view), view, confidence_view, info, str(npy_path), state
 
 
# Pose + depth: MediaPipe Lite finds the body joints, YOLO depth says how far each one is.
POSE_MODEL = POSE_ROOT / "models" / "pose_landmarker_lite.task"
_pose_estimators: dict[str, object] = {}
 
 
def get_pose_estimator(model_path: str):
    from pose_estimator import PoseEstimator         # needs mediapipe
    if model_path not in _pose_estimators:
        _pose_estimators[model_path] = PoseEstimator(model_path)
    return _pose_estimators[model_path]
 
 
def run_pose_depth(image, yolo_name, device, yolo_imgsz, max_side=DEFAULT_MAX_SIDE):
    nothing = (None, None, None)
    if image is None:
        return *nothing, "Upload an image first."
    image, shrunk = limit_size(image, max_side)
 
    try:
        pose_est = get_pose_estimator(str(POSE_MODEL))
    except ImportError as e:
        return *nothing, f"**Couldn't import the pose estimator.**\n\n```\n{e}\n```\n\n`pip install mediapipe`"
    except FileNotFoundError:
        return *nothing, (f"**Pose model missing.** Download `pose_landmarker_lite.task` from the MediaPipe "
                          f"Pose Landmarker page and put it in:\n\n`{POSE_MODEL.parent}`")
    try:
        depth_est = get_yolo_depth(yolo_name, device, yolo_imgsz)
    except ImportError as e:
        return *nothing, import_error_message(e)
    except Exception as e:
        return *nothing, f"**Couldn't load YOLO depth.**\n\n```\n{e}\n```"
 
    frame = {"frame_id": 0, "source_id": "gradio-upload", "capture_timestamp_ns": 0}
    image = np.ascontiguousarray(image)
    t0 = perf_counter()
    pose = pose_est.estimate_frame(image, **frame)
    t1 = perf_counter()
    depth = depth_est.estimate_frame(image, **frame)
    t2 = perf_counter()
    result = fuse_pose_depth(pose, depth)
 
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    on_photo = to_rgb(draw_pose_depth(bgr, result))
    on_depth = to_rgb(draw_pose_depth(colorize(result.depth), result))
    if result.detected:
        found = sum(row[5] is not None for row in joint_table(result))
        person = (f"**Person:** about **{result.body_depth:.2f} m** away (torso median) · "
                  f"{found}/{len(result.joint_depth)} joints have a depth")
    else:
        person = "**No person detected.**"
    info = (
        f"**Pose:** {(t1 - t0) * 1000:.0f} ms · **Depth** ({Path(depth_est.model_name).name}): "
        f"{(t2 - t1) * 1000:.0f} ms · {image.shape[1]}×{image.shape[0]} px{shrunk}  \n"
        + person + "  \n"
        + "Depths are **metres** from the camera."
    )
    return on_photo, on_depth, joint_table(result), info
 
 
# Create a custom theme with white background for the Gradio interface
 
theme = gr.themes.Default(
    primary_hue=gr.themes.colors.green,
).set(
    body_background_fill="#ffffff",
    block_background_fill="#ffffff",
)
 
# Gradio 6 moved `theme` from gr.Blocks(...) to demo.launch(...); support both.
GRADIO_6 = int(gr.__version__.split(".")[0]) >= 6
theme_in_blocks = {} if GRADIO_6 else {"theme": theme}
theme_in_launch = {"theme": theme} if GRADIO_6 else {}
 
with gr.Blocks(title="Depth Estimator", **theme_in_blocks) as demo:
    gr.Markdown(
        "# Depth estimator\n"
        "Upload a photo in .jpg or .png format. "
    )
    max_side = gr.Slider(0, 4096, value=DEFAULT_MAX_SIDE, step=64,
                         label="Max image side (px): bigger photos are shrunk to this first. "
                               "Lower = faster; 0 = never shrink")

    with gr.Tabs():
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
 
        with gr.Tab("Pose + depth"):
            with gr.Row():
                with gr.Column(scale=1):
                    pd_image = gr.Image(label="Input image", type="numpy", image_mode="RGB",
                                        sources=["upload", "clipboard", "webcam"])
                    pd_yolo = gr.Dropdown(YOLO_DEPTH_MODELS, value=YOLO_DEPTH_MODELS[0], allow_custom_value=True,
                                          label="YOLO depth model")
                    pd_device = gr.Dropdown(available_devices(), value="auto", label="Device")
                    pd_imgsz = gr.Slider(256, 1024, value=DEFAULT_YOLO_IMGSZ, step=32, label="YOLO input size (lower = faster)")
                    pd_go = gr.Button("Estimate pose + depth", variant="primary")
                    pd_info = gr.Markdown()
                with gr.Column(scale=2):
                    with gr.Row():
                        pd_photo = gr.Image(label="Skeleton + distances", interactive=False)
                        pd_depth = gr.Image(label="Skeleton on the depth map", interactive=False)
                    pd_table = gr.Dataframe(headers=["#", "joint", "x px", "y px", "visibility", "depth (m)"],
                                            label="Every joint", interactive=False, wrap=True)
 
            pd_inputs = [pd_image, pd_yolo, pd_device, pd_imgsz, max_side]
            pd_outputs = [pd_photo, pd_depth, pd_table, pd_info]
            pd_go.click(run_pose_depth, pd_inputs, pd_outputs)
            pd_image.upload(run_pose_depth, pd_inputs, pd_outputs)
 
if __name__ == "__main__":
    demo.launch(inbrowser=True, **theme_in_launch)   # first free port from 7860 up