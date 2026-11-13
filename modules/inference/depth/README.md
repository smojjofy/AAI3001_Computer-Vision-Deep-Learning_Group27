# Depth module (YOLO depth + MediaPipe pose)
 
`PoseDepthPipeline` = MediaPipe Pose Lite (body joints) + depth in metres (ZED if given, else YOLO depth).
 
All commands: run from `modules\inference\depth`, with the venv active.
 
First time only (every person / every PC):
 
```powershell
python -m pip install -r requirements.txt
python -m pip install openvino
 
# Export YOLO depth to OpenVINO (~5 s): same model, same 640 input, ~25-40% faster on CPU.
# Creates models\yolo26n-depth_openvino_model\ (git-ignored, so each PC does this once).
python -c "from ultralytics import YOLO; YOLO('models/yolo26n-depth.pt').export(format='openvino', imgsz=640)"
```
 
Using the OpenVINO model: `PoseDepthPipeline(yolo_model="yolo26n-depth_openvino_model")`,
`--yolo-model yolo26n-depth_openvino_model` in the scripts, or pick it in the app's YOLO model box.
Without the export, use `yolo26n-depth.pt` (works the same, just slower).
 
## 1. Check everything works
 
```powershell
python -m pytest tests -q                 # expect: 58 passed
```
 
## 2. Gradio app
 
```powershell
python app.py                             # opens the browser; Ctrl+C to stop
```
 
- **YOLO depth** tab: depth map, click it to read metres.
- **Pose + depth** tab: skeleton + distance per joint.
- Keep YOLO input size at **640**: other sizes give wrong metres (384 reads ~60% too far).
- If the browser shows an old version: close old terminals, rerun, then Ctrl+F5.
## 3. Scripts
 
```powershell
# Depth on one image
python scripts\test_depth_image.py --image photo.jpg
 
# Depth on a folder (saves PNG + .npy per image)
python scripts\run_depth_folder.py --input my_photos --output my_photos_out
 
# Pose + depth on one image (YOLO depth)
python scripts\test_pose_depth_image.py --image photo.jpg
 
# Pose + depth using a ZED depth map (.npy) instead of YOLO
python scripts\test_pose_depth_image.py --image zed_left.png --zed-depth zed_depth.npy
#   ZED depth in millimetres? add: --depth-scale 0.001
 
# Speed (real numbers, warm-up excluded); --resize 1280 720 = ZED size
   python scripts\benchmark_latency.py --images zed_fake --resize 1280 720 --frames 20 --pipeline --yolo-model yolo26n-depth_openvino_model (static folder)  
#   --pipeline      = whole pipeline (pose + YOLO depth)
#   --images can be a folder or one image
#   --yolo-model yolo26n-depth_openvino_model = whole pipeline using yolo-model openvino
```
 
Useful flags: `--no-show` (don't open a window), `--device cuda` (GPU), `--imgsz 640`.
 
## 4. Using it in code (for the frame processor / GUI)
 
**Rules:**
1. Create the pipeline **once** at startup, then call `process_frame` for every frame.
   Creating it inside the frame loop reloads the models every frame (seconds each).
2. Code outside this folder must put the depth folder on Python's path first (see the import below).
3. Needs the packages in `requirements.txt` and `pose_landmarker_lite.task` in `modules/inference/pose/models/`.
**Import** (from anywhere in the repo):
 
```python
import sys
from pathlib import Path
 
REPO_ROOT = Path(__file__).resolve().parents[N]       # N = how many folders up the repo root is from this file
sys.path.insert(0, str(REPO_ROOT / "modules" / "inference" / "depth"))
 
from src import PoseDepthPipeline, zed_to_rgb
```
 
**Frame loop** (names like `frame.image` / `gui.show_...` are placeholders for the stream handler and GUI):
 
```python
pipe = PoseDepthPipeline(depth_scale=0.001)   # ONCE. depth_scale=0.001 if ZED depth is in mm; leave out if metres
 
def on_new_frame(frame):                      # called by the stream handler for every frame
    rgb = zed_to_rgb(frame.image)             # ZED images are BGRA -> RGB
    out = pipe.process_frame(rgb, frame.frame_id, frame.source_id, frame.timestamp_ns,
                             depth=frame.depth)               # ZED depth; depth=None -> YOLO depth instead
    gui.show_pose(out.pose.landmarks)
    gui.show_depth(out.depth.depth)
    gui.show_text(f"Person: {out.body_depth:.2f} m · {out.timings_ms['total']:.0f} ms")
 
# when the app closes:
pipe.close()
```
 
**What `out` contains:**
 
```python
out.pose.landmarks   # 33 joints (x, y in 0..1 of the image; multiply by width/height for pixels)
out.depth.depth      # depth map [H, W], metres, NaN = invalid
out.joint_depth      # metres per joint (NaN = not visible / no depth)
out.body_depth       # how far the person is (torso median)
out.depth_source     # "zed" or "yolo"
out.timings_ms       # {"pose", "depth", "fuse", "total"} in ms
```
 
**Ready-made drawings** (BGR images for OpenCV / the GUI):
 
```python
from src.pose_depth import draw_pose_depth
from src.utils import colorize
 
skeleton_img = draw_pose_depth(bgr_image, out.frame)       # skeleton + distances on the photo
depth_img = draw_pose_depth(colorize(out.depth), out.frame)  # skeleton on the coloured depth map
```
 
Optional: `PoseDepthPipeline(intrinsics=CameraIntrinsics(fx, fy, cx, cy))` (from the ZED calibration)
also gives `out.joint_xyz`: each joint's 3D position in metres.
 
### Integration notes (for the other modules)
 
| Who | What the pipeline expects / offers |
|---|---|
| **Stream handler** | Per frame: an RGB uint8 image (or the ZED's BGRA image → `zed_to_rgb()`), `frame_id`, `source_id`, `timestamp_ns`, and the ZED depth map. ZED depth in **millimetres** (the SDK default) → create the pipeline with `depth_scale=0.001`; in **metres** (`coordinate_units = sl.UNIT.METER`) → leave it out. |
| **GUI / frame processor** | `out.pose.landmarks` = the 33 joints. `draw_pose_depth(...)` and `colorize(...)` give ready-made skeleton and depth images (see above). |
| **Everyone** | The pose model `pose_landmarker_lite.task` must be in `modules/inference/pose/models/`. That folder is git-ignored, so each person downloads it once (MediaPipe Pose Landmarker page). |
| **ZED vs YOLO comparison** | Save a few ZED frames as a left image + depth `.npy` (see below), then run `scripts\test_pose_depth_image.py` with and without `--zed-depth` on the same image and compare the joint tables. |
 
Saving a ZED frame (needs the ZED SDK's Python API, `pyzed`):
 
```python
import numpy as np, cv2, pyzed.sl as sl
 
zed = sl.Camera()
zed.open(sl.InitParameters(coordinate_units=sl.UNIT.METER, depth_mode=sl.DEPTH_MODE.NEURAL))
image, depth = sl.Mat(), sl.Mat()
if zed.grab() == sl.ERROR_CODE.SUCCESS:
    zed.retrieve_image(image, sl.VIEW.LEFT)
    zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
    cv2.imwrite("zed_left.png", image.get_data())      # BGRA
    np.save("zed_depth.npy", depth.get_data())         # metres
zed.close()
```
 
## 5. Speed (measured: laptop CPU, 1280x720, 20 frames after warm-up, median)
 
| Pipeline | Pose | Depth | **Total** |
|---|---|---|---|
| Pose + YOLO depth `.pt` (imgsz 640) | ~32-40 ms | ~188-224 ms | **~222-267 ms (~3.7-4.4 fps)** |
| Pose + YOLO depth **OpenVINO** (imgsz 640) | ~20 ms | ~174 ms | **~196 ms (~5.1 fps)** |
| Pose + ZED depth | ~27 ms | ~69 ms (cleanup only) | **~98 ms (~10 fps)** |
 
Live webcam inside `run_live_scheduler.py --pose-depth` (OpenVINO, RAFT running too): ~240 ms per refresh.
First frame after starting is ~3-4 s (model warm-up), then normal. Timings vary by ±30 ms on the laptop. Faster: GPU (`--device cuda`), smaller frames.
Keep `imgsz` at **640** (YOLO depth's training size): other sizes change the metres (384 read ~60% too far).
