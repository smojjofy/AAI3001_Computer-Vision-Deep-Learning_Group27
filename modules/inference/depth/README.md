# Depth module (YOLO depth + MediaPipe pose)
 
`PoseDepthPipeline` = MediaPipe Pose Lite (body joints) + depth in metres (ZED if given, else YOLO depth).
 
All commands: run from `modules\inference\depth`, with the venv active.
 
## Startup
 
```powershell
cd C:\Users\User\Github-Projects\AAI3001_Computer-Vision-Deep-Learning_Group27
.venv\Scripts\Activate.ps1          # prompt must show (.venv)
cd modules\inference\depth
```
 
```powershell
python -m pip install -r requirements.txt
```
 
Also needs `pose_landmarker_lite.task` in `modules\inference\pose\models\` (from the MediaPipe Pose Landmarker page).
Use `python -m pip`, never plain `pip`.
 
## 1. Tests
 
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
python scripts\benchmark_latency.py --images my_photo_folder --resize 1280 720 --frames 20 --pipeline --pipeline-zed
#   --pipeline      = whole pipeline (pose + YOLO depth)
#   --pipeline-zed  = whole pipeline with depth handed in like the ZED (fake ZED depth, speed only)
#   --images can be a folder or one image
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
