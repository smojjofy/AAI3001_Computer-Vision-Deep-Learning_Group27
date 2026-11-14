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
 
## 4. Speed (measured: laptop CPU, 1280x720, 20 frames after warm-up, median)
 
| Pipeline | Pose | Depth | **Total** |
|---|---|---|---|
| Pose + YOLO depth `.pt` (imgsz 640) | ~32-40 ms | ~188-224 ms | **~222-267 ms (~3.7-4.4 fps)** |
| Pose + YOLO depth **OpenVINO** (imgsz 640) | ~20 ms | ~174 ms | **~196 ms (~5.1 fps)** |
| Pose + ZED depth | ~27 ms | ~69 ms (cleanup only) | **~98 ms (~10 fps)** |
 
Live webcam inside `run_live_scheduler.py --pose-depth` (OpenVINO, RAFT running too): ~240 ms per refresh.
First frame after starting is ~3-4 s (model warm-up), then normal. Timings vary by ±30 ms on the laptop. Faster: GPU (`--device cuda`), smaller frames.
Keep `imgsz` at **640** (YOLO depth's training size): other sizes change the metres (384 read ~60% too far).
