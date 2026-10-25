# MediaPipe Pose Estimator Prototype

Input:
- Single RGB frame from video.

Output:
- Pose estimation for the same frame.
- MediaPipe body landmarks.
- Visual skeleton overlay for prototype validation.

Prototype milestones:
- [x] Run MediaPipe Pose Landmarker on a single image frame.
- [x] Return detected pose landmarks.
- [x] Draw pose landmarks and connections for visual validation.
- [x] Process recorded video frame-by-frame.
- [x] Measure pose inference latency and FPS.
- [x] Preserve frame identity/timestamp when integrated with the pipeline.