# MediaPipe Pose Estimator Prototype

Input:
- Single RGB frame from the Reranker.

Output:
- Pose estimation for the same frame.
- MediaPipe body landmarks.
- Visual skeleton overlay for prototype validation.

Selected model:
- MediaPipe Pose Landmarker Lite.
- Pose Landmarker Full is retained only for benchmarking and comparison.
- Lite was selected as the project's runtime model after testing normal,
  cropped-body, multi-person, low-light, motion-blur, and occlusion scenarios.
- Full detected one additional extreme motion-blur case, but Lite provided
  lower inference latency while maintaining comparable pose quality across
  the project's typical operating scenarios.
- The team prioritised Lite's lower computational cost and real-time suitability.

Prototype milestones:
- [x] Run MediaPipe Pose Landmarker on a single image frame.
- [x] Return detected pose landmarks.
- [x] Draw pose landmarks and connections for visual validation.
- [x] Test front-facing pose.
- [x] Test side-facing articulated pose.
- [x] Test lower-body cutoff.
- [x] Test multiple people with single-pose output.
- [x] Test subjects at varying distances.
- [x] Test low-light conditions.
- [x] Test motion blur.
- [x] Test facial and object occlusion.
- [x] Compare Pose Landmarker Lite against Full.
- [x] Select Lite as the project runtime model.
- [x] Preserve frame identity and timestamp for pipeline integration.

NOTE: compare_models.py requires the following:

- models/pose_landmarker_lite.task
- models/pose_landmarker_full.task