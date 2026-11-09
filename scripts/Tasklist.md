# Scripts

- Purpose: repeatable setup and validation commands for each development device.
- `setup_device.ps1` creates a local Python environment and installs the Stream Handler, Optical Flow, and Reranker modules in editable mode.
- Optional RAFT setup selects a PyTorch CPU or NVIDIA wheel for the current device; the default setup needs only OpenCV DIS.
- Inputs: Python 3.11+, network access for initial package installation, and an NVIDIA driver when CUDA RAFT is requested.
- Outputs: an ignored `.venv/` environment. The script does not change shared packet schemas or produce recordings.
- `run_live_scheduler.py` exercises live capture, newest-frame sampling, reranking, and a simulated refresh worker without requiring depth or pose implementations.
- Remaining work: connect the scheduler to real depth/pose, temporal cache, and splat construction while preserving selected frame identity.
