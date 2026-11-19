# Scripts

## Device setup and live diagnostics

- Purpose: repeatable setup and validation commands for each development device.
- `setup_system.ps1` is the canonical setup script. It creates the local Python environment, installs the pipeline modules, PyTorch, the runtime packages, the GUI build, and the native components, then runs the test suite. `setup_device.ps1` forwards to it.
- Optional RAFT setup selects a PyTorch CPU or NVIDIA wheel for the current device; the default setup needs only OpenCV DIS.
- Inputs: Python 3.11+, network access for initial package installation, and an NVIDIA driver when CUDA RAFT is requested.
- Outputs: an ignored `.venv/` environment. The script does not change shared packet schemas or produce recordings.
- `run_live_scheduler.py` exercises live capture, newest-frame sampling, reranking, and a simulated refresh worker without requiring depth or pose implementations.
- Remaining work: connect the scheduler to real depth/pose, temporal cache, and splat construction while preserving selected frame identity.

## Splat Constructor utilities

- [x] Add a checksum-verified APC sample downloader for baseline training.
- [x] Add C++ build, fixture preparation, and 3DGS PLY inspection utilities.
- [x] Add direct source-view construction and reprojection validation command.
- [x] Add a deterministic recorded-video replay command for baseline learning data.
- [ ] Add trained-model evaluation to that replay command.
- [ ] Add schema validation and interface checks.
- [ ] Add fixture preparation and cleanup helpers.
- [ ] Keep scripts cross-platform where practical.
