# Run the live web application

Run these commands in **PowerShell from the repository root**. Use Windows,
Python 3.11 to 3.13, Node.js with `npm`, and a webcam or OBS Virtual Camera.
The setup script creates `.venv`, installs the application packages, builds
the native constructor and web interface, and runs the test suite. First
setup needs internet access for packages and the MediaPipe Lite model.
The first webcam refresh can download YOLO depth weights when they are absent.

## First setup

```powershell
& .\scripts\setup_system.ps1
```

## Start the webcam application

```powershell
& .\start_live_demo.ps1
```

The launcher opens the web application in a browser. Keep its application
window open during the run. Press `Ctrl+C` in that window to stop capture.

## Select another source

Use one of these commands after first setup:

```powershell
& .\start_live_demo.ps1 -Source obs
& .\start_live_demo.ps1 -Source zed
& .\start_live_demo.ps1 -Source video -VideoPath .\path\to\clip.mp4
```

Start OBS Virtual Camera before the OBS command. The ZED command requires a
working ZED SDK and a `pyzed` wheel that matches the Python interpreter and
SDK version. Check the camera with:

```powershell
& .\start_live_demo.ps1 -ListZed
```

Use `-RebuildGui` on the launcher if the web interface needs a fresh build.
The launcher uses port `8765` by default; use `-Port <number>` when needed.

For implementation details and measurements, see the
[integration technical record](integration-technical-history.md). For the
constructor-to-browser data contract, see the
[splat representation guide](splat-representation-and-runtime.md).
