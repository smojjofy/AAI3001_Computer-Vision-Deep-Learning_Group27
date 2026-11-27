# Live Splat Viewer (GUI)

The milestone web app displays the captured frame, selected mask, depth and
pose previews, diagnostics, and the newest Gaussian splat state. Its main
viewer uses `gsplat`. An optional Three.js view can import a `.glb`
environment and display the same splat frames. The browser does not open
the camera. In the integrated run, the backend serves the built GUI and
local HTTP API on `127.0.0.1:8765`. The Vite `/api` proxy serves development
runs. The [milestone workflow](../../docs/single-camera-workflow.md)
describes the full path.

## Panels

- **Original Input** — the latest RGB preview frame (`/api/preview`) with frame
  ID, source ID, and capture time. Shows LIVE, CAMERA CLOSED, or PREVIEW
  DISCONNECTED in the top bar.
- **Frame Processor Output** — three live tabs:
  - `RGBA`: the same preview JPEG,
  - `Mask`: `/api/mask` selected person mask (PNG),
  - `Depth`: `/api/pose-depth` colourised depth image.
  Every tab shows a waiting state when the endpoint returns HTTP 204.
- **Inference Diagnostics** — live `/api/diagnostics`: reuse/refresh decision and
  reason, flow score against threshold, cache age, refresh counters, stage
  timings, splat count, mask area/confidence, pose confidence, person detection,
  queue depth/drops, last error, and stream health.
- **Splat Constructor Output** — the live splat state (`/api/splat`) rendered with
  `gsplat` (splat.js) and gsplat OrbitControls. The frame, splat count, render
  FPS, and `display_alpha` are in the footer. When `/api/motion` provides a reuse
  flow field, the panel samples it at each splat source pixel and offsets only the
  displayed positions in view space; base positions are never changed. See the
  [representation guide](../../docs/splat-representation-and-runtime.md) for
  the coordinate conversion and the body-region warp heuristic.

The splat state is fetched only when `/api/diagnostics` reports a new
`splat.frame_id` (diagnostics poll: 300 ms). No mock splat data is rendered.

## Environment compositor

Use **Environment compositor** in the top bar to open the optional 3D view.
Import one self-contained `.glb` file of 30 MB or less. The browser keeps the
file local. The view uses the same published splat frames as the default panel.
One splat viewer stays visible while a second viewer sorts the newest frame.
The view swaps them only after the new splat buffer is ready. A source or
generation change holds the last drawn splat until the next state is ready.
The status text identifies a held frame. Close the view to clear that frame.
The default panel continues to run when the compositor is closed.

The imported model is centered and its longest side is set to 4 m. Use the
scale and X, Y, and Z controls to place it around the subject. Both the model
and the splats use the constructor world: +X right, +Y up, and camera forward
-Z. The view reads the capture focal length to set its vertical field of view.
Opaque model surfaces can hide splats behind them. Transparent model
surfaces can give incorrect depth results. The optional view does not apply the
default panel's optical-flow display warp or person-loss fade. Close the view
to release its renderer resources.

## Run

Start the integrated pipeline (one window):

```powershell
cd live-splat-pipeline
.\.venv\Scripts\python.exe scripts\run_integration_app.py --source webcam --device 0
```

Start the GUI (second window):

```powershell
cd live-splat-pipeline\applications\gui
npm ci
npm run dev
```

Open the printed Vite URL. The Vite config forwards `/api` to
`http://127.0.0.1:8765`.

## Checks and screenshots

```powershell
cd applications\gui
npx tsc --noEmit
npm run build
npm run check:environment
```

`tools/capture-screenshot.ps1` captures the running page with headless Firefox
through Selenium. Firefox starts with an isolated temporary profile, so a
capture never touches the user's own browser. The helper waits until the page
shows a live frame, waits real time, optionally clicks a tab, and saves a PNG.
It replaces the earlier Edge/DevTools capture, which stopped every running
Microsoft Edge process. Install the Python dependency once:

```powershell
..\..\.venv\Scripts\python.exe -m pip install selenium
```

For example:

```powershell
./tools/capture-screenshot.ps1 -Url http://127.0.0.1:5180/ `
    -Out screenshots/gui-live-main.png -WaitMs 10000
```

`tools/inspect_capture.py` is a developer helper that reads the live preview and
captured screenshots and writes brightened crops for inspection. It is not part
of the built application.

The preview JPEG is display-only and capped at 640 pixels wide. The original RGB
packets stay in memory; the GUI does not record them.
