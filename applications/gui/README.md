# Live Splat Viewer (GUI)

Vite + TypeScript viewer for the integrated local pipeline. The browser does not
open the camera. It polls the local HTTP API on `127.0.0.1:8765` through the Vite
`/api` proxy.

## Panels

- **Original Input** — the latest RGB preview frame (`/api/preview`) with frame
  ID, source ID, and capture time. Shows LIVE, CAMERA CLOSED, or PREVIEW
  DISCONNECTED in the top bar.
- **Frame Processor Output** — four live tabs:
  - `RGBA`: the same preview JPEG,
  - `Mask`: `/api/mask` depth-threshold person mask (PNG),
  - `Depth`: `/api/pose-depth` colourised depth image,
  - `Motion`: `/api/motion` reuse flow field drawn as a magnitude heat map with
    direction arrows.
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
  comments in `src/components/SplatOutputPanel.ts` for the coordinate conversion
  and the body-region warp heuristic.

The splat state is fetched only when `/api/diagnostics` reports a new
`splat.frame_id` (diagnostics poll: 300 ms). No mock splat data is rendered.

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
