# Live original input preview

The **Original Input** panel shows a display-only JPEG from the same Stream
Handler `FramePacket` that is available to the optical-flow and reranker queue.
The browser does not open the camera itself. The processor and splat panels
still use Brejesh's demonstration data.

From the repository root, run the live Stream Handler, optical flow,
reranker, and GUI preview in one PowerShell window:

```powershell
.\.venv\Scripts\python.exe scripts\run_live_scheduler.py --source camera --device 0 --camera-backend dshow --config config\optical_flow_raft.example.json --sample-ms 300 --max-reference-age-ms 1000 --refresh-work-ms 1000 --duration-s 0 --gui-preview --summary-only
```

For a raw camera preview without optical flow or reranking, use:

```powershell
.\.venv\Scripts\python.exe scripts\run_gui_preview.py --source camera --device 0 --camera-backend dshow
```

For OBS Virtual Camera, start the virtual camera in OBS and use `--source obs`
with its device index. Keep OBS closed when testing a physical webcam directly
if OBS is holding that webcam. A recorded fixture can be used with
`--source video --path data\recordings\flow_eval.mp4`.

In another PowerShell window, run the GUI:

```powershell
cd applications\gui
npm ci
npm run dev
```

Open the local URL printed by Vite. The Vite development server forwards
`/api/preview` to the Python server on `127.0.0.1:8765`. Preview frames are
limited to 640 pixels wide and sent at the browser's polling rate. The
original RGB8 packets remain in memory and are not recorded or sent to a
remote service. Stop either server with Ctrl+C.

The preview transport is defined in `shared/schemas/preview_frame.schema.json`.
`capture_timestamp_ns` is a decimal string because JavaScript numbers cannot
always represent nanosecond timestamps exactly. The GUI displays time since
the first captured frame from the current source.
