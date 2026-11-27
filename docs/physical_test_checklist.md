# Physical test checklist (person present)

## Milestone evidence

P1 to P5 ran with a real person on 2026-11-19. Those results covered the
depth-band mask used in that rehearsal. The live mask later changed to the
pose-gradient method. Do not treat the earlier pass as a physical test of
the newer mask. The [enhancement evidence](enhancement-evidence.md) keeps
the measured rehearsal result, and the
[milestone reflection](Integration-Enhancements.md) states the accepted
quality claim. The steps below remain the repeatable physical test protocol.

Run these checks when a person is in front of the camera. They are needed for
the real-person evidence gate. Automated checks and recorded-video runs do not
replace them.

## Prerequisites

- Repository root: `live-splat-pipeline/`.
- Windows PowerShell, Firefox installed, `selenium` installed in `.venv`.
- Camera is not used by another application.
- Note the camera index. The default is `0` with the `dshow` backend.
- Set a writable temporary directory for pytest only if you also run pytest:

```powershell
New-Item -ItemType Directory -Force .tmp | Out-Null
$env:TMP = (Resolve-Path .tmp).Path
$env:TEMP = $env:TMP
$env:PYTHONNOUSERSITE = '1'
```

Run every command from the repository root unless a step says otherwise.

## Test P1 - camera sanity, empty scene

Purpose: check that an empty scene publishes no person state. A missing pose
can reject the refresh before the mask-area guard runs.

1. Point the camera at the room with no person.
2. Run:

```powershell
.\.venv\Scripts\python.exe scripts\run_integration_app.py `
  --source camera --device 0 --camera-backend dshow `
  --threshold 0.002 --window-ms 100 --duration-s 20 `
  --port 8820 --summary-json .tmp\p1_empty.json
```

Expected:
- `refresh_published` is 0 or small.
- `last_error` contains `mask area ... outside [0.005, 0.600]`.
- `/api/splat` returns 204 while the run lasts.
- No person state is displayed.

Record: the printed summary block.

## Test P2 - person present, standing still (reuse)

Purpose: prove the reuse path with a real person and a stable splat.

1. Stand in frame, facing the camera, upper torso visible.
2. Keep still for the full run.
3. Run:

```powershell
.\.venv\Scripts\python.exe scripts\run_integration_app.py `
  --source camera --device 0 --camera-backend dshow `
  --threshold 0.002 --window-ms 100 --duration-s 30 `
  --port 8821 --summary-json .tmp\p2_still.json `
  2>&1 | Tee-Object -FilePath .tmp\p2_still.log
```

Expected:
- `reuse` is greater than zero.
- `refresh_failed` is 0 or small.
- Splat count is stable, about 27,000 for an upper-torso framing.
- Warm refresh total is about 240-340 ms.

Record: summary block, `reuse`, `refresh_published`, `refresh_failed`,
`capture_fps`, and the splat counts from the `{"refresh": ...}` lines.

## Test P3 - person swaying side to side (refresh plus failure rate)

Purpose: prove refresh dominates when the person moves and measure pose dropout.

1. Stand in frame and sway side to side for the full run.
2. Run:

```powershell
.\.venv\Scripts\python.exe scripts\run_integration_app.py `
  --source camera --device 0 --camera-backend dshow `
  --threshold 0.002 --window-ms 100 --duration-s 30 `
  --port 8822 --summary-json .tmp\p3_sway.json `
  2>&1 | Tee-Object -FilePath .tmp\p3_sway.log
```

Expected:
- `reuse` is 0 or small.
- `refresh_published` is greater than in P2.
- Note `refresh_failed` and the `last_error` text.

Record: the same fields as P2, plus the failure reason. If failures grow when
you rotate to a side, that is evidence for the partial-body pose item; note the
direction and the approximate angle.

## Test P4 - leave and return (hold, fade, recovery)

Purpose: prove the loss hold, the fade to zero, and recovery to full alpha.

1. Run the app with a longer duration:

```powershell
.\.venv\Scripts\python.exe scripts\run_integration_app.py `
  --source camera --device 0 --camera-backend dshow `
  --threshold 0.002 --window-ms 100 --duration-s 45 `
  --port 8823 --summary-json .tmp\p4_loss.json `
  2>&1 | Tee-Object -FilePath .tmp\p4_loss.log
```

2. Stand in frame for 15 s.
3. Step out of frame for 8 s.
4. Step back in for the rest of the run.
5. While the run lasts, observe the splat panel status line.
6. Capture the page during the absence:

```powershell
cd applications\gui
..\..\.venv\Scripts\python.exe tools\capture-firefox.py `
  --url http://127.0.0.1:8823/ --out screenshots\p4_absent.png --wait-ms 2000
```

Expected:
- During absence, the diagnostic shows `person_loss_age_ms` increasing and
  `/api/splat` keeps the last state.
- The panel status shows `Alpha: 0.00` after the hold and fade.
- After you return, a new state publishes and the status returns to `1.00 alpha`.

Record: `.tmp\p4_absent.png`, the page status text, and the diagnostic values
before and after the return.

## Test P5 - depth tab consistency

Purpose: confirm that depth is shown only for the displayed splat state.

1. Start the app as in P2.
2. Capture the Depth tab with Firefox:

```powershell
cd applications\gui
..\..\.venv\Scripts\python.exe tools\capture-firefox.py `
  --url http://127.0.0.1:8821/ --out screenshots\p5_depth.png `
  --wait-ms 5000 --click-selector '.processor-tab[data-view="depth"]'
```

Expected:
- The Depth tab shows the depth image and the skeleton or pose overlay.
- The footer frame label is not greater than the active splat frame.
- No old depth image remains after the splat frame advances.

Record: `.tmp\p5_depth.png` and the page text printed by the capture tool.

## Evidence to return

Copy these files into `docs/evidence/` and list their paths in the test report:

- `p1_empty.json`
- `p2_still.json` and `p2_still.log`
- `p3_sway.json` and `p3_sway.log`
- `p4_loss.json`, `p4_loss.log`, `p4_absent.png`
- `p5_depth.png`

State which tests passed and which failed, with the printed numbers. Do not
mark a test passed while no person was in frame; a whole-frame mask rejection is
P1, not P2-P5.
