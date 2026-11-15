import './style.css'

import {
  decodeJpegBase64,
  fetchDiagnostics,
  fetchMask,
  fetchMotion,
  fetchPoseDepth,
  fetchPreview,
  fetchSplat,
} from './api/liveApi'

import { FrameProcessorPanel } from './components/FrameProcessorPanel'
import { InferencePanel } from './components/InferencePanel'
import { OriginalInputPanel } from './components/OriginalInputPanel'
import { SplatOutputPanel } from './components/SplatOutputPanel'

import type {
  DiagnosticsPayload,
  MotionPayload,
  PoseDepthFrame,
} from './api/types'

const PREVIEW_POLL_MS = 100
const STATE_POLL_MS = 300

const app = document.querySelector<HTMLDivElement>('#app')

if (!app) {
  throw new Error('App root element not found')
}

app.innerHTML = `
  <div class="app-shell">
    <header class="top-bar">
      <div>
        <h1>Real-Time 2.5D Reconstruction Viewer</h1>
        <p>Single Camera Pipeline · live local API</p>
      </div>
      <div class="status" id="capture-status">
        <span class="status-dot"></span>
        <span class="status-label">WAITING FOR INPUT</span>
      </div>
    </header>
    <main class="viewer-grid" id="viewer-grid"></main>
  </div>
`

const viewerGrid = document.querySelector<HTMLElement>('#viewer-grid')
const status = document.querySelector<HTMLElement>('#capture-status')
const statusLabel = status?.querySelector<HTMLElement>('.status-label')

if (!viewerGrid || !status || !statusLabel) {
  throw new Error('Viewer elements not found')
}

const originalInputPanel = new OriginalInputPanel()
viewerGrid.appendChild(originalInputPanel.getElement())

const frameProcessorPanel = new FrameProcessorPanel()
viewerGrid.appendChild(frameProcessorPanel.getElement())

const inferencePanel = new InferencePanel()
viewerGrid.appendChild(inferencePanel.getElement())

const splatOutputPanel = new SplatOutputPanel()
viewerGrid.appendChild(splatOutputPanel.getElement())

type PreviewState = 'unknown' | 'live' | 'waiting' | 'disconnected'

let previewState: PreviewState = 'unknown'
let previewHealth = 'unknown'
let pipelineHealth = 'unknown'

let previousPreviewFrameId: number | null = null
let previousSourceId: string | null = null
let previousTimestampOrigin: string | null = null
let firstCaptureNs: bigint | null = null

// Fetch gating. A requested id is recorded only when its fetch actually
// starts. If a newer id arrives while a fetch is running, it is kept as
// pending and fetched when the current one finishes, so no state is skipped.
// The rendered id is taken from the response, because /api/splat always
// returns the latest state and may differ from the requested id.
let splatRenderedFrameId: string | null = null
let splatPendingFrameId: string | null = null
let splatFetchInFlight = false
let splatRetryAt = 0

let maskRenderedCount = -1
let maskPendingCount: number | null = null
let maskFetchInFlight = false
let maskRetryAt = 0

let lastMotionFrameId: number | null = null
let activeMotion: MotionPayload | null = null
let poseDepthRenderedFrameId: number | null = null
let poseDepthPendingFrameId: number | null = null
let poseDepthFetchInFlight = false
let poseDepthRetryAt = 0

/**
 * Viewer-clock anchor for the server's latched invalid-person age.
 * The panel holds the last valid state for 500 ms, then fades it over 500 ms.
 */
let personLossAnchorMs: number | null = null

function applyStatus(): void {
  let label = 'WAITING FOR INPUT'
  let live = false

  if (previewState === 'disconnected') {
    label = 'PREVIEW DISCONNECTED'
  } else if (previewHealth === 'closed' || pipelineHealth === 'closed') {
    label = 'CAMERA CLOSED'
  } else if (previewHealth === 'stalled' || previewHealth === 'reconnecting') {
    label = previewHealth.toUpperCase()
  } else if (previewState === 'live') {
    label = 'LIVE'
    live = true
  }

  statusLabel!.textContent = label
  status!.classList.toggle('is-live', live)
}

function describeError(error: unknown): string {
  if (error instanceof Error) {
    return `${error.name}: ${error.message}`
  }
  return String(error)
}

async function pollPreview(): Promise<void> {
  const result = await fetchPreview()

  if (result.kind === 'ok') {
    const frame = result.data
    previewHealth = frame.stream_health

    if (frame.stream_health !== 'healthy') {
      previewState = 'waiting'
      originalInputPanel.showWaiting(`Camera ${frame.stream_health}`)
      frameProcessorPanel.showPreviewWaiting(`Camera ${frame.stream_health}`)
      applyStatus()
      window.setTimeout(() => {
        void pollPreview()
      }, PREVIEW_POLL_MS)
      return
    }

    previewState = 'live'
    const sourceChanged = frame.source_id !== previousSourceId
    const isNewFrame =
      frame.frame_id !== previousPreviewFrameId || sourceChanged

    if (isNewFrame) {
      previousPreviewFrameId = frame.frame_id
      previousSourceId = frame.source_id

      const captureNs = BigInt(frame.capture_timestamp_ns)
      if (
        firstCaptureNs === null ||
        sourceChanged ||
        frame.timestamp_origin !== previousTimestampOrigin
      ) {
        firstCaptureNs = captureNs
      }
      previousTimestampOrigin = frame.timestamp_origin
      const timestamp = `Capture +${(
        Number(captureNs - firstCaptureNs) / 1_000_000_000
      ).toFixed(2)} s`

      try {
        const image = await decodeJpegBase64(frame.jpeg_base64)
        const meta = {
          frameId: frame.frame_id,
          sourceId: frame.source_id,
          timestamp,
        }
        originalInputPanel.showImage(image, frame.width, frame.height, meta)
        frameProcessorPanel.setPreviewFrame(image, meta)
      } catch (error) {
        const message = `Preview decode failed: ${describeError(error)}`
        originalInputPanel.showWaiting(message)
        frameProcessorPanel.showPreviewWaiting(message)
      }
    }
  } else if (result.kind === 'empty') {
    previewState = 'waiting'
    originalInputPanel.showWaiting('Waiting for camera frames')
    frameProcessorPanel.showPreviewWaiting('Waiting for the camera preview')
  } else {
    previewState = 'disconnected'
    originalInputPanel.showWaiting(
      'Start the pipeline server on 127.0.0.1:8765',
    )
    frameProcessorPanel.showPreviewWaiting('Preview disconnected')
  }

  applyStatus()
  window.setTimeout(() => {
    void pollPreview()
  }, PREVIEW_POLL_MS)
}

async function refreshSplat(frameId: string): Promise<void> {
  if (splatFetchInFlight) {
    // Keep the newest request; it is fetched when the current fetch ends.
    splatPendingFrameId = frameId
    return
  }
  if (performance.now() < splatRetryAt) {
    splatPendingFrameId = frameId
    return
  }
  splatFetchInFlight = true
  splatPendingFrameId = null
  try {
    const result = await fetchSplat()
    if (result.kind === 'ok') {
      // Record the identity that actually arrived, not the one requested.
      const rendered = String(result.data.header.frame_id)
      splatRenderedFrameId = rendered
      splatOutputPanel.setSplatFrame(result.data)
      // A refresh clears the motion payload; clear the warp when this state is
      // at least as new as the active motion (the panel also checks identity).
      if (
        activeMotion !== null &&
        Number(rendered) >= activeMotion.frame_id
      ) {
        activeMotion = null
        splatOutputPanel.setMotionWarp(null)
      }
      splatRetryAt = 0
    } else if (result.kind === 'empty') {
      splatRenderedFrameId = null
    } else {
      console.warn(`Splat fetch failed: ${result.error}`)
      splatRetryAt = performance.now() + 500
    }
  } catch (error) {
    console.warn(`Splat frame ${frameId} could not be rendered: ${describeError(error)}`)
    splatRetryAt = performance.now() + 500
  } finally {
    splatFetchInFlight = false
  }
  const next = splatPendingFrameId
  if (next !== null && next !== splatRenderedFrameId) {
    splatPendingFrameId = null
    void refreshSplat(next)
  }
}

async function refreshMask(refreshCount: number): Promise<void> {
  if (maskFetchInFlight) {
    maskPendingCount = refreshCount
    return
  }
  if (performance.now() < maskRetryAt) {
    maskPendingCount = refreshCount
    return
  }
  maskFetchInFlight = true
  maskPendingCount = null
  try {
    const result = await fetchMask()
    if (result.kind === 'ok') {
      maskRenderedCount = refreshCount
      frameProcessorPanel.setMaskImage(result.data, { refreshCount })
      maskRetryAt = 0
    } else {
      frameProcessorPanel.setMaskImage(null, null)
      maskRetryAt = performance.now() + 500
    }
  } finally {
    maskFetchInFlight = false
  }
  const next = maskPendingCount
  if (next !== null && next !== maskRenderedCount) {
    maskPendingCount = null
    void refreshMask(next)
  }
}

async function refreshPoseDepth(poseDepth: PoseDepthFrame): Promise<void> {
  if (poseDepthFetchInFlight) {
    poseDepthPendingFrameId = poseDepth.frame_id
    return
  }
  if (performance.now() < poseDepthRetryAt) {
    poseDepthPendingFrameId = poseDepth.frame_id
    return
  }
  poseDepthFetchInFlight = true
  poseDepthPendingFrameId = null
  try {
    const image = await decodeJpegBase64(poseDepth.depth_jpeg_base64)
    poseDepthRenderedFrameId = poseDepth.frame_id
    frameProcessorPanel.setDepthFrame(image, poseDepth)
    poseDepthRetryAt = 0
  } catch {
    frameProcessorPanel.setDepthFrame(null, poseDepth)
    poseDepthRetryAt = performance.now() + 500
  } finally {
    poseDepthFetchInFlight = false
  }
  const next = poseDepthPendingFrameId
  if (next !== null && next !== poseDepthRenderedFrameId) {
    poseDepthPendingFrameId = null
  }
}

/** Anchor the server's latched person-loss age on the viewer clock. */
function schedulePersonLoss(ageMs: number | null): void {
  if (ageMs === null) {
    personLossAnchorMs = null
    splatOutputPanel.setPersonLoss(null)
    return
  }
  const now = performance.now()
  const currentAge =
    personLossAnchorMs === null ? 0 : now - personLossAnchorMs
  if (personLossAnchorMs === null || ageMs < currentAge - 50) {
    // First loss, or the server latch restarted after a recovery.
    personLossAnchorMs = now - Math.max(0, ageMs)
  }
  splatOutputPanel.setPersonLoss(now - personLossAnchorMs)
}

function tickDisplayClocks(): void {
  splatOutputPanel.tickPersonLoss()
  window.requestAnimationFrame(tickDisplayClocks)
}

function hasDiagnostics(payload: DiagnosticsPayload): boolean {
  return typeof payload.refresh_count === 'number'
}

async function pollState(): Promise<void> {
  const [diagnosticsResult, motionResult, poseDepthResult] = await Promise.all([
    fetchDiagnostics(),
    fetchMotion(),
    fetchPoseDepth(),
  ])

  if (diagnosticsResult.kind === 'ok' && hasDiagnostics(diagnosticsResult.data)) {
    const diagnostics = diagnosticsResult.data
    pipelineHealth = diagnostics.stream_health
    inferencePanel.update(diagnostics)

    const splatFrameId = diagnostics.splat?.frame_id ?? null
    if (
      splatFrameId !== null &&
      splatFrameId !== splatRenderedFrameId &&
      splatFrameId !== splatPendingFrameId
    ) {
      void refreshSplat(splatFrameId)
    }
    if (
      diagnostics.refresh_count !== maskRenderedCount &&
      diagnostics.refresh_count !== maskPendingCount &&
      diagnostics.refresh_count > 0
    ) {
      void refreshMask(diagnostics.refresh_count)
    }
    // Server-latched invalid-person age. Anchor it once on the viewer clock;
    // the browser then runs the 500 ms hold and 500 ms fade without refetching.
    schedulePersonLoss(diagnostics.person_loss_age_ms ?? null)
  } else {
    inferencePanel.update(null)
    pipelineHealth =
      diagnosticsResult.kind === 'error' ? 'disconnected' : 'waiting'
  }

  if (motionResult.kind === 'ok') {
    const motion = motionResult.data
    const motionKey = motion.frame_id * 1_000_003 + (motion.reference_frame_id ?? 0) % 1000
    if (motionKey !== lastMotionFrameId) {
      lastMotionFrameId = motionKey
      activeMotion = motion
      frameProcessorPanel.setMotion(motion)
      splatOutputPanel.setMotionWarp(motion)
    }
  } else if (lastMotionFrameId !== null) {
    // 204: the refresh path replaced the reuse state, so clear the warp.
    lastMotionFrameId = null
    activeMotion = null
    frameProcessorPanel.setMotion(null)
    splatOutputPanel.setMotionWarp(null)
  }

  if (poseDepthResult.kind === 'ok') {
    const poseDepth = poseDepthResult.data
    if (
      poseDepth.frame_id !== poseDepthRenderedFrameId &&
      poseDepth.frame_id !== poseDepthPendingFrameId
    ) {
      void refreshPoseDepth(poseDepth)
    }
  } else if (poseDepthResult.kind === 'empty' && poseDepthRenderedFrameId !== null) {
    poseDepthRenderedFrameId = null
    frameProcessorPanel.setDepthFrame(null, null)
  }

  applyStatus()
  window.setTimeout(() => {
    void pollState()
  }, STATE_POLL_MS)
}

void pollPreview()
void pollState()
window.requestAnimationFrame(tickDisplayClocks)
