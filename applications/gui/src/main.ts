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

let lastSplatFrameRequested: string | null = null
let splatFetchInFlight = false

let lastMaskRefreshCount = -1
let maskFetchInFlight = false

let lastMotionFrameId: number | null = null
let activeMotion: MotionPayload | null = null
let lastPoseDepthFrameId: number | null = null
let poseDepthFetchInFlight = false

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
    return
  }
  splatFetchInFlight = true
  try {
    const result = await fetchSplat()
    if (result.kind === 'ok') {
      splatOutputPanel.setSplatFrame(result.data)
      // Keep the reuse warp only while it is newer than this splat state.
      // A refresh always replaces the motion payload with 204, which clears it.
      if (
        activeMotion !== null &&
        Number(result.data.header.frame_id) >= activeMotion.frame_id
      ) {
        activeMotion = null
        splatOutputPanel.setMotionWarp(null)
      }
    } else if (result.kind === 'empty') {
      lastSplatFrameRequested = null
    } else {
      console.warn(`Splat fetch failed: ${result.error}`)
      lastSplatFrameRequested = null
    }
  } catch (error) {
    console.warn(`Splat frame ${frameId} could not be rendered: ${describeError(error)}`)
    lastSplatFrameRequested = null
  } finally {
    splatFetchInFlight = false
  }
}

async function refreshMask(refreshCount: number): Promise<void> {
  if (maskFetchInFlight) {
    return
  }
  maskFetchInFlight = true
  try {
    const result = await fetchMask()
    if (result.kind === 'ok') {
      frameProcessorPanel.setMaskImage(result.data, { refreshCount })
    } else {
      frameProcessorPanel.setMaskImage(null, null)
    }
  } finally {
    maskFetchInFlight = false
  }
}

async function refreshPoseDepth(poseDepth: PoseDepthFrame): Promise<void> {
  if (poseDepthFetchInFlight) {
    return
  }
  poseDepthFetchInFlight = true
  try {
    const image = await decodeJpegBase64(poseDepth.depth_jpeg_base64)
    frameProcessorPanel.setDepthFrame(image, poseDepth)
  } catch {
    frameProcessorPanel.setDepthFrame(null, poseDepth)
  } finally {
    poseDepthFetchInFlight = false
  }
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
    if (splatFrameId !== null && splatFrameId !== lastSplatFrameRequested) {
      lastSplatFrameRequested = splatFrameId
      void refreshSplat(splatFrameId)
    }
    if (
      diagnostics.refresh_count !== lastMaskRefreshCount &&
      diagnostics.refresh_count > 0
    ) {
      lastMaskRefreshCount = diagnostics.refresh_count
      void refreshMask(diagnostics.refresh_count)
    }
  } else {
    inferencePanel.update(null)
    pipelineHealth =
      diagnosticsResult.kind === 'error' ? 'disconnected' : 'waiting'
  }

  if (motionResult.kind === 'ok') {
    const motion = motionResult.data
    if (motion.frame_id !== lastMotionFrameId) {
      lastMotionFrameId = motion.frame_id
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
    if (poseDepth.frame_id !== lastPoseDepthFrameId) {
      lastPoseDepthFrameId = poseDepth.frame_id
      void refreshPoseDepth(poseDepth)
    }
  } else if (poseDepthResult.kind === 'empty' && lastPoseDepthFrameId !== null) {
    lastPoseDepthFrameId = null
    frameProcessorPanel.setDepthFrame(null, null)
  }

  applyStatus()
  window.setTimeout(() => {
    void pollState()
  }, STATE_POLL_MS)
}

void pollPreview()
void pollState()
