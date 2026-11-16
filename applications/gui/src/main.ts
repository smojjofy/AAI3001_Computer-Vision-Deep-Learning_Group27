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

import { isKeyAtLeast, retryDelayMs, splatKey } from './api/identity'
import { FrameProcessorPanel } from './components/FrameProcessorPanel'
import { InferencePanel } from './components/InferencePanel'
import { OriginalInputPanel } from './components/OriginalInputPanel'
import { SplatOutputPanel } from './components/SplatOutputPanel'

import type {
  DiagnosticsPayload,
  MotionPayload,
  PoseDepthFrame,
  SplatHeader,
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

// Fetch gating. Identity is the complete (source, generation, frame) tuple,
// because a source or process restart can reuse a frame id. A requested id is
// recorded only when its fetch starts; a newer id waits and is fetched when the
// current fetch ends. A failed fetch retries the same identity after 500 ms.
let splatRenderedKey: string | null = null
let splatPendingKey: string | null = null
let splatFetchInFlight = false
let splatRetryAt = 0
let splatRetryTimer: number | null = null

let maskRenderedKey: string | null = null
let maskPendingKey: string | null = null
let maskPendingCount: number | null = null
let lastMaskRefreshCount = -1
let maskFetchInFlight = false
let maskRetryAt = 0
let maskRetryTimer: number | null = null
const maskStaleAttempts = new Map<string, number>()

let lastMotionFrameId: number | null = null
let activeMotion: MotionPayload | null = null

let poseDepthRenderedKey: string | null = null
let poseDepthPending: PoseDepthFrame | null = null
let poseDepthFetchInFlight = false
let poseDepthRetryAt = 0
let poseDepthRetryTimer: number | null = null

/**
 * Viewer-clock anchor for the server's latched invalid-person age.
 * The panel holds the last valid state for 500 ms, then fades it over 500 ms.
 * The clock is released only when a state that is new for the loss generation
 * has been installed.
 */
let personLossAnchorMs: number | null = null
let personLossGeneration: number | null = null
let personLossLastFrameId = 0

/** Identity of the splat state currently shown in the panel. */
let splatRenderedGeneration = 0
let splatRenderedFrameInt = 0

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

function poseKey(pose: PoseDepthFrame): string {
  const generation = Number((pose as PoseDepthFrame & { generation?: number }).generation ?? 0)
  return splatKey(pose.source_id, generation, pose.frame_id)
}

async function refreshSplat(key: string): Promise<void> {
  if (splatFetchInFlight) {
    splatPendingKey = key
    return
  }
  if (performance.now() < splatRetryAt) {
    splatPendingKey = key
    scheduleSplatRetry()
    return
  }
  splatFetchInFlight = true
  splatPendingKey = null
  try {
    const result = await fetchSplat()
    if (result.kind === 'ok') {
      const header = result.data.header
      const rendered = splatKey(
        header.source_id,
        Number(header.generation ?? 0),
        header.frame_id,
      )
      splatRenderedKey = rendered
      splatRenderedGeneration = Number(header.generation ?? 0)
      splatRenderedFrameInt = Number(header.frame_id_int ?? header.frame_id)
      splatOutputPanel.setSplatFrame(result.data)
      // Release the loss clock only when a state that is new for this run has
      // actually been installed. Clearing it from diagnostics would let the old
      // splats regain full alpha before the new state arrives.
      releasePersonLossIfNewState(header)
      if (activeMotion !== null && Number(header.frame_id) >= activeMotion.frame_id) {
        activeMotion = null
        splatOutputPanel.setMotionWarp(null)
      }
      splatRetryAt = 0
    } else if (result.kind === 'empty') {
      splatRenderedKey = null
    } else {
      console.warn(`Splat fetch failed: ${result.error}`)
      splatRetryAt = performance.now() + 500
      splatPendingKey = key
      scheduleSplatRetry()
    }
  } catch (error) {
    console.warn(`Splat ${key} could not be rendered: ${describeError(error)}`)
    splatRetryAt = performance.now() + 500
    splatPendingKey = key
    scheduleSplatRetry()
  } finally {
    splatFetchInFlight = false
  }
  const next = splatPendingKey
  if (next !== null && next !== splatRenderedKey) {
    splatPendingKey = null
    void refreshSplat(next)
  }
}

/** Fetch the pending splat identity when the retry delay ends. */
function scheduleSplatRetry(): void {
  if (splatRetryTimer !== null) {
    return
  }
  splatRetryTimer = window.setTimeout(() => {
    splatRetryTimer = null
    const pending = splatPendingKey
    if (pending !== null && !splatFetchInFlight) {
      splatPendingKey = null
      void refreshSplat(pending)
    }
  }, retryDelayMs(splatRetryAt, performance.now()))
}

async function refreshMask(key: string, refreshCount: number): Promise<void> {
  if (maskFetchInFlight) {
    maskPendingKey = key
    maskPendingCount = refreshCount
    return
  }
  if (performance.now() < maskRetryAt) {
    maskPendingKey = key
    maskPendingCount = refreshCount
    scheduleMaskRetry()
    return
  }
  maskFetchInFlight = true
  const requested = key
  maskPendingKey = null
  maskPendingCount = null
  try {
    const result = await fetchMask()
    if (result.kind === 'ok') {
      // The mask headers carry the identity of the state that produced the
      // image. Never display a mask older than the requested state; retry once
      // because a publish can land between the diagnostics and mask reads.
      const identity = result.data.identity
      const identityKey = identity === null
        ? null
        : splatKey(identity.source_id, identity.generation, identity.frame_id)
      const acceptable = identityKey === null || identityKey === requested
        || isKeyAtLeast(identityKey, requested)
      if (!acceptable) {
        const attempts = maskStaleAttempts.get(requested) ?? 0
        if (attempts < 3) {
          maskStaleAttempts.set(requested, attempts + 1)
          maskPendingKey = requested
          maskPendingCount = refreshCount
          maskRetryAt = performance.now() + 300
          scheduleMaskRetry()
          return
        }
        console.warn(`Mask identity ${identityKey} is older than ${requested}; accepting after 3 tries`)
      }
      maskRenderedKey = identityKey ?? requested
      maskStaleAttempts.clear()
      frameProcessorPanel.setMaskImage(result.data.bitmap, { refreshCount })
      maskRetryAt = 0
    } else {
      frameProcessorPanel.setMaskImage(null, null)
      maskRetryAt = performance.now() + 500
      maskPendingKey = requested
      maskPendingCount = refreshCount
      scheduleMaskRetry()
    }
  } finally {
    maskFetchInFlight = false
  }
  const next = maskPendingKey
  if (next !== null && next !== maskRenderedKey) {
    const nextCount = maskPendingCount ?? refreshCount
    maskPendingKey = null
    maskPendingCount = null
    void refreshMask(next, nextCount)
  }
}

function scheduleMaskRetry(): void {
  if (maskRetryTimer !== null) {
    return
  }
  maskRetryTimer = window.setTimeout(() => {
    maskRetryTimer = null
    const pending = maskPendingKey
    if (pending !== null && !maskFetchInFlight) {
      const count = maskPendingCount ?? lastMaskRefreshCount
      maskPendingKey = null
      maskPendingCount = null
      void refreshMask(pending, count)
    }
  }, retryDelayMs(maskRetryAt, performance.now()))
}

async function refreshPoseDepth(poseDepth: PoseDepthFrame): Promise<void> {
  if (poseDepthFetchInFlight) {
    poseDepthPending = poseDepth
    return
  }
  if (performance.now() < poseDepthRetryAt) {
    poseDepthPending = poseDepth
    schedulePoseDepthRetry()
    return
  }
  poseDepthFetchInFlight = true
  poseDepthPending = null
  try {
    const image = await decodeJpegBase64(poseDepth.depth_jpeg_base64)
    poseDepthRenderedKey = poseKey(poseDepth)
    frameProcessorPanel.setDepthFrame(image, poseDepth)
    poseDepthRetryAt = 0
  } catch {
    frameProcessorPanel.setDepthFrame(null, poseDepth)
    poseDepthRetryAt = performance.now() + 500
    poseDepthPending = poseDepth
    schedulePoseDepthRetry()
  } finally {
    poseDepthFetchInFlight = false
  }
  const next = poseDepthPending
  if (next !== null && poseKey(next) !== poseDepthRenderedKey) {
    poseDepthPending = null
    void refreshPoseDepth(next)
  }
}

function schedulePoseDepthRetry(): void {
  if (poseDepthRetryTimer !== null) {
    return
  }
  poseDepthRetryTimer = window.setTimeout(() => {
    poseDepthRetryTimer = null
    const pending = poseDepthPending
    if (pending !== null && !poseDepthFetchInFlight) {
      poseDepthPending = null
      void refreshPoseDepth(pending)
    }
  }, retryDelayMs(poseDepthRetryAt, performance.now()))
}

/** Release the viewer loss clock only after a new state for this run is shown. */
function releasePersonLossIfNewState(header: SplatHeader): void {
  if (personLossAnchorMs === null) {
    return
  }
  const generation = Number(header.generation ?? 0)
  const frameId = Number(header.frame_id_int ?? header.frame_id)
  if (
    personLossGeneration === null ||
    generation !== personLossGeneration ||
    frameId > personLossLastFrameId
  ) {
    personLossAnchorMs = null
    personLossGeneration = null
    splatOutputPanel.setPersonLoss(null)
  }
}

/** Anchor the server's latched person-loss age on the viewer clock. */
function schedulePersonLoss(ageMs: number | null): void {
  if (ageMs === null) {
    // Do not clear here. releasePersonLossIfNewState owns the release so the
    // old splats cannot regain full alpha before the new state is displayed.
    return
  }
  const now = performance.now()
  const currentAge =
    personLossAnchorMs === null ? 0 : now - personLossAnchorMs
  if (personLossAnchorMs === null || ageMs < currentAge - 50) {
    // First loss, or the server latch restarted after a recovery.
    personLossAnchorMs = now - Math.max(0, ageMs)
    personLossGeneration = splatRenderedGeneration
    personLossLastFrameId = splatRenderedFrameInt
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
    const activeIdentity = diagnostics.active_identity ?? null
    if (splatFrameId !== null && activeIdentity !== null) {
      const key = splatKey(
        activeIdentity.source_id,
        activeIdentity.generation,
        splatFrameId,
      )
      if (key !== splatRenderedKey && key !== splatPendingKey) {
        void refreshSplat(key)
      }
      if (diagnostics.refresh_count > 0) {
        const maskKey = splatKey(
          activeIdentity.source_id,
          activeIdentity.generation,
          activeIdentity.frame_id,
        )
        if (maskKey !== maskRenderedKey && maskKey !== maskPendingKey) {
          lastMaskRefreshCount = diagnostics.refresh_count
          void refreshMask(maskKey, diagnostics.refresh_count)
        }
      }
    }
    // The server latches the invalid-person event. Anchor it once on the
    // viewer clock; the browser then owns the hold and fade. Do not clear the
    // clock from a null age: the old splats would regain full alpha too early.
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
    const key = poseKey(poseDepth)
    const pendingKey = poseDepthPending === null ? null : poseKey(poseDepthPending)
    if (key !== poseDepthRenderedKey && key !== pendingKey) {
      void refreshPoseDepth(poseDepth)
    }
  } else if (poseDepthResult.kind === 'empty' && poseDepthRenderedKey !== null) {
    poseDepthRenderedKey = null
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
