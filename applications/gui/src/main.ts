import './style.css'

import { InferencePanel } from './components/InferencePanel'
import { createMockPipelineFrame } from './data/mockPipelineFrame'
import { OriginalInputPanel } from './components/OriginalInputPanel'
import { FrameProcessorPanel } from './components/FrameProcessorPanel'
import { SplatOutputPanel } from './components/SplatOutputPanel'


interface PreviewFrame {
  schema_version: '1.0'
  frame_id: number
  source_id: string
  capture_timestamp_ns: string
  timestamp_origin: 'source_presentation' | 'decode_monotonic'
  width: number
  height: number
  jpeg_base64: string
  stream_health: 'closed' | 'connecting' | 'healthy' | 'reconnecting' | 'stalled'
}

const app = document.querySelector<HTMLDivElement>('#app')

if (!app) {
  throw new Error('App root element not found')
}

app.innerHTML = `
  <div class="app-shell">
    <header class="top-bar">
      <div>
        <h1>Real-Time 2.5D Reconstruction Viewer</h1>
        <p>Single Camera Pipeline</p>
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

const mockFrame =
  createMockPipelineFrame(
    1,
    'mock-camera',
    '0'
  )

// These panels remain fixture-driven until their own backends are connected.
const frameProcessorPanel =
  new FrameProcessorPanel()

viewerGrid.appendChild(
  frameProcessorPanel.getElement()
)

frameProcessorPanel.setProcessorData(
  mockFrame.processor
)

const inferencePanel =
  new InferencePanel()

viewerGrid.appendChild(
  inferencePanel.getElement()
)

const splatOutputPanel =
  new SplatOutputPanel()

viewerGrid.appendChild(
  splatOutputPanel.getElement()
)
inferencePanel.setInferenceData(
  mockFrame.inference
)

splatOutputPanel.setSplatData(
  mockFrame.splat,
  mockFrame.frameId,
)

function showStatus(label: string, live: boolean): void {
  statusLabel!.textContent = label
  status!.classList.toggle('is-live', live)
}

let previousFrameId: number | null = null
let previousSourceId: string | null = null
let firstCaptureNs: bigint | null = null
let previousOrigin: string | null = null

async function pollPreview(): Promise<void> {
  try {
    const response = await fetch('/api/preview', { cache: 'no-store' })
    if (response.status === 204) {
      const health = response.headers.get('X-Stream-Health')
      showStatus(health === 'closed' ? 'CAMERA CLOSED' : 'WAITING FOR INPUT', false)
      originalInputPanel.showWaiting(health === 'closed' ? 'Camera could not open or capture ended' : 'Waiting for camera frames')
      return
    }
    if (!response.ok) {
      throw new Error(`Preview HTTP ${response.status}`)
    }

    const frame = await response.json() as PreviewFrame
    if (frame.schema_version !== '1.0') {
      throw new Error('Unsupported preview frame version')
    }
    if (frame.stream_health !== 'healthy') {
      showStatus(frame.stream_health.toUpperCase(), false)
      originalInputPanel.showWaiting(`Camera ${frame.stream_health}`)
      return
    }

    showStatus('LIVE INPUT', true)
    if (frame.frame_id === previousFrameId && frame.source_id === previousSourceId) {
      return
    }
    const captureNs = BigInt(frame.capture_timestamp_ns)
    if (frame.source_id !== previousSourceId || frame.timestamp_origin !== previousOrigin || firstCaptureNs === null) {
      firstCaptureNs = captureNs
    }

    const image = new Image()
    image.src = `data:image/jpeg;base64,${frame.jpeg_base64}`
    await image.decode()
    originalInputPanel.showImage(image, frame.width, frame.height, {
      frameId: frame.frame_id,
      sourceId: frame.source_id,
      timestamp: `Capture +${(Number(captureNs - firstCaptureNs) / 1_000_000_000).toFixed(2)} s`,
    })
    previousFrameId = frame.frame_id
    previousSourceId = frame.source_id
    previousOrigin = frame.timestamp_origin
  } catch {
    showStatus('PREVIEW DISCONNECTED', false)
    originalInputPanel.showWaiting('Start the Python preview server')
  } finally {
    window.setTimeout(pollPreview, 100)
  }
}

void pollPreview()
