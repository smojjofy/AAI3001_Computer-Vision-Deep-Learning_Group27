import './style.css'

import { OriginalInputPanel } from './components/OriginalInputPanel'
import { FrameProcessorPanel } from './components/FrameProcessorPanel'
import { SplatOutputPanel } from './components/SplatOutputPanel'
import {createMockPipelineFrame,} from './data/mockPipelineFrame'

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

      <div class="status">
        <span class="status-dot"></span>
        OFFLINE
      </div>
    </header>

    <main class="viewer-grid" id="viewer-grid">
    </main>
  </div>
`

const viewerGrid = document.querySelector<HTMLElement>('#viewer-grid')

if (!viewerGrid) {
  throw new Error('Viewer grid not found')
}

const pipelineFrame = createMockPipelineFrame(1)

/*
 * Original Input
 */
const originalInputPanel = new OriginalInputPanel()
viewerGrid.appendChild(originalInputPanel.getElement())

originalInputPanel.showImage(
  pipelineFrame.original.image,
  pipelineFrame.original.width,
  pipelineFrame.original.height,
  {
    frameId:
      pipelineFrame.frameId,

    timestamp:
      pipelineFrame.timestamp,
  }
)
/*
 * Frame Processor Output
 */
/*
 * Frame Processor Output
 */
const frameProcessorPanel = new FrameProcessorPanel()
viewerGrid.appendChild(frameProcessorPanel.getElement())

frameProcessorPanel.setProcessorData(pipelineFrame.processor)

/*
 * Splat Constructor Output
 */
/*
 * Splat Constructor Output
 */
const splatOutputPanel =
  new SplatOutputPanel()

viewerGrid.appendChild(
  splatOutputPanel.getElement()
)

splatOutputPanel.setSplatData(pipelineFrame.splat,pipelineFrame.frameId)