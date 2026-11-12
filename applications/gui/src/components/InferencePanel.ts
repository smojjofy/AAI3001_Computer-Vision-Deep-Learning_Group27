import type {
  InferenceFrameData,
} from '../types/InferenceFrame'

type InferenceView =
  | 'pose'
  | 'depth'
  | 'segmentation'
  | 'flow'

const VIEW_NAMES: Record<
  InferenceView,
  string
> = {
  pose: 'Pose',
  depth: 'Depth',
  segmentation: 'Segmentation',
  flow: 'Optical Flow',
}

export class InferencePanel {
  private element: HTMLElement

  private canvas: HTMLCanvasElement
  private placeholder: HTMLElement

  private frameLabel: HTMLElement
  private processingLabel: HTMLElement
  private sourceLabel: HTMLElement
  private timestampLabel: HTMLElement

  private activeView:
    InferenceView = 'pose'

  private currentData:
    InferenceFrameData | null = null

  constructor() {
    this.element =
      document.createElement('section')

    this.element.className =
      'viewer-panel'

    this.element.innerHTML = `
      <div class="panel-header">
        <h2>Inference Output</h2>
      </div>

      <div class="viewer-content processor-content">
        <div class="processor-tabs">
          <button
            class="processor-tab"
            data-view="pose"
            type="button"
          >
            Pose
          </button>

          <button
            class="processor-tab"
            data-view="depth"
            type="button"
          >
            Depth
          </button>

          <button
            class="processor-tab"
            data-view="segmentation"
            type="button"
          >
            Segmentation
          </button>

          <button
            class="processor-tab"
            data-view="flow"
            type="button"
          >
            Optical Flow
          </button>
        </div>

        <div class="processor-viewport">
          <canvas
            class="processor-canvas"
          ></canvas>

          <div
            class="placeholder processor-placeholder"
          >
            Waiting for inference output
          </div>
        </div>
      </div>

      <div class="panel-footer">
        <span class="inference-frame-label">
          Frame: --
        </span>
        <span class="inference-source-label">
          Source: --
        </span>

        <span class="inference-timestamp-label">
          Capture: --
        </span>
        <span class="inference-processing-label">
          Processing: -- ms
        </span>
      </div>
    `

    this.canvas =
      this.element.querySelector<HTMLCanvasElement>(
        '.processor-canvas'
      )!

    this.placeholder =
      this.element.querySelector<HTMLElement>(
        '.processor-placeholder'
      )!

    this.frameLabel =
      this.element.querySelector<HTMLElement>(
        '.inference-frame-label'
      )!

    this.sourceLabel =
      this.element.querySelector<HTMLElement>(
        '.inference-source-label'
      )!

    this.timestampLabel =
      this.element.querySelector<HTMLElement>(
        '.inference-timestamp-label'
      )!

    this.processingLabel =
      this.element.querySelector<HTMLElement>(
        '.inference-processing-label'
      )!

    const buttons =
      this.element.querySelectorAll<HTMLButtonElement>(
        '.processor-tab'
      )

    buttons.forEach((button) => {
      button.addEventListener(
        'click',
        () => {
          const view =
            button.dataset
              .view as InferenceView

          this.setActiveView(view)
        }
      )
    })

    this.setActiveView('pose')
  }

  getElement(): HTMLElement {
    return this.element
  }

  setInferenceData(
    data: InferenceFrameData
  ): void {
    this.currentData = data

    this.frameLabel.textContent =
      `Frame: ${data.frameId}`

    this.sourceLabel.textContent =
      `Source: ${data.sourceId ?? '--'}`

    this.timestampLabel.textContent =
      `Capture: ${data.captureTimestampNs ?? '--'}`
    
    this.renderActiveView()
  }

  setActiveView(
    view: InferenceView
  ): void {
    this.activeView = view

    const buttons =
      this.element.querySelectorAll<HTMLButtonElement>(
        '.processor-tab'
      )

    buttons.forEach(
      (button) => {
        button.classList.toggle(
          'active',
          button.dataset.view === view
        )
      }
    )

    this.renderActiveView()
  }

  private renderActiveView(): void {
    if (!this.currentData) {
      this.showPlaceholder(
        `Waiting for ${
          VIEW_NAMES[this.activeView]
        } output`
      )

      return
    }

    switch (this.activeView) {
      case 'pose':
        this.renderPose()
        break

      case 'depth':
        this.renderDepth()
        break

      case 'segmentation':
        this.renderImage(
          this.currentData.segmentation?.image,
          this.currentData.segmentation?.processingMs
        )
        break

      case 'flow':
        this.renderImage(
          this.currentData.flow?.image,
          this.currentData.flow?.processingMs
        )
        break
    }
  }

  private renderImage(
    source: CanvasImageSource | undefined,
    processingMs: number | undefined
  ): void {
    if (
      !this.currentData ||
      !source
    ) {
      this.showPlaceholder(
        `${
          VIEW_NAMES[this.activeView]
        } output unavailable`
      )

      return
    }

    const context =
      this.canvas.getContext('2d')

    if (!context) {
      throw new Error(
        'Unable to create inference canvas context'
      )
    }

    this.canvas.width =
      this.currentData.width

    this.canvas.height =
      this.currentData.height

    context.clearRect(
      0,
      0,
      this.canvas.width,
      this.canvas.height
    )

    context.drawImage(
      source,
      0,
      0,
      this.canvas.width,
      this.canvas.height
    )

    this.showCanvas()

    this.setProcessingTime(
      processingMs
    )
  }

  private renderPose(): void {
    if (
      !this.currentData ||
      !this.currentData.poseDepth
    ) {
      this.showPlaceholder(
        'Pose output unavailable'
      )

      return
    }

    const pose =
      this.currentData.poseDepth

    if (!pose.detected) {
      this.showPlaceholder(
        'No pose detected'
      )

      this.setProcessingTime(
        pose.timingsMs?.pose
      )

      return
    }

    const context =
      this.canvas.getContext('2d')

    if (!context) {
      throw new Error(
        'Unable to create inference canvas context'
      )
    }

    this.canvas.width =
      this.currentData.width

    this.canvas.height =
      this.currentData.height

    context.fillStyle = '#081018'

    context.fillRect(
      0,
      0,
      this.canvas.width,
      this.canvas.height
    )

    const landmarks =
      pose.landmarks

    /*
     * Mock-fixture skeleton connections.
     * These indices correspond only to the
     * simplified mock landmark set.
     */
    const connections: Array<
      [number, number]
    > = [
      [0, 1],
      [0, 2],
      [1, 2],
      [1, 3],
      [2, 4],
      [1, 5],
      [2, 6],
      [5, 6],
      [5, 7],
      [6, 8],
    ]

    context.strokeStyle = '#22d3ee'
    context.lineWidth = 4

    connections.forEach(
      ([startIndex, endIndex]) => {
        const start =
          landmarks[startIndex]

        const end =
          landmarks[endIndex]

        if (!start || !end) {
          return
        }

        context.beginPath()

        context.moveTo(
          start.x * this.canvas.width,
          start.y * this.canvas.height
        )

        context.lineTo(
          end.x * this.canvas.width,
          end.y * this.canvas.height
        )

        context.stroke()
      }
    )

    context.fillStyle = '#f59e0b'

    landmarks.forEach(
      (landmark) => {
        context.beginPath()

        context.arc(
          landmark.x *
            this.canvas.width,
          landmark.y *
            this.canvas.height,
          6,
          0,
          Math.PI * 2
        )

        context.fill()
      }
    )

    context.fillStyle =
      'rgba(0, 0, 0, 0.65)'

    context.fillRect(
      18,
      18,
      180,
      50
    )

    context.fillStyle = '#ffffff'

    context.font =
      '600 20px system-ui'

    context.fillText(
      'Pose Output',
      32,
      50
    )

    this.showCanvas()

    this.setProcessingTime(
      pose.timingsMs?.pose
    )
  }

 private renderDepth(): void {
  if (!this.currentData) {
    this.showPlaceholder(
      'Depth output unavailable'
    )

    return
  }

  const poseDepth =
    this.currentData.poseDepth

  if (!poseDepth || !poseDepth.depthImage) {
    this.showPlaceholder(
      'Depth output unavailable'
    )

    return
  }

  const depthImage =
    poseDepth.depthImage

  const context =
    this.canvas.getContext('2d')

  if (!context) {
    throw new Error(
      'Unable to create inference canvas context'
    )
  }

  this.canvas.width =
    this.currentData.width

  this.canvas.height =
    this.currentData.height

  context.clearRect(
    0,
    0,
    this.canvas.width,
    this.canvas.height
  )

  context.drawImage(
    depthImage,
    0,
    0,
    this.canvas.width,
    this.canvas.height
  )

  context.fillStyle =
    'rgba(0, 0, 0, 0.65)'

  context.fillRect(
    18,
    18,
    220,
    50
  )

  context.fillStyle = '#ffffff'

  context.font =
    '600 20px system-ui'

  const depthLabel =
    poseDepth.depthKind === 'metric_depth'
      ? 'Metric Depth'
      : poseDepth.depthKind === 'relative_depth'
        ? 'Relative Depth'
        : 'Depth Output'

  context.fillText(
    depthLabel,
    32,
    50
  )

  this.showCanvas()

  this.setProcessingTime(
    poseDepth.timingsMs?.depth
  )
}




  private showCanvas(): void {
    this.canvas.style.display =
      'block'

    this.placeholder.style.display =
      'none'
  }

  private showPlaceholder(
    message: string
  ): void {
    this.canvas.style.display =
      'none'

    this.placeholder.style.display =
      'flex'

    this.placeholder.textContent =
      message

    this.processingLabel.textContent =
      'Processing: -- ms'
  }

  private setProcessingTime(
    processingMs: number | undefined
  ): void {
    this.processingLabel.textContent =
      processingMs !== undefined
        ? `Processing: ${processingMs.toFixed(1)} ms`
        : 'Processing: -- ms'
  }
}