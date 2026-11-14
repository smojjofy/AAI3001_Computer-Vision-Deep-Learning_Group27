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

const POSE_CONNECTIONS: Array<
  [number, number]
> = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 7],

  [0, 4],
  [4, 5],
  [5, 6],
  [6, 8],

  [9, 10],

  [11, 12],

  [11, 13],
  [13, 15],
  [15, 17],
  [15, 19],
  [15, 21],
  [17, 19],

  [12, 14],
  [14, 16],
  [16, 18],
  [16, 20],
  [16, 22],
  [18, 20],

  [11, 23],
  [12, 24],
  [23, 24],

  [23, 25],
  [24, 26],

  [25, 27],
  [26, 28],

  [27, 29],
  [28, 30],

  [29, 31],
  [30, 32],

  [27, 31],
  [28, 32],
]

const LABELLED_JOINTS: Record<
  number,
  string
> = {
  0: 'Nose',
  15: 'L wrist',
  16: 'R wrist',
  25: 'L knee',
  26: 'R knee',
  27: 'L ankle',
  28: 'R ankle',
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
        this.renderSegmentation()
        break

      case 'flow':
        this.renderFlow()
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

    context.strokeStyle = '#22d3ee'
    context.lineWidth = 4

    POSE_CONNECTIONS.forEach(
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

   landmarks.forEach(
    (landmark, index) => {
      const x =
        landmark.x *
        this.canvas.width

      const y =
        landmark.y *
        this.canvas.height

      const depth =
        pose.jointDepth[index]

      const hasDepth =
        depth !== null &&
        depth !== undefined &&
        Number.isFinite(depth)

      context.fillStyle =
        hasDepth
          ? '#f59e0b'
          : '#6b7280'

      context.beginPath()

      const landmarkRadius =
        index <= 10
          ? 3
          : 5

      context.arc(
        x,
        y,
        landmarkRadius,
        0,
        Math.PI * 2
      )

      context.fill()

      const label =
        LABELLED_JOINTS[index]

      if (
        label &&
        hasDepth
      ) {
        const unit =
          pose.depthKind === 'metric_depth'
            ? ' m'
            : ''

        context.fillStyle =
          '#ffffff'

        context.font =
          '12px system-ui'

        const labelX =
          index === 0
            ? x + 14
            : x + 8

        const labelY =
          index === 0
            ? y - 14
            : y - 6

        context.fillText(
          `${label} ${depth.toFixed(2)}${unit}`,
          labelX,
          labelY
        )
      }
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
      'Pose + Depth',
      32,
      48
    )

    context.font =
      '13px system-ui'

    const depthUnit =
      pose.depthKind === 'metric_depth'
        ? ' m'
        : ''

    const bodyDepthText =
      pose.bodyDepth !== null &&
      pose.bodyDepth !== undefined &&
      Number.isFinite(pose.bodyDepth)
        ? `${pose.bodyDepth.toFixed(2)}${depthUnit}`
        : '--'

    context.fillText(
      `Body depth: ${bodyDepthText}`,
      32,
      70
    )

    context.fillText(
      `Depth source: ${pose.depthSource ?? '--'}`,
      32,
      90
    )

    const totalMs =
      pose.timingsMs?.total

    context.fillText(
      totalMs !== undefined
        ? `Total: ${totalMs.toFixed(1)} ms`
        : 'Total: -- ms',
      32,
      110
    )

    this.showCanvas()

    this.setProcessingTime(
      pose.timingsMs?.pose
    )
  }

private renderFlow(): void {
  if (!this.currentData) {
    this.showPlaceholder(
      'Optical Flow output unavailable'
    )

    return
  }

  const flow =
    this.currentData.flow

  if (
    !flow ||
    !flow.image
  ) {
    this.showPlaceholder(
      'Optical Flow output unavailable'
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
    flow.image,
    0,
    0,
    this.canvas.width,
    this.canvas.height
  )

  context.fillStyle =
    'rgba(0, 0, 0, 0.72)'

  context.fillRect(
    18,
    18,
    230,
    70
  )

  context.fillStyle = '#ffffff'

  context.font =
    '600 20px system-ui'

  context.fillText(
    'Optical Flow',
    32,
    48
  )

  context.font =
    '13px system-ui'

  const processingMs =
    flow.processingMs

  context.fillText(
    processingMs !== undefined
      ? `Processing: ${processingMs.toFixed(1)} ms`
      : 'Processing: -- ms',
    32,
    70
  )

  this.showCanvas()

  this.setProcessingTime(
    processingMs
  )
}

private renderSegmentation(): void {
  if (!this.currentData) {
    this.showPlaceholder(
      'Segmentation output unavailable'
    )

    return
  }

  const segmentation =
    this.currentData.segmentation

  if (
    !segmentation ||
    !segmentation.image
  ) {
    this.showPlaceholder(
      'Segmentation output unavailable'
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
    segmentation.image,
    0,
    0,
    this.canvas.width,
    this.canvas.height
  )

  context.fillStyle =
    'rgba(0, 0, 0, 0.72)'

  context.fillRect(
    18,
    18,
    230,
    70
  )

  context.fillStyle = '#ffffff'

  context.font =
    '600 20px system-ui'

  context.fillText(
    'Segmentation Mask',
    32,
    48
  )

  context.font =
    '13px system-ui'

  const processingMs =
    segmentation.processingMs

  context.fillText(
    processingMs !== undefined
      ? `Processing: ${processingMs.toFixed(1)} ms`
      : 'Processing: -- ms',
    32,
    70
  )

  this.showCanvas()

  this.setProcessingTime(
    processingMs
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

  if (
    !poseDepth ||
    !poseDepth.depthImage
  ) {
    this.showPlaceholder(
      'Depth output unavailable'
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
    poseDepth.depthImage,
    0,
    0,
    this.canvas.width,
    this.canvas.height
  )

  /*
   * Depth metadata overlay.
   */
  context.fillStyle =
    'rgba(0, 0, 0, 0.72)'

  context.fillRect(
    18,
    18,
    260,
    112
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
    48
  )

  context.font =
    '13px system-ui'

  const sourceLabel =
    poseDepth.depthSource
      ? poseDepth.depthSource.toUpperCase()
      : '--'

  context.fillText(
    `Source: ${sourceLabel}`,
    32,
    70
  )

  context.fillText(
    `Model: ${poseDepth.depthModelName ?? '--'}`,
    32,
    90
  )

  const depthMs =
    poseDepth.timingsMs?.depth

  context.fillText(
    depthMs !== undefined
      ? `Depth processing: ${depthMs.toFixed(1)} ms`
      : 'Depth processing: -- ms',
    32,
    110
  )

  this.showCanvas()

  this.setProcessingTime(
    depthMs
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