import type {
  ProcessorFrameData,
} from '../types/ProcessorFrame'

export class FrameProcessorPanel {
  private element: HTMLElement

  private canvas: HTMLCanvasElement
  private placeholder: HTMLElement

  private frameLabel: HTMLElement
  private processingLabel: HTMLElement
  private sourceLabel: HTMLElement
  private timestampLabel: HTMLElement

  private currentData:
    ProcessorFrameData | null = null

  constructor() {
    this.element =
      document.createElement('section')

    this.element.className =
      'viewer-panel'

    this.element.innerHTML = `
      <div class="panel-header">
        <h2>Frame Processor Output</h2>
      </div>

      <div class="viewer-content processor-content">
        <div class="processor-viewport">
          <canvas
            class="processor-canvas"
          ></canvas>

          <div
            class="placeholder processor-placeholder"
          >
            Waiting for processed frame
          </div>
        </div>
      </div>

      <div class="panel-footer">
        <span class="processor-frame-label">
          Frame: --
        </span>

        <span class="processor-source-label">
          Source: --
        </span>

        <span class="processor-timestamp-label">
          Capture: --
        </span>

        <span class="processor-processing-label">
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
        '.processor-frame-label'
      )!
    
    this.sourceLabel =
      this.element.querySelector<HTMLElement>(
        '.processor-source-label'
      )!

    this.timestampLabel =
      this.element.querySelector<HTMLElement>(
        '.processor-timestamp-label'
      )!
    
    this.processingLabel =
      this.element.querySelector<HTMLElement>(
        '.processor-processing-label'
      )!
  }

  getElement(): HTMLElement {
    return this.element
  }

  setProcessorData(
    data: ProcessorFrameData
  ): void {
    this.currentData = data

    this.frameLabel.textContent =
      `Frame: ${data.frameId}`

    this.sourceLabel.textContent =
      `Source: ${data.sourceId ?? '--'}`

    this.timestampLabel.textContent =
      `Capture: ${data.captureTimestampNs ?? '--'}`

    this.processingLabel.textContent =
      data.processingMs !== undefined
        ? `Processing: ${data.processingMs.toFixed(1)} ms`
        : 'Processing: -- ms'

    this.render()
  }

  private render(): void {
    if (
      !this.currentData ||
      !this.currentData.image
    ) {
      this.showPlaceholder(
        'Processed frame unavailable'
      )

      return
    }

    const context =
      this.canvas.getContext('2d')

    if (!context) {
      throw new Error(
        'Unable to create processor canvas context'
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
      this.currentData.image,
      0,
      0,
      this.canvas.width,
      this.canvas.height
    )

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
  }
}