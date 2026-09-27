import type {
  ProcessorFrameData,
  ProcessorView,
} from '../types/ProcessorFrame'

const VIEW_NAMES: Record<
  ProcessorView,
  string
> = {
  rgba: 'RGBA',
  mask: 'Foreground Mask',
  depth: 'Depth',
  motion: 'Motion / Optical Flow',
}

export class FrameProcessorPanel {
  private element: HTMLElement

  private canvas: HTMLCanvasElement
  private placeholder: HTMLElement

  private frameLabel: HTMLElement
  private processingLabel: HTMLElement

  private activeView:
    ProcessorView = 'rgba'

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
        <div class="processor-tabs">
          <button
            class="processor-tab"
            data-view="rgba"
            type="button"
          >
            RGBA
          </button>

          <button
            class="processor-tab"
            data-view="mask"
            type="button"
          >
            Mask
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
            data-view="motion"
            type="button"
          >
            Motion
          </button>
        </div>

        <div class="processor-viewport">
          <canvas
            class="processor-canvas"
          ></canvas>

          <div
            class="placeholder processor-placeholder"
          >
            Waiting for RGBA output
          </div>
        </div>
      </div>

      <div class="panel-footer">
        <span class="processor-frame-label">
          Frame: --
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

    this.processingLabel =
      this.element.querySelector<HTMLElement>(
        '.processor-processing-label'
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
              .view as ProcessorView

          this.setActiveView(
            view
          )
        }
      )
    })

    this.setActiveView(
      'rgba'
    )
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

    this.processingLabel.textContent =
      data.processingMs !== undefined
        ? `Processing: ${data.processingMs.toFixed(1)} ms`
        : 'Processing: -- ms'

    this.renderActiveView()
  }

  setActiveView(
    view: ProcessorView
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
          VIEW_NAMES[
            this.activeView
          ]
        } output`
      )

      return
    }

    const source =
      this.currentData.outputs[
        this.activeView
      ]

    if (!source) {
      this.showPlaceholder(
        `${
          VIEW_NAMES[
            this.activeView
          ]
        } output unavailable`
      )

      return
    }

    const context =
      this.canvas.getContext(
        '2d'
      )

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
      source,
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