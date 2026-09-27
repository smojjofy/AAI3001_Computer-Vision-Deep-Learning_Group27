export interface OriginalFrameMeta {
  frameId?: number | string
  timestamp?: string
  sourceId?: string
}

export class OriginalInputPanel {
  private element: HTMLElement
  private canvas: HTMLCanvasElement
  private placeholder: HTMLElement
  private frameLabel: HTMLElement
  private timestampLabel: HTMLElement

  constructor() {
    this.element = document.createElement('section')
    this.element.className = 'viewer-panel'

    this.element.innerHTML = `
      <div class="panel-header">
        <h2>Original Input</h2>
      </div>

      <div class="viewer-content">
        <div class="media-container">
          <canvas class="input-canvas"></canvas>

          <div class="placeholder original-placeholder">
            Waiting for original input
          </div>
        </div>
      </div>

      <div class="panel-footer">
        <span class="frame-label">Frame: --</span>
        <span class="timestamp-label">Waiting for input</span>
      </div>
    `

    this.canvas = this.element.querySelector<HTMLCanvasElement>(
      '.input-canvas'
    )!

    this.placeholder = this.element.querySelector<HTMLElement>(
      '.original-placeholder'
    )!

    this.frameLabel = this.element.querySelector<HTMLElement>(
      '.frame-label'
    )!

    this.timestampLabel = this.element.querySelector<HTMLElement>(
      '.timestamp-label'
    )!

    this.showWaiting()
  }

  getElement(): HTMLElement {
    return this.element
  }

  showWaiting(): void {
    this.canvas.style.display = 'none'
    this.placeholder.style.display = 'flex'

    this.frameLabel.textContent = 'Frame: --'
    this.timestampLabel.textContent = 'Waiting for input'
  }

  showImage(
    image: CanvasImageSource,
    width: number,
    height: number,
    meta: OriginalFrameMeta = {}
  ): void {
    const context = this.canvas.getContext('2d')

    if (!context) {
      throw new Error('Unable to create 2D canvas context')
    }

    this.canvas.width = width
    this.canvas.height = height

    context.clearRect(0, 0, width, height)
    context.drawImage(image, 0, 0, width, height)

    this.canvas.style.display = 'block'
    this.placeholder.style.display = 'none'

    this.frameLabel.textContent =
      `Frame: ${meta.frameId ?? '--'}`

    this.timestampLabel.textContent =
      meta.timestamp ?? 'Input received'
  }
}