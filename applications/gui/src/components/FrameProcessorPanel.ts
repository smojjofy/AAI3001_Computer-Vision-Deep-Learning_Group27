/*
 * Frame Processor Output panel.
 *
 * Four tabs show live data from the integrated pipeline:
 *   RGBA   - the same JPEG preview that the Original Input panel shows
 *   Mask   - GET /api/mask (depth-threshold person mask, PNG)
 *   Depth  - GET /api/pose-depth depth_jpeg_base64
 *   Motion - GET /api/motion flow field drawn as a 2D arrow field
 *
 * Every tab has an explicit waiting state for HTTP 204.
 */

import type {
  MotionPayload,
  PoseDepthFrame,
} from '../api/types'
import { skeletonSegments, visibleJoints } from '../api/skeleton'

type ProcessorView = 'rgba' | 'mask' | 'depth' | 'motion'

const TAB_LABELS: Record<ProcessorView, string> = {
  rgba: 'RGBA',
  mask: 'Mask',
  depth: 'Depth',
  motion: 'Motion',
}

export interface PreviewMeta {
  frameId: number
  sourceId: string
  timestamp: string
}

export interface MaskMeta {
  refreshCount: number
}

const PLACEHOLDER_WIDTH = 640
const PLACEHOLDER_HEIGHT = 480

export class FrameProcessorPanel {
  private element: HTMLElement
  private canvas: HTMLCanvasElement
  private placeholder: HTMLElement
  private footerItems: HTMLElement[]
  private tabs = new Map<ProcessorView, HTMLButtonElement>()

  private activeView: ProcessorView = 'rgba'

  private previewImage: HTMLImageElement | null = null
  private previewMeta: PreviewMeta | null = null
  private maskImage: ImageBitmap | null = null
  private maskMeta: MaskMeta | null = null
  private depthImage: HTMLImageElement | null = null
  private depthMeta: PoseDepthFrame | null = null
  private motion: MotionPayload | null = null

  private waiting: Record<ProcessorView, string> = {
    rgba: 'Waiting for the camera preview',
    mask: 'Waiting for a depth-threshold mask',
    depth: 'Waiting for pose and depth output',
    motion: 'Waiting for reuse motion (refresh path active)',
  }

  constructor() {
    this.element = document.createElement('section')
    this.element.className = 'viewer-panel'

    const tabButtons = (Object.keys(TAB_LABELS) as ProcessorView[])
      .map(
        (view) => `
          <button class="processor-tab" data-view="${view}" type="button">
            ${TAB_LABELS[view]}
          </button>
        `,
      )
      .join('')

    this.element.innerHTML = `
      <div class="panel-header">
        <h2>Frame Processor Output</h2>
      </div>

      <div class="viewer-content processor-content">
        <div class="processor-tabs">
          ${tabButtons}
        </div>

        <div class="processor-viewport">
          <canvas class="processor-canvas"></canvas>

          <div class="placeholder processor-placeholder">
            Waiting for processed frame
          </div>
        </div>
      </div>

      <div class="panel-footer processor-footer">
        <span class="processor-footer-item">Frame: --</span>
        <span class="processor-footer-item">Source: --</span>
        <span class="processor-footer-item">Capture: --</span>
        <span class="processor-footer-item">Processing: --</span>
      </div>
    `

    this.canvas = this.element.querySelector<HTMLCanvasElement>(
      '.processor-canvas',
    )!
    this.placeholder = this.element.querySelector<HTMLElement>(
      '.processor-placeholder',
    )!
    this.footerItems = Array.from(
      this.element.querySelectorAll<HTMLElement>('.processor-footer-item'),
    )

    const buttons =
      this.element.querySelectorAll<HTMLButtonElement>('.processor-tab')
    buttons.forEach((button) => {
      const view = button.dataset.view as ProcessorView
      this.tabs.set(view, button)
      button.addEventListener('click', () => {
        this.setActiveView(view)
      })
    })

    this.setActiveView('rgba')
  }

  getElement(): HTMLElement {
    return this.element
  }

  setActiveView(view: ProcessorView): void {
    this.activeView = view
    this.tabs.forEach((button, tabView) => {
      button.classList.toggle('active', tabView === view)
    })
    this.renderActiveView()
  }

  /** Live camera preview JPEG, shared with the Original Input panel. */
  setPreviewFrame(image: HTMLImageElement, meta: PreviewMeta): void {
    this.previewImage = image
    this.previewMeta = meta
    this.waiting.rgba = 'Waiting for the camera preview'
    if (this.activeView === 'rgba') {
      this.renderActiveView()
    }
  }

  showPreviewWaiting(message: string): void {
    this.previewImage = null
    this.waiting.rgba = message
    if (this.activeView === 'rgba') {
      this.renderActiveView()
    }
  }

  setMaskImage(image: ImageBitmap | null, meta: MaskMeta | null): void {
    if (this.maskImage !== null) {
      this.maskImage.close()
    }
    this.maskImage = image
    this.maskMeta = meta
    if (this.activeView === 'mask') {
      this.renderActiveView()
    }
  }

  setDepthFrame(
    image: HTMLImageElement | null,
    meta: PoseDepthFrame | null,
  ): void {
    this.depthImage = image
    this.depthMeta = meta
    if (meta === null) {
      this.waiting.depth = 'Waiting for pose and depth output'
    }
    if (this.activeView === 'depth') {
      this.renderActiveView()
    }
  }

  setMotion(motion: MotionPayload | null): void {
    this.motion = motion
    if (motion === null) {
      this.waiting.motion = 'Waiting for reuse motion (refresh path active)'
    }
    if (this.activeView === 'motion') {
      this.renderActiveView()
    }
  }

  private renderActiveView(): void {
    switch (this.activeView) {
      case 'rgba':
        if (this.previewImage === null) {
          this.showPlaceholder(this.waiting.rgba)
          this.setFooter([
            'Frame: --',
            'Source: --',
            'Capture: --',
            'Processing: --',
          ])
          return
        }
        this.drawImage(this.previewImage)
        this.setFooter([
          `Frame: ${this.previewMeta?.frameId ?? '--'}`,
          `Source: ${this.previewMeta?.sourceId ?? '--'}`,
          `Capture: ${this.previewMeta?.timestamp ?? '--'}`,
          'Processing: live JPEG',
        ])
        return

      case 'mask':
        if (this.maskImage === null) {
          this.showPlaceholder(this.waiting.mask)
          this.setFooter([
            'State: --',
            'Mask: --',
            'Source: depth threshold',
            'Refresh: --',
          ])
          return
        }
        this.drawImage(this.maskImage, this.maskImage.width, this.maskImage.height)
        this.drawTag('Depth threshold mask', [
          `Size: ${this.maskImage.width} x ${this.maskImage.height}`,
        ])
        this.setFooter([
          `State: refresh ${this.maskMeta?.refreshCount ?? '--'}`,
          `Mask: ${this.maskImage.width} x ${this.maskImage.height}`,
          'Source: /api/mask',
          'Refresh: --',
        ])
        return

      case 'depth':
        if (this.depthImage === null || this.depthMeta === null) {
          this.showPlaceholder(this.waiting.depth)
          this.setFooter([
            'Frame: --',
            'Source: --',
            'Depth: --',
            'Processing: --',
          ])
          return
        }
        this.drawImage(this.depthImage)
        this.drawSkeletonOverlay(this.depthMeta)
        this.drawDepthOverlay(this.depthMeta)
        this.setFooter([
          `Frame: ${this.depthMeta.frame_id}`,
          `Source: ${this.depthMeta.depth_source}`,
          `Depth: ${formatRange(this.depthMeta.near_m, this.depthMeta.far_m)} m`,
          `Pose: ${this.depthMeta.joints.length} joints  ·  ${formatNumber(
            this.depthMeta.timings_ms.total ?? 0,
            1,
          )} ms`,
        ])
        return

      case 'motion':
        if (this.motion === null) {
          this.showPlaceholder(this.waiting.motion)
          this.setFooter([
            'Frame: --',
            'Reference: --',
            'Grid: --',
            'Max offset: --',
          ])
          return
        }
        this.drawMotion(this.motion)
        this.setFooter([
          `Frame: ${this.motion.frame_id}`,
          `Reference: ${this.motion.reference_frame_id ?? '--'}`,
          `Source: ${this.motion.source_width} x ${this.motion.source_height}`,
          `Max offset: ${this.motion.max_offset_ndc.toFixed(2)} NDC`,
        ])
        return
    }
  }

  private drawImage(
    source: CanvasImageSource,
    width?: number,
    height?: number,
  ): void {
    const context = this.canvas.getContext('2d')
    if (!context) {
      throw new Error('Unable to create the processor canvas context')
    }
    const targetWidth =
      width ?? (source instanceof HTMLImageElement ? source.naturalWidth : PLACEHOLDER_WIDTH)
    const targetHeight =
      height ?? (source instanceof HTMLImageElement ? source.naturalHeight : PLACEHOLDER_HEIGHT)

    this.canvas.width = Math.max(1, targetWidth)
    this.canvas.height = Math.max(1, targetHeight)
    context.clearRect(0, 0, this.canvas.width, this.canvas.height)
    context.drawImage(source, 0, 0, this.canvas.width, this.canvas.height)
    this.showCanvas()
  }

  private drawTag(title: string, lines: string[]): void {
    const context = this.canvas.getContext('2d')
    if (!context) {
      return
    }
    const boxWidth = Math.min(360, this.canvas.width - 24)
    const boxHeight = 26 + lines.length * 18
    context.fillStyle = 'rgba(0, 0, 0, 0.72)'
    context.fillRect(12, 12, boxWidth, boxHeight)
    context.fillStyle = '#ffffff'
    context.font = '600 15px system-ui'
    context.fillText(title, 24, 32)
    context.font = '12px ui-monospace, Consolas, monospace'
    context.fillStyle = '#cbd5e1'
    lines.forEach((line, index) => {
      context.fillText(line, 24, 50 + index * 18)
    })
  }

  /** Draw the pose skeleton from the joints in the same depth payload. */
  private drawSkeletonOverlay(meta: PoseDepthFrame): void {
    const context = this.canvas.getContext('2d')
    if (!context || meta.joints.length === 0) {
      return
    }
    const width = this.canvas.width
    const height = this.canvas.height
    const toX = (joint: { x: number | null }): number => (joint.x ?? 0) * width
    const toY = (joint: { y: number | null }): number => (joint.y ?? 0) * height

    context.lineWidth = Math.max(1.5, width / 320)
    context.strokeStyle = 'rgba(80, 240, 180, 0.9)'
    context.fillStyle = 'rgba(255, 255, 255, 0.95)'
    for (const [start, end] of skeletonSegments(meta.joints)) {
      context.beginPath()
      context.moveTo(toX(start), toY(start))
      context.lineTo(toX(end), toY(end))
      context.stroke()
    }
    const radius = Math.max(2, width / 240)
    for (const joint of visibleJoints(meta.joints)) {
      context.beginPath()
      context.arc(toX(joint), toY(joint), radius, 0, Math.PI * 2)
      context.fill()
    }
  }

  private drawDepthOverlay(meta: PoseDepthFrame): void {
    const depthLabel =
      meta.depth_kind === 'metric_depth' ? 'Metric depth' : 'Relative depth'
    const lines = [
      `Source: ${meta.depth_source}`,
      `Near/far: ${formatRange(meta.near_m, meta.far_m)} m`,
      `Valid: ${formatPercent(meta.valid_fraction)}`,
      `Person: ${meta.person_detected ? 'yes' : 'no'}`,
      `Body depth: ${
        meta.body_depth_m === null ? '--' : `${meta.body_depth_m.toFixed(2)} m`
      }`,
      `Total: ${formatNumber(meta.timings_ms.total, 1)} ms`,
    ]
    this.drawTag(depthLabel, lines)
  }

  private drawMotion(motion: MotionPayload): void {
    const context = this.canvas.getContext('2d')
    if (!context) {
      throw new Error('Unable to create the processor canvas context')
    }

    const width = Math.max(2, motion.source_width)
    const height = Math.max(2, motion.source_height)
    this.canvas.width = width
    this.canvas.height = height

    context.fillStyle = '#081018'
    context.fillRect(0, 0, width, height)

    const columns = Math.max(1, motion.width)
    const rows = Math.max(1, motion.height)
    const cellWidth = width / columns
    const cellHeight = height / rows
    const maxOffset = Math.max(1e-6, motion.max_offset_ndc)

    const magnitude = (cell: number): number => {
      const deltaX = motion.flow_ndc[cell * 2] ?? 0
      const deltaY = motion.flow_ndc[cell * 2 + 1] ?? 0
      return Math.hypot(deltaX, deltaY)
    }

    // Magnitude heat cells: cyan (small motion) to orange (large motion).
    let validCells = 0
    for (let row = 0; row < rows; row++) {
      for (let column = 0; column < columns; column++) {
        const cell = row * columns + column
        if (!motion.valid[cell]) {
          continue
        }
        validCells++
        const ratio = Math.min(1, magnitude(cell) / maxOffset)
        const hue = 190 - 180 * ratio
        const lightness = 24 + 34 * ratio
        context.fillStyle = `hsl(${hue} 85% ${lightness}%)`
        context.globalAlpha = 0.55
        context.fillRect(
          column * cellWidth,
          row * cellHeight,
          cellWidth + 0.5,
          cellHeight + 0.5,
        )
      }
    }
    context.globalAlpha = 1

    // Direction arrows, scaled by the normalized offset magnitude.
    context.strokeStyle = 'rgba(234, 246, 255, 0.92)'
    context.fillStyle = 'rgba(234, 246, 255, 0.92)'
    context.lineWidth = 1.2
    for (let row = 0; row < rows; row++) {
      for (let column = 0; column < columns; column++) {
        const cell = row * columns + column
        if (!motion.valid[cell]) {
          continue
        }
        const deltaX = motion.flow_ndc[cell * 2] ?? 0
        const deltaY = motion.flow_ndc[cell * 2 + 1] ?? 0
        const mag = Math.hypot(deltaX, deltaY)
        if (mag < 1e-6) {
          continue
        }
        const startX = (column + 0.5) * cellWidth
        const startY = (row + 0.5) * cellHeight
        // flow_ndc is clip-space: +x right, +y up. Canvas y grows down.
        const pixelX = deltaX * (width / 2)
        const pixelY = -deltaY * (height / 2)
        const directionLength = Math.hypot(pixelX, pixelY)
        const unitX = pixelX / directionLength
        const unitY = pixelY / directionLength
        const arrowLength = 4 + Math.min(1, mag / maxOffset) * 26
        const endX = startX + unitX * arrowLength
        const endY = startY + unitY * arrowLength

        context.beginPath()
        context.moveTo(startX, startY)
        context.lineTo(endX, endY)
        context.stroke()

        const angle = Math.atan2(unitY, unitX)
        const head = 4
        context.beginPath()
        context.moveTo(endX, endY)
        context.lineTo(
          endX - head * Math.cos(angle - Math.PI / 6),
          endY - head * Math.sin(angle - Math.PI / 6),
        )
        context.lineTo(
          endX - head * Math.cos(angle + Math.PI / 6),
          endY - head * Math.sin(angle + Math.PI / 6),
        )
        context.closePath()
        context.fill()
      }
    }

    // Pose joints carried with the motion payload, when a person was detected.
    if (motion.joints.length > 0) {
      context.strokeStyle = '#22d3ee'
      context.lineWidth = 2
      motion.joints.forEach((joint) => {
        if (joint.x === null || joint.y === null) {
          return
        }
        context.beginPath()
        context.arc(joint.x * width, joint.y * height, 4, 0, Math.PI * 2)
        context.stroke()
      })
    }

    this.drawTag('Reuse motion field', [
      `Frame: ${motion.frame_id}  <- reference ${motion.reference_frame_id ?? '--'}`,
      `Grid: ${columns} x ${rows}   Valid cells: ${validCells}/${columns * rows}`,
      `Max |offset|: ${maxOffset.toFixed(2)} NDC   Source: ${width} x ${height}`,
    ])

    this.showCanvas()
  }

  private setFooter(parts: [string, string, string, string]): void {
    parts.forEach((text, index) => {
      const item = this.footerItems[index]
      if (item) {
        item.textContent = text
      }
    })
  }

  private showCanvas(): void {
    this.canvas.style.display = 'block'
    this.placeholder.style.display = 'none'
  }

  private showPlaceholder(message: string): void {
    this.canvas.style.display = 'none'
    this.placeholder.style.display = 'flex'
    this.placeholder.textContent = message
  }
}

function formatNumber(value: number | null | undefined, digits: number): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return '--'
  }
  return value.toFixed(digits)
}

function formatRange(
  near: number | null | undefined,
  far: number | null | undefined,
): string {
  const nearText = formatNumber(near, 2)
  const farText = formatNumber(far, 2)
  return `${nearText} - ${farText}`
}

function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return '--'
  }
  return `${(value * 100).toFixed(1)}%`
}
