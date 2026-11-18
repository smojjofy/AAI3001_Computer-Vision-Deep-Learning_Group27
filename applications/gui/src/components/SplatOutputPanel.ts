/*
 * Splat Constructor Output panel.
 *
 * Renders the live Gaussian splat state with gsplat.js (splat.js 1.2.9) and
 * gsplat OrbitControls. The splat state is only fetched by main.ts when
 * /api/diagnostics reports a new `splat.frame_id`.
 *
 * Coordinate conventions (shared/coordinate_system/README.md):
 *   constructor world: right-handed, +X right, +Y up, camera forward -Z,
 *                      camera at the origin, positions in metres.
 *   gsplat view frame: +X right, +Y down, +Z forward.
 *
 * gsplat OrbitControls owns the camera pose (it rewrites camera.position and
 * camera.rotation every update), so the scene is uploaded in a display world
 * that the OrbitControls convention understands:
 *
 *   R0 = 180 degrees about X  =>  p_display = (x, -y, -z)
 *
 * The display +Y axis therefore points down, like an image, and a camera at the
 * origin with an identity view looks along +Z into the scene. Splat rotations
 * are rotated with the same R0: for a constructor quaternion (w, x, y, z) in
 * wxyz order, q_display = R0 * q = (-x, w, -z, y).
 *
 * scales stay linear metres: exp(log_scales). colors: RGB from the spherical
 * harmonics DC term, alpha from the opacity logit times header.display_alpha.
 *
 * Optional reuse warp (see setMotionWarp): the /api/motion flow field offsets
 * only the displayed positions, in view space, and the base positions are never
 * mutated.
 */

import {
  Camera,
  Color32,
  OrbitControls,
  Scene,
  Splat,
  SplatData,
  Vector3,
  WebGLRenderer,
} from 'gsplat'

import type { MotionPayload, PoseJoint, SplatFrame } from '../api/types'

/** gsplat camera/view limits in metres. */
const NEAR_M = 0.05
const FAR_M = 100

/** Spherical harmonics DC coefficient of the first basis function. */
const SH_C0 = 0.28209479177387814

/** Default source size when the header principal point is not usable. */
const DEFAULT_SOURCE_WIDTH = 640
const DEFAULT_SOURCE_HEIGHT = 480

/**
 * Body-region heuristic for the warp strength.
 *
 * A splat is "near the body" when its normalized source pixel lies within
 * BODY_NEAR_RADIUS (0.25 normalized image units, both axes) of one of the
 * head, shoulder, or hip joints of the pose. Those splats get the full flow
 * offset; every other splat gets BODY_FACTOR_FAR. When no person is detected
 * (empty joints list) every splat uses BODY_FACTOR_FAR.
 */
const BODY_FACTOR_NEAR = 1.0
const BODY_FACTOR_FAR = 0.6
const BODY_NEAR_RADIUS = 0.25
const BODY_JOINT_INDICES = new Set([
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, // head and face
  11, 12, // shoulders
  23, 24, // hips
])

interface Intrinsics {
  fx: number
  fy: number
}

/**
 * Converted splat arrays for one frame, kept separately from the gsplat
 * SplatData because the gsplat data worker replaces (SplatData.reattach) the
 * live arrays after every pack.
 */
interface SplatDescriptor {
  basePositions: Float32Array
  rotations: Float32Array
  scales: Float32Array
  colors: Uint8Array
}

export class SplatOutputPanel {
  private element: HTMLElement
  private viewport: HTMLElement
  private canvas: HTMLCanvasElement
  private placeholder: HTMLElement
  private statusLabel: HTMLElement
  private resetButton: HTMLButtonElement

  private frameLabel: HTMLElement
  private splatLabel: HTMLElement
  private fpsLabel: HTMLElement
  private alphaLabel: HTMLElement
  private latencyLabel: HTMLElement

  private renderer: WebGLRenderer | null = null
  private scene = new Scene()
  private camera = new Camera()
  private controls: OrbitControls | null = null

  private activeSplat: Splat | null = null
  private activeFrameId: string | null = null
  private basePositions: Float32Array | null = null
  private sourcePixels: Uint32Array | null = null
  private bodyFactors: Float32Array | null = null
  private activeMotion: MotionPayload | null = null
  private pendingDescriptor: SplatDescriptor | null = null
  private descriptorDirty = false
  private descriptorReadyAt = 0
  private warpDirty = false

  private intrinsics: Intrinsics = { fx: 800, fy: 800 }
  private sourceWidth = DEFAULT_SOURCE_WIDTH
  private sourceHeight = DEFAULT_SOURCE_HEIGHT
  private splatCount = 0
  private displayAlpha = 1
  private activeIdentity: { sourceId: string; generation: number; frameId: number } | null = null
  private pendingOpacityLogits: Float32Array | null = null
  /** Viewer clock anchor for the hold-then-fade rule (ms since first loss). */
  private lossAnchorMs: number | null = null
  private lossHoldMs = 500
  private lossFadeMs = 500

  private hasFitted = false
  private animationFrameId = 0
  private animationFrameSeen = false
  private fallbackTimer = 0
  private frameCounter = 0
  private lastFpsUpdate = performance.now()
  private resizeObserver: ResizeObserver | null = null
  private disposed = false
  private lastError = ''

  constructor() {
    this.element = document.createElement('section')
    this.element.className = 'viewer-panel'

    this.element.innerHTML = `
      <div class="panel-header">
        <h2>Splat Constructor Output</h2>
      </div>

      <div class="viewer-content splat-content">
        <div class="splat-viewport">
          <canvas class="splat-canvas"></canvas>

          <div class="placeholder splat-placeholder">
            Waiting for the splat state
          </div>

          <div class="splat-status">Warp: off</div>

          <button class="reset-view-button" type="button">
            Reset View
          </button>
        </div>
      </div>

      <div class="panel-footer splat-footer">
        <span class="splat-footer-item splat-frame">Frame: --</span>
        <span class="splat-footer-item splat-count">Splats: --</span>
        <span class="splat-footer-item splat-fps">Render FPS: --</span>
        <span class="splat-footer-item splat-alpha">Alpha: --</span>
      </div>
    `

    this.viewport =
      this.element.querySelector<HTMLElement>('.splat-viewport')!
    this.canvas = this.element.querySelector<HTMLCanvasElement>('.splat-canvas')!
    this.placeholder = this.element.querySelector<HTMLElement>(
      '.splat-placeholder',
    )!
    this.statusLabel = this.element.querySelector<HTMLElement>(
      '.splat-status',
    )!
    this.resetButton = this.element.querySelector<HTMLButtonElement>(
      '.reset-view-button',
    )!
    this.frameLabel = this.element.querySelector<HTMLElement>('.splat-frame')!
    this.splatLabel = this.element.querySelector<HTMLElement>('.splat-count')!
    this.fpsLabel = this.element.querySelector<HTMLElement>('.splat-fps')!
    this.alphaLabel = this.element.querySelector<HTMLElement>('.splat-alpha')!
    this.latencyLabel = this.element.querySelector<HTMLElement>('.splat-latency')!

    this.resetButton.addEventListener('click', () => {
      this.fitView()
    })

    try {
      // The renderer creates a WebGL2 context on the supplied canvas and sizes
      // itself from the canvas client size.
      this.renderer = new WebGLRenderer(this.canvas)
      this.renderer.backgroundColor = new Color32(9, 11, 16, 255)
      this.camera.data.near = NEAR_M
      this.camera.data.far = FAR_M
      this.controls = new OrbitControls(this.camera, this.canvas)
      this.updateProjection()
    } catch (error) {
      this.showError(error)
    }

    if (this.controls !== null) {
      this.controls.update()
    }

    this.resizeObserver = new ResizeObserver(() => {
      this.resize()
    })
    this.resizeObserver.observe(this.viewport)
    this.resize()

    this.animationFrameId = requestAnimationFrame(this.tick)
    // Headless screenshot runs normally tick rAF. The timer is a safety net for
    // environments that throttle requestAnimationFrame while loading.
    this.fallbackTimer = window.setInterval(() => {
      if (!this.animationFrameSeen) {
        this.renderFrame()
      }
    }, 33)
  }

  getElement(): HTMLElement {
    return this.element
  }

  /** Replace the rendered splat state with one decoded /api/splat frame. */
  /** Show the measured capture-to-display latency for the installed state. */
  setCaptureToDisplay(latencyMs: number | null): void {
    if (latencyMs === null || !Number.isFinite(latencyMs)) {
      this.latencyLabel.textContent = 'Display: --'
      return
    }
    this.latencyLabel.textContent = `Display: ${Math.round(latencyMs)} ms`
  }

  setSplatFrame(frame: SplatFrame): void {
    if (this.renderer === null || this.controls === null) {
      this.showError(new Error('gsplat renderer is not available'))
      return
    }
    const { header } = frame
    const count = header.count
    if (count <= 0) {
      this.showPlaceholder('The splat state is empty')
      return
    }

    // Positions: constructor metres -> display world (see the file header).
    const basePositions = new Float32Array(count * 3)
    for (let index = 0; index < count; index++) {
      const offset = index * 3
      basePositions[offset] = frame.positions[offset]
      basePositions[offset + 1] = -frame.positions[offset + 1]
      basePositions[offset + 2] = -frame.positions[offset + 2]
    }

    // Rotations: wxyz, q_display = R0 * q.
    const rotations = new Float32Array(count * 4)
    for (let index = 0; index < count; index++) {
      const offset = index * 4
      const w = frame.rotations[offset]
      const x = frame.rotations[offset + 1]
      const y = frame.rotations[offset + 2]
      const z = frame.rotations[offset + 3]
      rotations[offset] = -x
      rotations[offset + 1] = w
      rotations[offset + 2] = -z
      rotations[offset + 3] = y
    }

    // Scales: natural log of metres -> linear metres.
    const scales = new Float32Array(count * 3)
    for (let index = 0; index < count * 3; index++) {
      scales[index] = Math.exp(frame.logScales[index])
    }

    // Colors: gsplat Uint8 RGBA, alpha = sigmoid(opacity) * display_alpha.
    const alpha = clamp01(header.display_alpha ?? 1)
    const colors = new Uint8Array(count * 4)
    for (let index = 0; index < count; index++) {
      const colorOffset = index * 4
      colors[colorOffset] = toByte(
        0.5 + frame.shDc[index * 3] * SH_C0,
      )
      colors[colorOffset + 1] = toByte(
        0.5 + frame.shDc[index * 3 + 1] * SH_C0,
      )
      colors[colorOffset + 2] = toByte(
        0.5 + frame.shDc[index * 3 + 2] * SH_C0,
      )
      colors[colorOffset + 3] = toByte(sigmoid(frame.opacityLogits[index]) * alpha)
    }

    // The transport header supplies the capture image size. Fall back to the
    // principal point only for an older payload that does not carry it.
    const [fx, fy, cx, cy] = header.intrinsics
    this.intrinsics = {
      fx: Number.isFinite(fx) && fx > 0 ? fx : 800,
      fy: Number.isFinite(fy) && fy > 0 ? fy : 800,
    }
    const headerWidth = Number(header.image_width ?? 0)
    const headerHeight = Number(header.image_height ?? 0)
    this.sourceWidth =
      Number.isFinite(headerWidth) && headerWidth > 0
        ? Math.round(headerWidth)
        : Number.isFinite(cx) && cx > 0
          ? Math.round(2 * cx + 1)
          : DEFAULT_SOURCE_WIDTH
    this.sourceHeight =
      Number.isFinite(headerHeight) && headerHeight > 0
        ? Math.round(headerHeight)
        : Number.isFinite(cy) && cy > 0
          ? Math.round(2 * cy + 1)
          : DEFAULT_SOURCE_HEIGHT
    this.sourcePixels = frame.sourcePixels
    this.splatCount = count
    this.displayAlpha = alpha

    const descriptor: SplatDescriptor = {
      basePositions,
      rotations,
      scales,
      colors,
    }
    this.pendingDescriptor = descriptor
    this.basePositions = basePositions
    this.activeFrameId = header.frame_id
    this.activeIdentity = {
      sourceId: String(header.source_id ?? ''),
      generation: Number(header.generation ?? 0),
      frameId: Number(header.frame_id),
    }
    this.pendingOpacityLogits = frame.opacityLogits

    if (
      this.activeSplat !== null &&
      this.activeSplat.data.vertexCount === count
    ) {
      // Same splat count: update the live SplatData buffers in place. This
      // avoids a scene rebuild on every refresh and the gsplat render data
      // rebuild that would start an asynchronous initial pack.
      this.descriptorDirty = true
      this.descriptorReadyAt = 0
    } else {
      // Count changed (or the first frame): replace the splat object. Pass
      // copies because gsplat detaches the arrays it owns on the first pack.
      const splatData = new SplatData(
        count,
        basePositions.slice(),
        rotations.slice(),
        scales.slice(),
        colors.slice(),
      )
      const splat = new Splat(splatData)
      if (this.activeSplat !== null) {
        this.scene.removeObject(this.activeSplat)
      }
      this.scene.addObject(splat)
      this.activeSplat = splat
      this.descriptorDirty = false
      // The new render data packs asynchronously while its WebAssembly module
      // loads. Wait before raising a change event, otherwise the deferred
      // initial pack reads buffers that the data worker already detached.
      this.descriptorReadyAt = performance.now() + 400
    }

    // A new splat state can arrive while the reuse warp is still current (the
    // motion poll and the 5 MB splat fetch are independent). Keep the warp and
    // re-apply it to the new base positions; main.ts clears it when
    // /api/motion returns 204, or when this state is at least as new as the
    // motion frame.
    if (this.activeMotion !== null) {
      this.bodyFactors = this.buildBodyFactors(this.activeMotion.joints)
      this.warpDirty = true
    } else {
      this.bodyFactors = null
      this.warpDirty = false
    }
    this.updateProjection()
    this.resize()
    // Fit the orbit camera once. Later frames keep the user's orbit pose.
    if (!this.hasFitted && this.fitView()) {
      this.hasFitted = true
    }

    this.placeholder.style.display = 'none'
    this.frameLabel.textContent = `Frame: ${header.frame_id}`
    this.splatLabel.textContent = `Splats: ${count}`
    this.alphaLabel.textContent = `Alpha: ${alpha.toFixed(2)}`
    this.lastError = ''
    this.updateStatus()
  }

  /**
   * True when motion describes the displayed splat state: same source and
   * generation, its reference is the displayed frame, and its candidate frame
   * is later than that reference.
   */
  private motionMatchesActiveState(motion: MotionPayload): boolean {
    const identity = this.activeIdentity
    if (identity === null) {
      return false
    }
    if ((motion.source_id ?? '') !== identity.sourceId) {
      return false
    }
    if (Number(motion.generation ?? 0) !== identity.generation) {
      return false
    }
    if (motion.reference_frame_id !== identity.frameId) {
      return false
    }
    return motion.frame_id > motion.reference_frame_id
  }

  /**
   * Start the hold-then-fade rule on the viewer clock from the server's
   * `person_loss_age_ms`. A repeated call with an age that continues the
   * same loss does not restart the hold. Passing null restores full alpha.
   */
  setPersonLoss(ageMs: number | null): void {
    if (ageMs === null) {
      this.lossAnchorMs = null
      this.applyDisplayAlpha(1)
      return
    }
    const now = performance.now()
    if (this.lossAnchorMs === null || ageMs < this.currentLossAgeMs(now) - 50) {
      // First loss, or the server latch restarted: anchor now.
      this.lossAnchorMs = now - Math.max(0, ageMs)
    }
    this.applyDisplayAlpha(this.alphaAt(now))
  }

  /** Recompute alpha on the viewer clock; call once per animation frame. */
  tickPersonLoss(): void {
    if (this.lossAnchorMs === null) {
      return
    }
    this.applyDisplayAlpha(this.alphaAt(performance.now()))
  }

  private currentLossAgeMs(now: number): number {
    return this.lossAnchorMs === null ? 0 : now - this.lossAnchorMs
  }

  private alphaAt(now: number): number {
    const age = this.currentLossAgeMs(now)
    if (age <= this.lossHoldMs) {
      return 1
    }
    return Math.max(0, Math.min(1, 1 - (age - this.lossHoldMs) / this.lossFadeMs))
  }

  /** Scale the stored colour alpha without refetching the splat arrays. */
  private applyDisplayAlpha(alpha: number): void {
    const next = Math.max(0, Math.min(1, alpha))
    if (Math.abs(next - this.displayAlpha) < 0.005) {
      return
    }
    this.displayAlpha = next
    const splat = this.activeSplat
    const descriptor = this.pendingDescriptor
    if (splat === null || descriptor === null || splat.data.detached) {
      this.alphaLabel.textContent = `Alpha: ${next.toFixed(2)}`
      return
    }
    const base = descriptor.colors
    const target = splat.data.colors
    if (target.length !== base.length) {
      return
    }
    for (let index = 3; index < base.length; index += 4) {
      target[index] = base[index]
    }
    // Colours were built with the original display alpha folded in, so rebuild
    // the alpha byte from the opacity logit and the new alpha.
    const logits = this.pendingOpacityLogits
    if (logits !== null && logits.length * 4 === base.length) {
      for (let index = 0; index < logits.length; index++) {
        target[index * 4 + 3] = toByte(sigmoid(logits[index]) * next)
      }
    }
    splat.data.changed = true
    splat.raiseChangeEvent()
    this.alphaLabel.textContent = `Alpha: ${next.toFixed(2)}`
    this.updateStatus()
  }

  /**
   * Apply (or clear) the optional reuse warp from one /api/motion payload.
   * Passing null restores the base splat positions.
   */
  setMotionWarp(motion: MotionPayload | null): void {
    if (motion !== null && !this.motionMatchesActiveState(motion)) {
      // RT-0019 item 2: motion created for a different state must never move
      // the displayed splats. Keep the current display and ignore the payload.
        this.activeMotion = null
      this.bodyFactors = null
      this.warpDirty = true
      this.updateStatus()
      return
    }
    this.activeMotion = motion
    if (motion === null) {
      this.bodyFactors = null
    } else {
      this.bodyFactors = this.buildBodyFactors(motion.joints)
    }
    this.warpDirty = true
    this.updateStatus()
  }

  /** Clear the warp on refresh: a new splat state replaces the old positions. */
  clearWarp(): void {
    this.setMotionWarp(null)
  }

  /**
   * Drop the displayed splat state, for example at a capture-run boundary.
   * The next published state rebuilds the scene and re-fits the orbit camera.
   */
  clearState(message = 'Waiting for the first refresh of this run'): void {
    if (this.activeSplat !== null && this.scene !== null) {
      this.scene.removeObject(this.activeSplat)
    }
    this.activeSplat = null
    this.pendingDescriptor = null
    this.basePositions = null
    this.sourcePixels = null
    this.pendingOpacityLogits = null
    this.splatCount = 0
    this.activeFrameId = null
    this.activeIdentity = null
    this.activeMotion = null
    this.bodyFactors = null
    this.warpDirty = false
    this.hasFitted = false
    this.displayAlpha = 1
    this.lossAnchorMs = null
    this.alphaLabel.textContent = 'Alpha: 1.00'
    this.frameLabel.textContent = 'Frame: --'
    this.splatLabel.textContent = 'Splats: --'
    this.setCaptureToDisplay(null)
    this.showPlaceholder(message)
    this.updateStatus()
  }

  private updateStatus(): void {
    if (this.lastError.length > 0) {
      this.statusLabel.textContent = `ERROR: ${this.lastError}`
      this.statusLabel.classList.add('is-error')
      return
    }
    this.statusLabel.classList.remove('is-error')
    if (this.activeFrameId === null) {
      this.statusLabel.textContent = 'Warp: off'
      return
    }
    if (this.activeMotion === null) {
      this.statusLabel.textContent = `Warp: off  ·  ${this.displayAlpha.toFixed(
        2,
      )} alpha`
      return
    }
    this.statusLabel.textContent = `Warp: on  ·  motion ${this.activeMotion.frame_id} <- ref ${
      this.activeMotion.reference_frame_id ?? '--'
    }`
  }

  private showPlaceholder(message: string): void {
    this.placeholder.style.display = 'flex'
    this.placeholder.textContent = message
  }

  private showError(error: unknown): void {
    this.lastError =
      error instanceof Error ? `${error.name}: ${error.message}` : String(error)
    this.placeholder.style.display = 'flex'
    this.placeholder.textContent = `Splat renderer error: ${this.lastError}`
    this.updateStatus()
  }

  /**
   * Compute a per-splat warp strength from the body-region heuristic.
   * See BODY_NEAR_RADIUS above for the documented rule.
   */
  private buildBodyFactors(joints: PoseJoint[]): Float32Array | null {
    const count = this.splatCount
    const sourcePixels = this.sourcePixels
    if (count <= 0 || sourcePixels === null) {
      return null
    }
    const factors = new Float32Array(count)
    factors.fill(BODY_FACTOR_FAR)

    const bodyJoints: Array<{ x: number; y: number }> = []
    for (const joint of joints) {
      if (
        BODY_JOINT_INDICES.has(joint.index) &&
        joint.x !== null &&
        joint.y !== null &&
        Number.isFinite(joint.x) &&
        Number.isFinite(joint.y)
      ) {
        bodyJoints.push({ x: joint.x, y: joint.y })
      }
    }
    if (bodyJoints.length === 0) {
      return factors
    }

    const radiusSquared = BODY_NEAR_RADIUS * BODY_NEAR_RADIUS
    for (let index = 0; index < count; index++) {
      const sourceX = sourcePixels[index * 2] + 0.5
      const sourceY = sourcePixels[index * 2 + 1] + 0.5
      const normalizedX = sourceX / this.sourceWidth
      const normalizedY = sourceY / this.sourceHeight
      let near = false
      for (const joint of bodyJoints) {
        const deltaX = normalizedX - joint.x
        const deltaY = normalizedY - joint.y
        if (deltaX * deltaX + deltaY * deltaY <= radiusSquared) {
          near = true
          break
        }
      }
      if (near) {
        factors[index] = BODY_FACTOR_NEAR
      }
    }
    return factors
  }

  /**
   * Write the pending frame arrays and warp offsets into the live SplatData
   * buffers. gsplat replaces those buffers (SplatData.reattach) after every
   * pack, so splat.data.* is re-read here and the converted descriptor arrays
   * are kept separately.
   */
  private applyPendingDescriptor(): void {
    const splat = this.activeSplat
    const base = this.basePositions
    const descriptor = this.pendingDescriptor
    if (splat === null || base === null) {
      return
    }
    if (!this.descriptorDirty && !this.warpDirty) {
      return
    }
    if (splat.data.detached) {
      // The gsplat data worker is still packing; retry on the next frame.
      return
    }
    if (performance.now() < this.descriptorReadyAt) {
      return
    }
    const data = splat.data
    const target = data.positions
    if (target.length !== base.length) {
      return
    }
    target.set(base)
    if (this.activeMotion !== null && this.bodyFactors !== null) {
      this.applyMotionOffsets(target)
    }
    if (this.descriptorDirty && descriptor !== null) {
      if (data.rotations.length === descriptor.rotations.length) {
        data.rotations.set(descriptor.rotations)
      }
      if (data.scales.length === descriptor.scales.length) {
        data.scales.set(descriptor.scales)
      }
      if (data.colors.length === descriptor.colors.length) {
        data.colors.set(descriptor.colors)
      }
      this.descriptorDirty = false
    }
    data.changed = true
    splat.raiseChangeEvent()
    this.warpDirty = false
  }

  /**
   * Offset each displayed splat in view space from the sampled flow field:
   *   dx_world = ndc_dx * z * image_width  / (2 * fx)
   *   dy_world = ndc_dy * z * image_height / (2 * fy)
   * where z is the camera depth in metres. Display +Y points down, so the
   * image-space dy is subtracted on the display Y axis.
   */
  private applyMotionOffsets(target: Float32Array): void {
    const motion = this.activeMotion
    const base = this.basePositions
    const sourcePixels = this.sourcePixels
    const factors = this.bodyFactors
    if (motion === null || base === null || sourcePixels === null || factors === null) {
      return
    }
    const columns = Math.max(1, motion.width)
    const rows = Math.max(1, motion.height)
    const flow = motion.flow_ndc
    const valid = motion.valid
    const sourceWidth = Math.max(1, motion.source_width)
    const sourceHeight = Math.max(1, motion.source_height)
    const scaleX = sourceWidth / (2 * this.intrinsics.fx)
    const scaleY = sourceHeight / (2 * this.intrinsics.fy)

    for (let index = 0; index < this.splatCount; index++) {
      const positionOffset = index * 3
      const depth = base[positionOffset + 2]
      if (!(depth > 0)) {
        continue
      }
      const pixelX = sourcePixels[index * 2]
      const pixelY = sourcePixels[index * 2 + 1]
      const gridX = ((pixelX + 0.5) / sourceWidth) * columns - 0.5
      const gridY = ((pixelY + 0.5) / sourceHeight) * rows - 0.5

      const cellX = Math.floor(gridX)
      const cellY = Math.floor(gridY)
      const fractionX = gridX - cellX
      const fractionY = gridY - cellY

      let weightedX = 0
      let weightedY = 0
      let weightSum = 0
      for (let offsetY = 0; offsetY <= 1; offsetY++) {
        for (let offsetX = 0; offsetX <= 1; offsetX++) {
          const sampleX = cellX + offsetX
          const sampleY = cellY + offsetY
          if (sampleX < 0 || sampleX >= columns || sampleY < 0 || sampleY >= rows) {
            continue
          }
          const cell = sampleY * columns + sampleX
          if (!valid[cell]) {
            continue
          }
          const weight =
            (offsetX === 0 ? 1 - fractionX : fractionX) *
            (offsetY === 0 ? 1 - fractionY : fractionY)
          if (weight <= 0) {
            continue
          }
          weightedX += (flow[cell * 2] ?? 0) * weight
          weightedY += (flow[cell * 2 + 1] ?? 0) * weight
          weightSum += weight
        }
      }
      if (weightSum < 1e-6) {
        continue
      }

      const bodyFactor = factors[index]
      const offsetX = (weightedX / weightSum) * depth * scaleX * bodyFactor
      const offsetY = (weightedY / weightSum) * depth * scaleY * bodyFactor

      target[positionOffset] = base[positionOffset] + offsetX
      target[positionOffset + 1] = base[positionOffset + 1] - offsetY
      target[positionOffset + 2] = base[positionOffset + 2]
    }
  }

  /** Fit the orbit camera to the current splat bounds. */
  private fitView(): boolean {
    const base = this.basePositions
    if (base === null || this.renderer === null || this.controls === null) {
      return false
    }
    const canvasHeight = this.renderer.canvas.height
    if (canvasHeight <= 0) {
      return false
    }

    let minX = Infinity
    let minY = Infinity
    let minZ = Infinity
    let maxX = -Infinity
    let maxY = -Infinity
    let maxZ = -Infinity
    for (let index = 0; index < base.length; index += 3) {
      const x = base[index]
      const y = base[index + 1]
      const z = base[index + 2]
      if (x < minX) minX = x
      if (y < minY) minY = y
      if (z < minZ) minZ = z
      if (x > maxX) maxX = x
      if (y > maxY) maxY = y
      if (z > maxZ) maxZ = z
    }
    if (!Number.isFinite(minX) || !Number.isFinite(maxX)) {
      return false
    }

    const center = new Vector3(
      (minX + maxX) / 2,
      (minY + maxY) / 2,
      (minZ + maxZ) / 2,
    )
    const spanY = maxY - minY
    const spanZ = maxZ - minZ
    const fovY = 2 * Math.atan(canvasHeight / (2 * this.camera.data.fy))
    const tangent = Math.max(0.05, Math.tan(fovY / 2))
    const halfExtent = 0.5 * Math.max(spanY, spanZ, 0.2)
    const distance = Math.max(0.4, (halfExtent / tangent) * 1.35)

    // Slightly above and to the right of the cloud for a readable 3D view.
    const direction = new Vector3(0.16, -0.28, -1).normalize()
    const cameraPosition = center.add(direction.multiply(distance))
    this.camera.position = cameraPosition
    this.controls.setCameraTarget(center)
    this.controls.update()
    return true
  }

  private updateProjection(): void {
    if (this.renderer === null) {
      return
    }
    const canvasHeight = this.renderer.canvas.height
    if (canvasHeight <= 0 || this.sourceHeight <= 0) {
      return
    }
    // Keep the capture camera's vertical field of view on the panel canvas and
    // use the header fx/fy ratio so pixels stay square in the 3D view.
    const scale = canvasHeight / this.sourceHeight
    this.camera.data.fx = this.intrinsics.fx * scale
    this.camera.data.fy = this.intrinsics.fy * scale
  }

  private resize(): void {
    if (this.renderer === null) {
      return
    }
    const width = this.viewport.clientWidth
    const height = this.viewport.clientHeight
    if (width <= 0 || height <= 0) {
      return
    }
    this.renderer.setSize(width, height)
    this.updateProjection()
  }

  private renderFrame(): void {
    if (this.disposed || this.renderer === null || this.controls === null) {
      return
    }
    this.applyPendingDescriptor()
    try {
      this.controls.update()
      this.renderer.render(this.scene, this.camera)
    } catch (error) {
      this.showError(error)
      return
    }
    this.frameCounter++
    this.updateFps()
  }

  private readonly tick = (): void => {
    this.animationFrameSeen = true
    this.renderFrame()
    if (this.disposed) {
      return
    }
    this.animationFrameId = requestAnimationFrame(this.tick)
  }

  private updateFps(): void {
    const now = performance.now()
    const elapsed = now - this.lastFpsUpdate
    if (elapsed >= 500) {
      const fps = (this.frameCounter * 1000) / elapsed
      this.fpsLabel.textContent = `Render FPS: ${fps.toFixed(0)}`
      this.frameCounter = 0
      this.lastFpsUpdate = now
    }
  }

  destroy(): void {
    if (this.disposed) {
      return
    }
    this.disposed = true
    cancelAnimationFrame(this.animationFrameId)
    window.clearInterval(this.fallbackTimer)
    this.resizeObserver?.disconnect()
    this.controls?.dispose()
    this.renderer?.dispose()
  }
}

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value))
}

function toByte(value: number): number {
  return Math.round(clamp01(value) * 255)
}

function sigmoid(value: number): number {
  if (value >= 0) {
    return 1 / (1 + Math.exp(-value))
  }
  const exponential = Math.exp(value)
  return exponential / (1 + exponential)
}
