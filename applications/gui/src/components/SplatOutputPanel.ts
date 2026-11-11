import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import type { SplatPointData } from '../data/mockSplatData'

export class SplatOutputPanel {
  private element: HTMLElement
  private viewport: HTMLElement

  private scene: THREE.Scene
  private camera: THREE.PerspectiveCamera
  private renderer: THREE.WebGLRenderer
  private controls: OrbitControls
  private pointCloud: THREE.Points | null = null

  private frameLabel: HTMLElement
  private splatLabel: HTMLElement
  private fpsLabel: HTMLElement

  private animationFrameId = 0
  private frameCounter = 0
  private lastFpsUpdate = performance.now()

  private resizeObserver: ResizeObserver

  constructor() {
    this.element = document.createElement('section')
    this.element.className = 'viewer-panel'

    this.element.innerHTML = `
      <div class="panel-header">
        <h2>Splat Constructor Output</h2>
      </div>

      <div class="viewer-content splat-content">
        <div class="splat-viewport">
            <button
            class="reset-view-button"
            type="button"
            >
            Reset View
            </button>
        </div>
        </div>

      <div class="panel-footer">
            <span class="splat-frame">
                Frame: --
            </span>

            <span class="splat-count">
                Splats: --
            </span>

            <span class="splat-fps">
                Render FPS: --
            </span>
        </div>
            `

    this.viewport = this.element.querySelector<HTMLElement>(
      '.splat-viewport'
    )!

    this.frameLabel = this.element.querySelector<HTMLElement>(
        '.splat-frame'
    )!

    this.splatLabel = this.element.querySelector<HTMLElement>(
      '.splat-count'
    )!

    this.fpsLabel = this.element.querySelector<HTMLElement>(
      '.splat-fps'
    )!

    // Three.js requires a scene, camera and renderer
    // before anything can be displayed.
    this.scene = new THREE.Scene()

    this.camera = new THREE.PerspectiveCamera(
      60,
      1,
      0.1,
      100
    )

    this.camera.position.set(0, 0, 5)

    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
    })

    this.renderer.setPixelRatio(
      Math.min(window.devicePixelRatio, 2)
    )

    this.viewport.appendChild(this.renderer.domElement)

    this.controls = new OrbitControls(
        this.camera,
        this.renderer.domElement
        )

        this.controls.target.set(0, 0, 0)

        this.controls.enableDamping = true
        this.controls.dampingFactor = 0.08

        this.controls.update()

        // Save this camera position as the default view.
        this.controls.saveState()

        const resetButton =
        this.element.querySelector<HTMLButtonElement>(
            '.reset-view-button'
        )!

        resetButton.addEventListener('click', () => {
        this.controls.reset()
        })

    this.splatLabel.textContent = 'Splats: --'

    // Resize the Three.js canvas when its GUI panel changes size.
    this.resizeObserver = new ResizeObserver(() => {
      this.resize()
    })

    this.resizeObserver.observe(this.viewport)

    this.animate()
  }

  getElement(): HTMLElement {
    return this.element
  }

  setSplatData(data: SplatPointData,frameId?: number | string): void {
  if (data.positions.length % 3 !== 0) {
    throw new Error(
      'Splat positions must contain XYZ triplets'
    )
  }

  const pointCount =
    data.positions.length / 3

  if (
    data.colors &&
    data.colors.length !== data.positions.length
  ) {
    throw new Error(
      'Splat colors must match the position array length'
    )
  }

  /*
   * Remove the old point cloud before replacing it.
   */
  if (this.pointCloud) {
    this.scene.remove(this.pointCloud)

    this.pointCloud.geometry.dispose()

    const oldMaterial =
      this.pointCloud.material

    if (oldMaterial instanceof THREE.Material) {
      oldMaterial.dispose()
    }
  }

  const geometry =
    new THREE.BufferGeometry()

  geometry.setAttribute(
    'position',
    new THREE.BufferAttribute(
      data.positions,
      3
    )
  )

  if (data.colors) {
    geometry.setAttribute(
      'color',
      new THREE.BufferAttribute(
        data.colors,
        3
      )
    )
  }

  const material =
    new THREE.PointsMaterial({
      size: 0.06,
      sizeAttenuation: true,

      vertexColors:
        data.colors !== undefined,
    })

  this.pointCloud =
    new THREE.Points(
      geometry,
      material
    )

  this.scene.add(this.pointCloud)

  this.frameLabel.textContent =
  `Frame: ${frameId ?? '--'}`

  this.splatLabel.textContent =
    `Splats: ${pointCount}`
}

  private resize(): void {
    const width = this.viewport.clientWidth
    const height = this.viewport.clientHeight

    if (width === 0 || height === 0) {
      return
    }

    this.camera.aspect = width / height
    this.camera.updateProjectionMatrix()

    this.renderer.setSize(
      width,
      height,
      false
    )
  }

  private animate = (): void => {
    this.animationFrameId =
      requestAnimationFrame(this.animate)

    // Temporary rotation to prove that
    // the render loop is functioning.
    this.controls.update()

    this.renderer.render(
      this.scene,
      this.camera
    )

    this.updateFps()
  }

  private updateFps(): void {
    this.frameCounter++

    const now = performance.now()
    const elapsed =
      now - this.lastFpsUpdate

    if (elapsed >= 500) {
      const fps =
        (this.frameCounter * 1000) /
        elapsed

      this.fpsLabel.textContent =
        `Render FPS: ${fps.toFixed(0)}`

      this.frameCounter = 0
      this.lastFpsUpdate = now
    }
  }

  destroy(): void {
    cancelAnimationFrame(
        this.animationFrameId
    )

    this.resizeObserver.disconnect()

    this.controls.dispose()

    if (this.pointCloud) {
        this.pointCloud.geometry.dispose()

        const material =
        this.pointCloud.material

        if (material instanceof THREE.Material) {
        material.dispose()
        }
    }

    this.renderer.dispose()
    }
}