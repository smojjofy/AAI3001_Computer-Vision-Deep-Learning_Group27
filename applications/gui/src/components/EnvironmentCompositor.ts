import * as THREE from 'three'
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { SceneRevealMode, SplatLoader, Viewer } from '@mkkellogg/gaussian-splats-3d'

import type { SplatFrame } from '../api/types'
import { encodeEnvironmentSplat } from './environmentSplat'
import { mayPresentPreparedFrame } from './environmentFramePolicy'

const MAX_GLB_BYTES = 30 * 1024 * 1024

function checkGlb(buffer: ArrayBuffer): void {
  const view = new DataView(buffer)
  if (buffer.byteLength < 20 || view.getUint32(0, true) !== 0x46546c67 ||
      view.getUint32(4, true) !== 2 || view.getUint32(8, true) !== buffer.byteLength ||
      view.getUint32(16, true) !== 0x4e4f534a) {
    throw new Error('Select a valid GLB 2.0 file')
  }
  const jsonLength = view.getUint32(12, true)
  if (jsonLength > buffer.byteLength - 20) throw new Error('The GLB file is incomplete')
  const json = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 20, jsonLength))) as {
    buffers?: { uri?: string }[]
    images?: { uri?: string }[]
  }
  const resources = [...(json.buffers ?? []), ...(json.images ?? [])]
  if (resources.some((resource) => resource.uri && !resource.uri.startsWith('data:'))) {
    throw new Error('The GLB must contain its textures and geometry')
  }
}

function disposeObject(root: THREE.Object3D): void {
  const textures = new Set<THREE.Texture>()
  root.traverse((object) => {
    if (!(object instanceof THREE.Mesh || object instanceof THREE.LineSegments)) return
    object.geometry.dispose()
    for (const material of (Array.isArray(object.material) ? object.material : [object.material])) {
      for (const value of Object.values(material)) {
        if (value instanceof THREE.Texture) textures.add(value)
      }
      material.dispose()
    }
  })
  for (const texture of textures) texture.dispose()
}

/** Optional 3D composite. The default live splat panel keeps its own renderer. */
export class EnvironmentCompositor {
  private dialog: HTMLDivElement
  private viewport: HTMLDivElement
  private fileInput: HTMLInputElement
  private status: HTMLSpanElement
  private scaleInput: HTMLInputElement
  private xInput: HTMLInputElement
  private yInput: HTMLInputElement
  private zInput: HTMLInputElement
  private scene = new THREE.Scene()
  private environmentRoot = new THREE.Group()
  private stage = new THREE.Group()
  private camera = new THREE.PerspectiveCamera(60, 1, 0.05, 100)
  private renderer: THREE.WebGLRenderer | null = null
  private controls: OrbitControls | null = null
  /** The inactive viewer sorts the next frame while the active viewer stays visible. */
  private viewers: [Viewer, Viewer] | null = null
  private activeViewer: Viewer | null = null
  private activeIdentity: { sourceId: string; generation: number; frameId: string } | null = null
  private environment: THREE.Object3D | null = null
  private environmentBaseScale = 1
  private latestFrame: SplatFrame | null = null
  private pendingFrame: SplatFrame | null = null
  private processingTask: Promise<void> | null = null
  private closeTask: Promise<void> | null = null
  private resizeObserver: ResizeObserver | null = null
  private animationFrame = 0
  private stateEpoch = 0
  private visible = false
  private session = 0
  private importSerial = 0

  constructor() {
    this.dialog = document.createElement('div')
    this.dialog.className = 'environment-dialog'
    this.dialog.hidden = true
    this.dialog.innerHTML = `
      <div class="environment-shell" role="dialog" aria-modal="true" aria-label="Environment compositor">
        <header class="environment-toolbar">
          <strong>Environment compositor</strong>
          <label class="environment-import">Import GLB <input type="file" accept=".glb,model/gltf-binary"></label>
          <button type="button" data-action="reset">Reset view</button>
          <button type="button" data-action="close">Close</button>
        </header>
        <div class="environment-view"></div>
        <footer class="environment-controls">
          <label>Scale <input data-transform="scale" type="number" min="0.05" max="20" step="0.1" value="1"></label>
          <label>X (m) <input data-transform="x" type="number" min="-20" max="20" step="0.1" value="0"></label>
          <label>Y (m) <input data-transform="y" type="number" min="-20" max="20" step="0.1" value="0"></label>
          <label>Z (m) <input data-transform="z" type="number" min="-20" max="20" step="0.1" value="0"></label>
          <span class="environment-status" role="status">Import an opaque GLB environment.</span>
        </footer>
      </div>
    `
    document.body.appendChild(this.dialog)
    this.viewport = this.dialog.querySelector<HTMLDivElement>('.environment-view')!
    this.fileInput = this.dialog.querySelector<HTMLInputElement>('input[type=file]')!
    this.status = this.dialog.querySelector<HTMLSpanElement>('.environment-status')!
    this.scaleInput = this.dialog.querySelector<HTMLInputElement>('[data-transform=scale]')!
    this.xInput = this.dialog.querySelector<HTMLInputElement>('[data-transform=x]')!
    this.yInput = this.dialog.querySelector<HTMLInputElement>('[data-transform=y]')!
    this.zInput = this.dialog.querySelector<HTMLInputElement>('[data-transform=z]')!
    this.dialog.querySelector('[data-action=close]')!.addEventListener('click', () => this.close())
    this.dialog.querySelector('[data-action=reset]')!.addEventListener('click', () => this.resetView())
    this.fileInput.addEventListener('change', () => void this.importEnvironment())
    for (const input of [this.scaleInput, this.xInput, this.yInput, this.zInput]) {
      input.addEventListener('input', () => this.applyEnvironmentTransform())
    }
    window.addEventListener('keydown', (event) => {
      if (event.key === 'Escape' && this.visible) this.close()
    })
  }

  async open(): Promise<void> {
    if (this.visible) return
    if (this.closeTask) await this.closeTask
    if (this.visible) return
    this.visible = true
    this.activeViewer = null
    this.activeIdentity = null
    const session = ++this.session
    this.dialog.hidden = false
    this.status.textContent = 'Preparing the compositor...'
    try {
      this.renderer = new THREE.WebGLRenderer({ antialias: false })
      this.renderer.setPixelRatio(1)
      this.renderer.setClearColor(0x10151d)
      this.viewport.appendChild(this.renderer.domElement)
      this.scene = new THREE.Scene()
      this.scene.background = new THREE.Color(0x10151d)
      this.scene.add(new THREE.HemisphereLight(0xffffff, 0x738090, 2))
      this.environmentRoot = new THREE.Group()
      this.scene.add(this.environmentRoot)
      this.makeStage()
      this.camera.position.set(0, 0.2, 2.4)
      this.camera.lookAt(0, 0, -1.8)
      this.controls = new OrbitControls(this.camera, this.renderer.domElement)
      this.controls.target.set(0, 0, -1.8)
      this.controls.enableDamping = true
      this.controls.update()
      this.resizeObserver = new ResizeObserver(() => this.resize())
      this.resizeObserver.observe(this.viewport)
      this.resize()
      const viewerOptions = {
        rootElement: this.viewport,
        threeScene: this.scene,
        renderer: this.renderer,
        camera: this.camera,
        selfDrivenMode: false as const,
        useBuiltInControls: false as const,
        sharedMemoryForWorkers: false as const,
        gpuAcceleratedSort: false as const,
        dynamicScene: true as const,
        ignoreDevicePixelRatio: true as const,
        sceneRevealMode: SceneRevealMode.Instant,
      }
      this.viewers = [new Viewer(viewerOptions), new Viewer(viewerOptions)]
      this.pendingFrame = this.latestFrame
      if (this.pendingFrame) this.startFrameWork()
      this.status.textContent = 'Import an opaque GLB environment. Live splats will update here.'
      this.draw(session)
    } catch (error) {
      this.status.textContent = `Compositor failed: ${String(error)}`
      console.error('Environment compositor failed:', error)
    }
  }

  close(): void {
    if (!this.visible) return
    this.visible = false
    ++this.session
    ++this.stateEpoch
    ++this.importSerial
    this.dialog.hidden = true
    cancelAnimationFrame(this.animationFrame)
    this.resizeObserver?.disconnect()
    this.resizeObserver = null
    this.pendingFrame = null
    const viewers = this.viewers
    const renderer = this.renderer
    const controls = this.controls
    const environment = this.environment
    const task = this.processingTask
    this.closeTask = (async () => {
      try {
        if (task) await task
        if (viewers) await Promise.all(viewers.map((viewer) => viewer.dispose()))
      } finally {
        controls?.dispose()
        if (environment) disposeObject(environment)
        disposeObject(this.stage)
        renderer?.dispose()
        renderer?.domElement.remove()
        if (this.viewers === viewers) this.viewers = null
        this.activeViewer = null
        this.activeIdentity = null
        if (this.renderer === renderer) this.renderer = null
        if (this.controls === controls) this.controls = null
        this.environment = null
      }
    })().catch((error) => console.warn('Compositor cleanup failed:', error))
  }

  setSplatFrame(frame: SplatFrame): void {
    this.latestFrame = frame
    if (!this.visible) return
    this.pendingFrame = frame
    this.startFrameWork()
  }

  /** A new capture generation does not erase the last drawn composite. */
  holdState(): void {
    ++this.stateEpoch
    this.pendingFrame = null
    if (this.visible) {
      this.status.textContent = this.activeIdentity
        ? `Holding ${this.activeIdentity.sourceId} frame ${this.activeIdentity.frameId} from generation ${this.activeIdentity.generation}`
        : 'Waiting for a new splat frame.'
    }
  }

  private startFrameWork(): void {
    if (this.processingTask || !this.viewers || !this.visible) return
    this.processingTask = this.processFrames().finally(() => {
      this.processingTask = null
      if (this.pendingFrame && this.visible) this.startFrameWork()
    })
  }

  private async processFrames(): Promise<void> {
    while (this.visible && this.pendingFrame) {
      const frame = this.pendingFrame
      this.pendingFrame = null
      const stateEpoch = this.stateEpoch
      const viewers = this.viewers!
      const nextViewer = this.activeViewer === viewers[0] ? viewers[1] : viewers[0]
      try {
        const data = encodeEnvironmentSplat(frame)
        const splatBuffer = await SplatLoader.loadFromFileData(data, 1, 0, false)
        if (!this.visible || viewers !== this.viewers) return
        await nextViewer.addSplatBuffers([splatBuffer], [{}], true, false, false, true)
        if (!this.visible || viewers !== this.viewers) return
        // Keep drawing the old viewer until this viewer finishes its first sort.
        // A later browser event can queue a frame while addSplatBuffers awaits.
        const pending = this.pendingFrame as SplatFrame | null
        if (!mayPresentPreparedFrame(
          { sourceId: frame.header.source_id, generation: Number(frame.header.generation ?? 0) },
          pending && { sourceId: pending.header.source_id, generation: Number(pending.header.generation ?? 0) },
          stateEpoch,
          this.stateEpoch,
        )) continue
        this.applyCaptureFov(frame)
        this.activeViewer = nextViewer
        this.activeIdentity = {
          sourceId: frame.header.source_id,
          generation: Number(frame.header.generation ?? 0),
          frameId: frame.header.frame_id,
        }
        if (this.visible) this.status.textContent = `Splat frame ${frame.header.frame_id} · ${frame.header.count} splats`
      } catch (error) {
        if (this.visible) this.status.textContent = `Splat update failed: ${String(error)}`
      }
    }
  }

  private applyCaptureFov(frame: SplatFrame): void {
    const fy = Number(frame.header.intrinsics?.[1])
    const cy = Number(frame.header.intrinsics?.[3])
    const height = Number(frame.header.image_height ?? (2 * cy + 1))
    if (!Number.isFinite(fy) || fy <= 0 || !Number.isFinite(height) || height <= 0) return
    const fov = 2 * Math.atan(height / (2 * fy)) * 180 / Math.PI
    if (fov >= 10 && fov <= 140) {
      this.camera.fov = fov
      this.camera.updateProjectionMatrix()
    }
  }

  private async importEnvironment(): Promise<void> {
    const file = this.fileInput.files?.[0]
    this.fileInput.value = ''
    if (!file || !this.visible) return
    if (!file.name.toLowerCase().endsWith('.glb') || file.size > MAX_GLB_BYTES) {
      this.status.textContent = 'Select a GLB file smaller than 30 MB.'
      return
    }
    const session = this.session
    const importSerial = ++this.importSerial
    this.status.textContent = `Loading ${file.name}...`
    try {
      const buffer = await file.arrayBuffer()
      checkGlb(buffer)
      const gltf = await new GLTFLoader().parseAsync(buffer, '')
      if (!this.visible || session !== this.session || importSerial !== this.importSerial) {
        disposeObject(gltf.scene)
        return
      }
      const bounds = new THREE.Box3().setFromObject(gltf.scene)
      if (bounds.isEmpty()) throw new Error('The GLB has no visible geometry')
      const size = bounds.getSize(new THREE.Vector3())
      const largest = Math.max(size.x, size.y, size.z)
      if (!Number.isFinite(largest) || largest <= 0) throw new Error('The GLB size is invalid')
      const center = bounds.getCenter(new THREE.Vector3())
      gltf.scene.position.sub(center)
      if (this.environment) {
        this.environmentRoot.remove(this.environment)
        disposeObject(this.environment)
      }
      this.environment = gltf.scene
      this.environmentRoot.add(gltf.scene)
      this.environmentBaseScale = 4 / largest
      this.scaleInput.value = '1'
      this.xInput.value = '0'
      this.yInput.value = '0'
      this.zInput.value = '-1.8'
      this.applyEnvironmentTransform()
      this.stage.visible = false
      let hasTransparentMaterial = false
      gltf.scene.traverse((object) => {
        if (!(object instanceof THREE.Mesh)) return
        for (const material of (Array.isArray(object.material) ? object.material : [object.material])) {
          if (material.transparent) hasTransparentMaterial = true
        }
      })
      this.status.textContent = hasTransparentMaterial
        ? `${file.name} loaded. Transparent surfaces may not occlude splats.`
        : `${file.name} loaded. The longest side is 4 m.`
    } catch (error) {
      this.status.textContent = `Import failed: ${String(error)}`
    }
  }

  private applyEnvironmentTransform(): void {
    const values = [this.scaleInput, this.xInput, this.yInput, this.zInput].map((input) => Number(input.value))
    if (values.some((value) => !Number.isFinite(value)) || values[0] <= 0) return
    this.environmentRoot.scale.setScalar(this.environmentBaseScale * values[0])
    this.environmentRoot.position.set(values[1], values[2], values[3])
  }

  private makeStage(): void {
    this.stage = new THREE.Group()
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(8, 8),
      new THREE.MeshStandardMaterial({ color: 0x343b48, roughness: 1 }),
    )
    floor.rotation.x = -Math.PI / 2
    floor.position.set(0, -1.2, -2)
    this.stage.add(floor)
    this.stage.add(new THREE.GridHelper(8, 16, 0x607087, 0x455063))
    this.stage.children[1].position.set(0, -1.195, -2)
    this.scene.add(this.stage)
  }

  private resetView(): void {
    this.camera.position.set(0, 0.2, 2.4)
    this.controls?.target.set(0, 0, -1.8)
    this.controls?.update()
  }

  private resize(): void {
    if (!this.renderer) return
    const width = Math.max(1, this.viewport.clientWidth)
    const height = Math.max(1, this.viewport.clientHeight)
    this.renderer.setSize(width, height, false)
    this.camera.aspect = width / height
    this.camera.updateProjectionMatrix()
  }

  private draw(session: number): void {
    if (!this.visible || session !== this.session || !this.renderer || !this.viewers) return
    this.controls?.update()
    this.activeViewer?.update()
    this.renderer.clear()
    if (this.activeViewer) this.activeViewer.render()
    else this.renderer.render(this.scene, this.camera)
    this.animationFrame = requestAnimationFrame(() => this.draw(session))
  }
}
