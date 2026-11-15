/*
 * Spike: render a dynamic Gaussian splat state inside a three.js scene.
 *
 * Approach A: custom three.js instanced-quad renderer with a ShaderMaterial.
 *             The 3D covariance is projected to screen space in the vertex
 *             shader (the standard 3DGS Jacobian approximation) and the
 *             Gaussian weight is evaluated in the fragment shader.
 *
 * Approach B: in-memory splat data fed to @mkkellogg/gaussian-splats-3d
 *             through SplatLoader.loadFromFileData() and
 *             Viewer.addSplatBuffers(), using DropInViewer to place the splat
 *             mesh inside a normal three.js scene.
 *
 * URL options:
 *   ?mode=a|b|both   show one approach only (default: both)
 *   ?sort=0          disable per-frame back-to-front sorting (Approach A)
 *   ?blend=add       additive blending instead of alpha blending (Approach A)
 */
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'

const SPLAT_COUNT = 5000
const SPLAT_OPACITY = 0.9

const params = new URLSearchParams(window.location.search)
const mode = params.get('mode') ?? 'both'
const sortEnabled = params.get('sort') !== '0'
const additiveBlend = params.get('blend') === 'add'

interface PersonSplatData {
  count: number
  positions: Float32Array
  scales: Float32Array
  quats: Float32Array
  colors: Uint8Array
}

/** Deterministic PRNG so every run and screenshot uses the same cloud. */
function createRandom(seed: number): () => number {
  let state = seed >>> 0
  return () => {
    state = (state + 0x6d2b79f5) >>> 0
    let t = state
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/**
 * Build 5000 synthetic person-shaped splats:
 * ellipsoid torso, sphere head and neck, and two capsule arms.
 * Convention: +X right, +Y up, camera forward -Z. Scales are linear metres.
 */
function buildPersonSplats(count: number): PersonSplatData {
  const rand = createRandom(0x5eed1234)
  const positions = new Float32Array(count * 3)
  const scales = new Float32Array(count * 3)
  const quats = new Float32Array(count * 4)
  const colors = new Uint8Array(count * 3)

  const torsoCount = Math.round(count * 0.4)
  const headCount = Math.round(count * 0.2)
  const armCount = Math.floor((count - torsoCount - headCount) / 2)

  const randomUnitVector = (out: THREE.Vector3): THREE.Vector3 => {
    const z = rand() * 2 - 1
    const phi = rand() * Math.PI * 2
    const r = Math.sqrt(Math.max(0, 1 - z * z))
    return out.set(r * Math.cos(phi), r * Math.sin(phi), z)
  }

  const tmp = new THREE.Vector3()
  const tmp2 = new THREE.Vector3()
  const write = (
    index: number,
    x: number,
    y: number,
    z: number,
    scale: number,
    color: [number, number, number],
  ): void => {
    const p = index * 3
    positions[p] = x
    positions[p + 1] = y
    positions[p + 2] = z
    const s = index * 3
    const jitter = 0.85 + rand() * 0.3
    scales[s] = scale * jitter
    scales[s + 1] = scale * jitter
    scales[s + 2] = scale * jitter
    const q = index * 4
    quats[q] = 1 // w (identity rotation, wxyz order)
    quats[q + 1] = 0
    quats[q + 2] = 0
    quats[q + 3] = 0
    const c = index * 3
    const tint = 0.94 + rand() * 0.12
    colors[c] = Math.min(255, Math.round(color[0] * tint))
    colors[c + 1] = Math.min(255, Math.round(color[1] * tint))
    colors[c + 2] = Math.min(255, Math.round(color[2] * tint))
  }

  // Torso: front-facing ellipsoid shell, centre 1.05 m, half-height 0.33 m.
  const torsoRadii = new THREE.Vector3(0.21, 0.33, 0.14)
  for (let i = 0; i < torsoCount; i += 1) {
    randomUnitVector(tmp)
    write(
      i,
      torsoRadii.x * tmp.x * (0.97 + rand() * 0.06),
      1.05 + torsoRadii.y * tmp.y * (0.97 + rand() * 0.06),
      torsoRadii.z * tmp.z * (0.97 + rand() * 0.06),
      0.016,
      [96, 132, 226],
    )
  }

  // Head: sphere at 1.52 m with radius 0.11 m.
  for (let i = torsoCount; i < torsoCount + headCount; i += 1) {
    randomUnitVector(tmp)
    write(
      i,
      0.11 * tmp.x,
      1.52 + 0.11 * tmp.y,
      0.11 * tmp.z,
      0.012,
      [222, 174, 148],
    )
  }

  // Arms: capsules from shoulder to hand.
  let index = torsoCount + headCount
  const arms: Array<[THREE.Vector3, THREE.Vector3]> = [
    [new THREE.Vector3(-0.225, 1.34, 0), new THREE.Vector3(-0.34, 0.78, 0.02)],
    [new THREE.Vector3(0.225, 1.34, 0), new THREE.Vector3(0.34, 0.78, 0.02)],
  ]
  for (const [shoulder, hand] of arms) {
    const axis = tmp2.copy(hand).sub(shoulder).normalize()
    for (let i = 0; i < armCount; i += 1) {
      const t = rand()
      // Random direction perpendicular to the arm axis.
      const any = new THREE.Vector3(1, 0, 0)
      if (Math.abs(axis.dot(any)) > 0.9) any.set(0, 1, 0)
      const perpendicular = new THREE.Vector3().crossVectors(axis, any).normalize()
      const angle = rand() * Math.PI * 2
      const radial = new THREE.Vector3()
        .addScaledVector(perpendicular, Math.cos(angle))
        .addScaledVector(new THREE.Vector3().crossVectors(axis, perpendicular), Math.sin(angle))
      const radius = 0.05
      write(
        index,
        shoulder.x + (hand.x - shoulder.x) * t + radial.x * radius,
        shoulder.y + (hand.y - shoulder.y) * t + radial.y * radius,
        shoulder.z + (hand.z - shoulder.z) * t + radial.z * radius,
        0.012,
        [222, 174, 148],
      )
      index += 1
    }
  }

  return { count: index, positions, scales, quats, colors }
}

/**
 * Build the six 3D covariance components (upper triangle) per splat from
 * scale and rotation: Sigma = R * diag(s^2) * R^T.
 * Layout: covA = (s00, s01, s02), covB = (s11, s12, s22).
 */
function buildCovariances(data: PersonSplatData): Float32Array {
  const cov = new Float32Array(data.count * 6)
  const matrix = new THREE.Matrix4()
  const rotation = new THREE.Matrix4()
  const quaternion = new THREE.Quaternion()
  const scaleMatrix = new THREE.Matrix4()
  const sigma = new THREE.Matrix4()
  for (let i = 0; i < data.count; i += 1) {
    const q = i * 4
    quaternion.set(data.quats[q + 1], data.quats[q + 2], data.quats[q + 3], data.quats[q])
    rotation.makeRotationFromQuaternion(quaternion)
    const s = i * 3
    scaleMatrix.makeScale(data.scales[s], data.scales[s + 1], data.scales[s + 2])
    sigma.copy(scaleMatrix).multiply(scaleMatrix)
    matrix.copy(rotation).multiply(sigma).multiply(
      new THREE.Matrix4().copy(rotation).transpose(),
    )
    const e = matrix.elements // column-major: e[0]=m00, e[1]=m10, e[4]=m01 ...
    const c = i * 6
    cov[c] = e[0]
    cov[c + 1] = e[4]
    cov[c + 2] = e[8]
    cov[c + 3] = e[5]
    cov[c + 4] = e[9]
    cov[c + 5] = e[10]
  }
  return cov
}

interface FramingSphere {
  center: THREE.Vector3
  radius: number
}

function computeBoundingSphere(positions: Float32Array, count: number): FramingSphere {
  const box = new THREE.Box3()
  const point = new THREE.Vector3()
  for (let i = 0; i < count; i += 1) {
    point.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2])
    box.expandByPoint(point)
  }
  const center = box.getCenter(new THREE.Vector3())
  return { center, radius: box.getSize(new THREE.Vector3()).length() * 0.5 }
}

/** Fit the perspective camera so the whole cloud is visible from the front. */
function frameCamera(
  camera: THREE.PerspectiveCamera,
  controls: OrbitControls | null,
  framing: FramingSphere,
): void {
  const direction = new THREE.Vector3(0.4, 0.16, 1).normalize()
  const halfFov = THREE.MathUtils.degToRad(camera.fov * 0.5)
  const distance = (framing.radius / Math.sin(halfFov)) * 1.15
  camera.position.copy(framing.center).addScaledVector(direction, distance)
  camera.near = Math.max(0.01, distance * 0.02)
  camera.far = distance * 20
  camera.updateProjectionMatrix()
  if (controls) {
    controls.target.copy(framing.center)
    controls.update()
    controls.saveState()
  } else {
    camera.lookAt(framing.center)
  }
}

function addGrid(scene: THREE.Scene, size: number): void {
  const grid = new THREE.GridHelper(size, 20, 0x2c3e50, 0x1b2733)
  grid.position.y = 0
  scene.add(grid)
  scene.background = new THREE.Color(0x0b0e12)
}

function createHud(): HTMLElement {
  const hud = document.createElement('div')
  hud.className = 'spike-hud'
  return hud
}

const vertexShader = /* glsl */ `
  // Per-splat attributes from the instanced BufferGeometry.
  attribute vec3 aCenter;
  attribute vec3 aCovA; // (sigma00, sigma01, sigma02)
  attribute vec3 aCovB; // (sigma11, sigma12, sigma22)
  attribute vec3 aColor;

  uniform vec2 uViewport; // 1 / drawing buffer size, in pixels
  uniform vec2 uFocal;    // focal lengths in pixels

  varying vec2 vPosition; // position inside the splat, in standard deviations
  varying vec3 vColor;

  void main() {
    vec4 viewCenter = modelViewMatrix * vec4(aCenter, 1.0);

    // Cull splats behind or at the camera.
    if (viewCenter.z > -0.02) {
      gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
      vPosition = vec2(0.0);
      vColor = vec3(0.0);
      return;
    }

    // 3D covariance matrix (symmetric).
    mat3 Vrk = mat3(
      aCovA.x, aCovA.y, aCovA.z,
      aCovA.y, aCovB.x, aCovB.y,
      aCovA.z, aCovB.y, aCovB.z
    );

    // Jacobian of the perspective projection, in pixels.
    float s = 1.0 / (viewCenter.z * viewCenter.z);
    mat3 J = mat3(
      uFocal.x / viewCenter.z, 0.0, -(uFocal.x * viewCenter.x) * s,
      0.0, uFocal.y / viewCenter.z, -(uFocal.y * viewCenter.y) * s,
      0.0, 0.0, 0.0
    );

    // Concatenate the projection Jacobian with the model-view rotation.
    mat3 W = transpose(mat3(modelViewMatrix));
    mat3 T = W * J;

    // Projected 2D covariance: cov2D = J^T * W^T * Sigma3D * W * J.
    mat3 cov2Dm = transpose(T) * Vrk * T;

    // Low-pass filter for sub-pixel splats (kernel2DSize = 0.3 px^2).
    float a = cov2Dm[0][0] + 0.3;
    float b = cov2Dm[0][1];
    float d = cov2Dm[1][1] + 0.3;

    float det = a * d - b * b;
    float traceOver2 = 0.5 * (a + d);
    float term2 = sqrt(max(0.1, traceOver2 * traceOver2 - det));
    float eigenValue1 = traceOver2 + term2;
    float eigenValue2 = traceOver2 - term2;

    // Eigenvector 1, with an explicit fallback for axis-aligned covariances.
    vec2 eigenVector1;
    if (abs(b) < 1e-6) {
      eigenVector1 = (a >= d) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    } else {
      eigenVector1 = normalize(vec2(b, eigenValue1 - a));
    }
    vec2 eigenVector2 = vec2(eigenVector1.y, -eigenVector1.x);

    // Quad corners sit at sqrt(8) standard deviations; the fragment shader
    // evaluates the Gaussian weight from vPosition.
    float maxSize = 256.0;
    vec2 basisVector1 = eigenVector1 * min(sqrt(8.0 * eigenValue1), maxSize);
    vec2 basisVector2 = eigenVector2 * min(sqrt(8.0 * eigenValue2), maxSize);

    vec4 clipCenter = projectionMatrix * viewCenter;
    vec3 ndcCenter = clipCenter.xyz / clipCenter.w;
    vec2 ndcOffset = (position.x * basisVector1 + position.y * basisVector2) * uViewport * 2.0;

    gl_Position = vec4(ndcCenter.xy + ndcOffset, ndcCenter.z, 1.0);

    vPosition = position.xy * sqrt(8.0);
    vColor = aColor;
  }
`

const fragmentShader = /* glsl */ `
  uniform float uOpacity;

  varying vec2 vPosition;
  varying vec3 vColor;

  void main() {
    float squaredDistance = dot(vPosition, vPosition);
    if (squaredDistance > 8.0) discard;
    float alpha = exp(-0.5 * squaredDistance) * uOpacity;
    if (alpha < 0.004) discard;
    gl_FragColor = vec4(vColor, alpha);
  }
`

let cachedData: PersonSplatData | null = null
function getPersonSplats(): PersonSplatData {
  if (!cachedData) cachedData = buildPersonSplats(SPLAT_COUNT)
  return cachedData
}

/**
 * Panel that renders the splat cloud with our own one-draw-call instanced
 * quad renderer and a custom ShaderMaterial.
 */
class CustomSplatPanel {
  readonly element: HTMLDivElement
  private readonly canvasHost: HTMLDivElement
  private readonly hud: HTMLElement
  private readonly status: HTMLElement

  private readonly renderer: THREE.WebGLRenderer
  private readonly scene = new THREE.Scene()
  private readonly camera: THREE.PerspectiveCamera
  private readonly controls: OrbitControls
  private readonly material: THREE.ShaderMaterial
  private readonly geometry: THREE.InstancedBufferGeometry
  private readonly mesh: THREE.Mesh

  private readonly data: PersonSplatData
  private readonly covariances: Float32Array
  private readonly depths: Float32Array
  private readonly order: Uint32Array
  private readonly sortedCenters: Float32Array
  private readonly sortedCovA: Float32Array
  private readonly sortedCovB: Float32Array
  private readonly sortedColors: Float32Array

  private readonly centerAttribute: THREE.InstancedBufferAttribute
  private readonly covAAttribute: THREE.InstancedBufferAttribute
  private readonly covBAttribute: THREE.InstancedBufferAttribute
  private readonly colorAttribute: THREE.InstancedBufferAttribute

  private animationId = 0
  private sortMs = 0
  private frameCounter = 0
  private fps = 0
  private lastFpsTime = performance.now()
  private lastFrameTime = performance.now()
  private readonly viewport = new THREE.Vector2()

  constructor() {
    this.element = document.createElement('div')
    this.element.className = 'spike-panel'
    this.element.innerHTML = `
      <div class="spike-panel-title">A: custom three.js ShaderMaterial</div>
      <div class="spike-canvas-host"></div>
      <div class="spike-status"></div>
    `
    this.canvasHost = this.element.querySelector<HTMLDivElement>('.spike-canvas-host')!
    this.status = this.element.querySelector<HTMLElement>('.spike-status')!
    this.hud = createHud()
    this.canvasHost.appendChild(this.hud)

    this.data = getPersonSplats()
    this.covariances = buildCovariances(this.data)
    this.depths = new Float32Array(this.data.count)
    this.order = new Uint32Array(this.data.count)
    this.sortedCenters = new Float32Array(this.data.count * 3)
    this.sortedCovA = new Float32Array(this.data.count * 3)
    this.sortedCovB = new Float32Array(this.data.count * 3)
    this.sortedColors = new Float32Array(this.data.count * 3)

    this.renderer = new THREE.WebGLRenderer({ antialias: true })
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    this.renderer.setSize(1, 1, false)
    this.renderer.outputColorSpace = THREE.SRGBColorSpace
    this.canvasHost.appendChild(this.renderer.domElement)

    this.camera = new THREE.PerspectiveCamera(45, 1, 0.05, 100)
    this.controls = new OrbitControls(this.camera, this.renderer.domElement)
    this.controls.enableDamping = true
    this.controls.dampingFactor = 0.08

    // Unit quad: two triangles with vertices in [-1, 1].
    const quad = new Float32Array([
      -1, -1, 0,
      1, -1, 0,
      1, 1, 0,
      -1, -1, 0,
      1, 1, 0,
      -1, 1, 0,
    ])
    this.geometry = new THREE.InstancedBufferGeometry()
    this.geometry.setAttribute('position', new THREE.BufferAttribute(quad, 3))
    this.centerAttribute = new THREE.InstancedBufferAttribute(this.sortedCenters, 3)
    this.covAAttribute = new THREE.InstancedBufferAttribute(this.sortedCovA, 3)
    this.covBAttribute = new THREE.InstancedBufferAttribute(this.sortedCovB, 3)
    this.colorAttribute = new THREE.InstancedBufferAttribute(this.sortedColors, 3)
    this.geometry.setAttribute('aCenter', this.centerAttribute)
    this.geometry.setAttribute('aCovA', this.covAAttribute)
    this.geometry.setAttribute('aCovB', this.covBAttribute)
    this.geometry.setAttribute('aColor', this.colorAttribute)
    this.geometry.instanceCount = this.data.count

    this.material = new THREE.ShaderMaterial({
      uniforms: {
        uViewport: { value: new THREE.Vector2(1, 1) },
        uFocal: { value: new THREE.Vector2(1, 1) },
        uOpacity: { value: SPLAT_OPACITY },
      },
      vertexShader,
      fragmentShader,
      transparent: true,
      depthTest: true,
      depthWrite: false,
      blending: additiveBlend ? THREE.AdditiveBlending : THREE.NormalBlending,
    })

    this.mesh = new THREE.Mesh(this.geometry, this.material)
    this.mesh.frustumCulled = false
    this.scene.add(this.mesh)

    const framing = computeBoundingSphere(this.data.positions, this.data.count)
    addGrid(this.scene, framing.radius * 4)
    frameCamera(this.camera, this.controls, framing)

    this.fillInstances(false)
    const gl = this.renderer.getContext()
    const debugInfo = gl.getExtension('WEBGL_debug_renderer_info')
    const rendererName = debugInfo
      ? String(gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL))
      : 'unknown'
    this.status.textContent =
      `GL: ${rendererName} | splats: ${this.data.count} | 1 draw call | ` +
      `sort: ${sortEnabled ? 'per frame' : 'off'} | blending: ${additiveBlend ? 'additive' : 'alpha'}`

    const resizeObserver = new ResizeObserver(() => this.resize())
    resizeObserver.observe(this.canvasHost)
    this.resize()
    this.animate()

    // Debug hook for headless automation and evidence capture.
    ;(window as unknown as Record<string, unknown>).__spikeA = this
  }

  private fillInstances(sort: boolean): void {
    const start = performance.now()
    const { count, positions, colors } = this.data
    const viewMatrix = this.camera.matrixWorldInverse.elements
    for (let i = 0; i < count; i += 1) {
      this.depths[i] =
        viewMatrix[2] * positions[i * 3] +
        viewMatrix[6] * positions[i * 3 + 1] +
        viewMatrix[10] * positions[i * 3 + 2] +
        viewMatrix[14]
      this.order[i] = i
    }
    if (sort) {
      // Ascending view-space Z: farthest first, so back-to-front alpha
      // compositing is correct.
      this.order.sort((left, right) => this.depths[left] - this.depths[right])
    }
    for (let k = 0; k < count; k += 1) {
      const i = this.order[k]
      const c = i * 3
      const dc = k * 3
      this.sortedCenters[dc] = positions[c]
      this.sortedCenters[dc + 1] = positions[c + 1]
      this.sortedCenters[dc + 2] = positions[c + 2]
      const cov = i * 6
      this.sortedCovA[dc] = this.covariances[cov]
      this.sortedCovA[dc + 1] = this.covariances[cov + 1]
      this.sortedCovA[dc + 2] = this.covariances[cov + 2]
      this.sortedCovB[dc] = this.covariances[cov + 3]
      this.sortedCovB[dc + 1] = this.covariances[cov + 4]
      this.sortedCovB[dc + 2] = this.covariances[cov + 5]
      this.sortedColors[dc] = colors[c] / 255
      this.sortedColors[dc + 1] = colors[c + 1] / 255
      this.sortedColors[dc + 2] = colors[c + 2] / 255
    }
    this.centerAttribute.needsUpdate = true
    this.covAAttribute.needsUpdate = true
    this.covBAttribute.needsUpdate = true
    this.colorAttribute.needsUpdate = true
    this.sortMs = performance.now() - start
  }

  private resize(): void {
    const width = this.canvasHost.clientWidth
    const height = this.canvasHost.clientHeight
    if (width === 0 || height === 0) return
    this.camera.aspect = width / height
    this.camera.updateProjectionMatrix()
    this.renderer.setSize(width, height, false)
  }

  private updateUniforms(): void {
    this.renderer.getDrawingBufferSize(this.viewport)
    const projection = this.camera.projectionMatrix.elements
    this.material.uniforms.uViewport.value.set(1 / this.viewport.x, 1 / this.viewport.y)
    this.material.uniforms.uFocal.value.set(
      projection[0] * 0.5 * this.viewport.x,
      projection[5] * 0.5 * this.viewport.y,
    )
  }

  private animate = (): void => {
    this.animationId = requestAnimationFrame(this.animate)
    const frameStart = performance.now()
    this.controls.update()
    this.camera.updateMatrixWorld()
    this.updateUniforms()
    this.fillInstances(sortEnabled)
    this.renderer.render(this.scene, this.camera)
    this.countFrame()
    this.lastFrameTime = performance.now() - frameStart
  }

  private countFrame(): void {
    this.frameCounter += 1
    const now = performance.now()
    const elapsed = now - this.lastFpsTime
    if (elapsed >= 500) {
      this.fps = (this.frameCounter * 1000) / elapsed
      this.frameCounter = 0
      this.lastFpsTime = now
    }
    const sizes = this.renderer.getDrawingBufferSize(this.viewport)
    this.hud.textContent =
      `FPS ${this.fps.toFixed(0)} | CPU/frame ` +
      `${this.lastFrameTime.toFixed(2)} ms | sort ` +
      `${this.sortMs.toFixed(2)} ms | buffer ` +
      `${sizes.x}x${sizes.y} | ${this.renderer.info.render.calls} draw call`
  }

  dispose(): void {
    cancelAnimationFrame(this.animationId)
    this.controls.dispose()
    this.geometry.dispose()
    this.material.dispose()
    this.renderer.dispose()
  }
}

/**
 * Panel that renders the same in-memory splat cloud with
 * @mkkellogg/gaussian-splats-3d 0.4.7 through a DropInViewer placed inside a
 * normal three.js scene.
 */
class LibrarySplatPanel {
  readonly element: HTMLDivElement
  private readonly canvasHost: HTMLDivElement
  private readonly hud: HTMLElement
  private readonly status: HTMLElement

  private renderer: THREE.WebGLRenderer | null = null
  private animationId = 0
  private splatCount = 0
  private sortMs = 0
  private frameCounter = 0
  private fps = 0
  private lastFpsTime = performance.now()

  constructor() {
    this.element = document.createElement('div')
    this.element.className = 'spike-panel'
    this.element.innerHTML = `
      <div class="spike-panel-title">B: @mkkellogg/gaussian-splats-3d 0.4.7 (in-memory)</div>
      <div class="spike-canvas-host"></div>
      <div class="spike-status">Loading library...</div>
    `
    this.canvasHost = this.element.querySelector<HTMLDivElement>('.spike-canvas-host')!
    this.status = this.element.querySelector<HTMLElement>('.spike-status')!
    this.hud = createHud()
    this.canvasHost.appendChild(this.hud)
    void this.start()
  }

  private async start(): Promise<void> {
    try {
      // @ts-expect-error -- the package ships JavaScript only (no type declarations).
      const library: any = await import('@mkkellogg/gaussian-splats-3d')

      // Build the same cloud as a binary in-memory .splat payload:
      // 32 bytes per splat: center f32x3, scale f32x3, RGBA u8, rotation u8x4.
      const data = getPersonSplats()
      const rowBytes = 32
      const buffer = new ArrayBuffer(data.count * rowBytes)
      const floats = new Float32Array(buffer)
      const bytes = new Uint8Array(buffer)
      for (let i = 0; i < data.count; i += 1) {
        const floatBase = i * (rowBytes / 4)
        floats[floatBase] = data.positions[i * 3]
        floats[floatBase + 1] = data.positions[i * 3 + 1]
        floats[floatBase + 2] = data.positions[i * 3 + 2]
        floats[floatBase + 3] = data.scales[i * 3]
        floats[floatBase + 4] = data.scales[i * 3 + 1]
        floats[floatBase + 5] = data.scales[i * 3 + 2]
        const byteBase = i * rowBytes + 24
        bytes[byteBase] = data.colors[i * 3]
        bytes[byteBase + 1] = data.colors[i * 3 + 1]
        bytes[byteBase + 2] = data.colors[i * 3 + 2]
        bytes[byteBase + 3] = Math.round(SPLAT_OPACITY * 255)
        bytes[byteBase + 4] = 255 // quaternion w = 1 (identity), wxyz order
        bytes[byteBase + 5] = 128
        bytes[byteBase + 6] = 128
        bytes[byteBase + 7] = 128
      }
      const splatBuffer = await library.SplatLoader.loadFromFileData(buffer, 1, 0, true)

      const scene = new THREE.Scene()
      const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 100)
      this.renderer = new THREE.WebGLRenderer({ antialias: true })
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
      this.renderer.setSize(1, 1, false)
      this.renderer.outputColorSpace = THREE.SRGBColorSpace
      this.canvasHost.appendChild(this.renderer.domElement)

      const framing = computeBoundingSphere(data.positions, data.count)
      addGrid(scene, framing.radius * 4)
      const orbit = new OrbitControls(camera, this.renderer.domElement)
      orbit.enableDamping = true
      frameCamera(camera, orbit, framing)

      // DropInViewer is a THREE.Group. Add it to a normal three.js scene.
      const dropIn = new library.DropInViewer({
        sharedMemoryForWorkers: false,
        gpuAcceleratedSort: false,
        dynamicScene: true,
        sceneRevealMode: library.SceneRevealMode.Instant,
        logLevel: library.LogLevel.None,
      })
      scene.add(dropIn)
      // Render one frame first so the viewer receives the external renderer.
      this.renderer.render(scene, camera)
      // In-memory splat buffer, no URL and no file.
      await dropIn.viewer.addSplatBuffers(
        [splatBuffer],
        [{}],
        true,
        false,
        false,
        true,
        true,
      )
      this.splatCount = dropIn.viewer.getSplatMesh().getSplatCount()
      this.status.textContent =
        `API: SplatLoader.loadFromFileData(ArrayBuffer) then ` +
        `DropInViewer.viewer.addSplatBuffers([SplatBuffer]) | ` +
        `splats: ${this.splatCount} | dynamicScene: true`

      const resizeObserver = new ResizeObserver(() => {
        const width = this.canvasHost.clientWidth
        const height = this.canvasHost.clientHeight
        if (width === 0 || height === 0 || !this.renderer) return
        camera.aspect = width / height
        camera.updateProjectionMatrix()
        this.renderer.setSize(width, height, false)
      })
      resizeObserver.observe(this.canvasHost)

      const animate = (): void => {
        this.animationId = requestAnimationFrame(animate)
        if (!this.renderer) return
        orbit.update()
        this.renderer.render(scene, camera)
        this.frameCounter += 1
        const now = performance.now()
        const elapsed = now - this.lastFpsTime
        if (elapsed >= 500) {
          this.fps = (this.frameCounter * 1000) / elapsed
          this.frameCounter = 0
          this.lastFpsTime = now
          this.sortMs = moduleLastSortTime(dropIn)
          const sizes = this.renderer.getDrawingBufferSize(new THREE.Vector2())
          this.hud.textContent =
            `FPS ${this.fps.toFixed(0)} | library sort ${this.sortMs.toFixed(2)} ms | ` +
            `buffer ${sizes.x}x${sizes.y} | ${this.renderer.info.render.calls} draw calls`
        }
      }
      animate()
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      this.status.textContent = `APPROACH B FAILED: ${message}`
      this.status.classList.add('is-error')
      this.hud.textContent = 'FAILED'
      console.error(error)
    }
  }

  dispose(): void {
    cancelAnimationFrame(this.animationId)
    this.renderer?.dispose()
  }
}

/** Read the library's last sort duration (milliseconds) without its types. */
function moduleLastSortTime(dropIn: any): number {
  const value = dropIn?.viewer?.lastSortTime
  return typeof value === 'number' ? value : 0
}

function installStyles(): void {
  const style = document.createElement('style')
  style.textContent = `
    :root { color-scheme: dark; }
    html, body { margin: 0; height: 100%; background: #0b0e12; overflow: hidden; }
    #spike-root { display: flex; gap: 8px; padding: 8px; height: calc(100% - 16px); box-sizing: border-box; }
    .spike-panel { position: relative; flex: 1 1 0; min-width: 0; display: flex;
      flex-direction: column; border: 1px solid #22303c; border-radius: 6px; overflow: hidden; }
    .spike-panel-title { flex: 0 0 auto; padding: 6px 10px; font: 600 12px/1.4 system-ui, sans-serif;
      color: #cfe3f5; background: #131a22; border-bottom: 1px solid #22303c; }
    .spike-canvas-host { position: relative; flex: 1 1 auto; min-height: 0; }
    .spike-canvas-host canvas { display: block; width: 100%; height: 100%; }
    .spike-hud { position: absolute; left: 8px; bottom: 8px; z-index: 2; padding: 3px 7px;
      border-radius: 4px; background: rgba(8, 12, 16, 0.72); color: #9ff0c0;
      font: 11px/1.45 ui-monospace, Consolas, monospace; pointer-events: none; }
    .spike-status { flex: 0 0 auto; padding: 6px 10px; font: 11px/1.45 ui-monospace, Consolas, monospace;
      color: #8fa8bd; background: #101720; border-top: 1px solid #22303c; }
    .spike-status.is-error { color: #ff8a8a; }
  `
  document.head.appendChild(style)
}

function main(): void {
  installStyles()
  const root = document.querySelector<HTMLDivElement>('#spike-root')
  if (!root) throw new Error('Spike root element not found')

  if (mode === 'a' || mode === 'both') {
    root.appendChild(new CustomSplatPanel().element)
  }
  if (mode === 'b' || mode === 'both') {
    root.appendChild(new LibrarySplatPanel().element)
  }

  // Expose a small hook so an external driver can verify the page state.
  ;(window as unknown as Record<string, unknown>).__spike = {
    mode,
    sortEnabled,
    additiveBlend,
    splatCount: SPLAT_COUNT,
  }
}

main()
