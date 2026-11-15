/*
 * Spike: prove that gsplat (splat.js) 1.2.9 can render a Gaussian splat cloud
 * that is built from raw constructor-convention arrays inside this Vite +
 * TypeScript application.
 *
 * Constructor convention (shared/coordinate_system/README.md):
 *   - right-handed, +X right, +Y up, camera forward -Z, camera at origin
 *   - rotations are unit quaternions in wxyz order
 *   - scales are natural logarithms of metres, opacity is a logit
 *
 * gsplat 1.2.9 internal conventions (read from node_modules/gsplat/dist):
 *   - SplatData(vertexCount, positions, rotations, scales, colors)
 *   - positions: Float32Array [x, y, z] per splat, metres
 *   - rotations: Float32Array [w, x, y, z] per splat (SplatData.serialize
 *     writes rotations[4i + 0] into byte 0 and Deserialize/PLY loader read it
 *     back as w)
 *   - scales: Float32Array [sx, sy, sz] per splat, LINEAR metres (the PLY
 *     loader applies Math.exp to 3DGS log scales before storing them)
 *   - colors: Uint8Array [r, g, b, a] per splat; rgb and alpha are used
 *     linearly (/255) by the fragment shader, alpha is the linear opacity
 *     (the PLY loader stores sigmoid(logit opacity) * 255 there)
 *   - camera view frame: +X right, +Y down, +Z forward (CameraData builds the
 *     projection with w = +z and -2*fy/height, so an identity camera sees
 *     points at positive Z and screen-up is -Y)
 */

import {
  Camera,
  Color32,
  Quaternion,
  Scene,
  Splat,
  SplatData,
  Vector3,
  WebGLRenderer,
} from 'gsplat'

const SPLAT_COUNT = 5000
const VERTICAL_FOV_DEG = 50
const CAMERA_DISTANCE_M = 2.4
const NEAR_M = 0.05
const FAR_M = 50
const UPDATE_INTERVAL_MS = 200 // 5 Hz stream update test
const OPACITY = 0.9
const OPACITY_LOGIT = Math.log(OPACITY / (1 - OPACITY))

interface RawSplatCloud {
  readonly vertexCount: number
  readonly positions: Float32Array // metres, constructor convention
  readonly rotations: Float32Array // wxyz, identity in this spike
  readonly logScales: Float32Array // ln(metres)
  readonly colors: Float32Array // linear RGB, 0..1
  readonly opacityLogits: Float32Array // logit space
}

interface BlobSpec {
  readonly count: number
  readonly center: readonly [number, number, number]
  readonly radii: readonly [number, number, number]
  readonly tiltZ: number
  readonly color: readonly [number, number, number]
}

const hudElement = document.querySelector<HTMLDivElement>('#hud')
const canvasElement = document.querySelector<HTMLCanvasElement>('#viewer')

if (!hudElement || !canvasElement) {
  throw new Error('spike-gsplat.html is missing #hud or #viewer')
}

const hud = hudElement
const canvas = canvasElement

// "?mode=replace" streams fresh SplatData arrays through new Splat objects.
// The default "inplace" mode mutates the existing SplatData buffers.
const updateMode: 'inplace' | 'replace' =
  new URLSearchParams(window.location.search).get('mode') === 'replace' ? 'replace' : 'inplace'

let frames = 0
let updates = 0
let skippedUpdates = 0
let lastError = ''

function describeError(error: unknown): string {
  if (error instanceof Error) {
    return `${error.name}: ${error.message}`
  }
  return String(error)
}

function showHud(): void {
  const lines = [
    'gsplat 1.2.9 spike - synthetic 5000-splat person cloud',
    `splats: ${SPLAT_COUNT}`,
    `frames rendered: ${frames}`,
    `${updateMode} 5 Hz updates: ${updates} (skipped while packing: ${skippedUpdates})`,
    `WebGL2: ${webglSummary}`,
    lastError ? `ERROR: ${lastError}` : 'status: ok',
  ]
  hud.textContent = lines.join('\n')
  hud.classList.toggle('error', lastError.length > 0)
}

window.addEventListener('error', (event) => {
  lastError = event.message
  showHud()
})

window.addEventListener('unhandledrejection', (event) => {
  lastError = describeError(event.reason)
  showHud()
})

function probeWebGL2(): string {
  const probeCanvas = document.createElement('canvas')
  const gl = probeCanvas.getContext('webgl2')
  if (!gl) {
    return 'UNAVAILABLE'
  }
  const debugInfo = gl.getExtension('WEBGL_debug_renderer_info')
  const rendererName = debugInfo
    ? String(gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL))
    : String(gl.getParameter(gl.RENDERER))
  return `${String(gl.getParameter(gl.VERSION))} | ${rendererName}`
}

const webglSummary = probeWebGL2()

function mulberry32(seed: number): () => number {
  let state = seed >>> 0
  return () => {
    state = (state + 0x6d2b79f5) >>> 0
    let value = state
    value = Math.imul(value ^ (value >>> 15), value | 1)
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61)
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296
  }
}

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value))
}

function toByte(value: number): number {
  return Math.round(clamp01(value) * 255)
}

function createRawSplatCloud(): RawSplatCloud {
  const random = mulberry32(0x5eed1234)

  // A simple person: head, ellipsoid torso, two tilted arms. Centre of the
  // body is at (0, 0, -CAMERA_DISTANCE_M) so the blob stays inside a 50 degree
  // vertical field of view from a camera at the origin.
  const parts: BlobSpec[] = [
    {
      count: 850,
      center: [0, 0.62, -CAMERA_DISTANCE_M],
      radii: [0.115, 0.14, 0.11],
      tiltZ: 0,
      color: [0.88, 0.68, 0.54], // skin
    },
    {
      count: 2300,
      center: [0, -0.08, -CAMERA_DISTANCE_M],
      radii: [0.23, 0.4, 0.14],
      tiltZ: 0,
      color: [0.2, 0.44, 0.92], // blue shirt
    },
    {
      count: 925,
      center: [-0.37, 0.08, -CAMERA_DISTANCE_M],
      radii: [0.23, 0.075, 0.075],
      tiltZ: (-128 * Math.PI) / 180, // green arm on world -X (camera left)
      color: [0.16, 0.85, 0.5],
    },
    {
      count: 925,
      center: [0.37, 0.08, -CAMERA_DISTANCE_M],
      radii: [0.23, 0.075, 0.075],
      tiltZ: (-52 * Math.PI) / 180, // amber arm on world +X (camera right)
      color: [0.95, 0.62, 0.14],
    },
  ]

  const positions = new Float32Array(SPLAT_COUNT * 3)
  const rotations = new Float32Array(SPLAT_COUNT * 4)
  const logScales = new Float32Array(SPLAT_COUNT * 3)
  const colors = new Float32Array(SPLAT_COUNT * 3)
  const opacityLogits = new Float32Array(SPLAT_COUNT)

  let index = 0
  for (const part of parts) {
    const [centerX, centerY, centerZ] = part.center
    const [radiusX, radiusY, radiusZ] = part.radii
    const cosTilt = Math.cos(part.tiltZ)
    const sinTilt = Math.sin(part.tiltZ)

    for (let i = 0; i < part.count; i++) {
      // Uniform point in the unit ball, stretched into an ellipsoid.
      const cosPhi = Math.cos(random() * Math.PI * 2)
      const sinPhi = Math.sin(random() * Math.PI * 2)
      const cosTheta = random() * 2 - 1
      const sinTheta = Math.sqrt(1 - cosTheta * cosTheta)
      const radius = Math.cbrt(random())

      const localX = sinTheta * cosPhi * radius * radiusX
      const localY = sinTheta * sinPhi * radius * radiusY
      const localZ = cosTheta * radius * radiusZ

      positions[index * 3 + 0] = centerX + localX * cosTilt - localY * sinTilt
      positions[index * 3 + 1] = centerY + localX * sinTilt + localY * cosTilt
      positions[index * 3 + 2] = centerZ + localZ

      // Identity rotation in wxyz order.
      rotations[index * 4 + 0] = 1
      rotations[index * 4 + 1] = 0
      rotations[index * 4 + 2] = 0
      rotations[index * 4 + 3] = 0

      // Isotropic Gaussian, 0.012 m .. 0.030 m radius, stored as log scale.
      const logScale = Math.log(0.012 + random() * 0.018)
      logScales[index * 3 + 0] = logScale
      logScales[index * 3 + 1] = logScale
      logScales[index * 3 + 2] = logScale

      const jitter = (random() - 0.5) * 0.12
      colors[index * 3 + 0] = clamp01(part.color[0] + jitter)
      colors[index * 3 + 1] = clamp01(part.color[1] + jitter)
      colors[index * 3 + 2] = clamp01(part.color[2] + jitter)

      opacityLogits[index] = OPACITY_LOGIT
      index++
    }
  }

  return { vertexCount: index, positions, rotations, logScales, colors, opacityLogits }
}

function sigmoid(value: number): number {
  return 1 / (1 + Math.exp(-value))
}

function toSplatData(cloud: RawSplatCloud): SplatData {
  const count = cloud.vertexCount
  const rotations = new Float32Array(count * 4)
  const scales = new Float32Array(count * 3)
  const colors = new Uint8Array(count * 4)

  for (let i = 0; i < count; i++) {
    // gsplat stores rotations as [w, x, y, z]; the constructor cloud is wxyz.
    rotations[i * 4 + 0] = cloud.rotations[i * 4 + 0]
    rotations[i * 4 + 1] = cloud.rotations[i * 4 + 1]
    rotations[i * 4 + 2] = cloud.rotations[i * 4 + 2]
    rotations[i * 4 + 3] = cloud.rotations[i * 4 + 3]

    // gsplat expects linear scale; the constructor cloud stores log scale.
    for (let axis = 0; axis < 3; axis++) {
      scales[i * 3 + axis] = Math.exp(cloud.logScales[i * 3 + axis])
    }

    // gsplat colors are linear RGBA bytes; alpha is linear opacity.
    colors[i * 4 + 0] = toByte(cloud.colors[i * 3 + 0])
    colors[i * 4 + 1] = toByte(cloud.colors[i * 3 + 1])
    colors[i * 4 + 2] = toByte(cloud.colors[i * 3 + 2])
    colors[i * 4 + 3] = toByte(sigmoid(cloud.opacityLogits[i]))
  }

  return new SplatData(count, cloud.positions, rotations, scales, colors)
}

function main(): void {
  const cloud = createRawSplatCloud()
  const splatData = toSplatData(cloud)

  // Renderer: owns its own WebGL2 context on the supplied canvas.
  const renderer = new WebGLRenderer(canvas)
  renderer.backgroundColor = new Color32(12, 16, 26, 255)

  const width = Math.max(1, Math.round(canvas.clientWidth || window.innerWidth))
  const height = Math.max(1, Math.round(canvas.clientHeight || window.innerHeight))
  renderer.setSize(width, height)

  // Scene graph: one Splat object that owns one SplatData buffer.
  const scene = new Scene()
  const camera = new Camera()

  const focalLength = height / (2 * Math.tan((VERTICAL_FOV_DEG * Math.PI) / 360))
  camera.data.fx = focalLength
  camera.data.fy = focalLength
  camera.data.near = NEAR_M
  camera.data.far = FAR_M

  // Camera at the constructor origin. gsplat's view frame is +Z forward with
  // -Y up on screen, so a 180 degree rotation about X maps our -Z forward /
  // +Y up world into it: world (-X,-Y,-Z) becomes view (+X,-Y,+Z).
  camera.position = new Vector3(0, 0, 0)
  camera.rotation = new Quaternion(1, 0, 0, 0)

  const splat = new Splat(splatData)
  scene.addObject(splat)
  let activeSplat = splat

  // 5 Hz update test, in-place mode: mutate the live SplatData buffers, mark
  // them changed, and raise the object change event so RenderData re-packs.
  // The position buffer is replaced by SplatData.reattach() after every pack,
  // so always read splat.data.positions again inside the update.
  const baselinePositions = new Float32Array(cloud.positions)

  const applyUpdate = (): void => {
    const data = activeSplat.data
    if (data.detached) {
      skippedUpdates++
      return
    }
    const positions = data.positions
    const bob = Math.sin(updates * 0.6) * 0.03
    for (let i = 0; i < SPLAT_COUNT; i++) {
      positions[i * 3 + 0] = baselinePositions[i * 3 + 0]
      positions[i * 3 + 1] = baselinePositions[i * 3 + 1] + bob
      positions[i * 3 + 2] = baselinePositions[i * 3 + 2]
    }
    data.changed = true
    activeSplat.raiseChangeEvent()
    updates++
  }

  // 5 Hz update test, replace mode: every "frame" is a fresh SplatData built
  // from new raw arrays, exactly as a stream consumer would supply it.
  const replaceSplat = (): void => {
    const nextSplat = new Splat(toSplatData(createRawSplatCloud()))
    scene.removeObject(activeSplat)
    scene.addObject(nextSplat)
    activeSplat = nextSplat
    updates++
  }

  if (updateMode === 'replace') {
    window.setInterval(replaceSplat, UPDATE_INTERVAL_MS)
  } else {
    window.setInterval(applyUpdate, UPDATE_INTERVAL_MS)
  }

  const renderFrame = (): void => {
    try {
      renderer.render(scene, camera)
      frames++
      if (frames % 15 === 0) {
        showHud()
      }
    } catch (error) {
      lastError = describeError(error)
      showHud()
    }
  }

  // Normal driving path: requestAnimationFrame. Headless Edge with
  // --virtual-time-budget does not tick requestAnimationFrame, so a timer
  // fallback renders the same frame until the first rAF callback arrives.
  let animationFrameSeen = false
  const frame = (): void => {
    animationFrameSeen = true
    renderFrame()
    requestAnimationFrame(frame)
  }

  window.setInterval(() => {
    if (!animationFrameSeen) {
      renderFrame()
    }
  }, 33)

  showHud()
  requestAnimationFrame(frame)
}

try {
  main()
} catch (error) {
  lastError = describeError(error)
  showHud()
}
