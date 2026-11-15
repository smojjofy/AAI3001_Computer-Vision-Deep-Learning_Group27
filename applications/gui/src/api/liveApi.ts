/*
 * Fetch helpers for the live local pipeline API.
 *
 * Every request is same-origin. In development the Vite proxy forwards /api to
 * the Python server at http://127.0.0.1:8765 (see vite.config.ts).
 */

import type {
  DiagnosticsPayload,
  MotionPayload,
  PoseDepthFrame,
  PreviewFrame,
  SplatFrame,
  SplatHeader,
} from './types'

export type ApiResult<T> =
  | { kind: 'ok'; data: T }
  | { kind: 'empty' }
  | { kind: 'error'; error: string }

function describeError(error: unknown): string {
  if (error instanceof Error) {
    return `${error.name}: ${error.message}`
  }
  return String(error)
}

async function request(
  path: string,
  accept: string,
): Promise<ApiResult<ArrayBuffer>> {
  try {
    const response = await fetch(path, {
      cache: 'no-store',
      headers: { Accept: accept },
    })
    if (response.status === 204) {
      return { kind: 'empty' }
    }
    if (!response.ok) {
      return { kind: 'error', error: `HTTP ${response.status}` }
    }
    return { kind: 'ok', data: await response.arrayBuffer() }
  } catch (error) {
    return { kind: 'error', error: describeError(error) }
  }
}

async function requestJson<T>(path: string): Promise<ApiResult<T>> {
  const result = await request(path, 'application/json')
  if (result.kind !== 'ok') {
    return result
  }
  try {
    const text = new TextDecoder().decode(result.data)
    return { kind: 'ok', data: JSON.parse(text) as T }
  } catch (error) {
    return { kind: 'error', error: `invalid JSON: ${describeError(error)}` }
  }
}

export function fetchPreview(): Promise<ApiResult<PreviewFrame>> {
  return requestJson<PreviewFrame>('/api/preview')
}

export function fetchDiagnostics(): Promise<ApiResult<DiagnosticsPayload>> {
  return requestJson<DiagnosticsPayload>('/api/diagnostics')
}

export function fetchPoseDepth(): Promise<ApiResult<PoseDepthFrame>> {
  return requestJson<PoseDepthFrame>('/api/pose-depth')
}

export function fetchMotion(): Promise<ApiResult<MotionPayload>> {
  return requestJson<MotionPayload>('/api/motion')
}

export async function fetchMask(): Promise<ApiResult<ImageBitmap>> {
  const result = await request('/api/mask', 'image/png')
  if (result.kind !== 'ok') {
    return result
  }
  try {
    const blob = new Blob([result.data], { type: 'image/png' })
    return { kind: 'ok', data: await createImageBitmap(blob) }
  } catch (error) {
    return { kind: 'error', error: describeError(error) }
  }
}

export async function fetchSplat(): Promise<ApiResult<SplatFrame>> {
  const result = await request('/api/splat', 'application/octet-stream')
  if (result.kind !== 'ok') {
    return result
  }
  try {
    return { kind: 'ok', data: decodeSplatPayload(result.data) }
  } catch (error) {
    return { kind: 'error', error: describeError(error) }
  }
}

/** Decode a base64 JPEG into a decoded image element. */
export async function decodeJpegBase64(
  base64: string,
): Promise<HTMLImageElement> {
  const image = new Image()
  image.src = `data:image/jpeg;base64,${base64}`
  await image.decode()
  return image
}

const SPLAT_ARRAY_NAMES = [
  'positions',
  'sh_dc',
  'opacity_logits',
  'log_scales',
  'rotations',
  'reconstruction_weights',
  'source_pixels',
] as const

type SplatArrayName = (typeof SPLAT_ARRAY_NAMES)[number]

/**
 * Decode one /api/splat body: little-endian uint32 header length, UTF-8 JSON
 * header, then the packed arrays described by the header.
 *
 * The JSON length is 4 + headerLength, which is not always 4-byte aligned, so
 * the packed payload is copied once into a fresh buffer before typed-array
 * views are created.
 */
export function decodeSplatPayload(buffer: ArrayBuffer): SplatFrame {
  if (buffer.byteLength < 8) {
    throw new Error('splat payload is too short')
  }
  const length = new DataView(buffer).getUint32(0, true)
  const headerStart = 4
  const headerEnd = headerStart + length
  if (headerEnd > buffer.byteLength) {
    throw new Error('splat header length exceeds the payload')
  }
  const headerText = new TextDecoder().decode(
    new Uint8Array(buffer, headerStart, length),
  )
  const header = JSON.parse(headerText) as SplatHeader

  if (!Array.isArray(header.arrays) || header.arrays.length === 0) {
    throw new Error('splat header has no array table')
  }
  if (!Array.isArray(header.intrinsics) || header.intrinsics.length !== 4) {
    throw new Error('splat header has no intrinsics')
  }

  const specs = new Map<string, { offset: number; count: number }>()
  for (const spec of header.arrays) {
    specs.set(spec.name, { offset: spec.byte_offset, count: spec.count })
  }
  for (const name of SPLAT_ARRAY_NAMES) {
    if (!specs.has(name)) {
      throw new Error(`splat payload is missing the "${name}" array`)
    }
  }

  const dataStart = headerEnd
  const payload = buffer.slice(dataStart)

  const floatArray = (name: SplatArrayName): Float32Array => {
    const spec = specs.get(name)!
    return new Float32Array(payload, spec.offset, spec.count)
  }
  const uintArray = (name: SplatArrayName): Uint32Array => {
    const spec = specs.get(name)!
    return new Uint32Array(payload, spec.offset, spec.count)
  }

  const count = header.count
  const arrays = {
    positions: floatArray('positions'),
    shDc: floatArray('sh_dc'),
    opacityLogits: floatArray('opacity_logits'),
    logScales: floatArray('log_scales'),
    rotations: floatArray('rotations'),
    reconstructionWeights: floatArray('reconstruction_weights'),
    sourcePixels: uintArray('source_pixels'),
  }

  if (
    count <= 0 ||
    arrays.positions.length !== count * 3 ||
    arrays.shDc.length !== count * 3 ||
    arrays.opacityLogits.length !== count ||
    arrays.logScales.length !== count * 3 ||
    arrays.rotations.length !== count * 4 ||
    arrays.reconstructionWeights.length !== count ||
    arrays.sourcePixels.length !== count * 2
  ) {
    throw new Error('splat array sizes do not match the header count')
  }

  return { header, ...arrays }
}
