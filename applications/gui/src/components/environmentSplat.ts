import type { SplatFrame } from '../api/types'

/** Standard .splat rows: position, scale, RGBA, then wxyz quaternion bytes. */
export function encodeEnvironmentSplat(frame: SplatFrame): ArrayBuffer {
  const count = frame.header.count
  if (!Number.isSafeInteger(count) || count < 0 || count > 1_000_000 ||
      frame.positions.length !== count * 3 || frame.logScales.length !== count * 3 ||
      frame.shDc.length !== count * 3 || frame.opacityLogits.length !== count ||
      frame.rotations.length !== count * 4) {
    throw new Error('The splat frame arrays do not match the splat count')
  }

  const buffer = new ArrayBuffer(count * 32)
  const view = new DataView(buffer)
  const shC0 = 0.28209479177387814
  const displayAlpha = Math.min(1, Math.max(0, frame.header.display_alpha ?? 1))
  const byte = (value: number): number => Math.max(0, Math.min(255, Math.round(value * 255)))

  for (let i = 0; i < count; i++) {
    const row = i * 32
    for (let axis = 0; axis < 3; axis++) {
      const position = frame.positions[i * 3 + axis]
      const scale = Math.exp(frame.logScales[i * 3 + axis])
      const color = frame.shDc[i * 3 + axis]
      if (!Number.isFinite(position) || !Number.isFinite(scale) || scale <= 0 || !Number.isFinite(color)) {
        throw new Error(`Splat ${i} has an invalid position, scale, or color`)
      }
      view.setFloat32(row + axis * 4, position, true)
      view.setFloat32(row + 12 + axis * 4, scale, true)
      view.setUint8(row + 24 + axis, byte(0.5 + color * shC0))
    }
    const opacity = frame.opacityLogits[i]
    if (!Number.isFinite(opacity)) {
      throw new Error(`Splat ${i} has an invalid opacity`)
    }
    view.setUint8(row + 27, byte((1 / (1 + Math.exp(-opacity))) * displayAlpha))
    let norm = 0
    for (let axis = 0; axis < 4; axis++) {
      const value = frame.rotations[i * 4 + axis]
      if (!Number.isFinite(value)) throw new Error(`Splat ${i} has an invalid rotation`)
      norm += value * value
    }
    if (norm < 1e-12) throw new Error(`Splat ${i} has an invalid rotation`)
    norm = Math.sqrt(norm)
    for (let axis = 0; axis < 4; axis++) {
      const value = frame.rotations[i * 4 + axis] / norm
      view.setUint8(row + 28 + axis, Math.max(0, Math.min(255, Math.round(value * 128 + 128))))
    }
  }
  return buffer
}
