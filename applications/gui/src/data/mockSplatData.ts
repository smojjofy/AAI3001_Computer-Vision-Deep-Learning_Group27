export interface SplatPointData {
  positions: Float32Array
  colors?: Float32Array
}

export function createMockSplatData(
  pointCount = 500
): SplatPointData {
  const positions = new Float32Array(pointCount * 3)
  const colors = new Float32Array(pointCount * 3)

  for (let i = 0; i < pointCount; i++) {
    const ratio = i / pointCount

    const angle = ratio * Math.PI * 20
    const radius = 1.5 * ratio

    const index = i * 3

    positions[index] =
      Math.cos(angle) * radius

    positions[index + 1] =
      (ratio - 0.5) * 3

    positions[index + 2] =
      Math.sin(angle) * radius

    /*
     * Temporary colours purely for development.
     */
    colors[index] = ratio
    colors[index + 1] = 1 - ratio
    colors[index + 2] = 0.8
  }

  return {
    positions,
    colors,
  }
}