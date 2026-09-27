export interface MockOriginalFrame {
  image: HTMLCanvasElement
  width: number
  height: number
  frameId: number
  timestamp: string
}

export function createMockOriginalFrame(
  frameId = 1
): MockOriginalFrame {
  const width = 640
  const height = 360

  const canvas = document.createElement('canvas')

  canvas.width = width
  canvas.height = height

  const context = canvas.getContext('2d')

  if (!context) {
    throw new Error(
      'Unable to create mock frame canvas context'
    )
  }

  /*
   * Background
   */
  const gradient = context.createLinearGradient(
    0,
    0,
    width,
    height
  )

  gradient.addColorStop(
    0,
    '#172554'
  )

  gradient.addColorStop(
    1,
    '#0f766e'
  )

  context.fillStyle = gradient

  context.fillRect(
    0,
    0,
    width,
    height
  )

  /*
   * Test objects
   */
  context.fillStyle = '#f59e0b'

  context.fillRect(
    90,
    120,
    120,
    100
  )

  context.fillStyle = '#ef4444'

  context.beginPath()

  context.arc(
    340,
    170,
    60,
    0,
    Math.PI * 2
  )

  context.fill()

  context.fillStyle = '#22c55e'

  context.fillRect(
    460,
    90,
    90,
    170
  )

  /*
   * Frame information
   */
  context.fillStyle = 'rgba(0, 0, 0, 0.55)'

  context.fillRect(
    18,
    18,
    245,
    72
  )

  context.fillStyle = '#ffffff'

  context.font =
    '600 20px system-ui'

  context.fillText(
    `Mock Camera Frame #${frameId}`,
    32,
    49
  )

  context.font =
    '14px system-ui'

  context.fillStyle =
    '#d1d5db'

  context.fillText(
    '640 × 360 development fixture',
    32,
    73
  )

  return {
    image: canvas,
    width,
    height,
    frameId,
    timestamp:
      new Date().toISOString(),
  }
}