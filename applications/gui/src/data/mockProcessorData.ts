import type {
  ProcessorFrameData,
} from '../types/ProcessorFrame'

function createCanvas(
  width: number,
  height: number
): {
  canvas: HTMLCanvasElement
  context: CanvasRenderingContext2D
} {
  const canvas =
    document.createElement('canvas')

  canvas.width = width
  canvas.height = height

  const context =
    canvas.getContext('2d')

  if (!context) {
    throw new Error(
      'Unable to create mock processor canvas'
    )
  }

  return {
    canvas,
    context,
  }
}

function createRgbaOutput(
  width: number,
  height: number
): HTMLCanvasElement {
  const { canvas, context } =
    createCanvas(width, height)

  const gradient =
    context.createLinearGradient(
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
   * Same simple objects as our
   * mock original frame.
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

  context.fillStyle =
    'rgba(0, 0, 0, 0.55)'

  context.fillRect(
    18,
    18,
    175,
    50
  )

  context.fillStyle = '#ffffff'

  context.font =
    '600 20px system-ui'

  context.fillText(
    'Processed Frame',
    32,
    50
  )

  return canvas
}

export function createMockProcessorData(
  frameId = 1,
  sourceId = 'mock-camera',
  captureTimestampNs = '0',
): ProcessorFrameData {
  const width = 640
  const height = 360

  return {
    frameId,
    sourceId,
    captureTimestampNs,
    processingMs: 24.7,

    width,
    height,

    image:
      createRgbaOutput(
        width,
        height
      ),
  }
}