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
    'RGBA Output',
    32,
    50
  )

  return canvas
}

function createMaskOutput(
  width: number,
  height: number
): HTMLCanvasElement {
  const { canvas, context } =
    createCanvas(width, height)

  /*
   * Background
   */
  context.fillStyle = '#000000'

  context.fillRect(
    0,
    0,
    width,
    height
  )

  /*
   * Foreground mask
   */
  context.fillStyle = '#ffffff'

  context.fillRect(
    90,
    120,
    120,
    100
  )

  context.beginPath()

  context.arc(
    340,
    170,
    60,
    0,
    Math.PI * 2
  )

  context.fill()

  context.fillRect(
    460,
    90,
    90,
    170
  )

  context.fillStyle =
    'rgba(30, 30, 30, 0.9)'

  context.fillRect(
    18,
    18,
    225,
    50
  )

  context.fillStyle = '#ffffff'

  context.font =
    '600 20px system-ui'

  context.fillText(
    'Foreground Mask',
    32,
    50
  )

  return canvas
}

function createDepthOutput(
  width: number,
  height: number
): HTMLCanvasElement {
  const { canvas, context } =
    createCanvas(width, height)

  /*
   * Development-only relative
   * depth visualization.
   */
  const gradient =
    context.createLinearGradient(
      0,
      0,
      width,
      0
    )

  gradient.addColorStop(
    0,
    '#ffffff'
  )

  gradient.addColorStop(
    1,
    '#111111'
  )

  context.fillStyle = gradient

  context.fillRect(
    0,
    0,
    width,
    height
  )

  /*
   * Add different grayscale values
   * to our three mock objects.
   */
  context.fillStyle = '#eeeeee'

  context.fillRect(
    90,
    120,
    120,
    100
  )

  context.fillStyle = '#999999'

  context.beginPath()

  context.arc(
    340,
    170,
    60,
    0,
    Math.PI * 2
  )

  context.fill()

  context.fillStyle = '#444444'

  context.fillRect(
    460,
    90,
    90,
    170
  )

  context.fillStyle =
    'rgba(0, 0, 0, 0.65)'

  context.fillRect(
    18,
    18,
    190,
    50
  )

  context.fillStyle = '#ffffff'

  context.font =
    '600 20px system-ui'

  context.fillText(
    'Relative Depth',
    32,
    50
  )

  context.font =
    '14px system-ui'

  context.fillText(
    'NEAR',
    25,
    height - 25
  )

  context.fillStyle = '#ffffff'

  context.fillText(
    'FAR',
    width - 55,
    height - 25
  )

  return canvas
}

function createMotionOutput(
  width: number,
  height: number
): HTMLCanvasElement {
  const { canvas, context } =
    createCanvas(width, height)

  context.fillStyle = '#081018'

  context.fillRect(
    0,
    0,
    width,
    height
  )

  /*
   * Draw a simple motion-vector field.
   */
  context.strokeStyle = '#22d3ee'
  context.fillStyle = '#22d3ee'
  context.lineWidth = 2

  const spacing = 55

  for (
    let y = 70;
    y < height - 30;
    y += spacing
  ) {
    for (
      let x = 40;
      x < width - 40;
      x += spacing
    ) {
      const dx =
        18 + (y / height) * 18

      const dy =
        Math.sin(x * 0.02) * 10

      const endX = x + dx
      const endY = y + dy

      context.beginPath()

      context.moveTo(
        x,
        y
      )

      context.lineTo(
        endX,
        endY
      )

      context.stroke()

      /*
       * Arrow head.
       */
      const angle =
        Math.atan2(
          endY - y,
          endX - x
        )

      context.beginPath()

      context.moveTo(
        endX,
        endY
      )

      context.lineTo(
        endX -
          7 *
            Math.cos(
              angle - Math.PI / 6
            ),
        endY -
          7 *
            Math.sin(
              angle - Math.PI / 6
            )
      )

      context.lineTo(
        endX -
          7 *
            Math.cos(
              angle + Math.PI / 6
            ),
        endY -
          7 *
            Math.sin(
              angle + Math.PI / 6
            )
      )

      context.closePath()
      context.fill()
    }
  }

  context.fillStyle =
    'rgba(0, 0, 0, 0.65)'

  context.fillRect(
    18,
    18,
    245,
    50
  )

  context.fillStyle = '#ffffff'

  context.font =
    '600 20px system-ui'

  context.fillText(
    'Motion / Optical Flow',
    32,
    50
  )

  return canvas
}

export function createMockProcessorData(
  frameId = 1
): ProcessorFrameData {
  const width = 640
  const height = 360

  return {
    frameId,
    processingMs: 24.7,

    width,
    height,

    outputs: {
      rgba:
        createRgbaOutput(
          width,
          height
        ),

      mask:
        createMaskOutput(
          width,
          height
        ),

      depth:
        createDepthOutput(
          width,
          height
        ),

      motion:
        createMotionOutput(
          width,
          height
        ),
    },
  }
}