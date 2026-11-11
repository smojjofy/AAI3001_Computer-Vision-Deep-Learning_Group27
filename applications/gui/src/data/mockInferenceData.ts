import type {
  InferenceFrameData,
} from '../types/InferenceFrame'

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
      'Unable to create mock inference canvas'
    )
  }

  return {
    canvas,
    context,
  }
}

function createSegmentationOutput(
  width: number,
  height: number
): HTMLCanvasElement {
  const { canvas, context } =
    createCanvas(width, height)

  context.fillStyle = '#000000'

  context.fillRect(
    0,
    0,
    width,
    height
  )

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
    'Depth Output',
    32,
    50
  )

  return canvas
}

function createFlowOutput(
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

export function createMockInferenceData(
  frameId = 1,
  sourceId = 'mock-camera',
  captureTimestampNs = '0',
): InferenceFrameData {
  const width = 640
  const height = 360

  return {
    frameId,
    sourceId,
    captureTimestampNs,
    width,
    height,

    segmentation: {
      image:
        createSegmentationOutput(
          width,
          height
        ),
      processingMs: 12.4,
    },

    depth: {
      image:
        createDepthOutput(
          width,
          height
        ),
      kind: 'relative_depth',
      processingMs: 28.6,
    },

    flow: {
      image:
        createFlowOutput(
          width,
          height
        ),
      processingMs: 17.2,
    },

    pose: {
      detected: true,
      processingMs: 18.8,

      landmarks: [
        {
          x: 0.50,
          y: 0.18,
          z: 0,
          visibility: 0.99,
          presence: 0.99,
        },
        {
          x: 0.44,
          y: 0.32,
          z: 0,
          visibility: 0.98,
          presence: 0.99,
        },
        {
          x: 0.56,
          y: 0.32,
          z: 0,
          visibility: 0.98,
          presence: 0.99,
        },
        {
          x: 0.40,
          y: 0.52,
          z: 0,
          visibility: 0.95,
          presence: 0.98,
        },
        {
          x: 0.60,
          y: 0.52,
          z: 0,
          visibility: 0.95,
          presence: 0.98,
        },
        {
          x: 0.45,
          y: 0.65,
          z: 0,
          visibility: 0.96,
          presence: 0.98,
        },
        {
          x: 0.55,
          y: 0.65,
          z: 0,
          visibility: 0.96,
          presence: 0.98,
        },
        {
          x: 0.44,
          y: 0.88,
          z: 0,
          visibility: 0.94,
          presence: 0.97,
        },
        {
          x: 0.56,
          y: 0.88,
          z: 0,
          visibility: 0.94,
          presence: 0.97,
        },
      ],
    },
  }
}