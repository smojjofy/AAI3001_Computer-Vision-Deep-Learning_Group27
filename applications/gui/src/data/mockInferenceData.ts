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


    flow: {
      image:
        createFlowOutput(
          width,
          height
        ),
      processingMs: 17.2,
    },

    poseDepth: {
      detected: true,

      landmarks: [
        { x: 0.50, y: 0.12, z: 0, visibility: 0.99, presence: 0.99 }, // 0 nose
        { x: 0.485, y: 0.105, z: 0, visibility: 0.99, presence: 0.99 }, // 1
        { x: 0.475, y: 0.105, z: 0, visibility: 0.99, presence: 0.99 }, // 2
        { x: 0.465, y: 0.11, z: 0, visibility: 0.99, presence: 0.99 }, // 3
        { x: 0.515, y: 0.105, z: 0, visibility: 0.99, presence: 0.99 }, // 4
        { x: 0.525, y: 0.105, z: 0, visibility: 0.99, presence: 0.99 }, // 5
        { x: 0.535, y: 0.11, z: 0, visibility: 0.99, presence: 0.99 }, // 6
        { x: 0.445, y: 0.13, z: 0, visibility: 0.98, presence: 0.99 }, // 7
        { x: 0.555, y: 0.13, z: 0, visibility: 0.98, presence: 0.99 }, // 8
        { x: 0.485, y: 0.145, z: 0, visibility: 0.99, presence: 0.99 }, // 9
        { x: 0.515, y: 0.145, z: 0, visibility: 0.99, presence: 0.99 }, // 10

        { x: 0.42, y: 0.28, z: 0, visibility: 0.99, presence: 0.99 }, // 11 L shoulder
        { x: 0.58, y: 0.28, z: 0, visibility: 0.99, presence: 0.99 }, // 12 R shoulder

        { x: 0.36, y: 0.43, z: 0, visibility: 0.98, presence: 0.99 }, // 13 L elbow
        { x: 0.64, y: 0.43, z: 0, visibility: 0.98, presence: 0.99 }, // 14 R elbow

        { x: 0.31, y: 0.57, z: 0, visibility: 0.97, presence: 0.98 }, // 15 L wrist
        { x: 0.69, y: 0.57, z: 0, visibility: 0.97, presence: 0.98 }, // 16 R wrist

        { x: 0.285, y: 0.595, z: 0, visibility: 0.95, presence: 0.97 }, // 17
        { x: 0.715, y: 0.595, z: 0, visibility: 0.95, presence: 0.97 }, // 18
        { x: 0.275, y: 0.575, z: 0, visibility: 0.95, presence: 0.97 }, // 19
        { x: 0.725, y: 0.575, z: 0, visibility: 0.95, presence: 0.97 }, // 20
        { x: 0.295, y: 0.555, z: 0, visibility: 0.95, presence: 0.97 }, // 21
        { x: 0.705, y: 0.555, z: 0, visibility: 0.95, presence: 0.97 }, // 22

        { x: 0.45, y: 0.55, z: 0, visibility: 0.99, presence: 0.99 }, // 23 L hip
        { x: 0.55, y: 0.55, z: 0, visibility: 0.99, presence: 0.99 }, // 24 R hip

        { x: 0.44, y: 0.72, z: 0, visibility: 0.98, presence: 0.99 }, // 25 L knee
        { x: 0.56, y: 0.72, z: 0, visibility: 0.98, presence: 0.99 }, // 26 R knee

        { x: 0.43, y: 0.88, z: 0, visibility: 0.97, presence: 0.98 }, // 27 L ankle
        { x: 0.57, y: 0.88, z: 0, visibility: 0.97, presence: 0.98 }, // 28 R ankle

        { x: 0.42, y: 0.91, z: 0, visibility: 0.96, presence: 0.98 }, // 29
        { x: 0.58, y: 0.91, z: 0, visibility: 0.96, presence: 0.98 }, // 30

        { x: 0.455, y: 0.94, z: 0, visibility: 0.96, presence: 0.98 }, // 31
        { x: 0.545, y: 0.94, z: 0, visibility: 0.96, presence: 0.98 }, // 32
      ],

      jointDepth: [
        1.82, 1.82, 1.82, 1.82, 1.82, 1.82, 1.82,
        1.83, 1.83, 1.83, 1.83,
        1.88, 1.88,
        1.90, 1.90,
        1.92, 1.92,
        1.93, 1.93, 1.93, 1.93, 1.93, 1.93,
        1.91, 1.91,
        1.96, 1.96,
        2.01, 2.01,
        2.03, 2.03,
        2.04, 2.04,
      ],

      bodyDepth: 1.90,

      depthImage:
        createDepthOutput(
          width,
          height
        ),

      depthKind: 'metric_depth',
      depthSource: 'yolo',
      depthModelName: 'mock-yolo-depth',

      timingsMs: {
        pose: 18.8,
        depth: 28.6,
        fuse: 1.2,
        total: 48.6,
      },
    },
  }
}