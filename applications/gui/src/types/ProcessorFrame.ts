export interface ProcessorFrameData {
  frameId: number | string
  sourceId?: string
  captureTimestampNs?: string

  processingMs?: number

  width: number
  height: number

  image?: CanvasImageSource
}