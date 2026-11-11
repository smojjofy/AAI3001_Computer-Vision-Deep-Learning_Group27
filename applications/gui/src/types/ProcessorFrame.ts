export type ProcessorView =
  | 'rgba'
  | 'mask'
  | 'depth'
  | 'motion'

export interface ProcessorFrameData {
  frameId: number | string
  processingMs?: number

  width: number
  height: number

  outputs: Partial<
    Record<ProcessorView, CanvasImageSource>
  >
}