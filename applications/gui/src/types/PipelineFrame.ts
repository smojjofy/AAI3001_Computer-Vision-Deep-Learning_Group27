import type {
  ProcessorFrameData,
} from './ProcessorFrame'

import type {
  SplatPointData,
} from '../data/mockSplatData'

export interface PipelineFrame {
  frameId: number | string
  timestamp: string

  original: {
    image: CanvasImageSource
    width: number
    height: number
  }

  processor: ProcessorFrameData

  splat: SplatPointData
}