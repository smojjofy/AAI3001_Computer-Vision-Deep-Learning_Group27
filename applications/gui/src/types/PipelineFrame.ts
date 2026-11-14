import type {
  ProcessorFrameData,
} from './ProcessorFrame'

import type {
  InferenceFrameData,
} from './InferenceFrame'

import type {
  SplatPointData,
} from '../data/mockSplatData'

export interface PipelineFrame {
  frameId: number | string
  sourceId?: string
  captureTimestampNs?: string

  original: {
    image: CanvasImageSource
    width: number
    height: number
  }

  processor: ProcessorFrameData

  inference: InferenceFrameData

  splat: SplatPointData
}