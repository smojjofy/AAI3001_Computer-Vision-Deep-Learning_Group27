export type DepthKind =
  | 'relative_depth'
  | 'metric_depth'

export interface PoseLandmarkData {
  x: number
  y: number
  z: number
  visibility?: number | null
  presence?: number | null
}

export interface PoseInferenceData {
  detected: boolean
  landmarks: PoseLandmarkData[]
  processingMs?: number
}

export interface DepthInferenceData {
  image?: CanvasImageSource
  kind?: DepthKind
  processingMs?: number
}

export interface SegmentationInferenceData {
  image?: CanvasImageSource
  processingMs?: number
}

export interface FlowInferenceData {
  image?: CanvasImageSource
  processingMs?: number
}

export interface InferenceFrameData {
  frameId: number | string
  sourceId?: string
  captureTimestampNs?: string

  width: number
  height: number

  pose?: PoseInferenceData
  depth?: DepthInferenceData
  segmentation?: SegmentationInferenceData
  flow?: FlowInferenceData
}