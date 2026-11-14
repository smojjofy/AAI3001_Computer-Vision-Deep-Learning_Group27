export type DepthKind =
  | 'relative_depth'
  | 'metric_depth'

export type DepthSource =
  | 'zed'
  | 'yolo'

export interface PoseLandmarkData {
  x: number
  y: number
  z: number
  visibility?: number | null
  presence?: number | null
}

export interface InferenceTimings {
  pose?: number
  depth?: number
  fuse?: number
  total?: number
}

export interface PoseDepthInferenceData {
  detected: boolean

  landmarks: PoseLandmarkData[]

  
  jointDepth: Array<number | null>

  
  bodyDepth?: number | null

  depthImage?: CanvasImageSource

  depthKind?: DepthKind
  depthSource?: DepthSource
  depthModelName?: string

  timingsMs?: InferenceTimings
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

  poseDepth?: PoseDepthInferenceData

  segmentation?: SegmentationInferenceData
  flow?: FlowInferenceData
}