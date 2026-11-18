/*
 * Payload types for the live local pipeline API.
 *
 * The values mirror applications/backend/pipeline_server.py and
 * applications/live_pipeline/*. Keep them in sync with those publishers.
 */

export interface PreviewFrame {
  schema_version: string
  frame_id: number
  source_id: string
  capture_timestamp_ns: string
  timestamp_origin: string
  width: number
  height: number
  jpeg_base64: string
  stream_health: string
}

export interface PoseJoint {
  index: number
  name: string
  x: number | null
  y: number | null
  visibility: number | null
  depth_m: number | null
}

export interface PoseDepthFrame {
  schema_version: string
  frame_id: number
  /** Capture run that produced this image. */
  generation?: number
  /** Splat state this image was published with. */
  splat_frame_id?: number
  source_id: string
  capture_timestamp_ns: string
  width: number
  height: number
  depth_kind: string
  depth_source: string
  near_m: number | null
  far_m: number | null
  valid_fraction: number | null
  person_detected: boolean
  body_depth_m: number | null
  joints: PoseJoint[]
  timings_ms: Record<string, number | null>
  depth_jpeg_base64: string
  skeleton_jpeg_base64: string
}

export interface MotionPayload {
  schema_version: string
  frame_id: number
  reference_frame_id: number | null
  /** Capture run that produced the flow. Must equal the displayed splat run. */
  source_id?: string
  generation?: number
  /** Frame id of the active splat state the motion may move. */
  splat_frame_id?: number | null
  width: number
  height: number
  source_width: number
  source_height: number
  max_offset_ndc: number
  /** Flat row-major [height, width, 2] clip-space offsets, normalized image units. */
  flow_ndc: number[]
  /** Flat row-major [height, width] validity flags (0 or 1). */
  valid: number[]
  joints: PoseJoint[]
}

export interface SplatMask {
  source?: string
  fallback_reason?: string | null
  background_area_fraction?: number | null
  near_m: number | null
  far_m: number | null
  area_fraction: number
  valid_fraction?: number
  confidence: number
  detected?: boolean
}

export interface SplatArraySpec {
  name: string
  dtype: string
  count: number
  byte_offset: number
  byte_length: number
}

export interface SplatHeader {
  schema_version: string
  frame_id: string
  /** Integer capture frame id. Use this for identity comparisons. */
  frame_id_int?: number
  source_id: string
  timestamp_ns: number
  count: number
  /** [fx, fy, cx, cy] in pixels for the capture image size. */
  intrinsics: [number, number, number, number]
  depth_semantics?: number
  depth_units?: number
  camera_to_world: number[]
  arrays: SplatArraySpec[]
  coordinate_system?: string
  rotation_order: string
  scale_encoding: string
  opacity_encoding: string
  byte_length: number
  timings_ms?: Record<string, number>
  /** Age from capture to server publish, milliseconds. Absent for recorded sources. */
  capture_age_ms?: number | null
  pose_confidence?: number
  person_detected?: boolean
  mask?: SplatMask
  depth_source?: string
  depth_units_metres?: boolean
  /** [width, height] of the capture image. The header supplies them directly. */
  image_width?: number
  image_height?: number
  state_age_ms: number
  /** 0 while the person is lost (the viewer owns the fade clock), else 1. */
  display_alpha: number
  person_lost?: boolean
  generation?: number
  active_identity?: ActiveStateIdentity | null
}

export interface ActiveStateIdentity {
  source_id: string
  generation: number
  frame_id: number
  reference_frame_id: number | null
}

/** One decoded mask image with the identity of the state that produced it. */
export interface MaskFrame {
  bitmap: ImageBitmap
  identity: ActiveStateIdentity | null
}

export interface SplatFrame {
  header: SplatHeader
  /** Constructor convention positions: metres, +X right, +Y up, forward -Z. */
  positions: Float32Array
  /** Spherical-harmonics DC term, 3 values per splat. */
  shDc: Float32Array
  /** Opacity logits, 1 value per splat. */
  opacityLogits: Float32Array
  /** Natural log of the linear scale in metres, 3 values per splat. */
  logScales: Float32Array
  /** Unit quaternions in wxyz order, 4 values per splat. */
  rotations: Float32Array
  /** Heuristic reconstruction weight in [0, 1], 1 value per splat. */
  reconstructionWeights: Float32Array
  /** Source pixel [u, v] pairs, 2 values per splat. */
  sourcePixels: Uint32Array
}

export interface DiagnosticsSplat {
  frame_id: string | null
  count: number | null
  timings_ms: Record<string, number> | null
  capture_age_ms?: number | null
  mask: SplatMask | null
  depth_source: string | null
  pose_confidence: number | null
  person_detected: boolean | null
}

export interface RefreshWorkerSnapshot {
  submitted_total?: number
  started_total?: number
  completed_total?: number
  replaced_total?: number
  discarded_total?: number
  active?: boolean
  pending_frame_id?: number | null
  last_error?: string | null
}

export interface CalibrationStatus {
  active: boolean
  status: string
  frames: number
  seconds: number
  baseline: {
    source_id: string
    generation: number
    width: number
    height: number
    frames_used: number
    valid_fraction: number
    median_depth_m: number
    p05_depth_m: number
    p95_depth_m: number
  } | null
  last_error: string | null
  events: number
}

export interface DiagnosticsPayload {
  schema_version: string
  action: string | null
  reason: string | null
  selected_frame_id: number | null
  flow_score: number | null
  flow_threshold: number | null
  candidates: number | null
  cache_age_ms: number | null
  active_state_frame_id: number | null
  refresh_count: number
  refresh_failures: number
  last_error: string | null
  /** Milliseconds since the first invalid-person event, or null. Latched. */
  person_loss_age_ms?: number | null
  person_lost?: boolean
  active_identity?: ActiveStateIdentity | null
  motion_active?: boolean
  calibration?: CalibrationStatus | null
  splat: DiagnosticsSplat | null
  counters: Record<string, number>
  refresh_worker: RefreshWorkerSnapshot | null
  stream_health: string
  queue: {
    depth: number
    dropped_total: number
    skipped_total: number
  }
}
