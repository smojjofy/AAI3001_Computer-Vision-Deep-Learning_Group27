import type {
  PipelineFrame,
} from '../types/PipelineFrame'

import {
  createMockOriginalFrame,
} from './mockOriginalFrame'

import {
  createMockProcessorData,
} from './mockProcessorData'

import {
  createMockInferenceData,
} from './mockInferenceData'

import {
  createMockSplatData,
} from './mockSplatData'

export function createMockPipelineFrame(
  frameId = 1,
  sourceId = 'mock-camera',
  captureTimestampNs = '0'
): PipelineFrame {
  const original =
    createMockOriginalFrame(frameId)

  const processor =
    createMockProcessorData(
      frameId,
      sourceId,
      captureTimestampNs
    )

  const inference =
    createMockInferenceData(
      frameId,
      sourceId,
      captureTimestampNs
    )

  const splat =
    createMockSplatData(500)

  return {
    frameId,
    sourceId,
    captureTimestampNs,

    original: {
      image: original.image,
      width: original.width,
      height: original.height,
    },

    processor,
    inference,
    splat,
  }
}