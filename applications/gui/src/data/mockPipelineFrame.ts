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
  createMockSplatData,
} from './mockSplatData'

export function createMockPipelineFrame(
  frameId = 1
): PipelineFrame {
  const timestamp =
    new Date().toISOString()

  const original =
    createMockOriginalFrame(frameId)

  const processor =
    createMockProcessorData(frameId)

  const splat =
    createMockSplatData(500)

  return {
    frameId,
    timestamp,

    original: {
      image: original.image,
      width: original.width,
      height: original.height,
    },

    processor,

    splat,
  }
}