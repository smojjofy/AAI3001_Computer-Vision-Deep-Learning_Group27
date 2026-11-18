/* Capture-to-display latency estimate for the viewer.
 *
 * The footer value must describe what the operator sees, so it is published only
 * after the renderer has applied that splat state. These helpers are pure so the
 * arithmetic and the labelling rule can be checked without a browser.
 *
 * Clock limits: the capture timestamp and the server age share the process
 * monotonic clock. The browser clock is separate, so browser-side phases are
 * measured inside the browser. The estimate therefore covers, end to end,
 * capture age at publish plus server state age plus the browser phases below,
 * and it is labelled as an estimate.
 */

export interface LatencyPhases {
  /** Capture timestamp to server publish (shared monotonic clock). */
  captureAgeMs: number
  /** Splat header state age, publish to response. */
  stateAgeMs: number
  /** Response received to the renderer applying the state. */
  browserMs: number
}

/** The footer label prefix. The value is always an estimate, never a measurement. */
export const LATENCY_LABEL_PREFIX = 'Display (est)'

/**
 * End-to-end capture-to-display estimate in milliseconds, or null when a phase
 * is missing or not finite.
 */
export function captureToDisplayMs(phases: LatencyPhases | null): number | null {
  if (phases === null) {
    return null
  }
  const values = [phases.captureAgeMs, phases.stateAgeMs, phases.browserMs]
  if (!values.every((value) => Number.isFinite(value) && value >= 0)) {
    return null
  }
  return values[0] + values[1] + values[2]
}

/** Footer text for the estimate; never shows a value for a missing phase. */
export function formatLatencyLabel(valueMs: number | null): string {
  if (valueMs === null || !Number.isFinite(valueMs)) {
    return `${LATENCY_LABEL_PREFIX}: --`
  }
  return `${LATENCY_LABEL_PREFIX}: ${Math.round(valueMs)} ms`
}
