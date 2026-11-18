/* Capture-to-display latency estimate for the viewer.
 *
 * The footer value describes what the operator sees, so it is published only
 * after the renderer has drawn the splat state it belongs to. These helpers are
 * pure so the arithmetic and the labelling rule are checked without a browser.
 *
 * Intervals are disjoint so no part of the timeline is counted twice:
 *   capture age   : capture timestamp -> server publish   (server clock)
 *   state age     : server publish     -> server response  (server clock)
 *   decode wait   : decoded response   -> applied render   (browser clock)
 *   render wait   : decoded response   -> first render     (browser clock)
 *
 * The server sends the response before the browser receives it. That transport
 * transit and the browser-to-server clock offset cannot be measured with the
 * available clocks, so the value is an estimate and is labelled as one.
 */

export interface LatencyPhases {
  /** Capture timestamp to server publish (server monotonic clock). */
  captureAgeMs: number
  /** Server publish to server response (server monotonic clock). */
  stateAgeMs: number
  /** Browser time when the decoded splat response arrived. */
  decodedAt: number
}

/** Browser render phases measured at publication. */
export interface BrowserPhases {
  /** Decoded response to the render that first shows the state. */
  renderWaitMs: number
}

export const LATENCY_LABEL_PREFIX = 'Display (est)'

/** Browser render wait for a render time; null when a timestamp is invalid. */
export function browserPhases(phases: LatencyPhases, renderAt: number): BrowserPhases | null {
  if (!Number.isFinite(phases.decodedAt) || !Number.isFinite(renderAt)) {
    return null
  }
  if (renderAt < phases.decodedAt) {
    return null
  }
  return { renderWaitMs: renderAt - phases.decodedAt }
}

/**
 * End-to-end capture-to-display estimate in milliseconds.
 *
 * The browser request interval is deliberately not added: a request that starts
 * between publish and server response would overlap `stateAgeMs` and count part
 * of the timeline twice. Only the decoded-response-to-render interval comes
 * from the browser clock.
 */
export function captureToDisplayMs(phases: LatencyPhases, renderAt: number): number | null {
  const browser = browserPhases(phases, renderAt)
  if (browser === null) {
    return null
  }
  const values = [phases.captureAgeMs, phases.stateAgeMs, browser.renderWaitMs]
  if (!values.every((value) => Number.isFinite(value) && value >= 0)) {
    return null
  }
  return values[0] + values[1] + values[2]
}

/** Footer text for the estimate; never shows a value when a phase is missing. */
export function formatLatencyLabel(valueMs: number | null): string {
  if (valueMs === null || !Number.isFinite(valueMs)) {
    return `${LATENCY_LABEL_PREFIX}: --`
  }
  return `${LATENCY_LABEL_PREFIX}: ${Math.round(valueMs)} ms`
}
