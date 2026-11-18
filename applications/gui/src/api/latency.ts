/* Capture-to-display latency estimate for the viewer.
 *
 * The footer value describes what the operator sees, so it is published only
 * after the renderer has drawn the splat state it belongs to. These helpers are
 * pure so the arithmetic and the labelling rule are checked without a browser.
 *
 * Clock limits: capture age and server state age use the process monotonic
 * clock on the server. The browser has a separate clock, so browser phases are
 * measured inside the browser. The transport time from the browser request to
 * the decoded response cannot be split into network and decode time with these
 * clocks, so it is reported as one phase and the total is labelled an estimate.
 */

export interface LatencyPhases {
  /** Capture timestamp to server publish (server monotonic clock). */
  captureAgeMs: number
  /** Server publish to server response (server monotonic clock). */
  stateAgeMs: number
  /** Browser request start time (browser clock). */
  requestStartedAt: number
  /** Browser time when the decoded response arrived (browser clock). */
  responseAt: number
}

/** Browser phases measured when the render completes. */
export interface BrowserPhases {
  /** Request start to decoded response: network transfer plus decode. */
  transferMs: number
  /** Decoded response to the render that first shows the state. */
  renderWaitMs: number
}

export const LATENCY_LABEL_PREFIX = 'Display (est)'

/** Browser phases for a render time; null when a timestamp is missing. */
export function browserPhases(phases: LatencyPhases, renderAt: number): BrowserPhases | null {
  if (!Number.isFinite(phases.requestStartedAt) || !Number.isFinite(phases.responseAt)) {
    return null
  }
  if (!Number.isFinite(renderAt) || renderAt < phases.responseAt || phases.responseAt < phases.requestStartedAt) {
    return null
  }
  return {
    transferMs: phases.responseAt - phases.requestStartedAt,
    renderWaitMs: renderAt - phases.responseAt,
  }
}

/** End-to-end capture-to-display estimate in milliseconds, or null when a value is invalid. */
export function captureToDisplayMs(phases: LatencyPhases, renderAt: number): number | null {
  const browser = browserPhases(phases, renderAt)
  if (browser === null) {
    return null
  }
  const values = [
    phases.captureAgeMs,
    phases.stateAgeMs,
    browser.transferMs,
    browser.renderWaitMs,
  ]
  if (!values.every((value) => Number.isFinite(value) && value >= 0)) {
    return null
  }
  return values[0] + values[1] + values[2] + values[3]
}

/** Footer text for the estimate; never shows a value when a phase is missing. */
export function formatLatencyLabel(valueMs: number | null): string {
  if (valueMs === null || !Number.isFinite(valueMs)) {
    return `${LATENCY_LABEL_PREFIX}: --`
  }
  return `${LATENCY_LABEL_PREFIX}: ${Math.round(valueMs)} ms`
}
