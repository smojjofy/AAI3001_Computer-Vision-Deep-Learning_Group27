/*
 * Pure decisions for the viewer's display epoch.
 *
 * The viewer clears its displayed run state when diagnostics report no active
 * state. Responses and whole polls that started before a clear must not restore
 * the old run. These decisions are pure so a delayed-response scenario can be
 * checked without a browser.
 */

/** True when a clear must run, including when only a request is in flight. */
export function shouldClearDisplayedState(
  hasDisplayedOrPending: boolean,
  inFlightCount: number,
): boolean {
  return hasDisplayedOrPending || inFlightCount > 0
}

/** True when a response that started at `startEpoch` still belongs to the view. */
export function shouldApplyResponse(startEpoch: number, currentEpoch: number): boolean {
  return startEpoch === currentEpoch
}

/**
 * True when the motion and pose-depth parts of a diagnostics poll may apply.
 *
 * With no active state in that poll, the auxiliary results can only belong to
 * the old run, so they are skipped and the next poll is scheduled.
 */
export function shouldApplyAuxiliary(hasActiveState: boolean): boolean {
  return hasActiveState
}
