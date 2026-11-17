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

export interface ActiveIdentityLike {
  source_id: string
  generation: number
  frame_id: number | string
}

export interface AuxiliaryIdentityLike {
  source_id?: string
  generation?: number
  splat_frame_id?: number | string | null
  frame_id?: number | string
}

/**
 * True when an auxiliary response belongs to the active splat state.
 *
 * Diagnostics, motion, and pose-depth are separate HTTP responses. A mixed
 * response order can pair an old motion or depth frame with a new active state.
 * Compare source, generation, and the splat frame before a panel uses it.
 */
export function auxMatchesActive(
  auxiliary: AuxiliaryIdentityLike | null,
  active: ActiveIdentityLike | null,
): boolean {
  if (auxiliary === null || active === null) {
    return false
  }
  if (auxiliary.source_id !== active.source_id) {
    return false
  }
  if (Number(auxiliary.generation ?? 0) !== Number(active.generation)) {
    return false
  }
  const auxiliaryFrame = auxiliary.splat_frame_id ?? auxiliary.frame_id
  if (auxiliaryFrame === undefined || auxiliaryFrame === null) {
    return false
  }
  return String(auxiliaryFrame) === String(active.frame_id)
}

/**
 * True when a pose-depth image belongs to the active run and is not older than
 * the active state. Separate responses can deliver an image that is newer than
 * the diagnostics sample; a newer image is safe, an older one is not.
 */
export function auxIsCurrentOrNewer(
  auxiliary: AuxiliaryIdentityLike | null,
  active: ActiveIdentityLike | null,
): boolean {
  if (auxiliary === null || active === null) {
    return false
  }
  if (auxiliary.source_id !== active.source_id) {
    return false
  }
  if (Number(auxiliary.generation ?? 0) !== Number(active.generation)) {
    return false
  }
  const auxiliaryFrame = auxiliary.splat_frame_id ?? auxiliary.frame_id
  if (auxiliaryFrame === undefined || auxiliaryFrame === null) {
    return false
  }
  return Number(auxiliaryFrame) >= Number(active.frame_id)
}

/** True when a splat request queued during a fetch should start now. */
export function shouldDrainSplat(nextKey: string | null, renderedKey: string | null): boolean {
  return nextKey !== null && nextKey !== renderedKey
}

/** True when a mask request queued during a fetch should start now. */
export function shouldDrainMask(nextKey: string | null, satisfiedKey: string | null): boolean {
  return nextKey !== null && nextKey !== satisfiedKey
}

/** True when a pose-depth request queued during a decode should start now. */
export function shouldDrainPoseDepth(nextKey: string | null, renderedKey: string | null): boolean {
  return nextKey !== null && nextKey !== renderedKey
}
