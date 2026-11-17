/*
 * Identity helpers for the live viewer.
 *
 * A frame id alone is not an identity: a source or process restart can reuse
 * it. Every fetch key and comparison uses (source, generation, frame). These
 * functions are pure so they can be checked without a browser.
 */

export function splatKey(
  sourceId: string,
  generation: number,
  frameId: number | string,
): string {
  return `${sourceId}|${generation}|${frameId}`
}

/** True when a later key of the same capture run is not older than the request. */
export function isKeyAtLeast(candidate: string, requested: string): boolean {
  const [candidateSource, candidateGeneration, candidateFrame] = candidate.split('|')
  const [requestedSource, requestedGeneration, requestedFrame] = requested.split('|')
  if (candidateSource !== requestedSource) {
    return false
  }
  if (Number(candidateGeneration) !== Number(requestedGeneration)) {
    return false
  }
  return Number(candidateFrame) >= Number(requestedFrame)
}

/** Wait until a retry deadline, never negative. */
export function retryDelayMs(deadline: number, now: number): number {
  return Math.max(0, deadline - now)
}

/**
 * Decide which identity to store after a mask response.
 *
 * `displayedKey` is the identity of the image the viewer holds. `satisfiedKey`
 * is the requested identity the fetch covered, even when the response is older
 * (after the stale retry limit). Keeping both means the viewer can name the
 * displayed image and still avoid repeating a settled request.
 */
export function settleMaskIdentity(
  requestedKey: string,
  responseIdentityKey: string | null,
): { displayedKey: string; satisfiedKey: string } {
  return {
    displayedKey: responseIdentityKey ?? requestedKey,
    satisfiedKey: requestedKey,
  }
}

/**
 * True when the viewer must request a mask for the active state.
 *
 * `satisfiedKey` is the requested identity that a completed fetcher has
 * covered, even when the server kept returning an older mask. Without it, a
 * fixed stale response would be requested again on every poll.
 */
export function needsMaskFetch(
  activeKey: string | null,
  satisfiedKey: string | null,
  pendingKey: string | null,
): boolean {
  if (activeKey === null) {
    return false
  }
  return activeKey !== satisfiedKey && activeKey !== pendingKey
}

/** Identity part of a fetch key, for display and logging. */
export function identityDescription(key: string): string {
  const [sourceId, generation, frameId] = key.split('|')
  return `${sourceId} gen ${generation} frame ${frameId}`
}
