/*
 * One async fetch step with an epoch guard.
 *
 * The viewer discards a splat, mask, or pose-depth response that started before
 * a run boundary clear. This runner holds that control flow so the same logic
 * can run in the browser and in an executable delayed-response check.
 */

export type FetchOutcome = 'applied' | 'discarded' | 'failed'

export interface FetchRunOptions<T> {
  /** Start the request. It may reject. */
  fetch: () => Promise<T>
  /** Display epoch captured before the request started. */
  startEpoch: number
  /** Read the current display epoch. */
  currentEpoch: () => number
  /** Apply a response that belongs to the current view. */
  apply: (value: T) => void
  /** Record a failure that belongs to the current view. */
  onFailure: (error: unknown) => void
}

/**
 * Run one fetch and report what happened.
 *
 * - A response that started before a clear is discarded and never applied.
 * - A failure that started before a clear is discarded and never recorded, so
 *   an old-run failure cannot schedule a retry in the new run.
 */
export async function runFetchWithEpoch<T>(options: FetchRunOptions<T>): Promise<FetchOutcome> {
  let value: T
  try {
    value = await options.fetch()
  } catch (error) {
    if (options.startEpoch !== options.currentEpoch()) {
      return 'discarded'
    }
    options.onFailure(error)
    return 'failed'
  }
  if (options.startEpoch !== options.currentEpoch()) {
    return 'discarded'
  }
  options.apply(value)
  return 'applied'
}
