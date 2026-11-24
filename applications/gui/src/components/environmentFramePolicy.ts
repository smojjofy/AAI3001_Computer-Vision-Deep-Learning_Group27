export interface EnvironmentRun {
  sourceId: string
  generation: number
}

/** A new run waits for its own prepared buffer; same-run frames may advance. */
export function mayPresentPreparedFrame(
  prepared: EnvironmentRun,
  queued: EnvironmentRun | null,
  preparedEpoch: number,
  currentEpoch: number,
): boolean {
  if (preparedEpoch !== currentEpoch) return false
  return queued === null || (
    queued.sourceId === prepared.sourceId && queued.generation === prepared.generation
  )
}
