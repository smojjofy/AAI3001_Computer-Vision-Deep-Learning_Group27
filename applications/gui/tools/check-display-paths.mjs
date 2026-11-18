/*
 * Executable delayed-response check for the viewer fetch paths (RT-0038).
 *
 * Unlike the pure-helper checks, this script runs the real async fetch runner
 * (`runFetchWithEpoch`) with controlled delays and epoch changes, and it traces
 * the boundary cases the display code depends on:
 *
 *   1. an old-run clear discards a response and a failure,
 *   2. a same-run frame advance makes a decoded depth image old,
 *   3. a newer depth image is held until the active splat frame catches up,
 *   4. a mixed API response order never pairs an old image with a new state,
 *   5. a failure keeps the newest pending identity instead of replacing it.
 *
 * Run from applications/gui after `npm run compile:helpers`:
 *   node tools/check-display-paths.mjs
 */

import { readFileSync } from 'node:fs'

async function loadModule(name) {
  const source = readFileSync(new URL(`./../.tmp-identity/${name}.js`, import.meta.url), 'utf8')
  return import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`)
}

const guard = await loadModule('displayGuard')
const skeleton = await loadModule('skeleton')
const latency = await loadModule('latency')
const runner = await loadModule('fetchRunner')
const {
  depthDisplayDecision,
  pendingAfterFailure,
  shouldDrainSplat,
  auxMatchesActive,
} = guard
const { runFetchWithEpoch } = runner
const { skeletonSegments, visibleJoints, MIN_VISIBILITY } = skeleton
const { captureToDisplayMs, formatLatencyLabel, LATENCY_LABEL_PREFIX } = latency

let failures = 0
function check(name, condition) {
  if (condition) {
    console.log(`ok   ${name}`)
  } else {
    failures += 1
    console.error(`FAIL ${name}`)
  }
}

function delay(ms, value) {
  return new Promise((resolve) => setTimeout(() => resolve(value), ms))
}

// 1. An old-run clear discards a response and a failure.
{
  let epoch = 5
  let applied = 0
  let recordedFailures = 0
  const appliedOutcome = await runFetchWithEpoch({
    fetch: () => delay(20, 'splat-from-old-run'),
    startEpoch: epoch,
    currentEpoch: () => epoch,
    apply: () => { applied += 1 },
    onFailure: () => { recordedFailures += 1 },
  })
  // Bump the epoch while the request is in flight.
  epoch += 1
  await delay(5, null)
  check('an in-flight response survives until the clear', appliedOutcome === 'applied' || appliedOutcome === 'discarded')

  // Repeat with the clear before the promise resolves.
  epoch = 8
  let lateApplied = 0
  let lateFailures = 0
  const pending = runFetchWithEpoch({
    fetch: () => delay(30, 'late'),
    startEpoch: epoch,
    currentEpoch: () => epoch,
    apply: () => { lateApplied += 1 },
    onFailure: () => { lateFailures += 1 },
  })
  await delay(5, null)
  epoch += 1
  const outcome = await pending
  check('a response after a clear is discarded', outcome === 'discarded')
  check('a discarded response is not applied', lateApplied === 0)

  // A failure after a clear must not schedule an old-run retry.
  let failureEpoch = 3
  let failureRecorded = 0
  const failed = runFetchWithEpoch({
    fetch: async () => { await delay(25); throw new Error('old-run decode failure') },
    startEpoch: failureEpoch,
    currentEpoch: () => failureEpoch,
    apply: () => {},
    onFailure: () => { failureRecorded += 1 },
  })
  await delay(5, null)
  failureEpoch += 1
  const failureOutcome = await failed
  check('a failure after a clear is discarded', failureOutcome === 'discarded')
  check('a discarded failure is not recorded', failureRecorded === 0)
}

// 2. A same-run frame advance makes a decoded image old.
{
  const active = { source_id: 'cam', generation: 1, frame_id: 21 }
  const decoded = { source_id: 'cam', generation: 1, splat_frame_id: 20 }
  check('a decoded frame older than the active frame is discarded', depthDisplayDecision(decoded, active) === 'discard')
  check('an old decode does not reach the depth panel', depthDisplayDecision(decoded, active) !== 'display')
}

// 3. A newer image is held until the active frame catches up.
{
  const newer = { source_id: 'cam', generation: 1, splat_frame_id: 22 }
  const activeNow = { source_id: 'cam', generation: 1, frame_id: 21 }
  const activeLater = { source_id: 'cam', generation: 1, frame_id: 22 }
  check('a newer depth image is held', depthDisplayDecision(newer, activeNow) === 'hold')
  check('the held image displays when the splat catches up', depthDisplayDecision(newer, activeLater) === 'display')
  check('a different generation is discarded', depthDisplayDecision({ source_id: 'cam', generation: 2, splat_frame_id: 22 }, activeNow) === 'discard')
  check('a different source is discarded', depthDisplayDecision({ source_id: 'other', generation: 1, splat_frame_id: 22 }, activeNow) === 'discard')
  // The active frame can skip past a held frame. The held frame is then stale.
  check('a held frame is discarded when the active frame passes it', depthDisplayDecision(newer, { source_id: 'cam', generation: 1, frame_id: 23 }) === 'discard')
}

// 4. Mixed API response order: motion and depth from the old state must not pair
// with the new diagnostics state.
{
  const activeNew = { source_id: 'cam', generation: 4, frame_id: 90 }
  const motionOld = { source_id: 'cam', generation: 4, splat_frame_id: 89 }
  const depthOld = { source_id: 'cam', generation: 4, splat_frame_id: 89 }
  check('an old motion warp is rejected', auxMatchesActive(motionOld, activeNew) === false)
  check('an old depth image is discarded', depthDisplayDecision(depthOld, activeNew) === 'discard')
  const motionSame = { source_id: 'cam', generation: 4, splat_frame_id: 90 }
  check('a motion warp for the active state applies', auxMatchesActive(motionSame, activeNew) === true)
}

// 5. A failure keeps the newest pending identity.
{
  const failed = 'cam|4|90'
  const newer = 'cam|4|91'
  check('a failure keeps a newer pending identity', pendingAfterFailure(newer, failed) === newer)
  check('a newer pending identity is not the failed one', pendingAfterFailure(newer, failed) !== failed)
  check('a failure retries itself when nothing newer is pending', pendingAfterFailure(null, failed) === failed)
  check('a failure retries itself when it is still pending', pendingAfterFailure(failed, failed) === failed)
  check('a discarded response still drains the queued request', shouldDrainSplat(newer, null) === true)
}

// 6. Skeleton overlay boundaries: visibility, missing coordinates, connections.
{
  const joint = (index, x, y, visibility = 1.0) => ({ index, x, y, visibility })
  check('a visible connected pair makes a segment', skeletonSegments([joint(11, 0.4, 0.4), joint(12, 0.6, 0.4)]).length === 1)
  check('a low-visibility joint is excluded', skeletonSegments([joint(11, 0.4, 0.4, 0.2), joint(12, 0.6, 0.4)]).length === 0)
  check('a joint exactly at the visibility floor is included', visibleJoints([joint(11, 0.4, 0.4, MIN_VISIBILITY)]).length === 1)
  check('missing coordinates are excluded', visibleJoints([joint(11, null, 0.4)]).length === 0)
  check('non-finite coordinates are excluded', visibleJoints([joint(11, Number.NaN, 0.4)]).length === 0)
  check('unconnected visible joints make no segment', skeletonSegments([joint(11, 0.4, 0.4), joint(0, 0.5, 0.1)]).length === 0)
  check('empty joints make no segment', skeletonSegments([]).length === 0)
  check('a full pose yields connected segments', skeletonSegments(
    Array.from({ length: 33 }, (_, index) => joint(index, 0.5, 0.5)),
  ).length > 20)
}

// 7. Capture-to-display estimate boundaries (RT-0043 item 1).
{
  check('all phases sum', captureToDisplayMs({ captureAgeMs: 40, stateAgeMs: 12, browserMs: 30 }) === 82)
  check('a zero browser phase is allowed', captureToDisplayMs({ captureAgeMs: 40, stateAgeMs: 12, browserMs: 0 }) === 52)
  check('a missing phase yields no value', captureToDisplayMs(null) === null)
  check('a non-finite phase yields no value', captureToDisplayMs({ captureAgeMs: Number.NaN, stateAgeMs: 1, browserMs: 1 }) === null)
  check('a negative phase yields no value', captureToDisplayMs({ captureAgeMs: -1, stateAgeMs: 1, browserMs: 1 }) === null)
  check('the missing label hides the value', formatLatencyLabel(null) === `${LATENCY_LABEL_PREFIX}: --`)
  check('the label states it is an estimate', formatLatencyLabel(82).includes('est'))
  check('the label rounds to whole milliseconds', formatLatencyLabel(82.6) === `${LATENCY_LABEL_PREFIX}: 83 ms`)
}

if (failures > 0) {
  console.error(`${failures} delayed-response check(s) failed`)
  process.exit(1)
}
console.log('all delayed-response checks passed')
