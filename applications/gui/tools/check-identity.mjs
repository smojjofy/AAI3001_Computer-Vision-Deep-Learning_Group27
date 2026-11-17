/*
 * Focused checks for the pure identity helpers used by the viewer.
 *
 * These helpers decide whether a fetched state belongs to the state the viewer
 * asked for (RT-0028 items 1 and 4, RT-0030 item 2). They are pure, so they are
 * checked here without a browser.
 *
 * Run from applications/gui:
 *   npm run check:identity
 * or, directly:
 *   npx tsc src/api/identity.ts src/api/displayGuard.ts --ignoreConfig --outDir .tmp-identity --target es2022 --module es2022
 *   node tools/check-identity.mjs
 *
 * `--ignoreConfig` is required because tsconfig.json exists and TypeScript
 * refuses a file list otherwise (TS5112).
 */
import { readFileSync } from 'node:fs'

async function loadModule(name) {
  const source = readFileSync(new URL(`./../.tmp-identity/${name}.js`, import.meta.url), 'utf8')
  return import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`)
}
const identity = await loadModule('identity')
const guard = await loadModule('displayGuard')
const { splatKey, isKeyAtLeast, retryDelayMs, needsMaskFetch, settleMaskIdentity } = identity
const {
  auxIsCurrentOrNewer,
  auxMatchesActive,
  shouldApplyAuxiliary,
  shouldApplyResponse,
  shouldClearDisplayedState,
  shouldDrainMask,
  shouldDrainPoseDepth,
  shouldDrainSplat,
} = guard

let failures = 0
function check(name, condition) {
  if (condition) {
    console.log(`ok   ${name}`)
  } else {
    failures += 1
    console.error(`FAIL ${name}`)
  }
}

// The mask request key uses the published frame id, never the refresh counter.
const active = splatKey('camera', 2, 42)
check('key uses source, generation and frame', active === 'camera|2|42')
check('same state matches', isKeyAtLeast(active, active))
check('newer frame in the same run is accepted', isKeyAtLeast(splatKey('camera', 2, 43), active))
check('older frame is rejected', !isKeyAtLeast(splatKey('camera', 2, 41), active))
check('other generation is rejected', !isKeyAtLeast(splatKey('camera', 3, 50), active))
check('other source is rejected', !isKeyAtLeast(splatKey('other', 2, 50), active))
check('frame id 0 is not confused with the counter form', splatKey('camera', 2, 0) === 'camera|2|0')

// Retry delay: the approved stale-mask rule is 300 ms, the failure rule 500 ms.
check('delay is the remaining time', retryDelayMs(1000, 700) === 300)
check('delay is never negative', retryDelayMs(1000, 1500) === 0)
check('failure retry is 500 ms', retryDelayMs(1500, 1000) === 500)

// A satisfied request is not fetched again, even when the server stayed stale.
check('satisfied active key needs no fetch', needsMaskFetch(active, active, null) === false)
check('new active identity needs a fetch', needsMaskFetch(splatKey('camera', 2, 43), active, null) === true)
check('pending key is not requested twice', needsMaskFetch(active, null, active) === false)
check('no active identity needs no fetch', needsMaskFetch(null, null, null) === false)
check('old satisfied key does not block a later frame', needsMaskFetch(active, splatKey('camera', 2, 41), null) === true)

// A settled stale response stores both the displayed and the satisfied key.
const staleSettled = settleMaskIdentity(active, splatKey('camera', 2, 41))
check('displayed key names the stale response', staleSettled.displayedKey === splatKey('camera', 2, 41))
check('satisfied key names the request', staleSettled.satisfiedKey === active)
check('a settled stale response is not requested again', needsMaskFetch(active, staleSettled.satisfiedKey, null) === false)
const freshSettled = settleMaskIdentity(splatKey('camera', 2, 43), splatKey('camera', 2, 43))
check('a fresh response displays its own identity', freshSettled.displayedKey === freshSettled.satisfiedKey)
check('a missing response identity falls back to the request', settleMaskIdentity(active, null).displayedKey === active)

// Delayed responses across a run clear (RT-0034). The epoch is 0 while a
// request is in flight; the boundary clear must invalidate it even when no
// displayed or pending key exists.
{
  const epochBefore = 7
  const inFlightOnlyClear = shouldClearDisplayedState(false, 1)
  check('a clear runs when only a request is in flight', inFlightOnlyClear === true)
  const epochAfter = inFlightOnlyClear ? epochBefore + 1 : epochBefore
  check('the delayed splat response is discarded', shouldApplyResponse(epochBefore, epochAfter) === false)
  check('the delayed mask response is discarded', shouldApplyResponse(epochBefore, epochAfter) === false)
  check('the delayed pose-depth response is discarded', shouldApplyResponse(epochBefore, epochAfter) === false)
  check('the delayed pose-depth decode failure is discarded', shouldApplyResponse(epochBefore, epochAfter) === false)
}

// A poll that spans a clear is discarded as a whole.
{
  const pollEpoch = 3
  const currentEpoch = 4
  check('a poll that spans a clear does not apply motion or depth', shouldApplyResponse(pollEpoch, currentEpoch) === false)
  check('a current poll applies normally', shouldApplyResponse(4, 4) === true)
}

// Auxiliary results from a poll without an active state are skipped.
check('no active state skips motion and depth', shouldApplyAuxiliary(false) === false)
check('an active state applies motion and depth', shouldApplyAuxiliary(true) === true)

// An idle clear has nothing to invalidate.
check('an idle clear does nothing', shouldClearDisplayedState(false, 0) === false)
check('displayed state always clears', shouldClearDisplayedState(true, 0) === true)

// A request queued behind an old in-flight response still drains (RT-0036).
{
  const oldEpoch = 11
  const newEpoch = 12
  const newRunKey = splatKey('camera', 1, 40)
  // The old response is discarded, so the rendered mask key stays empty; the
  // queued new-run request must still start from the drain path.
  check('discarded splat response still drains a queued request', shouldDrainSplat(newRunKey, null) === true)
  check('discarded mask response still drains a queued request', shouldDrainMask(newRunKey, null) === true)
  check('discarded pose-depth response still drains a queued request', shouldDrainPoseDepth(newRunKey, null) === true)
  check('the queued request is from the new run', shouldApplyResponse(newEpoch, newEpoch) === true)
  check('the old response is from the old run', shouldApplyResponse(oldEpoch, newEpoch) === false)
}

// Motion and pose-depth must match the active diagnostics identity (RT-0036).
{
  const activeState = { source_id: 'camera', generation: 2, frame_id: 77 }
  check('same-run motion applies', auxMatchesActive({ source_id: 'camera', generation: 2, splat_frame_id: 77 }, activeState) === true)
  check('old-generation motion is rejected', auxMatchesActive({ source_id: 'camera', generation: 1, splat_frame_id: 77 }, activeState) === false)
  check('old-frame motion is rejected', auxMatchesActive({ source_id: 'camera', generation: 2, splat_frame_id: 60 }, activeState) === false)
  check('other-source motion is rejected', auxMatchesActive({ source_id: 'other', generation: 2, splat_frame_id: 77 }, activeState) === false)
  check('same-run pose-depth applies', auxMatchesActive({ source_id: 'camera', generation: 2, splat_frame_id: 77 }, activeState) === true)
  check('pose-depth without identity is rejected', auxIsCurrentOrNewer(null, activeState) === false)
  check('newer pose-depth for the same run applies', auxIsCurrentOrNewer({ source_id: 'camera', generation: 2, splat_frame_id: 80 }, activeState) === true)
  check('older pose-depth is rejected', auxIsCurrentOrNewer({ source_id: 'camera', generation: 2, splat_frame_id: 70 }, activeState) === false)
  check('old-generation pose-depth is rejected', auxIsCurrentOrNewer({ source_id: 'camera', generation: 1, splat_frame_id: 77 }, activeState) === false)
  check('auxiliary without an active state is rejected', auxMatchesActive({ source_id: 'camera', generation: 2, splat_frame_id: 77 }, null) === false)
}

if (process.env.CHECK_IDENTITY_SELF_TEST === '1') {
  // Deliberate failure: proves the command exits non-zero when a check fails.
  check('self-test: deliberate failure is detected', false)
}

if (failures > 0) {
  console.error(`${failures} identity check(s) failed`)
  process.exit(1)
}
console.log('all identity checks passed')
