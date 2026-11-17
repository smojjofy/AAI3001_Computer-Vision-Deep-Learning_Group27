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
 *   npx tsc src/api/identity.ts --ignoreConfig --outDir .tmp-identity --target es2022 --module es2022
 *   node tools/check-identity.mjs
 *
 * `--ignoreConfig` is required because tsconfig.json exists and TypeScript
 * refuses a file list otherwise (TS5112).
 */
import { readFileSync } from 'node:fs'

const source = readFileSync(new URL('./../.tmp-identity/identity.js', import.meta.url), 'utf8')
const module = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`)
const { splatKey, isKeyAtLeast, retryDelayMs, needsMaskFetch, settleMaskIdentity } = module

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

if (failures > 0) {
  console.error(`${failures} identity check(s) failed`)
  process.exit(1)
}
console.log('all identity checks passed')
