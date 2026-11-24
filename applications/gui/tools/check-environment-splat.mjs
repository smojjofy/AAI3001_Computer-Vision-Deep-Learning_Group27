import assert from 'node:assert/strict'
import { encodeEnvironmentSplat } from '../.tmp-environment/components/environmentSplat.js'
import { mayPresentPreparedFrame } from '../.tmp-environment/components/environmentFramePolicy.js'

const buffer = encodeEnvironmentSplat({
  header: { count: 1, display_alpha: 1 },
  positions: new Float32Array([0.25, 0.5, -2]),
  logScales: new Float32Array([Math.log(0.1), Math.log(0.2), Math.log(0.3)]),
  shDc: new Float32Array([0, 0, 0]),
  opacityLogits: new Float32Array([0]),
  rotations: new Float32Array([1, 0, 0, 0]),
})
const view = new DataView(buffer)
assert.equal(buffer.byteLength, 32)
assert.equal(view.getFloat32(0, true), 0.25)
assert.equal(view.getFloat32(8, true), -2)
assert.ok(Math.abs(view.getFloat32(12, true) - 0.1) < 1e-6)
assert.deepEqual([...new Uint8Array(buffer, 24, 4)], [128, 128, 128, 128])
assert.deepEqual([...new Uint8Array(buffer, 28, 4)], [255, 128, 128, 128])
console.log('environment splat conversion: ok')

const first = { sourceId: 'zed', generation: 1 }
const nextRun = { sourceId: 'zed', generation: 2 }
assert.equal(mayPresentPreparedFrame(first, null, 1, 1), true)
assert.equal(mayPresentPreparedFrame(first, first, 1, 1), true)
assert.equal(mayPresentPreparedFrame(first, nextRun, 1, 1), false)
assert.equal(mayPresentPreparedFrame(first, null, 1, 2), false)
console.log('environment frame handoff: ok')
