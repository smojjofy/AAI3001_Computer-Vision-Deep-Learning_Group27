/*
 * Skeleton segments for the Depth tab.
 *
 * The pose joints arrive with the same pose-depth payload as the depth image, so
 * the overlay always belongs to the displayed depth state. This module is pure,
 * which lets the segment and visibility rules be checked without a browser.
 */

export interface PoseJointLike {
  index: number
  x: number | null
  y: number | null
  visibility: number | null
}

/** MediaPipe Pose landmark connections. */
const POSE_CONNECTIONS: ReadonlyArray<readonly [number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 7],
  [0, 4], [4, 5], [5, 6], [6, 8],
  [9, 10], [11, 12],
  [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
  [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
  [11, 23], [12, 24], [23, 24],
  [23, 25], [24, 26], [25, 27], [26, 28],
  [27, 29], [28, 30], [29, 31], [30, 32],
  [27, 31], [28, 32],
]

export const MIN_VISIBILITY = 0.3

export function jointIsVisible(joint: PoseJointLike): boolean {
  if (joint.x === null || joint.y === null) {
    return false
  }
  if (!Number.isFinite(joint.x) || !Number.isFinite(joint.y)) {
    return false
  }
  return (joint.visibility ?? 1) >= MIN_VISIBILITY
}

export type SkeletonSegment = [PoseJointLike, PoseJointLike]

/** Visible skeleton segments for a set of pose joints. */
export function skeletonSegments(joints: ReadonlyArray<PoseJointLike>): SkeletonSegment[] {
  const byIndex = new Map<number, PoseJointLike>()
  for (const joint of joints) {
    if (jointIsVisible(joint)) {
      byIndex.set(joint.index, joint)
    }
  }
  const segments: SkeletonSegment[] = []
  for (const [startIndex, endIndex] of POSE_CONNECTIONS) {
    const start = byIndex.get(startIndex)
    const end = byIndex.get(endIndex)
    if (start !== undefined && end !== undefined) {
      segments.push([start, end])
    }
  }
  return segments
}

/** Visible joints that the overlay draws as points. */
export function visibleJoints(joints: ReadonlyArray<PoseJointLike>): PoseJointLike[] {
  return joints.filter(jointIsVisible)
}
