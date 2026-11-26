# Reranker: Flow Result to Selection Decision

The reranker is a one-reference replacement cache, not a reconstruction-quality
model. It receives reference-to-current `FlowResult` data for selected frames
and owns exactly one retained camera frame: the last frame it marked
significant, together with
its identity and comparison metadata. It does not retain every frame and never
reads the Temporal Cache's splat state. For each incoming frame it asks:

```text
Is this frame significantly different from the retained significant frame?
```

It then emits a `SelectionDecision` with exactly one action. The intended
downstream routing is:

- `reuse`: the Frame Processor sends the original `FramePacket` and its
  matching `FlowResult` to the Temporal Cache. No depth, foreground, or motion
  estimator runs for that decision.
- `refresh`: the Frame Processor creates a `ProcessedFrame` from the selected
  original packet, then requests depth, foreground segmentation, and optional
  human motion/pose. Those results, plus matching flow identity, feed the Splat
  Constructor.

On the first frame, or when the difference is significant, the reranker flushes
the retained reference, copies the current frame into its one-entry cache, and
emits `refresh`. An insignificant difference emits `reuse` and leaves the
retained reference unchanged. If flow is unavailable, the conservative initial
policy is `refresh`.

The reference also expires after `max_reference_age_ns` of capture time
(default 500 ms). This keeps the same time limit across 30, 60, and variable-FPS
sources when timestamps share an origin and increase. An optional
`max_reference_gap_frames` provides a second frame-ID bound. Expiry emits
`reference_expired` and refreshes without running flow against an old reference.
Neither limit requires storing the intervening images. Validate the time limit
against subject motion and downstream refresh cost.

`SelectionDecision` preserves `source_id`, selected frame ID, capture
timestamp, and the matching flow's current-frame identity. A bounded queue may
skip IDs, so consumers must not infer adjacency from frame numbers. Estimator
completion order must be handled by downstream consumers using this identity.

The implementation is `modules/reranker`. `StreamRerankerConsumer` directly
links the Stream Handler queue to a DIS or RAFT estimator: it passes the
reranker's retained frame as the previous flow frame and the dequeued packet as
the current frame. This ensures the difference score is reference-to-current,
not merely adjacent-frame motion.

The current implementation makes deterministic threshold decisions and records
the measured difference score when flow is valid. A learned ranking model is
not part of the implemented path. The threshold and scheduling decisions are
recorded in [the integration history](../Integration.md) and the live diagnostic
results in [the integration guide](../integration_runbook.md).
