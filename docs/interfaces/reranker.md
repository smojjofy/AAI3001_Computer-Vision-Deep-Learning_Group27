# Reranker: Flow Result to Selection Decision

The reranker is a one-reference frame selector, not a reconstruction-quality
model. It receives reference-to-current `FlowResult` data and retains one
significant camera frame with its identity. It does not retain every frame
and does not read splat state. For each incoming frame it asks:

```text
Is this frame significantly different from the retained significant frame?
```

It emits a `SelectionDecision` with exactly one action. The integrated
application uses these actions:

- `reuse`: keep the active splat state and send matching flow for a browser
  display warp. Depth, pose, mask selection, and construction do not run.
- `refresh`: use the selected original frame for depth, MediaPipe pose,
  person-mask selection, and native splat construction. Publish the new
  state only when construction is valid.

The standalone reranker module replaces its retained frame when it emits
`refresh`. The integrated application advances the retained refresh
reference only after a valid construction. A failed refresh keeps the last
valid state and reference. Insignificant difference emits `reuse`.
Unavailable flow requests a refresh.

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
recorded in [the integration evidence](../integration-evidence.md) and the live diagnostic
results in [the technical record](../integration-technical-history.md).
