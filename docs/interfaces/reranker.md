# Reranker: Flow Observation to Frame Processor

The reranker is a one-reference replacement cache, not a reconstruction-quality
model. It receives each `FlowObservation` from Optical Flow and owns exactly
one retained camera frame: the last frame it marked significant, together with
its identity and comparison metadata. It does not retain every frame and never
reads the Temporal Cache's splat state. For each incoming frame it asks:

```text
Is this frame significantly different from the retained significant frame?
```

It then emits a `SelectionDecision` with exactly one action:

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

`SelectionDecision` must preserve `source_id`, selected frame ID, capture
timestamp, and the matching flow's current-frame identity. A bounded queue may
skip IDs, so consumers must not infer adjacency from frame numbers. Estimator
completion order is irrelevant: refresh results are joined by this identity
before Splat Construction.

The implementation is `modules/reranker`. `StreamRerankerConsumer` directly
links the Stream Handler queue to a DIS or RAFT estimator: it passes the
reranker's retained frame as the previous flow frame and the dequeued packet as
the current frame. This ensures the difference score is reference-to-current,
not merely adjacent-frame motion.

The initial implementation should make deterministic threshold decisions and
record the measured difference score. It should not learn a ranking model until
recorded replay shows that a hand-tuned threshold cannot meet latency and
reconstruction-stability targets.
