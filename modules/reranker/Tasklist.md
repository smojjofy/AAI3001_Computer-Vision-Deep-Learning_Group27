# Reranker

- Purpose: compare each consumed camera frame with one retained significant frame and choose `reuse` or `refresh`.
- Inputs: Stream Handler `FramePacket` and optional reference-to-current optical `FlowResult`.
- Outputs: `SelectionDecision` from `shared/schemas/selection_decision.schema.json`.
- Direct dependencies: Stream Handler queue protocol and optical-flow estimator protocol; no splat cache input.
- Reference policy: refresh when motion is significant, flow is unavailable, or the reference reaches a maximum elapsed capture time. An optional frame-ID gap limit can provide a second bound. Neither requires storing intervening images.
- Remaining work: evaluate time and motion thresholds on recorded clips, then test with downstream refresh and reuse stages.
- Live scheduling: application orchestration may drain to the newest available packet and sample by capture time before invoking optical flow. Reranker itself remains a one-reference decision module.
