# Applications

- Purpose: compose existing modules into a live pipeline without moving model dependencies into shared contracts.
- `live_pipeline/` schedules latest-frame optical flow and limits downstream refresh work to one running job and one replaceable pending job.
- The live scheduler can expose its own Stream Handler's latest RGB packet to the GUI without a second camera capture.
- Inputs: timestamped `FramePacket` values and `SelectionDecision` objects; the refresh worker is injected by the application.
- Outputs: selected observations and refresh jobs. The current command uses a simulated worker, so no depth, pose, cache, or splat data is produced.
- Remaining work: attach real estimators and downstream state updates, including gating reuse until a matching cache state is available.
