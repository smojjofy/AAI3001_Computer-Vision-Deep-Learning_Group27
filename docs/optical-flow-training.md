# Milestone optical-flow decision

The integrated application uses OpenCV DIS optical flow by default. The
repository also supports pretrained Torchvision RAFT Small for comparison.
The team did not train or fine-tune an optical-flow model for this milestone.
Flow ranks candidate frames and supplies the browser reuse warp. It does not
set a person's depth or replace a valid splat refresh.

## Measured comparison

Both backends processed the same 300-frame local clip at a 320 x 180 maximum
flow size, with ten warm-up pairs excluded. The result was measured on this
computer, not on an independently labelled optical-flow dataset.

| Backend | Median pair time | 95th-percentile pair time | Valid flow fraction |
| --- | ---: | ---: | ---: |
| OpenCV DIS, fast mode | 83.051 ms | 89.161 ms | 0.969 |
| RAFT Small, 12 updates | 200.564 ms | 230.591 ms | 0.911 |

RAFT Small made subject movement easier to see in inspected previews. The
recording has no ground-truth flow field, so this visual observation does
not establish lower motion error. These pair times exclude capture, depth,
pose, splat construction, and browser display. RAFT took about 2.4 times
the median DIS pair time in this isolated test. The integrated application
therefore keeps DIS as its default and RAFT as an optional research backend.

The [device verification guide](optical-flow-device-verification.md) gives
the benchmark and visual checks. The [training design record](optical-flow-training-record.md)
preserves the proposed datasets, paired augmentation rules, and acceptance
criteria for any later fine-tuning work.

## End-of-milestone limit

The comparison does not prove that one backend selects better refresh frames
or improves splat quality. A later model change needs labelled motion or an
independent task measure, plus a live latency test with the downstream
pipeline running. No optical-flow training claim is part of this milestone.
