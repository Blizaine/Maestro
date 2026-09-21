# H3 memory and enhancement latency — v2.2.3

Validated 17 September 2026 against v2.2.2; included in v2.2.3.

## Reference-sequence normalization

[Issue #139](https://github.com/Blizaine/Maestro/issues/139) reports an A100 40 GB
allocation failure in RMSNorm with 264,654 packed rows. A single FP32 copy of
that 5,376-wide hidden state requires 5.30 GiB. The existing transformer bounded
QKV and MLP projections but normalized the entire hidden sequence at once.

Normalization now uses bounded token slices during inference, preserving the
native norm operation, weights, dtype promotion and offload hooks. Training keeps
the original autograd path. No reference, resolution or residency settings change.

An isolated CUDA operation test used the reported shape, BF16 activations/weights,
PyTorch 2.7.1+cu128 and the development RTX 4090. It loaded no video model and
rendered no media:

| Operation | Peak allocated VRAM, including input | Additional allocation above input | Time |
| --- | ---: | ---: | ---: |
| Original RMSNorm | 18.554 GiB | 15.903 GiB | 0.611 s |
| Bounded RMSNorm | 5.793 GiB | 3.143 GiB | 0.135 s |

Sampled outputs were bit-identical. CPU tests also check FP32/BF16/FP16,
mixed input/weight dtypes, noncontiguous inputs, unchanged gradients, and a full
transformer block plus its final normalization. The timing is a single operation
measurement, not a video throughput benchmark. A full generation on the reporter's
A100 and reference set remains unverified; other stages still require VRAM.

## Enhancement latency

Saved development traces confirm avoidable retry work, but do not establish a
universal v2.2.2 slowdown. One three-window conversation used eleven writer calls
and 288.969 seconds on Qwen3.6 27B, including repeated density-only rewrites.
Another used separate shortening requests for multiple overlong turns.

These changes permit one successful development pass for sparse speech,
reserve further attempts for malformed output or actual content/timing problems,
try a text edit directly when existing speech is overlong, and batch the remaining
overlong turns. Tests verify call counts, retention of useful drafts after failed
repairs, speaker/event ownership, exact quotations, missing-topic repair, and
final timing validation. Word targets remain in the initial writing instructions.

For context, the same neighborhood case improved between the saved v2.2.1 and
v2.2.2 tests: 427.656 to 100.125 seconds with Qwen3.6, and 82.781 to 24.547 seconds
with Gemma E4B. These were enhancement-only test runs, not a controlled repeated
performance benchmark. They do not measure Qwen3.8 or the reporting user's setup.
No new live LLM run or end-to-end speedup is claimed for this isolated latency
investigation. Later reference-dialogue checks are documented separately.

## Automated checks

- 276 affected Python regressions passed: H3 runtime math, story/camera planning,
  dialogue writing/placement, exact-quote and review behavior, shared guides, and
  promptbench story timing.
- The isolated CUDA memory comparison above passed.
- This investigation made no launcher changes, application restart, model
  downloads, user generations, public push or GitHub comments. Publication is
  recorded in the combined v2.2.3 release validation record.
