# H3 placement on 16GB RTX 50-series GPUs

Dev follow-up, 23 October 2026. The external RTX 5080 slowdown report has not
been reproduced on that machine; this change addresses two identifiable Auto
placement differences without claiming a measured speedup.

## Baseline and diagnosis

Public Main moved from v2.6.0 (`31d7396`) to v3.0.0 (`0678b44`) and then
v3.1.0 (`726f758`). The unreleased v2.7 candidate was renamed v3.0. Comparing
against that candidate misses the MMGP 4 and generation-aware planner changes
that first reached Main in v3.0.
The reporter's exact rollback commit is still unknown; publication history
alone does not identify their local checkout.

Two placement differences matter for the reported class:

- A nominal 64GB host can report 63.7 GiB usable RAM. The newer tier tolerance
  puts it in the high-RAM class, changing the saved Auto profile from 4 to 2.
  Profile 2 attempts to pin all managed models; Profile 4 prioritizes the main
  transformer. The change can affect host pressure and weight transfers.
- For feasible H3 jobs, the pre-load workspace guard calculates an explicit
  resident-weight allowance. The newer planner replaced it with a cold-start
  ceiling of 3 GiB on hosts below 96 GiB, even when the guard allowed more.
  Turning off the optional allocator does not restore the previous planner.

These are code-path differences, not proof of the reporter's entire runtime
increase. Attention, checkpoint precision, references, available RAM, actual
allocator and selected INT8 backend still need a matched comparison.

## Upstream comparison

Reviewed WanGP 18.21 at immutable revision
[`9d25c601`](https://github.com/deepbeepmeep/Wan2GP/tree/9d25c601f3c7074ca03c50f41b307a7c517835f9).
Its default video profile is 4; it does not select a separate scheduler from
the RTX 5080 name. Profiles and preload settings control residency, with
activation pressure governing optional dynamic preload. Its
[CLI guide](https://github.com/deepbeepmeep/Wan2GP/blob/9d25c601f3c7074ca03c50f41b307a7c517835f9/docs/CLI.md#dynamic-vram-preload)
explains that dynamic preload needs its optimized allocator, measures a new
workload before retaining extra blocks, and may provide little benefit when
long video steps already hide transfers.

Maestro retains its adapted MMGP runtime and H3 workspace guard. This fix does
not import the whole newer engine, force dynamic preload, enable RAM spilling,
or copy upstream performance claims.

## Targeted compatibility policy

For Auto-owned native H3 video placement on Windows, a nominal 16GB Blackwell
GPU paired with nominal 64GB RAM uses Profile 4. Feasible jobs retain their
existing guard-approved resident-weight candidate instead of the extra 3 GiB
ceiling. Foreign GPU usage can further reduce that allowance. MMGP still
enforces its own working-VRAM limit.

When the activation reserve is clamped, the override stays zero and the
profile's streaming default remains in force. Low available host RAM can still
select Profile 5. Local learning cannot reselect Profile 2 for this compatibility
class. The final plan and its reasons appear in Last generation and output
metadata; saved settings are not rewritten.

Manual profiles, manual/dynamic preload, explicit command-line budgets and
Auto-off remain authoritative. The tested 8GB, 12GB and 24GB policies are
unchanged. FastH3, Viggle, audio-only and image routes are outside this change.
Sampling, steps, attention, visual quality settings and source references are
unchanged.

## Validation and reporter test

CPU regressions exercise the real guard, per-job worker and loader handoff,
including retained workspace, manual overrides, clamped workloads, foreign VRAM,
model reload/reuse and restoration after completion or failure. Synthetic
hardware inputs do not reproduce GPU throughput or Windows shared-memory
paging. No 16GB RTX 5080 benchmark or speed-recovery claim is made.

After updating Dev, restart Maestro to load the changed Python code. Compare
with Auto-tune enabled and video preload set to Default to exercise this
policy; manual placement continues to bypass it. Use
the same checkpoint, encoder, references, resolution, actual frame count,
sampling recipe, steps, preview setting and seed. Record loading, per-step
denoising and VAE decoding separately. Include the `[VRAM]`, `[Auto-tune]`,
`[INT8] Backend` and H3 performance lines, plus the allocator's **Active** value.
Changing an allocator requires a restart; selecting PyTorch for INT8 kernels
and selecting PyTorch's memory allocator are separate choices.
