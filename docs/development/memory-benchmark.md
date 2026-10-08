# Memory and speed test bench

`app/scripts/benchmark_memory.py` runs a bounded, serial matrix through a local
Maestro server. It records complete generations, device-wide NVML VRAM, system
RAM, API phase timings, output filenames, and relevant terminal evidence.
Python 3.10+ is sufficient; the runner needs no GPU packages of its own.

Start Maestro normally and copy its local URL. Use installed checkpoints and
check that Studio and Director are idle. Edit a copy of
`app/scripts/benchmark_memory.example.json` for the desired checkpoint, inputs,
resolution, duration, and settings. From the repository root:

```powershell
python app/scripts/benchmark_memory.py --base-url http://127.0.0.1:42012 --matrix app/scripts/benchmark_memory.example.json --output-dir .codex-tmp/my-memory-test --dry-run
python app/scripts/benchmark_memory.py --base-url http://127.0.0.1:42012 --matrix app/scripts/benchmark_memory.example.json --output-dir .codex-tmp/my-memory-test --timeout-seconds 1800 --poll-seconds 1
```

Replace the port with the running server's port. The output directory must be
new or empty. Dry-run validates the plan without HTTP requests or file writes.
The example requires the Fused checkpoint and its runtime assets to be installed;
generation uses Maestro's normal first-use download behavior if assets are absent.

Common `request` fields apply to every case. Case `request` fields override them.
Case `settings` are system settings; omitted settings reset to their initial
values between cases. Set `request.override_profile` explicitly to match
`settings.video_profile`, including when testing older running backends whose
default profile was cached at startup. `reload: true` releases resident models
before the first repeat; `repeats: 2` then runs a second generation without that
release. A model release does not flush OS file caches or compiled kernels.
Changing the profile or preview components can trigger a model reload even
when the case does not request one.

The examples include `settings_version` so the backend interprets duration
and window controls using the current schema. After completion, the runner
checks saved effective parameters and actual media dimensions/frame counts.
Only `benchmark_eligible: true` with `output_validation.status: verified`
qualifies as a performance observation. A shortened window is reported as
`workload_mismatch`; missing metadata is `unverified`. The original job's
terminal status is retained separately, and settings restoration still runs.
Symbolic resolutions are recorded without inventing an expected pixel size.
For H3 sequences, valid saved completion timing identifies cumulative
intermediate files. They remain in the evidence; eligibility requires a
verified final output with every window complete and the full frame count.

Inspect the actual saved window size before comparing long H3 runs: automatic
VRAM protection can shorten a requested window. A deliberate
`sliding_window_memory_override: true` comparison uses the manual window lock
and can exceed the card's conservative automatic recommendation. It is not a
way to establish a generally safe duration.

Results are written incrementally to `results.json` and `results.csv`. A private
settings backup is written before changes. The runner disables auto-tune during
the matrix and restores the changed settings and original auto-tune preference
after its jobs settle. Ctrl+C and timeouts cancel only the runner's own job.
If another job starts, the runner aborts its comparison and keeps the recovery
snapshot if settings cannot safely be restored. After Maestro becomes idle:

```powershell
python app/scripts/benchmark_memory.py --output-dir .codex-tmp/my-memory-test --restore-only
```

Recovery uses the saved local URL unless explicitly overridden with the same
origin. The runner refuses allocator changes: changing the allocator requires a
Maestro process restart. Select the allocator before starting a matrix and check
the UI's Active line. For Dynamic preload, use an active MMGP allocator and an
asynchronous profile such as 2, 4, or 5. Full and 4.5 do not apply Dynamic.

For meaningful comparisons:

- Keep the checkpoint, inputs, prompt, seed, steps, canvas, duration, LoRAs,
  preview mode, and attention backend constant while changing a memory setting.
- Compare warm runs and denoising separately from model loading, encoding, and
  decoding. Dynamic preload learns the workload, so retain its first/warm labels
  and case order when interpreting results.
- Fused H3 uses SLA or its explicit SDPA reference option. Attention Head Split
  does not apply to its SLA path; use a compatible non-Fused checkpoint and
  Sage2 to evaluate that setting. Custom quantization/adapter paths can fall back.
- Read terminal evidence for the resolved INT8 backend, attention mode, and
  residency plan. Requested UI values alone do not prove an optimization applied.
- NVML VRAM includes the desktop and other processes. Polling can miss brief
  peaks; system RAM includes OS file caches and other applications. Timings are
  measured from API polling, with the stated sampling resolution.
- Open the generated videos to evaluate quality. A successful generation and
  lower memory use do not establish visual or audio equivalence.

The test bench compares settings in the installed version. It does not establish
the speedup over a previous release, a universal OOM threshold, or results for
other GPUs. Local inputs, generated videos, and result snapshots belong in
ignored directories rather than the public repository.

Run its CPU-only safety checks with:

```powershell
python -m unittest discover -s tests -p test_memory_benchmark.py
```

## Measurements on October 7, 2026

The first screening used an RTX 4090 with 24 GiB VRAM, 128 GB system RAM,
driver 610.88, PyTorch 2.10 / CUDA 13, and an active MMGP Optimized allocator.
Live Tiny VAE video remained enabled. The Fused References checkpoint generated
the same 1280x704, 124-frame (5.17-second), eight-step clip with one image
reference and a fixed prompt and seed. No prompt writer ran during the tests.

| Fused H3 settings | Lifecycle | Total seconds | Denoising seconds | Peak VRAM GiB |
| --- | --- | ---: | ---: | ---: |
| Full / Default / Triton INT8 | Warm | 134.5 | 93.6 | 21.10 |
| Full / Default / Auto INT8 | Warm | 101.1 | 63.6 | 21.11 |
| Balanced / Default / Auto INT8 | Reload | 123.8 | 63.5 | 20.77 |
| Balanced / Dynamic / Auto INT8 | Reload | 119.1 | 63.5 | 23.49 |
| Balanced / Manual 8192 MiB / Auto INT8 | Reload | 118.6 | 63.5 | 14.89 |

Auto resolved to Comfy Kitchen CUDA. In the paired warm trials it reduced total
time by 24.8% and denoising time by 32.1% relative to Triton. Balanced with Manual
8192 MiB saved 5.88 GiB compared with Balanced Default, with essentially equal
denoising time and the same reload lifecycle. The manual value is a model-weight
preload budget, not a cap on the whole generation's VRAM.

Dynamic preload adapted its weight residency and used more available VRAM; it
did not improve denoising time in this screening. Profile 4 and Profile 5
Dynamic trials also completed. Smart Memory Pinning and Read Ahead toggles did
not show a clear benefit from disabling either. These toggle trials ran in a
process with retained Dynamic measurements and warm file caches, so they do not
isolate the individual toggle's effect or establish first-load disk performance.

A separate installed DaSiWa Hybrid Turbo V3 checkpoint used the same short clip
with Balanced / Default / Auto INT8. Fixed preload kept adaptive weight growth
out of the head-splitting comparison. All rows below are warm trials:

| DaSiWa attention | Total seconds | Denoising seconds | Peak VRAM GiB |
| --- | ---: | ---: | ---: |
| Sage2 / Head Split Off | 109.5 | 76.5 | 20.26 |
| Sage2 / Head Split Medium | 111.2 | 78.8 | 20.44 |
| Sage2 / Head Split High | 110.9 | 78.8 | 20.07 |
| Sol / Head Split Off | 103.6 | 70.4 | 20.24 |

The terminal confirmed 56 query heads in eight groups for Medium and fourteen
groups for High. Neither reduced the observed overall peak substantially, and
both added about 3% to denoising time. Sol reduced warm total time by 5.5% and
denoising by 7.9% in this single pair; it can produce different visual results.
These are option comparisons within the current build, not a comparison with
the pre-port release. Successful generation and representative decoded frames
are useful checks, but users should assess the complete videos and sound.

The longer confirmation used Fused References at the same 1280x704 canvas,
243 frames (10.125 seconds), ten steps, a fixed locked prompt and seed, two
active image references, and the Combat LoRA at 0.80. A supplied voice reference
was omitted by Maestro's normal speaker scoping because the prompt specified
silent characters. All cases used SLA, 67,281 packed rows and split projections,
with no cache or prompt-writing pass. Each row is one model-reload trial:

| Longer Fused H3 settings | Total seconds | Denoising seconds | Peak VRAM GiB |
| --- | ---: | ---: | ---: |
| Full / Default / Triton INT8 | 384.8 | 281.7 | 21.32 |
| Full / Default / Auto INT8 | 280.2 | 198.7 | 21.30 |
| Balanced / Manual 8192 MiB / Auto INT8 | 278.4 | 199.1 | 17.58 |

Auto reduced total time by 27.2% and denoising by 29.5% versus Triton in this
longer comparison. Manual preload with Balanced kept essentially the same
speed as Full Auto while reducing the observed peak by 3.72 GiB. That makes
**Video Profile 2 + Auto INT8 + Manual video preload 8192 MiB** a useful starting
point for this machine's tested H3 workloads. Keep Head Split Off for these
workloads; keep Smart Memory Pinning and Read Ahead enabled unless a separate
test shows a reason to change them. Full with Auto is also a simple improvement
over the tested Full/Triton setup. These single confirmation trials do not give
an uncertainty interval or establish the best budget for every checkpoint.

All **21 completed generations** passed output stream checks and complete CPU
audio/video decoding, and none reported an OOM. Representative frames were
reviewed separately; these checks do not establish perceptual quality or audio
equivalence. One earlier DaSiWa attempt was cancelled before denoising after a
Windows-console encoding error in the runner; that error was fixed and the
cancelled attempt was excluded from performance results. Original memory
settings and auto-tune were restored and the benchmark model cache released.
