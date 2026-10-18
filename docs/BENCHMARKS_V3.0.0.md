# Maestro v3.0.0 release benchmark pilot

Measured October 17–18, 2026 on three Windows systems. These are installed-release
comparisons of v2.6.0 Main and v3.0.0 Dev, not isolated tests of one optimization.

## Same-input H3 Fused References results

The primary metric is each completed output's `generation_time`, with
`generation_time_basis=active`, available in both releases. Warm results are the
median of two consecutive runs with the model retained.

| GPU / system RAM | 720p clip | Main warm median | Dev warm median | Main / Dev time ratio |
| --- | --- | ---: | ---: | ---: |
| RTX 4090 24GB / 128GB | 5.17s | 89s | 54s | 1.65× |
| RTX 4090 24GB / 128GB | 8.00s | 132.5s | 84s | 1.58× |
| RTX 3080 Ti 12GB / 32GB | 5.17s | 349.5s | 168.5s | 2.07× |
| RTX 3080 Ti 12GB / 32GB | 8.00s | 900.5s | 254.5s | 3.54× |
| RTX 5070 Laptop 8GB / 32GB | 5.17s | No completed warm baseline | 283s | Not calculated |
| RTX 5070 Laptop 8GB / 32GB | 8.00s | Not run | 453.5s | Not calculated |

For these warm workloads, Dev used **37–39% less active generation time on the
4090** and **52–72% less on the 3080 Ti**. The 3080 Ti Main 8s repeats varied from
721 to 1080 seconds; the small sample does not establish a confidence interval.

### Separate completed 8GB Main follow-up

The original laptop Main 5.17s trial reached its 35-minute bound during VAE
decoding after 1979 seconds of denoising. It was cancelled without verified
finished media. Its five remaining Main trials were not run; none is recorded
as an OOM or an unsupported duration.

One separately authorized rerun with a 60-minute bound completed:

| Laptop 5.17s References, model reload | Active generation |
| --- | ---: |
| Main follow-up | 1476s / 24m36s |
| Prior matched Dev reload | 288s / 4m48s |

The observed Main/Dev ratio is **5.13×**, or 80.5% less time on Dev. This is one
separate reload observation per release, not a new warm median. Host-memory,
OS-cache, thermal and paging state were not identical between sessions. The
earlier cancellation is retained and the variation is not explained by this
single completion. Main References at 8s and above remains untested on this
8-GB laptop configuration.

Main follow-up denoising took 1349 seconds (about 337 seconds per step). The
remaining 127 seconds includes encoding/loading, VAE decoding and saving; it is
not an exact VAE-decode timer. Job elapsed time was 1485 seconds.

## Memory observations

The following device-wide sampled peaks cover the original 8s case; they include
other desktop allocations and may miss brief spikes. RAM is whole-system
physical use, not Maestro process memory.

| GPU | Main peak VRAM | Dev peak VRAM | Main peak RAM used | Dev peak RAM used |
| --- | ---: | ---: | ---: | ---: |
| RTX 4090 | 23.34GiB | 14.39GiB | 84.37GiB | 68.54GiB |
| RTX 3080 Ti | 11.90GiB | 7.78GiB | 31.65GiB | 31.67GiB |
| RTX 5070 Laptop | Not measured at 8s | 6.44GiB | Not measured at 8s | 30.80GiB |

The completed laptop Main follow-up separately peaked at 7.92GiB VRAM and
30.68GiB RAM, with only 0.34GiB available physical RAM at the lowest sample.
Both 32GB systems approached their host-RAM limits, including on Dev. Faster
completion does not mean RAM pressure is solved, or that free VRAM can all be
used for weights without affecting workspaces.

## Recipe and verification

- H3 Fused References INT8 ConvRot transformer, Q2_K text/vision encoder, original
  full native INT8 video VAE and native audio VAE; four steps, SLA, seed 424242.
- 1280×704 at 24fps, 124 or 192 frames, the same fixed driver/car prompt and two
  byte-matched image references. Prompt enhancement and live previews were off.
- Auto-tune enabled in both releases; Dev INT8 set to Auto. Main actually used
  MMGP 3.8.2 and Triton INT8; Dev used MMGP 4.0.0 and Auto resolved to WGPKernels.
- Dev used the active optimized allocator and smart pinning, hardware/workload
  placement, and no head splitting for these short cases. Read-ahead was enabled
  on the 4090, disabled on the two 32GB hosts.
- The plan called for one model release/reload and two warm repeats per
  case/release; the cancelled and not-run laptop Main rows are disclosed above.
  A reload does
  not flush OS or kernel caches. Branches were serial per GPU; machines parallel.
- Main baseline: `31d7396eb3b5586c1828fc69a6af5eb2a9e83bff`.
  Dev baseline: `6fa4f1b02075a79dc446043a15381bbf54996ee2`.
  The 4090 checkout also had the subsequently released duration/SSH/test/docs
  changes; fixed requests and effective runtime plans were recorded.
- Actual loaded transformer, encoder and both VAE paths were checked against
  recorded component hashes. Some hashes were reused after frozen-file path/size
  checks; the separate follow-up also checked recorded mtimes. An older duplicate
  VAE found by inventory was corrected using the actual loaded runtime path.

The original batch retained all 36 planned rows: **30 completed**, **one bounded
cancellation**, **five not run**. All 30 completed videos and the separate Main
follow-up passed full CPU software decoding and exact dimensions/frame-count
checks. Five sampled frames per video showed no blank or visibly corrupt sample.
Audio streams decoded but were not listened to or rated. Settings and the original
Dev-on/Main-off application state were restored after testing.

## Interpretation and limits

These results support shorter generation times for the recorded H3 workloads.
They do not verify every model, LoRA, reference load, 1080p, long window, Director
music video, Linux runtime, 16GB RAM configuration or generated-media quality.
Outputs can differ across kernels; decoding and sampled stills do not establish
motion continuity, prompt adherence, singing or lip-sync quality.

Earlier manual laptop Frames observations remain separate from this controlled
References pilot: Fused Main/Dev ratios were about 1.70× at 5.17s and 1.89× at 8s;
Pruned at 8s was about 3.32×. Seeds, encoders, some prompts and the Pruned attention
backend differed. Those observations cannot fill missing References trials or
isolate a release effect.

The new release recommendations are documented in
[Duration recommendations](Duration-recommendations.md). The 8GB/32GB Fused 720p
10.1s starting point is supported by earlier recipe-specific completions, not an
extrapolation from this short benchmark. No longer default is established here.
