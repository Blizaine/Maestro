# Automatic H3 clip and window durations

The shared duration matrix was refreshed on October 14, 2026 for v2.7.0. Its
previous update was October 12, when the 12 GB GPU / 32 GB RAM 720p Pruned and
Fused recommendations were raised to 345 frames, displayed as 14.4 seconds.

Studio and Director use the same policy, including Frames, References and
Director music-video and short-film clip limits. Auto uses the detected GPU
memory, system RAM, actual canvas area and checkpoint. A nominal resolution
label alone is not enough: a square or wide canvas can have a different area.
Saved manual durations, locked windows and existing project timings remain in
control. Return the duration or clip limit to Auto to adopt the current policy.

## Starting points

This table covers built-in **H3 Pruned and H3 Fused**, with the noted Fused
Frames exception at 1080p. Durations are per generated pass, not the total
video or song. Longer projects use consecutive windows or clips.

| GPU VRAM / system RAM | 480p | 540p | 720p | Native 1080p | Basis |
| --- | ---: | ---: | ---: | ---: | --- |
| 6 GB / 32 GB | 5.2s | 5.2s | Choose a smaller canvas | Choose a smaller canvas | Estimated |
| 8 GB / 16 GB | 5.2s | Start at 480p | Start at 480p | Start at 480p | Estimated; host RAM is a limitation |
| 8 GB / 32 GB | 10.1s | 10.1s | 5.2s | Choose a smaller canvas | Estimated |
| 10 GB / 32 GB | 14.4s | 14.4s | 10.1s | Choose a smaller canvas | Estimated |
| 12 GB / 32 GB | 14.4s | 14.4s | 14.4s | 5.2s | Repeated reported 720p successes; conservative 1080p start |
| 16 GB / 32 GB | 14.4s | 14.4s | 14.4s | 6.6s | Estimated |
| 20 GB / 32 GB | 14.4s | 14.4s | 14.4s | 8.0s | Estimated |
| 20 GB / 64 GB | 14.4s | 14.4s | 14.4s | 10.1s | Estimated |
| 24 GB / 32 GB | 14.4s | 14.4s | 14.4s | 6.6s | Estimated; less host RAM than the 4090 test machine |
| 24 GB / 64 GB | 14.4s | 14.4s | 14.4s | 10.1s | Estimated |
| 24 GB / 128 GB | 14.4s | 14.4s | 14.4s | Fused Frames: **14.4s**; Pruned or Fused References: **10.1s** | Tested Fused routes; other combinations estimated |

The representative canvases are 864×480, 960×544, 1280×704 and 1920×1088.
Portrait canvases with the same pixel area use the same tier. Padding and
aspect ratio matter: for example, a 12 GB / 32 GB system now gets an estimated
294-frame / 12.3-second starting point at 1344×768 instead of falling back to
the older 5.2-second recommendation. Intermediate canvas sizes have explicit
shorter tiers.

The UI distinguishes **Tested recommendation**, **Reported recommendation**,
**Estimated starting point**, and **Conservative starting point**. Going beyond
a starting point is additional, unverified load; it is not automatically
labelled a likely out-of-memory failure. When no automatic profile is provided,
the UI suggests a smaller canvas and retains the explicit manual override.

## What is measured

- The 12 GB RTX 3080 Ti / approximately 31.76 GiB usable RAM completed measured
  shorter 720p Pruned/Fused workloads, and the user repeatedly completed
  345-frame 720p Pruned/Fused generations. The exact 345-frame row is reported
  evidence, not a controlled timing or peak-memory study.
- The 24 GB RTX 4090 / 128 GB RAM completed Fused References 345-frame music
  and image-reference clips, including a 1792×768 canvas. Large reference
  videos and additional LoRAs can change the memory requirement.
- The 24 GB machine completed an unconditioned Fused Frames 1920×1088,
  362-frame test with six steps, SLA, Q2_K encoder, Profile 4 and head splitting
  off. It took about 16 minutes and peaked at 18.47 GiB. The refreshed default
  stays at the shorter 345-frame native envelope. This does not establish the
  same limit for Full, Pruned or reference-heavy recipes.
- Longer native-1080p Fused clips completed on the 12 GB machine only with
  suitable memory settings; other configurations failed. Long Pruned dense
  runs could take hours. This is why the 12 GB native-1080p automatic start
  remains 5.2 seconds, while the UI explains that longer clips are possible
  conditional experiments. The older dense timing warning is not applied to
  SLA or Veda as though it were a sparse-attention ETA.
- The October 14 4090 comparison reproduced the same 10.1-second reference
  clip eight times. Auto INT8 reduced warm denoising time by 10.7%; increasing
  fixed preload and Dynamic preload did not improve speed. Spare VRAM alone
  is not a reason to raise a duration recommendation.

There is no physical 8 GB GPU / 16 GB RAM result in this evidence set.
Intermediate and smaller resource classes are explicitly estimates. They
assume memory-appropriate quantized encoders and optimized offloading, with
shorter starts where host RAM is limited. Available RAM, disk paging, other
applications, drivers, references, LoRAs and attention settings affect the
actual outcome. Recommendations describe a starting duration, not a memory
guarantee or an estimate of rendering time.

Full, imported, alternate, audio-only and fixed-workflow checkpoints retain
their separate conservative profiles. Other model families have their own
duration rules; these H3 observations do not justify changing LongCat or LTX
window limits. Unknown RAM retains the existing conservative fallback.

## Duration and performance are different limits

The native H3 pass remains capped at 345 frames (14.375 seconds, displayed as
14.4). Fitting a longer pass in memory does not establish model quality beyond
that envelope. The separate **Allow 30s clips — Experimental** option remains
explicitly opt-in.

Packed-row guidance describes native versus chunked **normalization**, not a
calibrated fast or slow rendering tier. The newer attention, projection and
offload optimizations can change the cost of each path. Actual render time
still depends on checkpoint, steps, attention, inputs and hardware.

The versioned source is
[h3_duration_recommendations.json](../app/shared/h3_duration_recommendations.json).
Backend policies, Studio controls and Director clip-limit responses all
consume the same rows and provenance. See also
[Performance Auto-Tune](Performance-auto-tune.md#automatic-clip-and-window-durations)
and the [memory benchmark guide](development/memory-benchmark.md).
