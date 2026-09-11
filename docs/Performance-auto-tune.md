# Performance Auto-Tune

Settings → Performance can apply recommendations for the detected GPU, VRAM
and system RAM. Changing a performance setting manually turns Auto off;
the system-config API does the same. Explicitly applying Auto again replaces
the performance settings with the current recommendations.

## Per-job H3 transformer residency

H3 and Viggle jobs reserve space for their packed video/audio sequence, then
pass the remaining transformer allowance to MMGP on profiles 2, 4, 4.5 and 5.
The safety coefficient caps residency; setting the transformer budget is what
lets MMGP retain more weights instead of repeatedly streaming them from RAM.
The requested allowance leaves margin under the effective coefficient ceiling.

This changes only the transformer budget. VAE, encoder and catch-all budgets,
manual preload choices and the automatic profile table retain their existing
behavior. A cached H3 model reloads when the requested budget changes, and the
job restores the base budget on completion, failure or cancellation.

A contributed RTX A4500 / 28 GB RAM Viggle test reported approximately 75 to
60 seconds per denoising step after this fix, compared with v2.1.5. This is a
specific workload measurement; see [v2.1.6 validation](VALIDATION_V2.1.6.md).

## RAM and streamed model weights

On machines with 12–23 GB VRAM, video and image generation use Profile 5 below
32 GB of detected system RAM, Profile 4 from 32 to below 64 GB, and Profile 2
from 64 GB. Profile 4 allows MMGP to pin transformer weights in RAM, which can
improve repeated transfers to the GPU. Pinning keeps those pages resident in
physical RAM. Profile 5 disables pinning to reduce that pressure.

A proposed Profile 4 recommendation for 24–31 GB hosts was held back after
the reported 28 GB RAM / RTX A4500 test failed before denoising. That test
confirmed partial pinning, but could not measure its effect on Maestro's
step time. Subsequent testing identified INT8 dispatch and insufficient
transformer residency as measurable contributors, addressed in v2.1.5 and
v2.1.6. The conservative automatic profile remains in place; manual Profile 4
remains available.

The residency change retains the existing VRAM safety coefficient calculations,
per-job workspace reserves, quantization and MMGP pinning ceiling.

MMGP can fall back to partial pinning. Its usual ceiling is 40% of physical
RAM on Windows and 50% elsewhere, unless overridden. That ceiling bounds its
pinned allocations, not the application's total RAM consumption. Other model
weights, media, temporary allocations and other programs still use memory.
The pinned tensors replace weight storage; adding the entire checkpoint size
to the pinned amount does not establish the process's resident footprint.
Measure actual use and allow for temporary allocations.

## Applying corrections to existing installs

Automatic settings record a recommendation revision and the values applied.
With Auto enabled, startup checks the revision before loading models. An older
revision refreshes once; ordinary restarts do not repeatedly retune the machine.
Fresh installs and the explicit Apply action record the same revision.

Legacy installs that recorded only `auto_performance_applied: true` can receive
the correction. Values that differ from the previous recommendation are
preserved. Auto-disabled installs are left alone. If CUDA detection fails,
startup leaves the update pending rather than replacing settings with a fallback.

## Checking a performance difference

For the reported Viggle comparison, use one 124-frame window (about 5.17 seconds
at 24 fps), three steps, and exactly the same source, edited frame, audio mode,
resolution, attention backend, checkpoint, LoRA and seed. Run both profiles in
the same Maestro installation so dependency versions do not change. Restart
between configurations, and repeat a successful job to record both first-run
and warmed denoising time. Also record total time, pinning logs, system RAM,
process memory and peak VRAM.

Profile 4 with `--perc-reserved-mem-max 0.25` is an **experimental comparison**,
not an automatic recommendation or a guaranteed OOM fix. The argument belongs
to the Maestro Python launch command. It caps pinning at 25% of physical RAM
(about 7 GB on a 28 GB host); actual pinning can be lower. Record the previous
value and restore it after testing. Do not compare a changed dependency stack
at the same time. A successful window is needed before attributing any speed
change to pinning.

The generic `CUDA error: out of memory` message does not establish host RAM
exhaustion. CUDA failures can be reported by a later operation because GPU
execution is asynchronous. See [PyTorch's CUDA notes](https://docs.pytorch.org/docs/2.7/notes/cuda.html#asynchronous-execution).
CPU audio preparation now scopes torchaudio's helper allocations to the CPU,
even when MMGP has selected CUDA as the default device. The encoder still uses
the GPU. This removes an unnecessary CUDA allocation path; an A4500 run is
still needed to establish whether it prevents the reported failure.

On a generation memory failure, Maestro logs available system RAM, process RSS,
Windows private committed memory when available, and PyTorch/CUDA VRAM readings
before unloading the model. Failed CUDA readings are marked unavailable. These
are diagnostic snapshots, not proof of which allocation originally failed.

## INT8 ConvRot kernel selection

ConvRot checkpoints retain FP32 weight scales. Quanto's native fallback uses
the scale dtype for both the activations and matrix multiplication, so passing
those scales unchanged promoted BF16/FP16 inference to FP32. Maestro now casts
the temporary scale to the activation dtype for dense inputs. Checkpoint data,
stored scales, activation rotation and quantized-activation math are preserved.

When INT8 Triton kernels are enabled, the first eligible ConvRot inference call
compares the corrected native fallback with Triton on the actual loaded weight,
activation shape and dtype. Compilation is warmed before timing. The decision
is cached by device, dtype, shape and input strides for the current kernel
session; at most 32 distinct shapes are checked. Native must measure at least
15% faster to replace Triton. This does not change the memory profile, pinning
budget or attention backend.

The check runs only for dense FP16/BF16 ConvRot inference, outside graph capture
or compilation. It skips probing when estimated GPU headroom is insufficient,
and a failed probe retains Triton. Native headroom is checked again on later
calls as model residency changes. If a selected native call raises a PyTorch
out-of-memory allocation error, that shape retries using Triton and stays on
Triton for the session. Generic CUDA driver errors are not treated as safe to
retry. The initial check adds a few layer evaluations; subsequent calls reuse
the decision.

Console output identifies the selection, for example:

```text
[Quanto][INT8] ConvRot kernel check M/K/N=8192/5376/28672 torch.bfloat16: native=16.23 ms, Triton=13.35 ms; using Triton.
```

For a controlled comparison, leave the hardware profile and job fixed. Disabling
INT8 kernels selects the corrected native fallback. With kernels enabled,
`MAESTRO_CONVROT_KERNEL_CHECK=0` disables this new comparison and retains the
existing Triton routing. This diagnostic override is not needed for ordinary use.

An isolated RTX 4090 test using the installed Viggle checkpoint and Maestro's
MMGP loader measured the corrected fallback about 3.0–3.1 times faster than the
old fallback across Q projection and both MLP projections. Triton remained faster
than the corrected fallback, and automatic selection retained it. These layer
tests exclude denoising, attention, LoRA work and weight streaming. They do not
predict an RTX A4500's full-window time or establish the cause of any remaining
Maestro/WanGP difference. A matched full-window comparison is still required.
