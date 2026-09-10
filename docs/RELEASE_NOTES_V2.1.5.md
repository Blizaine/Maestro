# Maestro v2.1.5

Released 10 September 2026.

This patch improves H3's handling of detailed authored prompts, fixes a Viggle
INT8 fallback precision problem, and addresses stereo audio, LLM lifecycle,
Director seamless timelines and performance-setting updates.

## Better AI Faithful adaptation

- Recognize explicit timed action blocks and inline Character A/B profiles,
  including pasted prompts whose original line breaks have been lost.
- Keep choreography in coherent phases instead of turning each sentence or
  production note into another scheduled story event. Preserve the sequence,
  relative phase durations, supplied character details and physical ending.
- Treat camera/VFX requirements as shared context. Silent action descriptions
  and ambiguous unquoted character notes no longer count as spoken dialogue;
  supplied quoted or tagged speech still receives exact-line and timing checks.
- Give the LLM a focused camera-writing request and enough output space for
  complete action descriptions. Bind each camera plan to its assigned events
  rather than later pairing it with unrelated replacement action text.
- Preserve unique action and sound details in the compiled H3 prompts. The
  prompt token target is advisory, rather than a reason to cut story content.
- Reject incomplete camera JSON and allow a focused repair. A remaining fallback
  still reports its cause for review instead of appearing to be an ordinary AI draft.

Maestro's selected duration/windows remain authoritative. A script that says
"30 seconds" can be adapted into a selected 28-second output; it does not
silently override the controls. **Exact H3 prompts** remains available for review
and editing. See [Studio controls](Studio-controls.md).

## Keep the LLM available through enhancement

Active calls and multi-pass H3 planning now hold an LLM-use reference through
validation, repairs and camera planning. A completed concurrent call cannot
start an idle unload while another request is still working, and a canceled
timer cannot unload a newer request. Normal idle memory release resumes after
the last active use finishes. Explicit model unloading remains available.

This addresses `LLM not loaded. Call load_model() first.` occurring between
otherwise successful enhancement passes.

## Viggle and H3 INT8 execution

This release also includes the control-window memory fix published after
v2.1.4: Viggle converts control frames in bounded chunks and pads the final
uint8 array, avoiding full-window floating-point copies while preserving pixels
and history alignment.

ConvRot checkpoints retain FP32 scales, but their inference activations normally
use BF16/FP16. Quanto's fallback selected matrix-multiplication precision from
the scale and promoted the operation to FP32. Maestro now casts the temporary
scale to the activation dtype, preserving the stored checkpoint scales and
the required ConvRot rotation and LoRA math.

With INT8 kernels enabled, eligible ConvRot inference calls compare the corrected
fallback with Triton using the actual loaded weights, activation shape and dtype.
Decisions are cached per device/shape for the session. Triton stays selected
unless native is measurably faster and has sufficient estimated GPU headroom.
The check is bounded and skipped during compilation/capture or when memory is
tight; it does not change the hardware profile or pinning budget.

On a local RTX 4090, three real Viggle checkpoint layers showed approximately
**3× faster fallback execution** after the precision fix. Triton remained faster
on that GPU, so automatic selection retained it. This is **not an end-to-end
Viggle speedup claim**, nor a measured result for an RTX A4500. See
[performance behavior](Performance-auto-tune.md) and the
[validation measurements](VALIDATION_V2.1.5.md).

## Preserve stereo audio and avoid unnecessary CUDA allocations

- Keep H3's 32 kHz stereo audio through video muxing and sliding-window joins.
  The fix is in the shared audio assembly path, so it applies beyond Viggle.
- Join stereo output with mono prefixes or silent gaps without forcing the
  assembled track to mono. Mono-only tracks remain mono.
- Keep CPU waveform resampling and its torchaudio helper tensors on CPU even
  when MMGP has selected CUDA as the default device. The audio encoder still
  uses the GPU when needed.

Existing exported mono files are not reconstructed by updating; generate a new
output to use the corrected audio assembly.

## Director seamless duration

Director's H3 Seamless workflow now validates the native inference window against
the saved frame increments and maximum shot length. The complete multi-window
movie is no longer rejected for exceeding one shot's maximum or having a
trimmed final tail. Independent shots retain their existing duration checks.
See [Director controls](Director-controls.md).

## Performance settings and memory diagnostics

- Auto performance settings record a recommendation revision and their previous
  automatic defaults. Eligible existing installs can receive a revised
  recommendation once at startup, while customized values and manual mode are
  preserved. Failed GPU detection leaves migration pending.
- Explicitly applying recommendations and creating a fresh configuration use
  the same revision bookkeeping. Manual settings changes through the API also
  disable Auto, matching the UI behavior.
- Memory failures log available system RAM, process memory and CUDA/PyTorch
  memory readings before cleanup. A generic CUDA out-of-memory message is no
  longer presented as proof that system RAM was the cause.

The conservative automatic profile for hosts below 32 GB RAM is retained.
This release does not force Profile 4 or assume that additional pinning will
solve a machine's performance problem.

## Updating and validation

Use **Update** in Pinokio, restart Maestro and refresh the browser. Models,
characters, outputs, workspaces and saved projects remain in place. No new
dependency installation or model download is required by these changes.

For prompt improvements, run **Enhance** again from the original source prompt;
an already prepared draft does not rewrite itself. Review the exact window
prompts before generating.

The [validation record](VALIDATION_V2.1.5.md) separates automated checks, real
LLM/layer measurements and remaining hardware/quality limits.
