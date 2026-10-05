# Maestro v2.6.0

October 4, 2026. Changes since v2.5.0.

## Watch generation develop

In **Settings → Performance → Generation Preview**, choose **Fast Frames**,
**Clearer Frames (Tiny VAE)** or **Live Video (Tiny VAE)**. Off remains the
default. Previews appear in Studio's generating card with progress, ETA and the
current clip/window. Pause or resume the looping video without pausing the render.
Phones play inline, with a Play button if the browser blocks automatic playback.

H3 previews use the predicted clean result so a scene can emerge before final
decoding. The setting lists support for the selected model: H3 requires Tiny VAE
and has no legacy Fast Frames path. Supported image models show still previews,
including when Live Video is selected. Audio-only jobs have no visual preview.
Temporary previews clear after the job and never become gallery assets. Preview
errors allow generation to continue; preview decoding can add GPU work.

See [preview support and API](Generation-preview.md).

## Import community MiniMax H3 checkpoints

Open **Model Browser → Checkpoints**, filter to **MiniMax H3**, then choose a
creator version and file. Maestro inspects the actual file before downloading
and verifies its full SHA-256 before installation. Compatible hybrid checkpoints
receive paired **Frames** and **References** entries sharing one transformer.
Matching installed weights are reused, including curated files with other names.

- Full and pruned joint video/audio H3 architectures.
- BF16/FP16 and supported scaled E4M3 FP8.
- Comfy INT8 ConvRot, including header-based descriptors such as Eros Max exports.
- DaSiWa/WanGP asymmetric W4A8 INT4, with compatible CUDA acceleration.
- Supported GGUF Q2_K/Q3_K/Q4_K/Q5_K/Q6_K, Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 weights.
- Grouped and head-interleaved QKV with matching native conditioning and
  compatible LoRA routing; FP32 rotary calculations.

Standard, baked Turbo and fused editions retain their sampling recipes. When
metadata is ambiguous, the browser asks for the creator's workflow, recipe or
QKV layout. Verification cannot make an incompatible file compatible and does
not certify its creative quality. Unsupported formats and files without required
verification data remain blocked. Import progress and failures are now visible.

See [supported files, recipes and validation](H3-checkpoint-import.md).

## More models and workflows

The experimental DaSiWa selections requested in [#167](https://github.com/Blizaine/Maestro/issues/167)
include H3 Hybrid v3 standard and baked Turbo in Frames/References, Krea 2
MirroredSkies RAW and DarkDesire Turbo UC images, Wan 2.2 SynthSeduction
Lightspeed v9 and LTX-2.3 DragonLeap v4 FP8. They download pinned creator files
with model-specific defaults; Mature mode controls applicable selections.
See the [DaSiWa model guide](DaSiWa-models.md).

**Singularity** now supports Frames text generation, start/end/timed images and
Control Video editing as well as References. Its companion entries reuse the
existing v1.3 checkpoint and recommended four-step adapter. See
[Singularity workflows](H3-Singularity.md).

DaSiWa H3 follows the configured dense Auto attention backend instead of forcing
SDPA. DaSiWa and compatible imported baked-Turbo checkpoints offer optional
**H3 Optimizations → Sol Engine** on supported hardware while keeping their
sampling recipe. Sol is approximate and can change the result. Additional
Turbo/PDD adapters and incompatible caches remain disabled for baked recipes.

## LongCat belongs in Avatar

**Video → Avatar** sits below References and above Extend. Select LongCat Single
or Multi and supply an anchor image and voice audio. Multi includes two voice
tracks and speaker-region controls. Missing inputs are caught before generation;
older saved LongCat Avatar settings migrate to the dedicated workflow.

Multi-speaker identification, audio offsets/padding and native window defaults
are corrected. Longer timelines continue through bounded passes. Sequential
guidance, activation-aware weight residency and earlier attention-buffer cleanup
reduce peak memory demand without changing model precision.

## Longer reference and control media

Frames control videos and masks advance with each window and separately rendered
clip, aligned with the source clock and trimmed tails. In H3 References, ordinary
video uploads default to **Follow window timeline**, including an enabled
embedded or attached soundtrack. Disable it to reuse a short appearance/motion
sample. Saved character and voice samples remain reusable.

**Duration → Auto** follows the longest timeline video and updates when it is
replaced or removed. A music/performance timeline retains soundtrack precedence.
Manual Time or Window choices retain your runtime. H3's shared 15-second video
reference budget still applies per window; overlap reuses the same source times,
and output beyond the source holds its last frame and pads audio with silence.

**Object / prop** images preserve an object's design, proportions and materials
without adding a cast identity. The prompt determines count, placement and
action; object-background isolation is available. **Sound effect reference**
reuses a short sound sample in each window to guide matching generated effects.
It does not loop the original waveform or set total duration.

See [long source media and reference controls](Studio-controls.md#long-source-media-across-windows).

## Duration controls that explain the work

- **Window** keeps the selected count fixed. Increasing Window Length updates
  total duration, accounting for continuation overlap and discarded frames.
- **Time** keeps the runtime fixed and recalculates the window count.
- **Auto** follows the prompt or source media. A note explains the active mode.
- Experimental 30-second H3 windows now retain Window behavior throughout the
  extended slider range and when the experiment is toggled. Auto exits it.
- H3 shows a normalization-path estimate separately from checkpoint/GPU memory
  guidance. Resolution, references, carried history and the largest actual
  generated pass contribute to the assessment. Expand the explanation for
  estimated packed rows and uncertainty.
- The duration band marks the recommended 14.4-second boundary and warns about
  longer experimental passes. Estimates are advisories, not fixed slowdown
  multipliers, guaranteed memory limits or a universal GPU capacity.

The locally incorporated change from [PR #152](https://github.com/Blizaine/Maestro/pull/152)
uses native RMSNorm through 75,000 ordinary packed rows, avoiding repeated
normalization launches. Larger workloads retain bounded normalization and
explicit chunk overrides. This threshold selects a normalization path; it is
not an overall render-speed or VRAM guarantee.

## Prompt and gallery reliability

Eligible long locked physical passages retain their complete source actions
while the writer supplies camera coverage. Camera repair targets failed cards
with more precise missing-clause feedback; safe event-level recovery retains
accepted cards. Semantic reviews are batched and bounded without bypassing
source, chronology or ownership checks.

Optical camera wording, character aliases, contact details and global style
directions are handled more accurately. Unclosed spoken quotes produce an input
error before writing/generation rather than silently losing a line. Reference
fallbacks preserve entrance order instead of placing every character in the
opening frame. See [enhancement input guidance](Studio-enhancement-api.md).

Gallery thumbnail clicks retain the intended asset through metadata loading and
new results, and the right strip follows selection reliably. Action menus fit
the viewport instead of clipping at the toolbar. Video inputs display posters.

## Downloads and runtime

Download buttons are separated from model-enable checkboxes. Controllable model
pre-downloads, CivitAI and Hugging Face imports offer **Cancel**, explicit
**Cancelling/Cancelled** status and retry. Complete files remain installed;
incomplete checkpoints are not registered. See [download controls](Download-controls.md).

[PR #156](https://github.com/Blizaine/Maestro/pull/156)'s download status waits for
changes and pauses on hidden pages, with polling fallback for older servers or
interrupted connections. Hardware polling also suspends when the page is hidden.
GPU utilization reports NVIDIA compute activity, consistent with `nvidia-smi`.

**Settings → Integrations** offers explicit loaded-model and unload controls for
LM Studio 0.4+ through its native API. Unload is verified and blocked while the
writer or generation is busy. See [LM Studio memory](LM-Studio-memory.md).

Startup clears stale Web UI addresses and shows failures when Windows App
Control/Device Guard prevents Python from starting. This does not resolve the
interpreter-trust problem in [#164](https://github.com/Blizaine/Maestro/issues/164).
Invalid/skipped generation tasks now produce failed jobs. Supported INT8
shared-memory launch failures retry with a smaller cached kernel tile.

MMGP is pinned to **3.8.2**. Optional promptbench sampling/reasoning experiments
remain developer-only; ordinary requests do not enable them.

## Update

Use **Update** in Pinokio, restart Maestro and refresh the browser. Updating
installs the new dependency pin and rebuilds the UI. Your weights, projects,
preferences and generated media remain in place. Tiny VAE decoders download on
first use; selecting a curated model does not immediately download its weights.

New community models/formats and extended windows retain their stated
experimental and hardware limitations. This release does not include PR #157
Reference Packs, PR #162 Director lyrics, or the deferred Veda-Sparse integration.

[Validation and limitations](VALIDATION_V2.6.0.md) · [Changelog](../CHANGELOG.md)
