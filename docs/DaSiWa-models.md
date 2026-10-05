# DaSiWa models (experimental)

Maestro includes eight selections for the four DaSiWa model families requested
in [issue #167](https://github.com/Blizaine/Maestro/issues/167). They use the
existing Studio and Director runtimes with the creator's pinned checkpoints.
They are experimental: configuration, routing and download checks have passed;
complete generation and visual quality still need testing with these weights.

| Family and edition | Where to select it | Starting recipe | New checkpoint download |
| --- | --- | --- | --- |
| H3 Hybrid v3 | Video → Frames or References | 25 steps, Euler | 21 GB |
| H3 Hybrid Turbo v3 | Video → Frames or References | 8 steps, Euler; supports 4–8 | 21 GB |
| Krea 2 MirroredSkies v1 RAW | Image → Generate | 52 steps, guidance 3.5 | 26 GB |
| Krea 2 DarkDesire v3 Turbo UC | Image → Generate; Mature mode | 8 steps, no CFG | 26 GB |
| Wan 2.2 SynthSeduction Lightspeed v9 | Video → Frames; Mature mode | Start image, 4 steps, Euler | 29 GB high/low pair |
| LTX-2.3 DragonLeap v4 FP8 | Video → Frames | 8 steps, native distilled runtime | 28 GB |

The download sizes above are decimal GB for the selected weights. Shared
encoders, VAEs and other required assets can add to the first-use download.
Weights download when generating, rather than when selecting a model.

## First test

1. Restart Maestro after installing the code changes and refresh the browser.
2. Select a DaSiWa model in the workflow listed above. If a selection was
   disabled, enable it under **Settings → Performance → Enabled Models**.
   The Turbo UC and SynthSeduction editions use Maestro's existing Mature-mode
   opt-in; they remain hidden while that mode is off.
3. For H3, try a short 480p clip before a longer project. Keep the preset's
   starting step count and use the same prompt, seed and inputs when comparing
   it with your usual model. For Wan, supply a start image.
4. The first generation retrieves the official creator files. If Civitai asks
   for authentication, set **Settings → Services → Civitai API key** and check
   that the account has access to the selected file.

Downloads use exact file identities, expected byte sizes and SHA-256 digests.
An incomplete download or HTML login response does not replace a checkpoint.
Existing shared model folders are still searched before downloading.

## H3 workflows and Turbo

The **Frames** entry supports text-only generation, start/end/timed images, and
Control Video editing. The **References** entry supports ordered image, video
and audio references. Switching between them retains the same standard or
Turbo edition, and both entries share a single checkpoint for that edition.

The standard Hybrid preset starts at 25 steps with video/audio shifts 11/4.
The Turbo preset has distillation baked into its weights and starts at 8 steps
with shifts 9/4. Its 4–8-step control remains adjustable, but adding another
Turbo/PDD accelerator is rejected. Audio refinement and sampling cache are
disabled for the baked recipe. Ordinary compatible H3 LoRAs can be tested;
their quality with this checkpoint is not yet verified.

Turbo reduces the number of denoising steps; the attention backend still
determines much of each step's cost. DaSiWa now follows the configured dense
Auto backend rather than forcing SDPA. For an optional speed comparison,
expand **H3 Optimizations** and enable **Sol Engine** on supported hardware.
Sol uses approximate sparse attention, so compare the same prompt, seed and
inputs with it off before choosing it for a project. First Block Cache and
additional Turbo/PDD adapters remain disabled for the baked recipe. Existing
downloads can use this setting without downloading the weights again.

The GPU status indicator reports NVIDIA compute utilization, matching
`nvidia-smi`. VRAM usage includes working tensors as well as resident weights;
MMGP may stream some weights from system RAM while generation runs.

The pinned H3 files use pruned INT8 ConvRot weights and grouped QKV with the
Ref2VA AdaLN basis. Frames changes the conditioning route while retaining that
basis. Full, INT4 and unrelated H3 files are not interchangeable with these
presets.

## Other families

Krea RAW and Turbo are separate text-to-image selections. The RAW edition
needs the longer undistilled recipe; Turbo hides the ineffective CFG control.
DaSiWa Identity Edit presets are not included in this initial integration.

Wan uses the creator's matching **High v9** and **Low v9** SafeTensors files,
with both guidance phases set to 1. The issue links a GGUF edition; this
integration uses the official compatible FP8 SafeTensors pair instead.
Do not mix high and low checkpoints from different editions or add another
distillation LoRA to the initial comparison.

LTX uses a full DragonLeap v4 checkpoint through the existing LTX-2.3 22B
distilled runtime. It supports text, frame guidance and native audio controls.
Its creator's ComfyUI workflow has additional sampler options; Maestro uses
its native distilled schedule. The separate DaSiWa LTX motion LoRA is not
substituted for the checkpoint.

## Pinned creator sources

| Selection | Official source | Version and file IDs |
| --- | --- | --- |
| H3 standard | [DaSiWa MiniMax H3](https://civitai.com/models/2877206/dasiwa-minimax-h3) | 3374445 / 3263052 |
| H3 baked Turbo | [DaSiWa MiniMax H3](https://civitai.com/models/2877206/dasiwa-minimax-h3) | 3374439 / 3263048 |
| Krea RAW | [DaSiWa Krea 2](https://civitai.com/models/2760803/dasiwa-krea2-or-turbo-or-raw) | 3115455 / 2997717 |
| Krea Turbo UC | [DaSiWa Krea 2](https://civitai.com/models/2760803/dasiwa-krea2-or-turbo-or-raw) | 3362930 / 3250905 |
| Wan high / low | [DaSiWa Wan 2.2 SafeTensors](https://civitai.com/models/1981116/dasiwa-wan-22-i2v-14b-lightspeed-synthseduction-v9) | 2555640 / 2455463 and 2555652 / 2455626 |
| LTX-2.3 | [DaSiWa LTX-2.3](https://civitai.com/models/2543443/dasiwa-ltx-23) | 3092188 / 2972051 |

The definitions record the published hashes and sizes. Downloads come from
the creator's official distribution and remain subject to the source terms.
