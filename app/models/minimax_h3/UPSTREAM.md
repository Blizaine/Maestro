# MiniMax H3 upstream components

## FastH3 V2 (experimental text-to-audio/video)

The optional `minimax_h3_fasth3` definition uses FastVideo's Comfy repack at
`FastVideo/FastVideo-FastH3-Comfy@ec1e3aa374a91c57b0b94a1623b7e657c0498cf2`:
`fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors`, 22,128,378,696 bytes,
SHA-256 `0922785978dc9bfe1adf27d8b291b0ca763f9f165f882e6cb297c72fbb6deda8`.
Weights download on demand and are checked before allocation. Shared H3
conditioning and VAE components retain their existing immutable sources.

Its sampling contract is `FastVideo/FastVideo-FastH3-8-Step-V2` revision
`3da2ddfe1954d9cda4c05b643dc0f26007a655c5`, `fastvideo_inference.json`:
T2AV only, no CFG, eight Euler forwards from unshifted noise rungs
`999,874,749,624,500,375,250,125` divided by 1000, shifted once with video/audio
10/3, followed by terminal zero. Nine grid points are eight forwards. The
uniform H3 schedule, RES, Turbo/PDD adapters and First Block Cache are excluded.

The native VSA policy follows Apache-2.0 FastVideo source at
`hao-ai-lab/FastVideo@d4f144581d6882d5bceb288ca13d767c054ca041`, particularly
`fastvideo/attention/backends/video_sparse_attn_h3.py` and
`fastvideo/models/dits/minimax_h3.py`. It preserves segment-pure text/audio
prefix tiles, generated-video 4×4×4 tiles, true partial-tile counts, 80% sparse
video top-k, dense prefix queries and exempt prefix keys. Dense pooled-tile
attention is multiplied by the raw trained compression projection and added
to fine attention. All 50 gate projections are loaded through MMGP; this is
not SLA or Veda and must never fall back to an ordinary H3 path.

Grouped QKV packing was independently checked with bounded HTTP range reads:
27 BF16 rows across Q/K/V and blocks 0/25/49 matched the original split
FastVideo tensors byte for byte. The INT8 file uses the same author's Comfy
ConvRot conversion convention. Native kernels and integration are experimental;
documented training support is not a benchmark of Maestro output quality.

OmniRef PDD is a separate model/recipe with 32 output heads, eight PDD blocks,
90% sparsity and 128-token tiles. The official documented weight repository was
not publicly readable during the initial review on 20 October 2026. It became
public later that day; the official formats are described below.
The published early real-time demo used eight B200 GPUs; the later OmniRef
release benchmarks report 5.6–11.9× end-to-end gains on eight GB200 GPUs.
Neither those results nor consumer-GPU launch profiles establish real-time performance on a 4090,
3080 Ti or 8 GB laptop.

## FastH3 OmniRef (experimental official Comfy formats)

The initial `minimax_h3_fasth3_omniref` integration pinned the community upload
[`lrigamai/FastH3-OmniRef-Comfy`](https://huggingface.co/lrigamai/FastH3-OmniRef-Comfy)
at `0d2c00a333ab1393d3ff18a008d7d9eda5536f11`, file
`diffusion_models/fasth3_omniref_8step_pruned_int8_convrot.safetensors`,
22,213,723,064 bytes, SHA-256
`851b2dc9670ed864b7f1e951ac9da047d83b1c04ded9bfcc1a9e5bba0ee33f80`.
That upload had no model card or exported inference sidecar. Its safetensors header
identifies `source=transformer_ref`, `config.transformer.pdd_steps=32`,
FP32 output banks (video 3072×5376, audio 1024×5376) and all 50 learned
compression gates. This establishes the structural PDD Ref2VA contract,
not provenance/quality parity with an unavailable official export at that time.

The official `FastVideo/FastH3-Omni-Comfy` release is now public at
`c6e7d1f212c44b21227bbd6e76bfff7da70e285d`. Its INT8 file has exactly the same
size and LFS SHA-256 as that community download, so Maestro now uses the official
source while reusing existing cached bytes. The same release's BF16 file is
44,240,780,704 bytes, SHA-256
`c0fac98c0317eb85215bd186b250be9b113675baab1323add40f26262f552ed9`.
Its source contract records Ref2VA, eight forwards, shifts 12/3, .999 grid,
32 PDD heads, VSA .9/tile128 and p2 multi-region reference routing. Header
geometry retains rank-8 AdaLN and FP32 residual PDD heads.

The additional official Comfy exports are pinned independently:

- `FastVideo/FastH3-Omni-FP8-Comfy@03127300d4803d5b66df66a22208b1d9238c8588`,
  `fasth3_omniref_fp8_comfy.safetensors`, 23,046,497,628 bytes, SHA-256
  `d3aa751e151df957b6248db7dbe4a159549dfefe78ba884056b12b0425b019f3`.
- `FastVideo/FastH3-Omni-NVFP4-QAD-Comfy@e1f39b9187e3e4059b7507e0a62fa22621ff77a5`,
  `fasth3_omniref_qad_nvfp4_comfy.safetensors`, 13,773,998,700 bytes, SHA-256
  `bf25ee58df65b4dfee349d612f7ac62a2fea0dddd1e68eb37e75c8ce8b166f5a`.

Their published quantization receipts describe **new native Comfy per-tensor
requantizations** with fused QKV. They must not be confused with a byte-preserving
repack of FastVideo's original per-row FP8 or independently scaled Q/K/V NVFP4.
The NVFP4 variant is QAD-trained and separately selectable. Header inspection
confirms U8 half-width packed weights, group-16 FP8 scales and a scalar second
scale. These use the existing MMGP native quantization handlers. All formats
retain source FP32 output banks; Maestro does not silently quantize those heads.
The trained sparse routing still uses Maestro's BF16/FP16 VSA kernel rather than
FastVideo's Blackwell FP4 attention/decoder stack. See [usage and hardware
limits](../../../docs/FastH3.md) for the distinction between model-format support
and measured performance. Conversion tools bundled in those repositories were
read as evidence only and are not vendored or executed by Maestro.

The native recipe follows the pinned FastVideo source above, particularly
`fastvideo/layers/pdd.py`, `docs/inference/fasth3-distilled.md` and
`fastvideo/tests/attention/test_vsa_h3_ref2va_regions.py`: fine grid
`linspace(.999,0,33)`, indices `[0,4,8,12,16,20,24,28,32]`, video/audio
shifts 12/3 with .999 fixed, target initial noise scaled by .999, FP32
integration-weighted output-head fusion and eight Euler forwards. The
ordinary Alibaba PDD LoRA grid and T2AV DMD ladder are not substituted.

The conversion's output storage differs from the FastVideo export: bank zero
is the complete base head, and banks 1–31 are residuals from it. Reconstruct
that base before integration. This convention is documented in ComfyUI's
`comfy/ldm/minimax/model.py`, `_pdd_head`, pinned at
`b1a247c9cf9dbfda4b2c0a0c1d50950667eeaa0c`. The local checkpoint's head RMS
magnitudes corroborate it (video .0804 base versus .0027–.0063 residuals).
An initial smoke render treated residuals as absolute heads and produced
noise; it is retained as a failed diagnostic, not a successful generation.

VSA uses 128-token 4×4×8 tiles, 90% target sparsity and 10% keep per
reference-video region; images, text and audio stay dense. Maestro's native
Triton implementation supports consumer CUDA hardware; it does not claim
the official Blackwell NVFP4 kernel's performance or quantized numerics.
Ordered references use the existing native Ref2VA conditioner and VAEs.
Start/end conditions, frozen target soundtrack, edit masks, continuation,
Director, additional adapters, First Block Cache, SLA and Veda are excluded. Weights
are fetched on demand and their pinned digest is checked before allocation.

### OmniRef validation (20 October 2026)

211 selected CPU regressions passed (two opt-in CUDA tests skipped in that
pass). A separate 15-test attention run included both CUDA tests, comparing
FP16/BF16 64- and 128-token kernels with explicit CPU attention oracles.
The UI type-check/build and undefined-name checks passed.

After the residual-head correction, two 4090 smoke cases produced fully
software-decoded 864×480, 124-frame, 24 fps clips with generated audio:
two image references (job `b6bd2617`, 86 s active generation / 144 s job
elapsed, sampled device-wide peak 12.79 GiB) and an image/video/audio mix
(job `b2e9db5d`, 243 s active / 261 s job elapsed, 13.69 GiB). Both used
seed 424242, eight forwards, Q2_K encoder, full INT8 VAE, Auto-tune Profile 2,
8 GiB transformer allowance and head splitting off. Saved preferences were
unchanged. Representative frames show a driver and white car in a coastal
tracking shot, without the initial diagnostic's noise.

These are functional smoke checks, not controlled throughput or quality
benchmarks. A short CPU regression pass overlapped the mixed-media case.
Stills and decoding cannot certify continuous motion, voice/sound-reference
fidelity or exact identity preservation. No 8/12 GB generation or real-time
performance was established. Existing H3 duration recommendations remain
unchanged; this experimental model does not inherit their benchmark tiers.

### Official-format checks (20 October 2026, after public release)

The four public checkpoint headers and LFS size/SHA pins were fetched with
bounded HTTP requests. Actual header contracts and schemas were inspected,
including fused QKV, all 50 compression gates, FP32 output banks and rank-8
AdaLN. No full BF16, FP8 or NVFP4 model was downloaded for this check.
The integrated CPU pass covered 50 recipe, pin, metadata, geometry, format
selection, attention, quantization, AdaLN and split/fused loader regressions
(two optional CUDA attention tests skipped in that pass).

Separate real-format checks copied only the first FP8 and NVFP4 compression
gate, then loaded it through the installed MMGP runtime on an RTX 5070 Laptop
GPU, SM120. FP8 loaded `ScaledFP8WeightTensor`; NVFP4 loaded
`NVFP4WeightTensor` and executed the lightx2v native kernel. Its comparison
with quantized-input/dequantized-weight math measured maximum absolute error
0.0009765625 and mean absolute error 0.0000587. Preferences and running Maestro
were preserved. The first private test omitted the application startup's
quantization-handler registration and failed before NVFP4 execution; that
test setup was corrected. These checks do not certify complete generation,
reference fidelity or 5090 throughput for the new variants.

### Separate T2AV V2 validation

Separate T2AV V2 validation on 20 October 2026: CPU recipe, input/capability, projection and
duration regressions passed, as did a small FP16/BF16 CUDA comparison against
the explicit attention oracle. A corrected full-prompt smoke render on an
RTX 4090 (24 GB; 128 GB system RAM; Windows; PyTorch 2.10/CUDA 13; Triton 3.6)
produced and fully decoded 124 frames at 864×480/24 fps plus audio, seed 424242,
eight Euler forwards, Q2_K encoder, full INT8 VAE and Auto-tune Profile 2.
It took about 50 seconds in denoising and 63 seconds of active generation;
the cold job including loading took 122 seconds. Sampled device-wide peak
VRAM was 12.63 GiB with an 8 GiB transformer allowance; this is not a minimum
VRAM requirement. Five representative stills showed a coherent traveler,
camera arc and birds; continuous motion and speech fidelity were not certified.
An earlier diagnostic exposed generic line splitting, which was fixed by
preserving the complete structured prompt as one native request. These are
smoke checks, not comparative performance or lower-memory benchmarks.

Viggle Animate adapts Wan2GP v12.72 at
`057f9ecab9ad57dfbec9768b2daf7a4426ce986c`: fixed prompt conditioning,
video-then-edited-image reference ordering, target-only control slicing after
the motion-history prefix, three Euler evaluations with shift 3, and fixed
124-frame windows with 18-frame overlap. The dedicated transformer, fixed
prompt and distillation adapter are pinned to
`DeepBeepMeep/MiniMax-H3@630975da14fd3c84169d6cd5092da3ce1287d137`.
The upstream rank-8 adapter map is pinned to the same Wan2GP commit and verified
with SHA-256 `89081fd235e9deff56274ab76b1b01fa4d9aeebcbd0746295b0fd086cf0e0fec`.
Maestro retains its own queue, media pipeline, MMGP offloading and continuation
implementation. See `docs/Viggle-Animate.md` and
`docs/development/Viggle-and-Studio-Controls.md` for usage and validation.

The September 2026 VDN, grouped-row masking/outpainting, audio-only voice
preset, and six-step audio refinement additions adapt Wan2GP v12.71 at
`1e1dd2757f24923f008593d9d4ec09062234be20` to Maestro's existing pipeline.
VDN module and adapter downloads use the immutable Hugging Face revision
`DeepBeepMeep/MiniMax-H3@630975da14fd3c84169d6cd5092da3ce1287d137`.
The earlier `304d34f7751f8ba9ca0eb55d5d10044234cdbfe2` pin stopped serving
these files after the Hub history consolidation. Maestro redirects only the
four known VDN module/default-adapter/Turbo-adapter URLs from that obsolete
pin, including URLs retained in saved presets. Existing local filenames
remain valid. On 21 October 2026, all four replacement assets were checked
anonymously against Hub LFS metadata and bounded download requests. Their
size and SHA-256 identities are recorded in `vdn_assets.py`.
See `THIRD_PARTY_NOTICES.md`, `app/LICENSES/WanGP-Community-2.0.txt`, and
`docs/development/wan2gp-12-71-port-plan.md` for licensing, integration and
local validation details.

The H3 Voice Audio extension adapts `models/minimax_h3/dialogue.py` at that
same revision, including Whisper medium loading and speech-boundary alignment.
Maestro adds automatic splitting of long monologues, complete-script budget
validation, its shared 2.8 words/second planning pace (maximum 3), private
temporary voice references, cancellation and native queue progress. Native
audio passes are limited to 45 seconds and assembled outputs to 300 seconds;
15 seconds remains the default. Whisper assets reuse the existing Scenema
`whisper_medium` download and MMGP offloading integration.

The video VAE, audio VAE, scheduler, FL2VA packing, and Ref2VA reference
preparation/packing in this directory are derived from the Hugging Face
Diffusers MiniMax H3 implementation at commit
`abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc`.

The optional learned 2x video decoder follows the MiniMax H3 X2 VAE integration
in Wan2GP commit `6479db36bdc2619a904a852bba9c2d78e1a83f82`. Its INT8 ConvRot
checkpoint is downloaded on demand from
`DeepBeepMeep/MiniMax-H3@630975da14fd3c84169d6cd5092da3ce1287d137`
(`minimax_h3/MiniMax-H3-X2-Detail-v1_int8_convrot.safetensors`, 2,834,844,787
bytes, SHA-256
`7ca1a1ad298e06e430da5c18bf721af4e0dcdf7f4ec7283391520bd0b6df7387`). The
original `adc81ccb71352192214d83d5fafb9487e860be39` download revision now
returns 404 for these files. The replacement revision preserves the same
checkpoint bytes and companion license/notice files. The
loader keeps the selected standard H3 VAE's encoder, quantization projection,
and latent conditioning path, then loads only `decoder.*` from the X2 file. It
filters the checkpoint's `detail_b32.*` branch because that path requires
source-image encoder features and is not available for generated H3 latents.
The finalized packed decoder output is blended on its native grid, pixel-shuffled
by two, normalized, and copied to CPU uint8 in short temporal chunks.

The upstream X2 asset's original repository is
`speach1sdef178/MiniMax-H3-X2-Detail-VAE@af8c92d267c6849fec5032c35a65d5766737b338`.
The downloaded `MiniMax-H3-X2-Detail-v1.LICENSE` and
`MiniMax-H3-X2-Detail-v1.NOTICE` are retained alongside the optional weights.
That Community License states that its grant excludes the United States, the
European Union, the United Kingdom, and South Korea and directs prospective
deployers in those territories to contact MiniMax. Review the included terms
before using or deploying the optional decoder.

Those files retain their upstream Apache-2.0 copyright and license headers.
Maestro-specific model loading, packing, memory management, and Studio
integration are implemented separately in this directory.

The default runtime stack is pinned to:

- `MiniMaxAI/MiniMax-H3` commit `5d9b308a59ab12e67147f191e184baf704185bd1`
  for the official processor and text-encoder configuration.
- `Comfy-Org/MiniMax-H3` commit `0543966fbdce5ba05709a8f2031c94bdba629b4a`
  for the scaled-FP8 FL2VA and Ref2VA transformers, NVFP4-AWQ conditioner,
  and compact video/audio VAE checkpoints.
- `DeepBeepMeep/MiniMax-H3` commit
  `fec7846aef352e58a1cfb699455e3d104281e68b` for the full 33B FL2VA/Ref2VA
  checkpoints and the optional BF16, Quanto INT8, and GGUF Qwen3-VL
  conditioners.

The dual full/pruned checkpoint probe, full-checkpoint split Q/K/V loading,
fused pruned-checkpoint attention, ConvRot QKV restoration, independent Qwen
language/vision profiling, and selectable text-encoder catalog adapt the
MiniMax H3 implementation shipped in WanGP v12.41
(`4ed4c744a396e43294f851f35cab769e11a89f2d`). Maestro retains its existing
compact Comfy checkpoint loader and local Studio APIs around those
memory-management pieces.

The INT8 ConvRot consumer is adapted from WanGP's
`shared/qtypes/int8_convrot.py` at `b382d0940cdbab29cff5d33301b34b337ad5517e`
(handler revision `6b92c54f92bde24d6d309d6f61249353b0ec783d`). The ConvRot export stores
its fused QKV rows and row scales in logical grouped `[Q, K, V]` order, which
Maestro splits contiguously into independent streamable projections. Active
LoRAs retain ConvRot's native activation rotation for the quantized base branch
and apply their deltas from the original, unrotated module input.

The Ref2VA runtime preserves the official ordered reference presentation,
shared rotary clock, soundtrack-before-video row ordering, sampled/noised
visual VAE conditioning, clean audio conditioning, and target-only denoising.
Maestro defaults reference images to an output-matched, downscale-only detail
policy for consumer GPUs; the official 2048-pixel-short-edge preparation is
also available as the Maximum reference detail option. Reference videos now
follow the same output-matched pixel-area policy by default; this prevents a
480p/544p generation from silently encoding its reference at a 768-pixel short
edge and more than doubling the packed attention working set.

Optional low-step support for LarryVRH's MiniMax H3 Turbo LoRA follows the
adapter and custom-sampler contract published at
`larryvrh/MiniMax-H3-Turbo-Lora` and
`Larryvrh/ComfyUI-MiniMax-H3-Turbo` (inspected 2026-08-06). Maestro's native
runtime already advances video and audio on the required independent shift-12
and shift-3 schedules. The adapter's logical grouped fused-QKV updates are split
contiguously with the instantiated H3 module, and the requested 4/6/8 steps are
treated as actual model evaluations. Full/Pruned AdaLN LoRA conversion is
adapted from WanGP commit `1830091bf4b27df2f901920d55b1fb748f33e7eb`.
Its small FL2VA/Ref2VA rank-8 and rank-64 affine packages are downloaded from
that immutable revision, size/hash verified, and stored in the user's checkpoint
area on first H3 LoRA use. They are not redistributed with Maestro.

The H3 shared-attention path, early release of projection inputs, bounded
projection chunks, and optional First Block Cache behavior also follow the
memory-conscious H3 runtime in WanGP commit
`1830091bf4b27df2f901920d55b1fb748f33e7eb`. Maestro adapts those ideas to its
batched Diffusers-style row layout, uses the exact generated-row boundary for
FL2VA and Ref2VA cache residuals, and keeps First Block Cache experimental and
disabled by default.

The optional H3 Sol Engine integration is adapted from WanGP commit
`7e45fe7e21105807b43f6285827d9ebb5fa72906`. The portable kernels under
`shared/sol_attn/triton_kernels` track NVIDIA Sana Sol-Attn commit
`46031940ba8af5d18054217e571149579424c0b1`; the optimized INT8-QK kernels
under `shared/sol_attn/saganaki` track ComfyUI-sol-attn v0.5.2 commit
`e2fc225`. Both sources are Apache-2.0 and retain their source headers and
bundled license. Maestro restricts this backend to compatible MiniMax H3
main-DiT attention, protects the complete packed conditioning prefix, and
falls back to the selected dense Sage/SDPA backend on unsupported calls or
kernel failure. RTX 40 users receive it through an isolated optional runtime;
the proven RTX 20/30/40 default environment is not upgraded in place.

The full FL2VA sliding-window continuation contract is adapted from WanGP
v12.44's H3 feature commit
`5c8b4ac3c5e15135b6510d9b6d4d57002e4bb5e4`, with follow-up fixes through
`639ee1351e5b57c5992903690199719607c3700e` (WanGP 12.441). A legal
`17*n+1` overlap is split into complete 17-frame visual-history chunks and one
regenerated boundary frame; the matching generated stereo-audio tail is VAE
encoded on the same rotary timeline. Maestro retains its exact-duration outer
assembler, prompt planner, and VRAM-aware pass limits around that native H3
conditioning. The same upstream follow-up also supplies the streaming video
VAE tile decoder used here, which decodes one tile at a time and preserves
already blended horizontal/vertical tails to reduce peak VRAM and avoid
multi-tile corner seams.

Phase 3 extends that v12.44 contract to Ref2VA. Each continuation pass keeps
the user's canonical ordered image/video/audio references, presents the prior
window boundary as an additional leading picture to Qwen, and packs clean
multi-frame video plus matching stereo-audio history ahead of the reference
rows. Picture labels in the generated Context-IR are shifted only for that
temporary boundary; canonical Video and Audio numbering remains stable.
Maestro's Multi-window sequence toggle uses this native path when continuity is
enabled and retains independent hard-cut clips when it is disabled. Window
prompts may come from the story-planning LLM or from an exact user-authored
one-non-empty-line-per-window mapping; the latter never loads the planner.

Phase 4 adapts WanGP v12.44's FL2VA multiple-frame injection contract. Extra
Studio Frame tiles are paired with exact target-frame anchors after carried
visual history is removed, while the H3 window planner gives those same images
window-local Picture numbers and timestamps. Start, final-end, continuation,
and injected pictures therefore share one deterministic presentation order
between Qwen conditioning, packed model conditions, saved plans, and resumed
jobs. Ref2VA deliberately does not advertise this FL2VA-only capability.

Phase 5 adapts WanGP v12.44's FL2VA media-source contract. An uploaded
soundtrack or the soundtrack extracted from a Control Video is VAE encoded
into clean, frozen target-audio rows so it can drive newly generated video.
The reciprocal video-to-audio mode VAE encodes the complete Control Video as
frozen target-video rows, skips visual denoising and decoding, and generates a
new synchronized soundtrack while returning the source pictures unchanged.
Maestro keeps its ordinary multi-window assembler around both paths,
de-duplicates the carried visual overlap from each combined guide window, and
uses H3's `17*n+5` frame offset when preparing Control Video windows.

Phase 6 adapts WanGP v12.44's FL2VA video-to-video editing contract. A Control
Video is VAE encoded as a clean reconstruction target and re-injected at the
matching scheduler noise level. Whole-frame mode uses denoising strength to
choose how late generation begins; masked modes use a white-edit/black-preserve
mask and masking strength to control how long protected pixels remain locked.
The same source can retain its soundtrack, generate a new synchronized stereo
track, or be driven by a separate soundtrack. Ref2VA remains reference-guided
and deliberately does not advertise this FL2VA-only editing pipeline.

The experimental Turbo selector is driven by `turbo_presets.json`; mutable
Hugging Face `main` is never used for generation. Maestro's current default is
the upstream-recommended `minimax_h3_turbo_v4_step600_ema.safetensors` at
immutable revision `afc0346516372a17162c14df3c5264de1d9aa1c0` (SHA-256
`5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3`),
six evaluations, and strength 1.00. The previous
`minimax_h3_turbo_4step_ckpt500.safetensors` preset remains available at
revision `7a44622816e16032cb0b6d044d8820da39a1dfdc` (SHA-256
`82d0acff583b04ad9a4238a7440b584b56094bfb7c4fdb2981f67c7a4784b62d`)
as a legacy rollback at six evaluations and strength 0.50. Users can tune
either selected adapter in Advanced. Both are listed virtually for Full and
Pruned checkpoints, downloaded and hash-verified on first use, and atomically
published with an integrity receipt.

The selector also exposes Alibaba PAI's Apache-2.0 MiniMax H3 Acc-LoRAs for
their matching FL2VA and Ref2VA workflows. These adapters use Parallel
Decoding Distillation rather than an ordinary low-rank-only sampler: 32
interval-specific video/audio output heads are fused four at a time into eight
model evaluations. Maestro adapts the official `minimax_h3_pdd.py` recipe from
`alibaba-pai/MiniMax-H3-Acc-LoRAs` revision
`78db175437ee05df7ec492ee366f01b68b8d20e6`, keeps the backbone updates in
MMGP's streamed LoRA path, and retains only the eight fused output-head pairs
in CPU memory. The head plans are rebuilt from the exact runtime video and
audio sigma boundaries, matching WanGP's PDD implementation introduced in
v12.645 instead of assuming that every future scheduler uses the original
uniform eight-evaluation grid. The FL2VA and Ref2VA files are pinned separately by immutable
revision, size, and Hugging Face LFS SHA-256; neither can appear for the wrong
workflow. Both PDD presets remain available on Full and Pruned checkpoints;
Maestro converts their canonical AdaLN adapters to the selected checkpoint at
load time. Ref2VA PDD defaults to Diffusers' official 2048px-short-edge
reference preparation, while an explicit Match output selection remains a
lower-memory, no-upscale option. Full and Pruned tests showed the same
composition promotion when the distilled Ref2VA interval heads were fed the
matched-detail references, ruling out checkpoint width as the cause. Alibaba's
published Ref2VA PDD examples currently cover 5.18s and 10.13s clips; Maestro
allows the ordinary H3 maximum but labels longer accelerated clips as
experimental and reports that fact in the generation log.

`.github/workflows/h3-turbo-upstream.yml` checks the public repository revision
daily and opens or updates one review issue when upstream changes. It never
edits the manifest or promotes weights automatically; promotion requires an
immutable revision, size/Hugging Face LFS SHA-256 capture (not the Xet hash or
download response ETag), and Full/Pruned, First/Last/Omni,
static/fast-motion, and audio testing.

Those model weights are downloaded at runtime and are not distributed in the
Maestro repository. They remain governed by their respective model terms and
any authorization or waiver required for the user's location.

The October 2026 repeat-conditioning cache and bounded VAE tile batching follow
Wan2GP v17.17 at `6479db36bdc2619a904a852bba9c2d78e1a83f82`, specifically
`models/minimax_h3/pipeline.py` and
`models/minimax_h3/components/video_autoencoder.py`. Maestro keeps a small,
CPU-only plain-text embedding cache and bypasses multimodal/Viggle conditioning.
It batches at most two VAE spatial tiles when the running CUDA device has
sufficient memory, retaining the serial path on smaller or busy devices.
The existing tile placement, temporal decoding and overlap blending are retained.
See `THIRD_PARTY_NOTICES.md`, `app/LICENSES/WanGP-Community-2.0.txt` and
`docs/development/h3-cross-app-benchmark.md` for provenance and validation scope.
