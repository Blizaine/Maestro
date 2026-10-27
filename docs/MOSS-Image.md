# MOSS Image

MOSS Image 5B Base and SFT are available in **Studio → Image → Generate**.
SFT is added to the model picker once on update; enable Base through **Manage
models** or **Settings → Performance → Enabled Models → Image → Generate**.
Saved selected models and existing visibility choices are retained.

Start at **1024 × 1024, 50 steps, guidance 3, flow shift 5, batch 1**. These
checkpoints are not step-distilled. SFT adds curated-data post-training to Base;
both use the same architecture and generation recipe.

## Reference editing

The separate **MOSS Image Edit SFT** and **MOSS Image Edit HQ v5** entries use
the SFT transformer with a required editing adapter. In **Studio → Image →
Generate**, add one to four source images in the desired order, select an editor,
and describe what to change and what to keep. These are community editors;
Base and SFT text-to-image entries retain their existing behavior.

Start with 50 Euler steps, shift 5 and batch 1. Edit SFT defaults to guidance 4;
HQ v5 defaults to guidance 3. The selected resolution sets the output canvas.
HQ fits each reference within 384 pixels per side; Edit SFT fits within 1024.
Fitting retains the full image and rounds dimensions to the VAE's 16-pixel
lattice. Small details can be lost when references are reduced.

For a localized edit, try: `Change only the jacket to dark blue. Preserve the
person's face, hair, pose, background and lighting.` For several sources, say
`Keep the person from image 1, use the jacket from image 2, and preserve the room
from image 3.` Explain each source's role. HQ is primarily intended for bounded
single-source edits; multi-image compositing is less established. Neither
editor guarantees unchanged pixels or provides masks, inpainting or outpainting.
The layout Prompt Helper remains exclusive to text-to-image generation.

Prompt enhancement is optional. The editor guide refines the requested changes
and preservation instructions instead of compiling a new scene or layout boxes.
The required editor adapter loads automatically; it is excluded from optional
LoRA choices to avoid applying it twice or mixing the two editor contracts.
Additional MOSS style adapters remain separate and can affect preservation.

Both editors reuse already downloaded SFT/text-encoder files. They also require
the full Flux2 VAE's image encoder and their own editing adapter; the small MOSS
decoder and normalization statistics stay in use. Model readiness checks include
these companions. The HQ preset pins the author's recommended armV5, rather
than experimental v6. See the [HQ model card](https://huggingface.co/akhaliq/MOSS-Image-Edit-HQ/tree/b4114b929f5e24904ae72697d9a719810aac1da9)
and [component notices](../app/models/moss_image/NOTICE). The original chinoll
adapter repository does not declare weight-license terms; its terms should not
be inferred from the SFT base's Apache-2.0 license.

### Reference cache

The editor's advanced controls offer **Disabled**, **Enabled**, **INT8** and
**INT8 in RAM**. Disabled is the default and recomputes reference features each
step. Enabled retains full-precision reference K/V for this generation. INT8
compresses those cached tensors, with a numerical-precision tradeoff; it does
not quantize the whole model or halve total job memory. RAM mode streams each
layer's cache and may be slower because of transfers.

Caches are local to one request and released on completion, cancellation or
failure. HQ keeps its guidance branches independent. Step-varying LoRAs require
fresh reference projections, so caching is disabled for those schedules.
The first denoising pass is uncached so MMGP can settle transformer residency;
the memory check then runs before caches are populated on subsequent passes.
Conservative memory checks can decline caching; the runtime log reports the
requested mode, actual mode and fallback reason. A selected cache option is not
proof that acceleration ran. Start at a smaller canvas and batch 1 on limited
VRAM; GPU speed and quality measurements for the new editors remain pending.

## Text-to-image prompts

Write a natural-language caption or tags. Describe the subject, composition,
lighting and materials. Quote exact requested lettering. Prompt enhancement is
optional and off by default; the enhancer follows MOSS's own caption/layout rules.

**Prompt Helper** draws and edits boxes on the selected canvas. Add Object or
Character or Text items, describe each item, enter exact visible text, and save the result into
the prompt. Drag boxes to move them or use their edges/corners to resize. Boxes
use `[ymin, xmin, ymax, xmax]` in pixels measured from the canvas center. For
1024 × 1024, both axes run from -512 to 512; rectangular canvases use their actual
width and height. **Fit Boxes to Canvas** brings existing boxes inside the current
canvas while retaining their size where possible. Changing resolution does not
silently rewrite the saved prompt. Extra JSON fields and `#` comment lines are
preserved. Boxes guide composition, rather than providing exact pixel masks.
Choose **Canvas arrangement → Single image, Multiple panels, or Collage** to
describe the intended composition. This guides the model; it does not divide the
canvas into exact pixel masks.

## Getting the most from MOSS

The [official Base user guide](https://github.com/OpenMOSS/MOSS-Image/blob/95559c93ce9f85967de70435deefc1153a1a2fab/USERGUIDE.md)
recommends at least 50 steps, Euler, shift 5, about 1–2 megapixels, and guidance
3–6. Maestro starts at guidance 3, which the guide says is generally sufficient
for long prompts. Try increasing it within that range for a short prompt that
misses details. Dimensions must be multiples of 16. These are starting points,
rather than a guarantee that more steps or guidance improve every image.

- Use **A Detailed Image Caption** for ordinary photographs and illustrations.
  Describe subjects and their relationships, framing, materials, lighting and
  intended style. Keep minimalist requests explicitly minimalist. The enhancer
  uses English descriptions by default unless another output language is requested.
- Use **An Image Layout with Bounding Boxes**, or ask the Studio enhancer for a
  layout, when precise text placement, multiple panels or relative object
  placement matters. Natural overlap and occlusion are allowed; a box does not
  instruct MOSS to draw a border or sticker.
- Supply exact visible words, punctuation, capitalization and line breaks. For
  example: `A ceramic mug on an ivory background. Dark-green text on the right
  reads "Take a moment" with "Make it yours" on the next line.` Text accuracy
  remains model-dependent. Quote only intended display text, distinguishing it
  from instructions or examples.
- If using tags, the guide suggests quality, score, medium/meta, year, character,
  series, artist, then general tags. Useful negative terms include `worst quality,
  bad anatomy, watermark, jpeg artifact`. Maestro leaves the saved negative prompt
  under your control; it does not silently add content exclusions.

Maestro follows the official
[prompt-enhancer skill](https://github.com/OpenMOSS/MOSS-Image/blob/95559c93ce9f85967de70435deefc1153a1a2fab/skills/moss-image-prompt-enhancer/SKILL.md)
and [layout draft specification](https://github.com/OpenMOSS/MOSS-Image/blob/95559c93ce9f85967de70435deefc1153a1a2fab/skills/moss-image-prompt-enhancer/references/layout.md).
New AI layout drafts use ordinary top-left pixel boxes; deterministic code checks
the schema and selected dimensions, encloses fractional edges, and converts to the
centered coordinates MOSS consumes. Width/height draft fields are removed before
generation. Invalid boxes are rejected rather than clipped. Studio enhancement
can make one repair attempt within the configured retry allowance. If it still
fails, it leaves the original prompt unchanged and reports the problem.
Existing compiled Prompt Helper layouts keep their centered format, extra fields,
comments and exact strings. The classic enhancer also compiles new drafts before
returning them and keeps literal `@` artist tags and email addresses in the prompt,
instead of interpreting them as system-instruction separators. Caption enhancement
preserves intentional style and counts and
avoids fabricated dates, venues, prices, placeholders or unsolicited lettering.

The current runtime already implements the recommended **two-pass negative-branch
skip at layer 22** when guidance is enabled (CFG other than 1). It does not expose
standard SLG or a selector for Off/22/24/26 yet. Those are potential quality
experiments, not features supplied by this prompt-enhancer update. The guide's
thinking-enabled VLM example is for captioning training images; it is separate
from the frozen generation text encoder and Maestro's interactive enhancer.

These releases are **text-to-image only**. They do not accept reference images,
start/end frames, pixel masks, or audio. Use another image model for reference
editing, Inpaint or Outpaint. Dimensions must be divisible by 16; about 1–2
megapixels is the suggested starting range, with an admission ceiling of 4,194,304
pixels. Larger canvases and batches use more memory.

## LoRAs

Open **Advanced → LoRAs & presets** to select compatible MOSS image-transformer
adapters and set their weights or per-step schedules. **Model Browser → LoRAs →
MOSS Image** searches both Civitai's MOSS Image (SFT) and MOSS Image Base labels.
Supported Civitai and Hugging Face imports use the shared `loras/moss_image`
folder, so adapters are available to both model entries.

Use the creator's recommended checkpoint and weight first. Base and SFT share
an architecture, but their post-training differs; the same adapter and weight
can produce different results. Adapters for unrelated models are not compatible.

## Downloads and memory

The **5B** name refers to the image transformer. A separate Qwen3.5-9B text
encoder and a small Flux2-family decoder are also required. Transformer files are
about 10.2 GB BF16 or 5.4 GB INT8; the encoder is about 15.9 GB BF16 or 9.0 GB
INT8. These are disk sizes, not VRAM requirements. Maestro uses its normal MMGP
offloading and saved precision settings; avoid assuming that a 5B model uses
only 5 GB of RAM or VRAM. Downloads occur only when selected or explicitly
requested. This integration does not require a Hugging Face account or gated
model access.

Both model entries use the public [WanGP prepared checkpoints](https://huggingface.co/DeepBeepMeep/Moss)
pinned to `0a508e986e920c8051b99df98d27329742fc3d0e`. Transformer downloads include
SHA-256 and byte-size checks. The port follows
[WanGP](https://github.com/deepbeepmeep/WanGP/tree/fdc63bdf89358fb13c93cf6b38693fd89a859f00/models/moss_image)
and [OpenMOSS MOSS-Image](https://github.com/OpenMOSS/MOSS-Image). Source/license
notices are retained in `app/models/moss_image`. The model weights are Apache 2.0;
WanGP integration portions retain their existing source notices.

## API example

Use the normal generation API with a dedicated workspace:

```json
{
  "model_type": "moss_image_sft",
  "generation_mode": "image",
  "image_mode": 1,
  "prompt": "A cream travel poster with blue mountains and the title \"EXPLORE\".",
  "resolution": "1024x1024",
  "num_inference_steps": 50,
  "guidance_scale": 3,
  "flow_shift": 5,
  "batch_size": 1,
  "seed": 424242,
  "video_prompt_type": "",
  "multi_prompts_gen_type": 2,
  "prompt_enhancer": "",
  "workspace": "MOSS-Image-Test"
}
```

POST this object to `/api/v1/generate`, then poll `/api/v1/status/{job_id}`. A layout
is serialized as the entire `prompt` string, retaining its newlines and JSON
braces. Prompt enhancement accepts the selected `resolution` so rectangular
layouts receive the correct bounds.

## Validation — 19 October 2026

Workstation 01 (RTX 4090 24 GB, 128 GB RAM) completed these normal-API tests
with the INT8 ConvRot transformer and Qwen text encoder, BF16 small decoder,
50 steps, guidance 3, flow shift 5, and base seed 424242. Auto-tune selected
profile 1; the user's saved Triton INT8 backend was retained.

| Test | Output | Job elapsed | Denoising | Sampled peak VRAM |
| --- | --- | ---: | ---: | ---: |
| SFT caption | 1024 × 1024, one image | 56.3 s | 44.5 s | 10.25 GB |
| Base caption | 1024 × 1024, one image | 57.3 s | 44.8 s | 10.16 GB |
| SFT layout | 1024 × 768, one image | 58.2 s | 44.5 s | 10.35 GB |
| SFT batch | 512 × 512, two images | 48.7 s | 45.6 s | 10.12 GB |

Job elapsed includes that job's loading, encoding and decoding; it excludes app
startup and downloads. These are smoke tests with different cache conditions,
not matched speed benchmarks. Device-wide memory sampling may miss brief peaks.
All five files decoded at the requested dimensions, retained their submitted
prompts and settings, and were visually reviewed. Both caption tests rendered
the requested “MAESTRO” sign; the layout placed the title above the mountain.
The two batch outputs were distinct. Native batches retain Maestro's existing
base-seed metadata and numbered-filename convention; MOSS samples image index
`i` with `base_seed + i`.

The first SFT attempt failed during text encoding because the existing SDPA
call path did not expand grouped key/value heads. The model-local compatibility
fix now handles the actual 16-query/4-KV-head configuration, with a CPU regression
and the successful subsequent GPU tests. No shared attention backend was changed.

Targeted Python regressions (85 tests), layout utility tests, desktop/mobile
browser flows, and the production UI build passed. Original system/service
settings fingerprints and the active workspace were unchanged; Maestro was
left running. Outputs are in workspace `MOSS-Image-Test-20261019`.

LoRA follow-up checks use a tiny MOSS transformer and the real MMGP CPU adapter
loader. Zero adapter weight matches the baseline; a nonzero scheduled weight
changes the output for both CFG branches and both batch images. Supported key
conversions and incompatible-shape rejection are covered as well. This verifies
the adapter plumbing, rather than a trained adapter's artistic effect.

BF16 transformer/encoder generation, LoRAs, and 8/12 GB GPU configurations have
not yet been GPU-tested. This port's smoke tests do not establish numerical
parity with the official pipeline or quality across arbitrary prompts.

The official-guide follow-up adds CPU coverage of draft compilation, rectangular
geometry, exact multilingual text, invalid-layout repair and legacy-layout
preservation, plus desktop/mobile checks for character and canvas-arrangement
controls. These checks do not measure the quality of live LLM rewrites or new
image generations; the existing GPU smoke tests above predate this follow-up.

## Reference-editing validation — 26 October 2026

The local editing implementation passed 248 offline CPU regressions spanning
MOSS generation/editing, LoRAs, cache storage and budgeting, Studio workflows,
and enhancement routing for MOSS, Flux and H3. Coverage includes cached versus
uncached tiny-model results, separate HQ guidance states and layer skipping,
INT8 numerical tolerance, changed-reference/request isolation, cancellation
cleanup, encoder latent packing, small overlapping tiles, and ordered enhancer
attachments. The production UI build and configured undefined-name checks for
the changed application modules also passed.

A separate meta-model audit compared the actual released safetensors JSON
headers to Maestro's native modules and LoRA preprocessor. Both chosen adapters
matched all 252 target modules / 504 matrices. All 108 full-VAE encoder keys and
143 small-decoder/normalization keys matched, with no missing keys or shape
mismatches. This audit fetched metadata and headers, without tensor payloads.

These checks establish integration and shape compatibility, not trained editing
quality or GPU acceleration. No new editor GPU jobs or full weight downloads
were run in this implementation pass. Visual editing, full-precision/INT8 cache
comparisons, memory use and actual speed on 24GB, 12GB and 8GB machines remain
pending. The earlier text-to-image GPU results above do not validate the new
reference editors.

## Reference-editing GPU follow-up — 27 October 2026

Eighteen bounded SFT/HQ editing and cache cases completed on RTX 4090 24GB,
RTX 3080 Ti 12GB, and RTX 5070 Laptop 8GB systems. The actual 512×512 outputs
were fully decoded on CPU and every contact sheet was reviewed. Original
preferences were preserved.

HQ's 16-bit, INT8-VRAM, and INT8-RAM caches activated on the 24GB and 12GB
machines. On 8GB all three requested cache modes safely declined the memory
budget and completed uncached. That is a functioning fallback, not an 8GB
cache acceleration result. Single-reference edits retained the robot while
changing the wall color. Two-reference composition added the cup but changed
its handle and the robot's antenna, with some cup shapes leaking into the
face. These fixtures do not establish general quality or a matched speedup;
the machines retained their different saved memory policies.

### HQ v5 conditioning correction

Further user testing exposed two gaps: a stale reference selector after switching
from Fibo could reject both editors, and the HQ port omitted image features in
the text encoder. The small robot outputs above exercised the old latent-only
HQ path; they do not establish faithful HQ v5 conditioning.

HQ v5 requires both VAE reference latents and Qwen3.5 vision features inside its
text input. Its separate vision component adds a roughly 912 MB download; the
existing MOSS text weights remain in use. Vision references use a 512×512 pixel
budget, while VAE references preserve more detail within a 1024-pixel maximum
side. These are separate from the selected output canvas and the VAE tile size.
SFT continues to use its original latent-only reference contract.
Both editors preserve each source image's native proportions through shared
preprocessing. A fixed output canvas does not letterbox the references; the
model's own bounded reference encoders perform their later resizing.

For HQ, start with one source and a localized edit. Multi-image compositing and
precise hand or text replacement remain quality-sensitive. A job completing
without an exception does not establish that its edit is visually successful.
