# MOSS Image

MOSS Image 5B Base and SFT are available in **Studio → Image → Generate**.
SFT is added to the model picker once on update; enable Base through **Manage
models** or **Settings → Performance → Enabled Models → Image → Generate**.
Saved selected models and existing visibility choices are retained.

Start at **1024 × 1024, 50 steps, guidance 3, flow shift 5, batch 1**. These
checkpoints are not step-distilled. SFT adds curated-data post-training to Base;
both use the same architecture and generation recipe.

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
