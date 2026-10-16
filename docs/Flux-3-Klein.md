# FLUX 3 Image Klein

Maestro's `flux3_klein` image model uses Black Forest Labs' distilled 5.6B
checkpoint and native 16-step flow-map Euler schedule. It supports text-to-image
and editing with up to ten reference images in **Studio → Image → Generate**.
Select **Flux 3 Klein 5.6B** (Distilled, 16 steps) or **Flux 3 Klein Base 5.6B** (50 steps, guidance 3.5), type a description, and generate. To edit, add the
source image under references and describe the desired change. Additional images
can supply other subjects or visual references.

## Prompt enhancement

Enhance uses separate [generation](../app/services/llm_guides/enhance/flux3_image.md)
and [reference/edit](../app/services/llm_guides/enhance/flux3_image_edit.md) guides
for Flux 3. A generation prompt describes the intended still image. An edit
states the final change and the relevant parts to preserve; a new composition
describes the whole target scene without locking an identity donor's original
background or pose. User-specified style, character counts, species, placement,
ownership and exact quoted lettering remain part of the brief.

Maestro supplies an ordered map to the LLM: the first UI reference is
`<ref_image_0>`, the second is `<ref_image_1>`, through `<ref_image_9>` for the
tenth. References can supply characters, objects, clothing, scenes or style;
the number of files is not automatically the number of people. The enhancer
uses only attached inputs and does not invent missing reference roles. The
output cleaner preserves legitimate repeated binding phrases rather than
truncating a multi-reference caption as a repetition loop.

These rules draw on the [Hugging Face Flux 3 article](https://huggingface.co/blog/flux-3),
its [source](https://github.com/huggingface/blog/blob/main/flux-3.md), the
[BFL prompting guide](https://docs.bfl.ai/guides/prompting_summary), and the
[official reference upsampler conventions](https://github.com/black-forest-labs/flux-3-image-klein/blob/45e57c9fb06f9751cdced29a65c5d2171003be05/src/flux3/pup_prompts.json).
Maestro applies them through the selected enhancement LLM; it does not download
the separate BFL 9B prompt upsampler. Prompt Helper creates optional layout JSON with element IDs, descriptions and
boxes. Enhancement preserves user-supplied layout and reference bindings; it
does not invent boxes for an ordinary caption.

## Installation

Models download on demand from [WanGP's public Flux 3 repository](https://huggingface.co/DeepBeepMeep/Flux3).
**No Hugging Face login or prior gate acceptance is required for these downloads.**
The BFL model license still applies; its [license](../app/models/flux3/WEIGHTS_LICENSE.md)
and [notice](../app/models/flux3/WEIGHTS_NOTICE.md) are included with Maestro.
The optional BFL prompt upsampler is not downloaded.

Transformer filenames distinguish Distilled (`_sd_`) from Base. Their pinned
public sources include SHA-256 digests and byte sizes. Components use public
revision `1e7c96984937fef03326e3b297fb44a28c30736d`; original downloaded weights
are retained. Public Qwen encoder files live under
`ckpts/flux3_klein/Qwen3-VL-4B-Instruct/`, and the public VAE under `ckpts/flux3/`.
Linked model folders and download cancellation remain supported.

## Settings and scope

| Variant | Starting steps | Guidance | Precision |
| --- | ---: | ---: | --- |
| Distilled | 16 | 1 (no CFG) | BF16 or INT8 ConvRot |
| Base | 50 | 3.5 (CFG) | BF16 or INT8 ConvRot |
| NVFP4 variants | Same recipe as above | Same as above | Supported RTX 50 GPU and NVFP4 kernel required |

Start at 1024 × 1024; dimensions follow the 16-pixel grid. INT8 reduces weight
memory and download size, but many references and large canvases still need
activation memory. Select **Ordered references** to retain the chosen output
shape, or **Main reference sets output shape** to derive it from Image 1.
Reference IDs are zero-based even though UI labels start at Image 1.

## Prompt Helper

Open **Prompt Helper** beside the image prompt. Write a complete scene caption,
add an element, describe it and drag its output box. Drag inside to move it;
drag the bottom-right corner to resize. Each element has an `<id>` caption tag.
Choose an uploaded image as its content source and switch to **Source region**
to identify the relevant area. **Remove this reference element** records a null
target. **Save prompt** writes the caption and complete layout JSON together;
reopening preserves extra fields and comments.

Boxes use `[top,left,bottom,right]`, integer corners on a 0–1000 grid. They guide
composition rather than imposing exact pixel masks. Text-to-image elements
use `bbox`; editing elements use `from`, `src_bbox`, and `tgt_bbox`. Example:

```text
A cream poster places the same dog <dog> beneath a navy title "BEACH DAY" <title>.
[{"id":"dog","from":"ref_image_0","src_bbox":[0,0,1000,1000],"tgt_bbox":[300,200,950,800],"desc":"The same golden dog"},{"id":"title","from":"target","src_bbox":null,"tgt_bbox":[80,100,220,900],"desc":"Navy lettering reading \"BEACH DAY\""}]
```

Native masked inpainting/outpainting and Flux 1/2 LoRAs remain unavailable.
Composition, exact lettering and reference preservation are generative rather
than guaranteed.

## Verified generation tests

On October 15, the official BF16 checkpoint passed three real Maestro API tests
on the RTX 4090 / 24-GB VRAM / 128-GB RAM workstation. All outputs were
1024 × 1024 RGB images at 16 steps and were reviewed visually.

| Test | API completion time | Denoising | Sampled peak VRAM |
| --- | ---: | ---: | ---: |
| Text-to-image, including initial model loading | 44.5 s | 8.2 s | 13.24 GB |
| Single-reference color edit, model already loaded | 25.6 s | 16.1 s | 14.14 GB |
| Two-reference composition, model already loaded | 35.9 s | 28.3 s | 15.08 GB |

VRAM figures are Maestro's device-wide NVML samples, including other GPU use;
brief peaks can be missed. API times include queue handoff and output saving.
The tests preserved the requested label and composition during the color edit
and combined both supplied objects in the two-reference image.

A subsequent ten-reference test at 1280 × 720 and 16 steps also completed on
the same workstation: 75.2 seconds API completion, 67.5 seconds denoising and
17.44 GB sampled peak VRAM. All ten supplied images reached native conditioning;
the result had nine visible people, so this is a runtime/capacity check rather
than full ten-subject fidelity acceptance. These earlier BF16 tests do not validate 8/12-GB GPUs, the new quantized variants or larger output sizes.

When switching from another image model or loading older settings, Maestro
normalizes Flux 3's reference selector to its native reference mode. Legacy
reference flags must not cause an otherwise valid edit to fail before generation.

A separate full-resolution numerical audit used one text-to-image prompt at
1088 × 1920 and 16 steps. With the DiT attention backend matched to BFL's
cuDNN backend, Maestro's real MMGP runtime matched all sixteen denoising steps
and every final output pixel exactly. The normal PyTorch efficient-attention
path produced a visually similar image with small numerical differences.
Both paths kept the VAE in FP32. This verifies that tested workload in the
existing runtime, rather than a general realism advantage over Flux 2.

The October 15 v18.10 port also verifies public INT8 ConvRot loading, including
the Qwen word-embedding rotation used by these weights. Its text features match
the original BF16 encoder closely (cosine 0.999967 for the tested prompt).
Real 4090 adapter checks produced the requested text-only images with both
Distilled at 16 steps and Base at 50 steps/CFG 3.5. The isolated 512-square
Profile 4 runs took 8.5 and 45.5 seconds to render, respectively. These are
adapter timings, not full API timings or a performance claim for 12-GB GPUs.

A fresh Distilled 1024-square test rendered in 8.7 seconds, with 4.34 GiB peak
PyTorch allocated memory and 7.40 GiB reserved. A 512-square reference test
preserved the two supplied colored shapes and their requested left/right
arrangement. An earlier vague still-life request changed the objects; reference
capacity is not a guarantee of content preservation. Device-wide usage can
exceed these allocator figures. The NVFP4 variants require separate hardware
validation.

## API

Submit through the existing `/api/v1/generate` endpoint, for example:

```json
{
  "model_type": "flux3_klein",
  "generation_mode": "image",
  "image_mode": 1,
  "prompt": "A tiny orange robot tending a rooftop garden at sunrise",
  "resolution": "1024x1024",
  "num_inference_steps": 16,
  "guidance_scale": 1.0,
  "video_prompt_type": "I",
  "image_refs": [],
  "seed": 424242
}
```

For editing, put uploaded image paths in `image_refs` and retain
`video_prompt_type: "I"`. Set it to an empty string to disable attached references.
Download status and cancellation use the standard model download endpoints.

The [reference inference code](https://github.com/black-forest-labs/flux-3-image-klein)
is Apache 2.0; the weights use the
[FLUX Kommunity License](https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/main/LICENSE.md).
