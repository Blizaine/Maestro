# FLUX 3 Image Klein

Maestro's `flux3_klein` image model uses Black Forest Labs' distilled 5.6B
checkpoint and native 16-step flow-map Euler schedule. It supports text-to-image
and editing with up to ten reference images in **Studio → Image → Generate**.
Select **Flux 3 Klein 5.6B**, type a description, and generate. To edit, add the
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
the separate BFL 9B prompt upsampler. Layout JSON and bounding-box annotations
are outside the verified adapter scope, so the enhancer expresses layout in
prose. This guide change requires one normal Maestro restart before testing.

## Installation

Open the [official model page](https://huggingface.co/black-forest-labs/flux-3-image-klein)
and request access while signed into your Hugging Face account. Use that same
account in Maestro's Hugging Face login. Files download only when selected for
generation or explicitly downloaded through the model picker. The BF16 model,
Qwen3-VL-4B encoder and VAE total about 20 GB. The optional BFL prompt upsampler
is not downloaded; Maestro's existing prompt enhancement remains available.

The main file is stored as `flux3_image_klein_sd_bf16.safetensors` to avoid
colliding with other models named `model.safetensors`. The download checks the
official checkpoint's SHA-256 and byte size. Shared components live in
`ckpts/flux3_klein/`; all files use pinned official revision
`8454017f9ebaada30c998fccd9f67a80c7580f68`. Linked checkpoint folders and normal
download cancellation continue to work.

## Settings and scope

Start at 1024 × 1024 and 16 steps. This distilled model does not use classifier-free
guidance or a negative prompt. Width and height follow the native 16-pixel grid.
Maestro uses MMGP for transformer, encoder and VAE placement. Large images and
many references need more activation memory; actual hardware support must be
established by generation tests rather than the checkpoint's file size.

This adapter initially uses BF16. BFL's FP8/NVFP4 and Comfy INT8 variants require
separate verified checkpoint mappings. Flux 1/2 adapters, masked inpainting,
outpainting and bounding-box layout controls are not exposed. Their presence in
the upstream model does not imply support in this initial Maestro adapter.

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
than full ten-subject fidelity acceptance. These tests do not validate 8/12-GB
GPUs or larger output sizes.

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
