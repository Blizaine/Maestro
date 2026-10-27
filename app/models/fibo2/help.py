# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
INFOS = """## Fibo 2.0 Dev

Generate images from text, edit one image, combine up to four ordered references, or replace a masked region.

### Generation and References
- Without references, describe the finished image.
- **Main Image Ref** derives output dimensions from the first reference. Later images supply additional subjects or visual references.
- **Use References with Selected Output Dimensions** keeps the selected output size.
- Number references in order as `<image_1>`, `<image_2>`, and so on, and explain each image's role.

### Masked Editing
Choose Inpainting, provide a Control Image and its mask, and describe the requested change. White marks the region to replace. Use one source image without additional references. The model generates the whole result; pixels outside the mask are not guaranteed to remain identical. Masked editing always uses the full sampling schedule. Outpainting is not exposed.

### Settings
Start with 50 steps, guidance 5, and approximately one megapixel, such as 1024×1024. Dimensions use a 16-pixel grid. Larger batches and more references require more memory and time. The base model is not a few-step model.

Prompt enhancement is optional and off by default. Natural-language prompts and structured JSON captions are supported. Proofread generated lettering and check preservation of faces, small details and unedited regions.
"""

PROMPT_INFOS = """## Prompt Guide

### Generate
Describe subject, composition, material, lighting and style. Example: `A red ceramic teapot on a wooden table beside a window, soft morning light, product photograph.`

### Edit
State the change and what to preserve: `Change the jacket to red leather; retain the face, pose, background and lighting.`
For several references: `Put the person from <image_1> in the dress from <image_2>.`

### Structured Captions and Written Text
The structured enhancer produces Fibo JSON with scene fields and an optional `text_render` list. Each text entry's `text` is the exact visible string; WanGP also renders these strings into conditioning images. Example:
`{"short_description":"A cream travel poster with blue mountains","text_render":[{"text":"EXPLORE"}]}`

An edit may combine a desired-result caption with an instruction:
`{"structured_caption":{"short_description":"A red leather jacket"},"edit_instruction":"Replace the jacket and preserve the rest of the image."}`

Use valid JSON, with double quotes escaped inside strings. Whole lines beginning with # are comments and ignored for generation. Inline # characters and hexadecimal colors are preserved. Structured prompts describe semantic composition, not a pixel-coordinate canvas.
"""

DEEPY_INFOS = """Generate or edit images with model_type="fibo2_dev", image_mode=1, num_inference_steps=50, guidance_scale=5, resolution="1024x1024", batch_size=1, prompt_enhancer="". No references: video_prompt_type="". Up to 4 ordered image_refs: "KI" derives output dimensions from the first reference; "I" keeps resolution. Masked editing: image_mode=2, video_prompt_type="VA", one image_guide and image_mask; no extra image_refs. White mask means replace. Full denoising only; outside-mask pixel identity is not guaranteed. No outpainting, audio or video. Use multiples of 16 at about 1MP. Negative prompt supported; empty selects the released mode-specific negative caption."""

DEEPY_PROMPT_INFOS = """Generation: describe the result. Editing: give the change and preservation constraints; refer to ordered sources as <image_1> etc. Natural language, structured scene JSON and {"structured_caption":{...},"edit_instruction":"..."} are accepted. text_render:[{"text":"EXACT WORDS"}] supplies visible text and enables glyph-image conditioning. Optional enhancer T/TI: classic generation/editing; T1/TI1: structured generation/editing. Enhancement stays opt-in. Use multi_prompts_gen_type="FG" for multiline JSON; whole # comment lines are ignored. No numeric-coordinate layout schema."""

TURBO_INFOS = """## Fibo 2.0 Edit Turbo

Edit an existing image in four steps, combine up to four ordered references, or replace a masked region. A source image is required; choose Dev for text-only generation.

### Reference Editing
- **Main Image Ref** derives output dimensions from the first reference. Later images supply additional subjects or visual references.
- **Use References with Selected Output Dimensions** keeps the selected output size.
- Refer to inputs as `<image_1>`, `<image_2>`, and so on, and explain each image's role.

### Masked Editing
Choose Inpainting, provide one Control Image and its mask, and describe the change. White marks the region to replace. Remove additional Reference Images. The model generates the whole result; pixels outside the mask are not guaranteed to remain identical. Outpainting is not exposed.

### Settings and Prompts
Start with 4 steps at approximately one megapixel, such as 1024×1024. Dimensions use a 16-pixel grid. Negative prompts and guidance adjustments are not used. More references and larger batches require more memory and time.

State the requested edit and what to preserve. Natural-language instructions and structured JSON edits are supported. Optional classic and structured editing enhancers are off by default. Proofread lettering and inspect faces, fine details and preserved regions.
"""

TURBO_PROMPT_INFOS = """## Editing Prompt Guide

### Describe the Change
State the edit and what to preserve: `Change the jacket to red leather; retain the face, pose, background and lighting.`
For several references: `Put the person from <image_1> in the dress from <image_2>.`
For a mask, describe the replacement in its surrounding context.

### Structured Edits and Written Text
An edit can combine a desired-result caption with an instruction:
`{"structured_caption":{"short_description":"A cat wearing a blue bow tie and a tag reading MILO","text_render":[{"text":"MILO"}]},"edit_instruction":"Add a blue bow tie and the tag to the cat in <image_1>; preserve the cat and background."}`

The structured editing enhancer can prepare this JSON. Each `text_render` entry's `text` is the exact visible string; WanGP also renders it into a conditioning image. Quote exact lettering in plain-language instructions too.

Use valid JSON and escape double quotes inside strings. Whole lines beginning with # are ignored; inline # and hexadecimal colors are preserved. Structured prompts describe semantic composition, not a pixel-coordinate canvas. Enhancement is optional.
"""

TURBO_DEEPY_INFOS = """Editing only: model_type="fibo2_edit_turbo", image_mode=1, num_inference_steps=4, guidance_scale=1, resolution="1024x1024", batch_size=1, prompt_enhancer="". Supply 1–4 ordered image_refs: video_prompt_type="KI" derives output dimensions from the first reference; "I" keeps resolution. Masked editing instead uses image_mode=2, video_prompt_type="VA", one image_guide and image_mask, without extra image_refs. White means replace; outside-mask pixel identity is not guaranteed. No text-only generation, outpainting, audio or video. Negative prompt and CFG are unused. Use multiples of 16 at about 1MP. For text-to-image choose fibo2_dev."""

TURBO_DEEPY_PROMPT_INFOS = """Give an editing instruction and preservation constraints; refer to ordered sources as <image_1>, <image_2>, etc. Natural language or {"structured_caption":{...},"edit_instruction":"..."} are accepted. Put exact visible strings in structured_caption.text_render:[{"text":"EXACT WORDS"}] for glyph-image conditioning. Optional enhancer TI/TI1 provides classic/structured reference editing; keep enhancement off unless requested. Use multi_prompts_gen_type="FG" for multiline JSON; whole # comment lines are ignored. No numeric-coordinate layout schema."""
