# Fibo 2.0

Maestro includes Fibo 2.0 Dev for text-to-image generation, multi-reference
editing, and masked editing, plus Fibo 2.0 Edit Turbo for instruction-based
editing. Dev defaults to 50 steps and CFG 5. Edit Turbo defaults to four steps
and CFG 1; it requires a source image and does not offer text-only generation.

Both variants accept up to four ordered reference images. A mask selects one
control image; white mask pixels preserve the source and black pixels are
replaced with the model's neutral gray conditioning value. Output dimensions
must be multiples of 16. References are passed to the Qwen3-VL encoder in the
same order used to construct the edit prompt. Structured captions support
scene fields and `text_render` entries; each requested string is rendered as a
conditioning image with the accompanying Aileron and DejaVu fonts.

The implementation adapts Bria's Diffusers transformer, caption processing,
and text rendering from commit
[`ab96e2b58c89adb9df418af74d81cc8b1e1be074`](https://github.com/huggingface/diffusers/tree/ab96e2b58c89adb9df418af74d81cc8b1e1be074).
It reuses Maestro's Qwen3-VL implementation and FLUX.2 VAE code, with Fibo-local
adapters for tensor operations, attention masks, runtime placement, and VAE
packing/tiling. The runtime requires the transformer, Qwen3-VL-4B encoder,
tokenizer and processor metadata, FLUX.2 VAE, and text-rendering fonts. The
asset definitions pin the compatible DeepBeepMeep/Fibo2 mirror to revision
`72bf0a8`; original Bria checkpoint revisions are recorded in
[`NOTICE.txt`](../app/models/fibo2/NOTICE.txt).

## Model terms

The Bria Fibo Dev and Edit Turbo weights are identified by their current
Hugging Face model cards as CC BY-NC 4.0. The compatible checkpoints hosted by
DeepBeepMeep are derivative rewrappings of those source revisions and inherit
the source model terms. Commercial use requires a separate agreement with
Bria. This is separate from the Apache-2.0 licenses for the Qwen3-VL-4B
weights and FLUX.2 VAE. The FLUX.2-dev transformer license does not apply to
the VAE file. See the [Fibo Dev model card](https://huggingface.co/briaai/Fibo-2.0-dev),
the [Fibo Edit Turbo model card](https://huggingface.co/briaai/Fibo-2.0-edit-turbo),
and [model licensing](https://platform.bria.ai/licensing/pricing?model=fibo-2).

This port has not been run on a GPU in this integration task. No quality or
speed claims are made here.
