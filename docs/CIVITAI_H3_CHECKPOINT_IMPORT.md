# Importing MiniMax H3 checkpoints from CivitAI

In the CivitAI browser, select a **Checkpoint** with base model **MiniMax H3**
and an individual `.safetensors` file. Choose the pipeline matching its model card:

- **H3 First / Last — Pruned** or **Full** for FL2VA/T2VA.
- **H3 Omni — Pruned** or **Full** for Ref2VA.

Choose the publisher's **QKV layout**. Grouped Q/K/V is used by Comfy/ConvRot
exports; head-interleaved applies to original exports. Check the instructions
for the specific file: tensor dimensions cannot distinguish these row orders,
and selecting the wrong order can corrupt generation. The importer deliberately
does not guess this setting from the filename or the CivitAI base-model label.

The importer supports the standard H3 fused-QKV transformer layout in BF16,
FP16, FP8 and INT8 (including ConvRot), with either Full AdaLN projections or
the Pruned curve table. It verifies the multimodal input/output projections,
all 50 transformer blocks, both token refiners, and the AdaLN dimensions before
publishing the downloaded file. FL2VA/Ref2VA partition metadata, when supplied,
must agree with the selected workflow. Without it, follow the model card.

This initial import support excludes GGUF, packed 4-bit/NVFP4 checkpoints,
archives, Diffusers shards/bundles, and separate text encoders or VAEs. LoRAs
continue to use the existing LoRA download path. An unsupported file is not
silently assigned to another model family.

Registration reuses the selected H3 template and its companion model handling,
while keeping the imported transformer's local filename and explicit QKV layout.
Built-in checkpoint migration aliases are disabled for the imported model.
Fused/distilled checkpoints may need additional publisher-specific sampler,
step-count, or LoRA settings; matching a tensor layout does not infer a recipe
or guarantee equivalent output quality.

## Validation reference

The dimensions were cross-checked using HTTP range reads of the pinned INT8
SafeTensor headers referenced by `app/defaults/minimax_h3.json` and
`app/defaults/minimax_h3_full.json` (DeepBeepMeep revision
`fec7846aef352e58a1cfb699455e3d104281e68b`). Regression tests use synthetic headers
and temporary finetune registrations; they do not allocate model weights or
require a CUDA GPU. Full inference remains a separate model-specific check.

## Early compatibility checks

The browser blocks H3 filenames explicitly marked INT4, FP4/NVFP4, 4-bit, or GGUF and explains that the user must choose another file/version. The download endpoint enforces the same restriction. Explicit workflow and size hints (FL2VA/T2VA/REF2VA plus Pruned/Full/33B) suggest a pipeline; ambiguous filenames still require a choice. A new model entry is created during import.

Checkpoint downloads validate the SafeTensor header at the start of the existing authenticated stream, before consuming the remaining weights. This also catches incompatible files whose names lack format hints, without depending on HTTP Range support. A failed check closes the response and removes the partial file. Header buffering is bounded to 256 MiB plus one download chunk; the complete file still receives the existing payload/offset validation before publication. QKV order remains an explicit publisher-informed choice.

Checkpoint browsing filters unsupported base families, non-model files, archives, and explicitly unsupported H3 quantizations from filenames or file precision metadata. Filtering is applied per version and file in search and detail responses; models without remaining files are hidden. Upstream pagination cursors are preserved so empty filtered pages can continue to subsequent results. These are metadata candidates, not a guarantee of tensor or runtime compatibility. LoRA browsing is unchanged.

Four-bit filename/precision filtering also recognizes W4A8 and W4A16 spellings, including separators and mixed case. W8A8 is not blocked by this rule.
