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
