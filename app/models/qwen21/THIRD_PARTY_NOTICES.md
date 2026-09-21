# Qwen Image 2.1 integration

The transformer, VAE and pipeline are vendored from
[Hugging Face Diffusers](https://github.com/huggingface/diffusers/tree/80c7ed262aeffbeb43ef13ae04baeb9b84515a69/src/diffusers),
revision `80c7ed262aeffbeb43ef13ae04baeb9b84515a69`. Their original Apache-2.0
copyright headers are retained. The Apache license is in `APACHE_LICENSE`.

Maestro changes import paths for local use, bridges Diffusers 0.36's missing
LoRA-scale decorator, captures only the final pre-normalization text activation,
supports the backbone without its unused LM head, and handles FP32/BF16 VAE
encoding separately from transformer precision. Runtime integration adds MMGP
offloading, cancellation, step progress, RGBA output and tiled VAE execution.
No global Diffusers or Transformers upgrade is required.

Configuration files originate from [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1),
revision `b3179ad355be050328e483a9dfdd9e60cd62adfa`. Weights download separately
from [DeepBeepMeep/Qwen_image_2](https://huggingface.co/DeepBeepMeep/Qwen_image_2)
and its shared [Qwen3-VL-8B encoder](https://huggingface.co/DeepBeepMeep/Ideogram4).
The repack's provenance identifies that same official source revision.

Qwen's materials use the **Qwen Research License**, included in
`QWEN_RESEARCH_LICENSE`: non-commercial research/evaluation only. Commercial use
requires a separate Qwen license. See `NOTICE` for the required attribution.
