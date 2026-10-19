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

Additional workflow code is adapted from Wan2GP's Qwen Image 2.1 integration
at commit `2345ae148f82740f66e82c41292dbbdd592e713d`:
`models/qwen21/pipeline.py`, `models/qwen21/vae.py`, and
`models/qwen21/qwen21_handler.py`. The adapted additions cover the Viggle
sampling schedules, masked denoising and LanPaint flow, red-canvas outpainting,
VRAM-based VAE tile selection, and streamed CPU uint8 decoding. Those additions
are licensed under the WanGP Community License 2.0; the license text is at
`../../LICENSES/WanGP-Community-2.0.txt`. Existing Diffusers-derived files
retain their Apache-2.0 notices and license.

Qwen's materials use the **Qwen Research License**, included in
`QWEN_RESEARCH_LICENSE`: non-commercial research/evaluation only. Commercial use
requires a separate Qwen license. See `NOTICE` for the required attribution.


Official Turbo sampling and registration are adapted from
[Maestro PR #175](https://github.com/Blizaine/Maestro/pull/175), by luisnomad,
head `cbe73fd1fae4d7c849a32949e6f8a96ba8e822d3`. Its eight sigma nodes and
scheduler settings were verified against
[Qwen/Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo),
revision `d65dbc9a7e8f6b5479e33dee6030eaab2a906509`. The BF16 and INT8 ConvRot
repack provenance is in `DeepBeepMeep/Qwen_image_2/qwen_image_21/turbo/provenance.json`
(repack revision `5b93c1409dc1ff02f666a83275c4a88c1aedbe2d`). It identifies the
same official revision. Turbo shares the existing architecture and component
loader; no global Diffusers or Transformers upgrade is required.
