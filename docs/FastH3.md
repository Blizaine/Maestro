# FastH3 in Maestro

FastH3 Omni turns a prompt plus ordered image, video and audio references into
video with generated stereo audio in eight transformer evaluations. In
Studio, select **References**, open **Manage models**, and enable **FastH3
OmniRef 8-Step (Experimental)**. Existing enabled-model preferences are retained.

Start at **480p / 5.2 seconds**, guidance 1, eight steps. The model uses its
own VSA-H3 attention, video/audio shifts 12/3, and 32 PDD output heads fused
into eight evaluations. Its .999 noise grid and residual-head reconstruction
are separate from FastH3 V2's text-only 10/3 recipe and ordinary H3 Turbo LoRAs.

## Official formats and hardware

| Model option | Diffusion weights | Intended use |
| --- | --- | --- |
| OmniRef, INT8 ConvRot | 22.2 GB | Practical starting format across supported consumer CUDA GPUs; streams through MMGP. |
| OmniRef, BF16 | 44.2 GB | Larger system-memory budget; no weight quantization. |
| OmniRef, FP8 | 23.0 GB | Native Comfy per-tensor FP8 export; accelerated math depends on the GPU. |
| OmniRef QAD NVFP4, separate model | 13.8 GB | Explicit experimental option for supported Blackwell GPUs with a working native NVFP4 kernel. |

The normal OmniRef option follows **Settings → Performance → Transformer
Quantization** for BF16, INT8 or FP8. The QAD NVFP4 option is separate so the
global INT8/FP8 selector cannot silently select BF16 instead. QAD is a distinct
trained weight variant; changing to it can change the output.

These sizes describe only the downloaded diffusion checkpoint. The text/vision
encoder, video and audio VAEs, activations and operating system also need
memory. Storage size is not a minimum VRAM requirement. MMGP can stream weights,
but smaller VRAM/RAM systems need their own tests. There is no proven 8GB or
12GB OmniRef duration tier yet. BF16 VSA uses CUDA compute capability 8.0 or
newer and Triton; the FP16 path requires capability 7.0 or newer. NVFP4 fails
with an actionable error when supported hardware or its native kernel is missing.

The sources are pinned to immutable revisions and checked by exact file size
and SHA-256 before allocation:

- [Official BF16 and INT8 Comfy release](https://huggingface.co/FastVideo/FastH3-Omni-Comfy).
- [Official FP8 Comfy release](https://huggingface.co/FastVideo/FastH3-Omni-FP8-Comfy).
- [Official QAD NVFP4 Comfy release](https://huggingface.co/FastVideo/FastH3-Omni-NVFP4-QAD-Comfy).

The official INT8 file is byte-identical to the earlier supported community
upload. Its filename and hash are retained, so an existing verified download
is reused. The other formats have their own pins and header checks. FP32 PDD
heads and retained FP32 AdaLN projections remain in their source precision.

## References and limitations

Use ordered `<Picture 1>`, `<Video 1>` and `<Audio 1>` references in the prompt.
At least one image or video reference is required. Reference audio guides the
generated sound; an exact frozen target soundtrack is not supported by this
recipe. Start/end frames, masks, continuation, Director, extra LoRAs, First
Block Cache, SLA, Veda and ordinary H3 audio refinement are unavailable.

Maestro keeps the trained sparse-attention pattern, using its BF16/FP16
Triton VSA implementation. Selecting NVFP4 quantizes transformer linear
layers; it does **not** enable FastVideo's separate FP4 sparse-attention
kernel or NVFP4 video decoder. The QAD Comfy format is also a separate
requantization from the original FastVideo QAD packed export.

## Performance claims and verification

[The OmniRef release benchmarks](https://haoailab.com/blogs/fastvideo-omniref-update/)
report 5.6–11.9× end-to-end speedups on **eight GB200 GPUs**. Their one-image,
10-second case averaged 9.37 seconds. Those results do not establish real-time
generation on a single 5090 or another consumer GPU.

[FastVideo's local recipes](https://github.com/hao-ai-lab/FastVideo/blob/main/docs/inference/fasth3_omniref.md)
describe a Blackwell NVFP4 stack for the RTX 5090, RTX PRO 6000 and DGX Spark,
with a different FP8/offloaded recipe for the RTX 4090. They combine the
matching model, sparse-attention kernel, encoder/decoder, AdaLN compression,
memory placement and repeated-reference caching. Their published runtime
recipes are not benchmarks of Maestro's native implementation.

Maestro's existing INT8 image-reference smoke test on a 4090 produced a fully
decoded 864×480, 124-frame, 24fps video with audio in 86 seconds of active
generation. A mixed image/video/audio test took 243 seconds. Those are
functional checks, not matched throughput benchmarks or evidence of real-time
generation. New BF16/FP8/NVFP4 format checks do not establish complete render
quality, identity or sound-reference fidelity, or performance on a 5090.

A bounded check on an RTX 5070 Laptop GPU loaded actual FP8 and NVFP4
compression-gate tensors through the same registered MMGP handlers used at
Maestro startup. Both produced finite outputs. NVFP4 selected the native
lightx2v kernel; comparison with the quantized-input/dequantized-weight
reference had maximum absolute error 0.00098. This checks one linear layer,
not an entire checkpoint, video render, memory tier or speed claim.

All variants remain experimental. Near-real-time generation and reliable
maximum durations need measured results on the actual GPU and complete
pipeline, including conditioning, denoising, decode and export.
