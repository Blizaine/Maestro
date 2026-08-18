# MiniMax-Music3 notices

The neural-network class definitions in this directory are adapted from the
MiniMax-Music3 integration contributed by the MiniMax and Hugging Face teams
to Diffusers and retain their Apache License 2.0 headers.

MiniMax-Music3 model weights are not distributed in the Maestro repository.
The original model and license are published at
<https://huggingface.co/MiniMaxAI/MiniMax-Music3>. Maestro's accelerated path
uses BF16 and ConvRot INT8 checkpoint conversions published by the WanGP
project at <https://huggingface.co/DeepBeepMeep/TTS>; they remain governed by
the [MiniMax-Music3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-Music3/blob/main/LICENSE)
and its Acceptable Use Policy. Maestro downloads the official license beside
the optimized model components.

The CUDA-graph/vLLM semantic decoder and its optimized component layout are
adapted from `deepbeepmeep/Wan2GP`'s MiniMax Music3 implementation. The vLLM
kernel path is selected only after both FlashAttention2 and Triton pass the
runtime probe; CUDA graphs with SDPA remain available on supported cards that
do not have a compatible FlashAttention binary.
