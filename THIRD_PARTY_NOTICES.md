# Third-party notices

This file supplements the license files distributed with Maestro and is not
an exhaustive replacement for dependency-specific notices in installed Python
or JavaScript packages.

## MiniMax H3 Sol Engine

Maestro's optional H3 Sol Engine includes adapted Apache-2.0-licensed source
from the following projects:

- **NVlabs/Sana Sol-Attn**, pinned to commit
  `46031940ba8af5d18054217e571149579424c0b1`.
  Source: https://github.com/NVlabs/Sana/tree/46031940ba8af5d18054217e571149579424c0b1/techniques/sparse_backends/sol_attn
- **Saganaki22/ComfyUI-sol-attn**, pinned to release `v0.5.2` / commit
  `e2fc225` for the optimized INT8-QK path.
  Source: https://github.com/Saganaki22/ComfyUI-sol-attn/tree/e2fc225

The applicable Apache License 2.0 text is distributed at
`app/shared/sol_attn/saganaki/LICENSE`. Adapted source files retain SPDX
license identifiers and upstream attribution.

## MiniMax H3 SLA sparse attention

Maestro's optional H3 SLA backend adapts MIT-licensed implementation work
from **PlagueKind/ComfyUI-PlagueKind-Nodes-only-sparse**, pinned to commit
`fd26ffb89dee294ca740a59632e5b3423b9a9d2a`. That implementation adapts
Apache-2.0-licensed SLA utilities and kernels from **ModelTC/LightX2V**.

- Source: https://github.com/ethanfel/ComfyUI-PlagueKind-Nodes-only-sparse/tree/fd26ffb89dee294ca740a59632e5b3423b9a9d2a
- LightX2V source: https://github.com/ModelTC/LightX2V
- The MIT license text is distributed at
  `app/models/minimax_h3/SLA_LICENSE.txt`.
- The Apache License 2.0 text covering the adapted LightX2V portions is
  distributed at `app/shared/sol_attn/saganaki/LICENSE`.

## MATLOWAI MiniMax H3 fused four-step checkpoint

The optional experimental model definitions
`minimax_h3_fused_turbo` and `minimax_h3_ref2va_fused_turbo` download the
same revision-pinned community checkpoint from
**MATLOWAI/minimax-h3-fused-turbo-int8-convrot**. It combines MiniMax H3,
the xmarre Ref2VA delta approximation, LightX2V Turbo, Mystic, and ConvRot
conversion components. No model weights are redistributed in this source
repository.

- Model revision: `3b51096a1bf67608d98131116558202208fcf195`
- Expected checkpoint SHA-256:
  `4262e4e9963c553fa00016bbe83961407a4fc0a888be95fd836c8d4f2304e48b`
- Source: https://huggingface.co/MATLOWAI/minimax-h3-fused-turbo-int8-convrot
- License: https://huggingface.co/MATLOWAI/minimax-h3-fused-turbo-int8-convrot/blob/main/LICENSE
- Required notices: https://huggingface.co/MATLOWAI/minimax-h3-fused-turbo-int8-convrot/blob/main/NOTICE

The matching experimental INT8 ConvRot video VAE is downloaded separately
from **Kijai/MiniMax-H3-experimental**. It is pinned to revision
`a3e7d8da4ae7ba8df0779094cf5ab9d6ee855fe4`, with expected SHA-256
`9bb2d96f218c76babd85e0611b85ca8fb330a90546c01a0005e8a58a59593410`.
Source: https://huggingface.co/Kijai/MiniMax-H3-experimental/blob/a3e7d8da4ae7ba8df0779094cf5ab9d6ee855fe4/minimax_h3_video_vae_int8_convrot.safetensors

Users must review the linked model license and NOTICE before downloading or
using this optional checkpoint; its terms and geographic scope differ from
Maestro's application license.
