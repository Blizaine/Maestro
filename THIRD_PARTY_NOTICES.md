# Third-party notices

This file supplements the license files distributed with Maestro and is not
an exhaustive replacement for dependency-specific notices in installed Python
or JavaScript packages.

## Wan2GP v12.71 media and H3 integrations

The VDN attention implementation, H3 dialogue Whisper boundary alignment,
DLSS native-worker adapter, offload registry, and optional DLSS installer are adapted from **deepbeepmeep/Wan2GP**, commit
`1e1dd2757f24923f008593d9d4ec09062234be20` (5 September 2026).
H3 grouped mask conditioning, outpaint margin quantization, audio refinement,
audio-only generation, and RIFE temporal interpolation also incorporate work
from this revision. Maestro-specific queue, API, and React integration is local.

The upstream WanGP Community License 2.0 covering these contributions is
preserved in `app/LICENSES/WanGP-Community-2.0.txt`. This notice does not replace
licenses applicable to older Maestro components or model weights.

- Source: https://github.com/deepbeepmeep/Wan2GP/tree/1e1dd2757f24923f008593d9d4ec09062234be20
- VDN weights: https://huggingface.co/DeepBeepMeep/MiniMax-H3/tree/304d34f7751f8ba9ca0eb55d5d10044234cdbfe2
- Native worker source and licenses: https://github.com/DeepBeepMeep/dlss5-visual-enhancer

Native DLSS binaries are not distributed in this repository. The optional
installer preserves their bundled notices and verifies pinned checksums.
The neural-rendering runtime includes community-modified, unsigned
NVIDIA-derived components outside the official NVIDIA SDK distribution.
See `docs/DLSS5.md` for installation requirements and the explicit installer
acknowledgment. NVIDIA, ReShade, RenoDX and model licenses remain applicable
to their respective files.

## Viggle Animate

Maestro's Viggle conditioning and integration adapt **deepbeepmeep/Wan2GP**
v12.72, commit `057f9ecab9ad57dfbec9768b2daf7a4426ce986c`, under the WanGP
Community License 2.0 preserved at `app/LICENSES/WanGP-Community-2.0.txt`.
This includes fixed-prompt reference ordering, control-window slicing and the
rank-8 affine compatibility map. Model assets are downloaded on demand from
`DeepBeepMeep/MiniMax-H3`, revision `fa7ed035f21d341439d4dd763a020fc4a2482c43`.
The dedicated Viggle weights retain their upstream MiniMax H3 Community model
terms; they are not bundled in Maestro.

Sources: https://huggingface.co/Viggle/Viggle-Animate and
https://github.com/deepbeepmeep/Wan2GP/tree/057f9ecab9ad57dfbec9768b2daf7a4426ce986c/models/minimax_h3

## H3 Face Refiner

Maestro's H3 Face Refiner also adapts the face detection, identity tracking,
crop preparation, stitch-back and refinement schedule from Wan2GP commit
`1e1dd2757f24923f008593d9d4ec09062234be20`. The face module is derived from
**Carasibana/ComfyUI-H3-FaceRefine** commit
`79a97ce5ee4b393ce26313bd1280b706fe8b4f2c`; its MIT license is preserved at
`app/postprocessing/h3_face_refiner/LICENSE.upstream`. WanGP's adaptation is
covered by the WanGP Community License noted above. Ultralytics is an external
AGPL-3.0 dependency. InsightFace model weights and H3/LightX2V weights retain
their respective upstream model terms; model weights are downloaded on demand.

Source: https://github.com/deepbeepmeep/Wan2GP/tree/1e1dd2757f24923f008593d9d4ec09062234be20/postprocessing/h3_face_refiner

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
