# H3 latent upscaler notices

`latent_upscaler.py` adapts the 3D latent-resizer architecture and temporal
chunking from [LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)
at source commit
[`98934c0b5f4a213323c4d1d0e47eb3d2f27cb743`](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/commit/98934c0b5f4a213323c4d1d0e47eb3d2f27cb743).
The source project is MIT licensed; its copyright and permission notice are
reproduced in `H3-LATENT-UPSCALER-MIT-LICENSE.txt`.

The optional model weights are separate from this source tree. They are fetched
only when H3 Two-Pass is explicitly enabled, from
[`LBH-123-AI/Minimax_h3_latent_Upscaler`](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)
at revision `09592c6221ec95cc8e0fae67842e34926c4e668b`, file
`minimax_h3_latent_upscaler_3d_bf16.safetensors`. The model card declares
Apache-2.0. Its pinned SHA-256 is
`4f57821f5837f32f7142b67d815606dbd7550f194e5c769f7d6c3f83b146a5e6`.

The adapted 3D path preserves the source's 32-latent temporal chunks, 5-latent
overlap, replicate padding, and weighted blending. GroupNorm spans the full
temporal axis, so chunked runs are the source implementation's memory-bounded
approximation to full-context inference; windows of at most 32 latent frames
run without temporal chunking. The 124-frame default H3 clip remains in the
full-context path.

The source network is the author's 24-channel H3 3D resizer with 12 residual
blocks on each side of trilinear resizing and temporal convolution every other
block. The implementation preserves its trained parameter names. The latent
upscaler is opt-in and does not alter the standard H3 generation path.
