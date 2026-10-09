# H3 VAE 2×

For a supported H3 video model, choose **H3 VAE 2×** in Studio’s
**Advanced → Finishing → Spatial Upsampling**, or Director’s
**Advanced → Video Upsampling**. The option is off by default.

Denoising and references use the selected base canvas. The learned decoder
doubles both output dimensions: 960×544 becomes 1920×1088. Live previews
retain their lightweight decoder and can have different detail from the final
video. This selection replaces the normal decoder; Maestro does not apply a
second 2× resize afterward.

The pinned INT8 ConvRot checkpoint is about 2.8 GB and downloads on first
selection. It is separate from transformer checkpoints and LoRAs. Saved
outputs retain the selected method and actual final dimensions. Switching the
decoder reloads the model profile while preserving the selected text encoder.

Current support covers standard H3 video generation, including Frames and
References, in Studio and Director. It excludes image mode, audio-only,
Viggle, and Audio from Control Video, which preserves the source frames.
The checkpoint’s source-image B32 detail branch is not used for generated
latents. Extra pixels do not establish the same detail as native higher
resolution denoising.

Both normal and x2 H3 decoding blend temporal chunks in the native decoder
grid and write finalized frames to CPU bytes. This avoids retaining the
complete floating-point decoded video on the GPU. The x2 decoder still adds
work and output memory, especially for long clips or further finishing.

## Model terms and provenance

The optional weights retain the [MiniMax H3 Community License](../app/models/minimax_h3/MiniMax-H3-X2-Detail-v1.LICENSE)
and [NOTICE](../app/models/minimax_h3/MiniMax-H3-X2-Detail-v1.NOTICE).
The license’s territory excludes the United States, EU, UK and South Korea
and directs interested deployers there to contact MiniMax. Review those terms
and any applicable authorization before use or deployment.
[Source revisions and the verified asset hash](../app/models/minimax_h3/UPSTREAM.md)
are recorded with the implementation. No model weights are distributed in the
Maestro source repository.
