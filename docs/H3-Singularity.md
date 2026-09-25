# H3 Singularity References (experimental)

Maestro includes the community **Singularity v1.3 Pruned INT8** checkpoint
as a separate H3 References model. It works through the existing Studio and
Director reference workflows, including image, video, and audio references.

## Test locally

1. Restart Maestro after updating and refresh the WebUI.
2. In Studio, select **Video → References → H3 Singularity — References
   (Experimental)**. In Director, choose the same model in its video selector.
3. Keep the recommended **LightX2V Ref2VA Turbo4 v0.1** preset enabled under
   **H3 Optimizations** for the first test. It selects four steps and LoRA
   strength 1.0 automatically.
4. Add your references and prompt, then generate. A short 480p reference clip
   is a useful first loading/quality check before trying a longer project.

The first generation downloads about **21 GB** of transformer weights and
**2 GB** for the Turbo adapter, plus any shared H3 encoder/VAE assets that
are not already installed. Downloads start on use; selecting the model does
not start generation. Existing H3 assets are reused.

The recommended recipe uses Euler, no CFG, video/audio shifts 12/3, and
Match reference sizing. The selected accelerator and strength are visible
in the LoRA controls. Turning Turbo off restores the ordinary 20-step
default; its step count can then be adjusted. Other H3 models retain their
existing defaults.

This initial integration supports the author's reference-focused workflow.
The Full and W4A8 files are not included. Compare quality with your usual H3
model using the same prompt, references, seed, duration, and resolution;
Singularity's quality in Maestro still needs generation testing.

## Sources and reproducibility

- [Singularity model card](https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity)
- [LightX2V Turbo model and workflow](https://github.com/ModelTC/Minimax-H3-Turbo)
- Checkpoint: `Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors`,
  revision `af671d9214a6e41ab8c2f43e9f871ea56246115f`.
- Adapter: `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors`,
  revision `3ec17a324ced54151364f24f8b5fb6bf7e26414f`.

The model definition and managed Turbo manifest record the published file
sizes and SHA-256 identities. The loader uses Singularity's grouped QKV and
pruned INT8 ConvRot layout and does not substitute a stock H3 checkpoint.
The adapter retains its published alpha/rank scaling at strength 1.0.
