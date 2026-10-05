# H3 Singularity (experimental)

Maestro includes the community **Singularity v1.3 Pruned INT8** checkpoint
through paired **Frames** and **References** entries in Studio and Director.
Both entries reuse the same checkpoint and recommended Turbo adapter.

| Workflow | Studio selection | Inputs |
| --- | --- | --- |
| Text to video | Video → Frames → H3 Singularity — Frames | Prompt, with the frame inputs empty |
| Image to video | Video → Frames → H3 Singularity — Frames | A start image, end image, or timed Frames |
| Video to video | Video → Frames → H3 Singularity — Frames | Control Video, with **Edit the whole source video** and denoising strength below 1 |
| References to video | Video → References → H3 Singularity — References | Ordered image, video, and audio references |

## Test locally

1. Restart Maestro after updating and refresh the WebUI.
2. Select the matching **H3 Singularity (Experimental)** entry from the table
   above. Director also exposes both entries in its video selector.
3. Keep the recommended **LightX2V Ref2VA Turbo4 v0.1** preset enabled under
   **H3 Optimizations** for the first test. It selects four steps and LoRA
   strength 1.0 automatically.
4. Add your prompt and the selected workflow's inputs, then generate. A short
   480p clip is a useful first loading/quality check before trying a longer project.

For video-to-video, open **Control Video** in Frames and choose **Edit the whole
source video**. Start with denoising strength **0.7**; a lower value preserves
more of the source, while a higher value allows more change. **Generate new
visuals from prompt** at strength 1 generates pictures without using the source as
visual guidance. References uses the source as a reference rather than as
the video being edited.

The first generation downloads about **21 GB** of transformer weights and
**2 GB** for the Turbo adapter, plus any shared H3 encoder/VAE assets that
are not already installed. Switching between Frames and References does not
download another Singularity checkpoint or adapter. Downloads start on use;
selecting the model does not start generation. Existing H3 assets are reused.

The recommended recipe uses Euler, no CFG, video/audio shifts 12/3, and
Match reference sizing. The selected accelerator and strength are visible
in the LoRA controls. Turning Turbo off restores the ordinary 20-step
default; its step count can then be adjusted. Other H3 models retain their
existing defaults.

The model card advertises native text, image, reference, and video conditioning.
Maestro exposes these through its existing H3 workflows. The Full and W4A8
files are not included. Compare quality with your usual H3 model using the
same prompt, inputs, seed, duration, and resolution.

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
Frames changes the conditioning route while retaining the checkpoint's Ref2VA
adapter compatibility and AdaLN conversion basis.

## Local validation

On October 1, 2026, short four-step renders completed for text-to-video,
start-and-end image-to-video, whole-frame Control Video editing at strength
0.7, and image-reference generation. All decoded at 608×352 and 24 fps with
audio and coherent first, middle, and final frames. The Start+End test retained
the shared workflow's default four-frame tail trim; the other tests produced
124 frames. Longer, higher-resolution and complex reference projects still
need quality evaluation.

The 200 backend regression checks, the focused Studio routing/payload fixture,
UI lint, and the production UI build passed. The paired routes were also
checked in the running browser UI.
