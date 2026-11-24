# Maestro v4.2.0

Released November 23, 2026 on Main and Dev. Update from the Maestro launcher
in Pinokio. If a legacy Python 3.10 install encounters the ONNX Runtime GPU
dependency error, retry Update with this release's launcher. [Issue #241](https://github.com/Blizaine/Maestro/issues/241)

## New video and image options

- In Studio Video → Frames, the **pi-H3 FL2VA** recipes apply the matching
  2- or 4-step LoRA to a compatible H3 model. Dedicated **pi-H3 DWD** models
  for Pruned and Full H3 load their own matching LoRA and decode the
  picture in the final denoising pass. The final Pruned LoRA fix was
  verified with both DWD variants on an RTX 4090 at 864 × 480 and 243 frames.
  [WanGP v20.10 source](https://github.com/deepbeepmeep/WanGP/tree/994642c87b02ef0abbdc1bc73dd14f6b5e4497e8)
- **H3 HyperFlow Flash 14B** is experimental in Frames and References.
  Choose 4, 6 or 8 denoising steps and optionally use two-pass generation.
- **MiniMax H3 Image** adds pDMD 2- and 4-step presets for text-to-image and
  reference editing alongside its native six-step preset.
- **Remake** can send up to four extracted keyframes to Studio Image for
  edits and return the results to their original slots. Experimental
  **BFS Aligned Swap** replaces a person using one identity image while
  keeping the source framing and soundtrack.
- **Keep source audio (output only)** in Studio References keeps the
  selected video's soundtrack in the output without using it to drive the
  H3 render. Choose this when you want to carry through sound such as an
  off-camera voice.

## Easier workflows for music, speech and video

- **Studio Recipes** have short playable previews and apply model settings
  with media and prompt guidance. Connected MCP agents can read the guides
  when preparing their own requests.
- In Director, **Sung by** changes a lead vocal or duet parts using SeedVC
  while keeping the backing track and song timing. Results depend on the
  source and its mix.
- **Director Delivery** exports timed lyrics as an SRT file that you can
  import into Editor Captions. It also queues upscaling for an individual
  take or an already joined cut.
- **Director planning** saves accepted music-video shots, repairs failed
  batches and limits visual and voice references to the shot's cast. H3
  References enhancement preserves written speaker assignments and validates
  dialogue across all planned windows before rendering. Supported shots can
  use a neighboring take's frame to anchor their start or end.
- Experimental **LongCat Avatar 1.6 Distilled** adds 25 fps audio-driven
  avatar video with 8-step sampling and streamed conditioning. Experimental
  **ZONOS3 Mini** adds speech in 43 languages, 27 built-in voices, voice
  cloning from references and delivery controls.

## Performance and reliability

Extreme attention chunking uses smaller token batches on supported model
paths to reduce peak VRAM, with a potential speed tradeoff. Optional SageGP
attention has GPU and runtime requirements. Auto-Tune improves H3 head
splitting and memory use during VAE conversion. Supported sliding-window
MP4s assemble incrementally with aligned video and audio. Face Refiner also
adds face selection, blending, audio guidance and before/after comparison.

See the [detailed changelog](../CHANGELOG.md#420--2026-11-23) and
[update history](UPDATE_HISTORY.md).
