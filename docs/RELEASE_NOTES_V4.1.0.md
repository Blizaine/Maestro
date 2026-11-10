# Maestro v4.1.0 — 2026-11-10

This release brings native MiniMax H3 Image to Studio and expands agent-led
filmmaking across Studio, Director and Editor.

- **MiniMax H3 Image (Experimental):** create images from text or edit with up
  to 16 ordered source images in six steps. Studio offers native 1K, 2K and 4K
  canvases. The model reuses Maestro's H3 32B text encoder, when installed.
- **Video workflows and finishing:** replace people or edit a scene with H3
  Remake, use Regenerate 2K and new H3 accelerators, render Director final takes
  and upscale video 2× with RelayVSR.
- **Title cards:** exact-text cards without loading an image model; optional
  Director opening, intertitle and end-card planning with persistent assembly.
- **Agent finishing:** build a new editable cut from takes/cards, narration,
  score and effects, with explicit mixing and software export. Custom graphic
  overlays retain editable text, panels and images on independent title tracks;
  named voiceover tracks and the source footage remain separate. The exported
  movie's gallery pencil reopens its saved Editor project.
- **Review and bounded polishing:** actual sampled images/audio, decode reports,
  concrete creative observations and durable selected-Director-take budgets.
- **Generation reliability:** PDD BF16 head compatibility, more accurate
  Director source-action validation, H3 reference handoff diagnostics and
  verified stale Agent Access PID recovery on Windows.
- **Settings reuse and Pika experiments:** recover recorded MCP/API clip timing
  and offer experimental 4–12 step Pika H3 Flex sampling. Six remains recommended;
  extra steps do not guarantee better quality.

See [Agent-led filmmaking](Agent-Filmmaking.md) for scope, usage and limitations.
Title cards are optional. Audio-locked intertitles require reserved timeline
space; opening/end holds extend the finished runtime. Sampled media does not
prove continuous motion, perfect lip-sync or correct character voices.
