# Maestro v2.1.0

Prepared locally on 7 September 2026; not yet published.

Maestro 2.1.0 brings a larger Studio prompt workspace, portable characters with
voice, automatic character preparation for Viggle Animate, and new H3 audio,
animation and finishing tools. This release includes the changes since the
public v2.0.1 update.

## Studio: more room to create

- Video, Image and Audio keep their complete workflow lists at the top.
  Compact media inputs sit above a large prompt field that fills the available
  space. Frames and References use consistent three-column input tiles.
- Media tabs, workflow selection, references and prompt share one vertical
  scroller, so short windows can reach every input. Long scripts expand the
  prompt without a second scrollbar; short prompts still fill the available
  writing space. Settings and Generate stay pinned, and typing keeps the
  active line visible as the mobile keyboard moves.
- Characters stays on the left of the bottom settings strip. Recipes,
  Resolution, Aspect, Duration and Advanced form a compact group on the right.
  Recipes sits immediately beside Resolution. Character buttons use the same
  person icon in Reference, Image, Animate and Speech.
- Resolution and Aspect open lists sized to their labels, directly above their
  buttons without redundant headers.
  All existing model-specific options remain available, including Auto.
- The model selector sits beside Generate. The Model Browser has its own globe
  shortcut, and Recipes opens directly from its book icon. Generate retains
  its separate Add to Queue action.
- Advanced uses independently collapsible Performance, Finishing, LoRAs and
  Generation sections. Empty sections are hidden; applicable controls remain
  available when disabled. Settings, active counts and unfinished preset
  drafts survive closing the panel. Active settings have circular count badges
  beside their section headings, and Advanced opens above its button on mobile
  without covering Generate.
- LoRA usage guides stay inside the visible screen in Studio and Director.
  Long guides scroll, and dismissing a guide leaves its parent settings open.
- Reference inputs retain their media roles, descriptions, ordering,
  background-isolation choices, saved voices and model-specific limits.
  Adding a reference reveals the next available drop zone.
- All three theme families, light/dark variants and automatic appearance remain
  supported. See [Studio controls](Studio-controls.md).

## Prompt enhancement and duration

- The magic button now explicitly enhances the visible prompt with **AI
  Faithful**. Its arrow offers Faithful or **AI Creative**. Users can review
  and edit the result before generation; Generate and Add to Queue no longer
  run an unseen enhancement based on an old prompt-mode selection.
- Faithful preserves the supplied events and dialogue. Creative can expand
  the idea. Both modes also work for images while preserving explicit facts,
  reference constraints, requested text and edit boundaries. Speech keeps its
  dedicated speech/dialogue enhancement menu.
- Long H3 enhancement prepares reviewable window prompts. **Exact H3 prompts**
  exposes them for editing; LTX retains one prompt line per window. Missing
  window prompts produce an actionable message before submission.
- H3 multi-window Faithful and Creative planning now share Studio's optional
  `enhance/nsfw_shared.md` guide when Mature mode is enabled. The same guide
  reaches story planning, long-form chapters, per-window writing and retries;
  it is omitted when Mature mode is off. The guide's wording is unchanged.
- Time mode adds a slider through model-aligned durations up to five minutes.
  Ordinary H3 starts at approximately 5.2s, 5.9s and 6.6s. Longer presets and
  editable timecodes still reach 60 minutes on supported video workflows.
- The compact video Duration popup uses **Time** and **Window** tabs. **Auto**
  and the current duration remain visible above both tabs. Auto dims manual
  time controls; grabbing the slider, using its arrow keys, choosing a preset
  or editing the timecode turns Auto off and applies the manual choice.
  Long presets are **10m, 15m, 30m, 60m and Custom**. Window retains its exact
  count controls. Window Length stays visible and adjusts automatically;
  overlap is collapsed by default in both tabs. The popup stays above its
  indicator as the keyboard moves, and slider dragging keeps a steady layout.
- Automatic window sizing follows the requested timeline up to the model/GPU
  recommendation or the user's override. Saved model/resolution overrides and
  Reference continuation settings remain available.
- Auto shows its recommended duration in the collapsed indicator. The Auto
  option inside Studio and Director duration controls also previews that
  recommendation before selection.
- Shared dialogue planning targets **2.8 words per second**, with admission up
  to **3 words per second**. Dense but valid lines can use the clip's available
  speech time instead of being compressed to reserve unnecessary silence.
- AI Creative writes developed character dialogue for conversations, tutorials,
  interviews and implied interactions. It receives a spoken-word target based
  on the actual window duration, with action and pauses accounted for. Sparse
  drafts receive a focused writing retry, and H3 allows up to six turns per
  window within the shared word budget. Supplied quotes remain exact anchors
  around which Creative can add supporting dialogue. Faithful, explicit silence
  and requests to use only the supplied lines retain their original contracts.
- H3 speech intervals account for the complete authored script, including
  Creative's supporting lines. A silent establishing shot or reaction no longer
  suppresses a later requested conversation. Brief tactical dialogue remains brief.
- Dialogue extraction excludes production headings such as Visual Direction
  and Sound, stops at closing quotes, and avoids double-counting mixed
  screenplay/tagged dialogue. This fixes inflated spoken-word requirements
  for prompts containing extensive action and cinematography instructions.
- Included a [14.4-second-window Reference tutorial example](prompts/blaine-maestro-tutorial-script.md)
  with native dialogue tags, presenter direction and screen-recording cues.

## Portable characters, RefMods and better reference images

- Import standard H3 RefMods or share a complete Maestro character as
  **`<character>.maestro.safetensors`**, for example
  **`blaine.maestro.safetensors`**. Exports include appearance, saved voice
  audio when present, name/description, and selected recovered images.
- Model Browser adds **Characters / RefMods**, local-file and Hugging Face
  imports, voice filters, and export. A saved collection opens
  **malcolmrey / MiniMax H3**. Exact file URLs select the requested file.
- Add or replace a character's saved voice, then export it with that audio
  embedded. Standard RefMods remain visual references; they are kept separate
  from trained LoRAs. Compatible upstream loaders can use the visual portion
  of Maestro files, while embedded voice is a Maestro extension.
- Multiple RefMods retain separate identities and voice bindings. Manual and
  enhanced prompts share name matching, including cleaned imported names and
  explicit Subject IDs. Speakers follow speaking order rather than character
  list order; ambiguous attribution asks for a name instead of guessing.
- RefMod recovery decodes native-resolution lossless PNG views. Existing
  original photos and video frames are preferred when available. Choose a
  cover, save selected views, and download individual PNGs or a ZIP.
- Recovery is cached and preserves the original RefMod tensor and saved audio.
  Portable exports retain selected PNGs and the cover so another Maestro
  installation can use them without repeating recovery.
- Saved characters and recovered views can be added to Image models that
  accept references. Their order and the model's reference limit are respected;
  appearance changes remain editable in the image prompt.
- Character browsers and image galleries scroll on mobile. Video characters
  use cached still thumbnails. Display names remove the `minimaxh3_` prefix,
  RefMod/safetensors suffixes and filename underscores without renaming files.

See [character sharing and recovery](Maestro-Characters.md).

## Viggle Animate and automatic character replacement

- **Video → Animate** adds the dedicated H3-based Viggle recipe: three Euler
  evaluations in 124-frame windows at 24 fps, approximately 5.2 seconds each.
  Longer videos use overlapping windows and trim to the requested duration.
- Supply a control video and an edited frame from the start or another point
  in the source video. The edited image conditions the animation; it does not
  have to be the opening frame.
- Choose a saved character, recovered RefMod view or uploaded character image.
  Maestro can use Flux 2 Klein to replace the subject in the selected source
  frame before running Viggle, with editable replacement instructions and an
  optional appearance prompt.
- Preview the replacement frame first or run preparation and animation in one
  queued job. Klein 9B is the default preparation model; 4B is also available.
  Preparation retains the source frame dimensions and runs serially with Viggle.
- Manual editing remains available through Image mode and **Apply & return**.
  The handoff prompt specifies source image first and character image second,
  preserving pose, orientation, props, background, framing, lighting and size.
- Load Settings retains the character view, preparation choices, frame time,
  edited image and audio guidance. Changed inputs invalidate stale previews.
- Viggle uses its required fixed motion-transfer prompt. Appearance instructions
  control Klein preparation; ordinary H3 LoRAs and acceleration recipes are
  not applied to Viggle. Source/custom audio conditioning remains experimental.

See [Viggle Animate](Viggle-Animate.md).

## H3 Voice Audio and saved voices throughout Speech

- **H3 Voice Audio — Pruned** generates speech, cloned voices or descriptive
  sound without decoding video. Output is **32 kHz stereo audio**.
- Individual generations support up to **45 seconds**; longer requests assemble
  into one output up to **five minutes**. Speech duration is a ceiling, so a
  short line does not fill the whole selection with silence.
- One or two voice references, named speaker turns, unspoken delivery cues,
  returning-speaker voice reuse and conservative Whisper boundary trimming
  support longer dialogue. Oversized scripts are rejected before queueing
  rather than losing their ending. Cancellation does not publish partial audio.
- `Sound:` prompts support sound effects and ambience. Native H3 prompts are
  also accepted; long speech requests extract dialogue into bounded turns.
  Generated wording and audio joins remain model-dependent.
- Every Speech workflow can access saved characters. Models keep their actual
  voice capabilities and speaker limits; Qwen preset/design variants offer an
  explicit switch to voice cloning. Character names reach prompt enhancement,
  and Load Settings restores names, references and character bindings.
- Characters can also be saved from Speech with an image/video and voice.

See [H3 Voice Audio](H3-Voice-Audio.md) and [TTS characters](TTS-Characters.md).

## H3 generation and refinement

- **H3 VDN First / Last — Full and Pruned** add the trained hybrid-attention
  path and dedicated eight-step presets. VDN requires Triton and additional
  VRAM. It has its own compatible adapters and does not combine with ordinary
  H3 Sol/SLA or Turbo controls. Upstream speed claims are not a Maestro benchmark.
- **H3 Fused 4-Step** Frames and References can use compatible H3 character,
  style and concept LoRAs, including in Director and restored settings.
  The normal four-step recipe is retained. Extra acceleration/VDN adapters and
  unsupported DoRA are excluded. Compatibility and quality remain experimental.
- **H3 Outpaint** extends video borders with ordinary Full/Pruned First/Last
  models, latent-aligned margins, protected source content, source-audio
  preservation and multi-window processing. Batch Outpaint is available too.
- Optional **H3 Audio Refinement Extra Phase** adds six steps at 0.5 audio
  denoising strength while locking the generated video latents. The pass omits
  LoRAs and reference/control reinjection. Fixed PDD/fused recipes and FL2VA
  source-soundtrack control are excluded.
- **H3 Face Refiner** detects, identity-tracks and refines up to five faces.
  Enable it in **Advanced → Finishing**, or use **Refine faces** on a gallery
  video/upload. Detect-and-map previews allow saved characters or RefMods to
  guide each face, retain its original identity, or skip it.
- Face refinement saves a new copy at the source resolution, frame count and
  frame rate with the original soundtrack. It shares the GPU queue and supports
  cancellation, model selection, strength and window settings.

See [fused H3 LoRAs](H3-Fused-LoRAs.md), [Face Refiner](H3-Face-Refiner.md)
and the [H3/media port record](development/wan2gp-12-71-port-plan.md).

## Media Flow, temporal upsampling and optional DLSS

- **Media Flow** batches image/video finishing through Maestro's existing
  queue with per-file progress, results and cancellation. Sources are retained
  and processing settings are saved with the outputs.
- **RIFE 4.26** adds x3 temporal upsampling alongside x2/x4. Generation finishing
  and batch tools retain the source duration and soundtrack while increasing FPS.
- Optional **DLSS 5 Neural Rendering** supports x1 refinement or x1.5–x3
  enlargement with intensity, depth and motion-estimation controls.
- Optional **DLSS Frame Generation** offers the factors reported by the native
  worker: x2–x4 on supported RTX 40/50 systems, and x5/x6 where supported on
  RTX 50. Unavailable factors remain disabled.
- Native DLSS requires a separate installation on compatible Windows 11/RTX
  hardware. The provided installer checks pinned downloads and presents the
  third-party runtime disclosure. Native binaries are not bundled or installed
  by normal Maestro Update. RIFE does not require that native runtime.

See [DLSS requirements and installation](DLSS5.md). Native DLSS image quality
and performance still need validation on a supported machine; the development
machine's Windows 10 environment cannot run that path.

## Fixes and polish

- Fixed the H3 Extend model-switch crash that could trigger React error #185
  when changing from a four-step model to Full/Pruned H3. Duration and native
  window reconciliation now use a stable model/GPU recommendation.
- Fixed prompt resizing, scrollbar flicker and text reflow during typing or
  background polling, including interactions with writing extensions.
- Fixed mobile keyboard geometry and internal scrolling so prompt fields stay
  reachable, including Animate's Image-mode handoff and appearance controls.
  The underlying media gallery stays in place while the drawer is open.
- Duration overlays fit the sidebar and keep a steady size. Single/multi-window
  transitions no longer move the slider being dragged; additional controls
  scroll inside the panel.
- Fixed Characters/output-control overlap at narrow widths and kept compact
  settings, enhancement menus and Generate anchored within the visible viewport.
- Fixed unintended horizontal scrolling in Director. Long messages, filenames,
  character/speaker fields and planning logs wrap within its vertical scroller.
- Model Browser's mobile toolbar and URL import form now fit the screen.
  Manual URL imports offer a **Destination LoRA folder**, defaulting to the
  best supported filename/metadata match and respecting an explicit override.
- Shared offload handling releases finishing/recovery models between GPU tasks.
  New queue jobs retain progress, cancellation and restorable output settings.
- Closed ETA-history database connections explicitly, fixing locked SQLite files
  on Windows. Wan/SCAIL model imports now defer default CUDA selection until
  encoder construction, allowing CPU-only validation without initializing a GPU.
- Added regression coverage for these model, character, dialogue, queue and UI
  paths, and cleaned up finishing-component refresh/state handling for release.

## Updating when the release is published

Use **Update** on Maestro's Pinokio page, then start normally and refresh the
browser. The application version is **2.1.0**; Pinokio's launcher schema version
is independent. Existing models, outputs, workspaces, characters, presets and
Director/Editor projects remain in their existing folders. No reset or reinstall
is part of this update.

Normal Update installs the new Face Refiner detector dependency. New model
assets download only when their features need them. The optional DLSS runtime
has its own installation procedure. Existing queued jobs retain their saved
prompt behavior; newly submitted Studio jobs use explicit enhancement.

This local preparation has not pushed `dev`, `main`, a tag, or a GitHub release.
See the [release validation record](VALIDATION_V2.1.0.md) for completed checks
and remaining hardware/device validation.
