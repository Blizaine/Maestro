# Maestro v2.4.0

September 24, 2026. Changes since v2.3.0.

## Gallery viewing and mobile

- Click an image to enlarge it. Browse images and videos in an immersive
  viewer with vertical swipes, keyboard navigation and a favorite button.
  Swipes move the media with your finger; native fullscreen is requested
  where supported by the browser.
- Keep heart and close controls over the media. Tap video to pause or resume;
  playback controls appear on interaction and fade while playing. Keep the
  selected sound setting when moving between clips.
- Prepare the next video during navigation and retain its preview until a
  decoded frame is ready, reducing the blank flash at the transition.
- Pinch to zoom images, drag to inspect details, then reset to resume swiping.
- Compare images using a before/after divider. Start with the active sidecar
  source, or choose another gallery image or local file for either side.
- Cache video first-frame posters so phones can display thumbnails without
  first decoding each video. Keep playback controls inside the visible mobile
  viewport as browser toolbars change size.
- Keep the sidecar collapsible in mobile landscape. Remember dismissal of the
  Home Screen tip, and use an opaque iOS Home Screen status bar so its shading
  does not overlap the app header.

Browser fullscreen and autoplay policies still apply. Opening Maestro from
the iPhone Home Screen removes Safari's toolbars.

## Gallery media as generation inputs

- Send gallery images and captured video frames into the active Studio or
  Director image input, including reference slots and start frames.
- Send whole videos into compatible control, reference, Animate, Retake,
  Repaint, Recast, outpaint and other video inputs.
- Menus name the available destinations, following the mode open in the
  sidecar instead of always assigning a Frames-mode start image.

## Qwen Image 2.1 improvements

- Add a dedicated CivitAI LoRA Browser filter and a separate Qwen 2.1 library
  destination for downloads and Hugging Face imports. Selecting a different
  model version uses that version's architecture for placement.
- Accept compatible AI Toolkit / ComfyUI LoRAs with fused feed-forward
  `gate_up` layers, retaining their trained weights and strength.
- Avoid large FP32 attention allocations during reference encoding on builds
  without native Flash Attention. Size the optional reference cache against
  available VRAM, recompute when necessary, and release it before VAE decoding.
  References and requested output dimensions are preserved.
- Allow cancellation between encoder layers.
- Show enabled, compatible Qwen image models, including Image 2.1, in
  Director's image selector. Image creation and reruns use the selected
  model's reference format, reference limit and native defaults.

Already on public main before v2.4.0: the Qwen Image 2.1 model shipped in
v2.3.0, followed by an unversioned hotfix for the first-use processor download
404. Those remain included and are not new v2.4.0 changes. Existing downloads
are reused. See the [Qwen guide](Qwen-Image-2.1.md).

## H3 Singularity — Experimental

- Add **Singularity v1.3 Pruned INT8 References** as a separate model in
  Studio and Director, using the existing reference workflow.
- Include the recommended **LightX2V Ref2VA Turbo4 v0.1** preset with four
  steps and strength 1.0. Turning Turbo off restores the ordinary 20-step
  default for this model.
- Download its checkpoint and adapter on first use, reuse shared H3 assets,
  and validate the managed file identities. Handle the checkpoint's grouped
  QKV / ConvRot layout and the adapter's alpha/rank scaling.

This is the reference-focused Pruned INT8 variant; Full and W4A8 variants are
not included. Quality and memory use need testing on the intended GPU.
See the [Singularity testing guide](H3-Singularity.md).

## H3 prompt enhancement and repair controls

- Better distinguish source actions from character descriptions, camera
  preferences and restrictions. Keep first-frame directions at the opening,
  and avoid treating scene references as additional characters.
- Improve checks for faithful paraphrases, ordered actions, distinct recurring
  events, dialogue ownership, starting poses and window-to-window state.
  Focus camera repairs on the affected event cards while retaining accepted
  neighboring work and exact source dialogue.
- Carry accepted visible action into the next window instead of treating an
  AI closing-state assertion as proof that an action happened. Retain source
  actions and continuous-shot instructions in compiled fallback prompts.
- With a **Music / performance timeline**, use the supplied audio for vocals
  and timing. Skip dialogue invention and word-count gates while retaining
  checks on the visual plan and performance timing.
- Add **Fidelity repair attempts** in **Settings → Integrations → Prompt
  Enhancement**. Choose 0–5 extra attempts for the story schedule and each
  failed camera window; the default is one. Single-window H3 repairs honor
  the same setting. More attempts may take longer.
- Add **Generate even if fidelity checks fail**, off by default. With Enhance
  on generation, this uses the saved usable draft after repairs instead of
  pausing for review. Warnings remain available. Loading failures, cancelled
  work, empty/invalid drafts and generation errors still stop the job.
- Queued jobs preserve their enhancement settings. Explicit retry/refresh
  picks up the current repair controls while retaining the job's writer,
  source prompt and references.

These changes reduce false warnings and improve repair; they do not guarantee
perfect choreography or continuity. Complex multi-window drafts can still
repeat an action, miss a prop placement or mishandle an audio-tail instruction,
including cases not caught by a warning. Keep review enabled for important
scenes. See [validation and remaining limits](VALIDATION_V2.4.0.md).

## Director music and dialogue

- Apply performance direction only to people actually visible and assigned a
  role in the shot. Narrative, dance and scenery shots no longer inherit a
  generic list of guitarists, bassists and drummers.
- Preserve visible performers' vocal roles, instrumental gaps and explicitly
  requested expressions. Remove the old injected musician boilerplate when
  recompiling saved music prompts.
- Repair malformed duplicate H3 dialogue tags using the structured exact lines
  and speakers, retaining surrounding visual action and per-line language.
- When supplied audio drives a clip, retain its transcript as timing metadata
  without generating duplicate speech.

## Reliability and integrations

- Release Face Refiner's mapped frame storage before Windows cleanup, retry
  transient file locks, and preserve completed work and original exceptions.
- Compose Recast masks frame by frame to avoid large whole-video temporary
  indexing allocations; retain color mappings, priority and overlap detection.
- Preserve real API keys containing ellipses, recognize saved masked values,
  wait for saving to succeed, and retain previous settings on persistence
  failure. Keep the editor open with an error if a save fails.
- Send image references to remote LLMs in both streaming and non-streaming
  requests, including Anthropic's native image payloads. Enhancement model
  overrides preserve the chosen provider and credentials.
- Bundle the DramaBox speech and dialogue guides to eliminate their
  missing-guide startup warnings.

## GitHub coverage

Implementation and regression coverage are included for:

- [#140](https://github.com/Blizaine/Maestro/issues/140): bundled DramaBox guides.
- [#143](https://github.com/Blizaine/Maestro/issues/143): API-key saving.
- [#145](https://github.com/Blizaine/Maestro/issues/145): Recast mask memory.
- [#146](https://github.com/Blizaine/Maestro/issues/146): Face Refiner cleanup.
- [#148](https://github.com/Blizaine/Maestro/issues/148): Director H3 dialogue
  markup, speaker/language and supplied-audio handling.
- [PR #136](https://github.com/Blizaine/Maestro/pull/136): remote LLM vision and
  provider routing support.

This is implementation coverage, not a claim that these issues or the PR have
been closed. The slower RTX 3060 investigation, queued Director shot reruns,
GB10/MLX contributions and localization are not included as completed fixes.

## Updating

After publication, use Pinokio's **Update**, restart Maestro and refresh the
browser. Existing models, references, recordings, LoRAs, projects and outputs
remain in place. New Singularity assets download only when that model is used.
Enhance again from the source prompt to apply the updated planning guidance to
a previously saved draft; existing generated clips are unchanged.
