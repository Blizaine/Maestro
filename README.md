# Maestro

A one-click AI **video, image, and audio studio** for creators. Maestro pairs a modern React UI with a powerful generation backend and adds a **Director mode** that uses an LLM to plan music videos and short films from a single prompt. Optimized for the latest LTX-2.3 models & LoRAs, with support for virtually all open weight models.  

![Maestro UI](Maestro_UI_02.jpg)

## What it does

### 🎬 Director Mode — automatic music videos and short films
The flagship feature. Drop in an audio track or write a story; a local LLM plans every shot, writes screenplays/lyrics, generates start frames & keyframes with character consistency, polishes prompts per model & LoRA-specific prompting guides, and runs the full multi-clip generation. Two skills:

- **Music Video** — beat-aware shot planning aligned to your audio. The LLM analyzes BPM, sections (verse/chorus/bridge), and energy, then writes shots that hit the downbeats. Speaker transcription & diarization lets you name and target different voices or singers.
- **Short Film** — screenplay-driven scenes with named characters, dialogue, and continuity across cuts. Pacing-bias slider controls cut frequency.
  
- **Auto Mode** runs the entire pipeline end-to-end (analyze → plan → generate images → generate clips → combine). Manual mode lets you review and edit at every step.
- **Director v2 architecture** with structured shot planning, mode-specific prompt renderers, and a 3-pass refinement (screenplay → shot breakdown → per-model polish). Director v2 optimizes what the LLM is being asked to do across several passes, with each pass optimizing the LLM request for creativity (when writing the screenplay), structured outputs (when outputting JSON), and prompt refinement, which injects LoRA prompting guides into the context.  

### ⚡ Performance Auto-Tune — zero-config setup
Detects your GPU, VRAM, and RAM on first launch and picks the right profile, quantization, VAE tiling, and VRAM safety coefficient. No more "Profile 1 vs 2 vs 4.5" guesswork. Power users still have full manual control under "Show advanced settings."

- **OOM recovery banner** auto-suggests lowering the VRAM headroom when a generation runs out, with one-click apply.
- **Live download status** during model setup ("Downloading transcription model (first use downloads ~300MB)..." instead of a vague spinner).

### 🎨 Studio Mode — full manual control
Direct access to every model and every knob:
- **Video** — MiniMax H3 with native synchronized audio, LTX-2.3, Wan1/2, Hunyuan, and many more.
- **Image** — Flux 2 Klein 9B, Krea 2 RAW/Turbo and Identity Edit, Qwen Image Edit, and many more
- **Audio** — TTS: Kugelaudio, Qwen3 TTS. Music: ACE-Step. SFX: MMAudio
- **Multi-clip generation** with per-clip prompts, seamless overlapping (sliding window) transitions, and shared LoRAs
- **Blend video Mode** Remember Sora 1 blend mode, where you could overlap two videos, and use AI to blend them together? 
- **Frames Injection (KFI)** for character continuity in long videos
- **Sliding window** for arbitrarily long generations
- **Spatial upsampling, film grain, codec selection** as post-processing options

### 🤖 Local LLM — built-in, no setup
Maestro auto-downloads `llama-server` (~600 MB one-time) and your chosen GGUF model on first use. Defaults to **Gemma 4 4B (Recommended)** — fast, capable, and runs comfortably on smaller GPUs. Auto-detects CUDA and binds the LLM to GPU when available.

- Pre-curated registry: Gemma 4 (2B / 4B / 26B MoE / 31B) and Qwen3.6 27B — uncensored/abliterated instruct variants tuned for creative prompting
- **External providers** also supported: OpenAI, Anthropic, custom OpenAI-compatible endpoints (currently experimental)
- **Vision support** so LLMs can enhance prompting based on reference images
- Auto-unloads after 60s idle to free VRAM for video gen

### 🛒 Built-in CivitAI LoRA browser
- Search, filter, and one-click install any LoRA from CivitAI without leaving Maestro
- **LoRA update detection** — Check button refreshes from CivitAI, shows update badges on outdated LoRAs
- **My LoRAs view** with filters for Updates and direct uninstall
- **AI-generated LoRA prompting guides** Helps remove the guesswork from LoRAs. AI generates LoRA guides when LoRA is downloaded based on CIVITAI and HuggingFace repos. The guides explain what each LoRA does and how to use it, provide prompt examples, and recommend weight settings that are automatically applied when LoRA is selected. 
- **Recommended weight ranges** (sourced from CivitAI sidecars, HuggingFace, or fallback heuristics) shown directly on the weight sliders
- **Multi-LoRA pack auto-extraction** for archives that bundle several LoRAs

### 🎭 Themes
Three theme families, each with a dark and a light variant, switchable in Settings → System:
- **Golden Hour** (default) — warm cinematic palette with sunset-gradient CTAs and spotlight bezels; warm paper with burnt orange in daylight
- **Classic** — the original cool charcoal palette with blue accents; cool paper in daylight
- **Onyx** — minimalist monochrome, pure black with neutral grey surfaces; white and grey in daylight

Appearance mode is **Dark / Light / Auto** — Auto follows your system's appearance and switches live when it changes.

### 🛠️ Edit Mode
- **Retake** — re-roll a section of an existing video with a new prompt
- **Edit Anything** — modify, add, or remove elements from existing videos using text prompts and In-Context LoRA models
- **Outpaint** — extend a video's frame in any direction while preserving its original action, timing, and audio
- **Repaint** — use SCAIL-2 to repaint characters, objects, or scenes while retaining the source motion and camera work
- **Recast** — map one or more people in a video to replacement characters, including multi-shot scenes and group shots

### 📂 Workspaces
Multiple isolated output directories with a quick switcher in the sidebar. Useful for separating client projects, NSFW vs SFW, or experiments. Pinned and favorited outputs are tracked per workspace.

### 🔒 Mature mode + experimental gate
- **NSFW mode** is opt-in with a disclaimer step. Disabled by default. Gates uncensored model variants, NSFW LoRAs in the CivitAI browser, and the Settings → Services NSFW toggle.
- **Experimental features gate** hides power-user toggles (external API keys, Voice Reference, Inpaint, Restyle, Wan2GP Enhancer) by default for a focused first-launch experience.

### 📊 Director Pipeline Dashboard
View all past Director runs with their full state — clip plans, generated images, generated clips, polish diffs. Re-run any clip without re-running the whole pipeline.

## Updates

The version you are running is shown next to the Maestro title in the UI. To update, use the launcher's Update button in Pinokio.

### v1.7.5 (2026-08-11)

**MiniMax H3 Performance Update**
- Added the experimental H3 Sol Engine sparse-attention backend for supported RTX 40- and 50-series GPUs, with cached kernel compilation and an automatic safe fallback.
- Added one collapsible H3 Optimizations panel for Turbo, Sol Engine, and First Block Cache; each can be enabled independently or combined.
- Updated the managed Turbo default to the newer v4-600 EMA LoRA at six steps and strength 1.0, while retaining the previous preset for rollback and comparison.
- Added pinned checksums, atomic downloads, local receipts, and a scheduled upstream-change monitor for managed H3 Turbo releases.
- Added the same Turbo preset, Sol Engine, and First Block Cache controls to Director. Settings now survive project saves, Dashboard regeneration, repair, and resume.
- Hardened RTX 50's CUDA 13/Triton runtime and made interrupted optional RTX 40 Sol installations repairable through Update without replacing the normal runtime.

### v1.7.2 (2026-08-11)

**MiniMax H3 sequence and compatibility fixes**
- Fixed legacy Director projects and uploaded audio/video producing ordinary frame counts that H3 rejected as outside its native frame lattice.
- H3 now repairs those clip schedules without accumulating timeline drift, and saved Director projects mark affected clips for safe Dashboard regeneration.
- Fixed manual First / Last multi-window generation applying the complete multiline prompt to every window; each line now drives exactly one window, with prompt-count validation before model loading.
- Fixed H3 GGUF image and video reference conditioning failing on mixed FP16/FP32 vision-encoder weights, without changing the established NVFP4/AWQ path.
- Expanded regression coverage for saved-project timing repair, media-derived clip lengths, manual window routing, and GGUF visual references.

### v1.7.1 (2026-08-10)

**MiniMax H3 memory stability**
- Fixed long 540p H3 generations becoming slower and exhausting VRAM while the same 720p workload succeeded.
- Rebalanced transformer residency and activation workspace smoothly across clip lengths and resolutions to avoid Windows shared-GPU-memory paging and excessive Copy activity.
- Added safer full-duration projection chunking for Full and Pruned checkpoints, validated in both First / Last and Omni workflows.

### v1.7.0 (2026-08-10)

**MiniMax H3 native multi-window generation**
- Added native multi-window continuation to both First / Last and Omni, carrying recent motion and synchronized stereo audio into each following window for smoother transitions.
- Added shared Multi-window controls with total duration, editable per-window prompts, optional AI planning, and independent hard-cut sequences.
- Improved H3 sequence planning so actions, dialogue, camera cuts, sound, and story events advance across windows instead of repeating or finishing in the first clip.
- Added exact-duration assembly, model-aware overlap handling, and saved runtime prompts so long generations can be reviewed, edited, and reproduced.

**New H3 media workflows**
- Added multiple timed frame injection for First / Last generations.
- Added audio-driven video from an uploaded soundtrack or a control video's audio.
- Added video-to-audio generation that preserves the source pictures while creating a new synchronized soundtrack.
- Added H3 video-to-video editing for the whole frame, inside a mask, or outside a mask, with adjustable denoise and mask strength.
- Fixed Omni music and performance references restarting from the beginning in every sequence clip; each window now receives the correct timeline segment while voice references remain reusable.

**Memory, performance, and RTX 50 support**
- Added VRAM-, model-, and resolution-aware H3 window recommendations with a native 14.4-second ceiling and saveable user overrides for proven hardware combinations.
- Improved transformer residency, activation workspace, streaming VAE decoding, RAM budgeting, and LoRA fallback behavior for Full and Pruned H3 models.
- Added a dedicated RTX 50 / Blackwell runtime with Python 3.11, PyTorch 2.10, CUDA 13, compatible acceleration kernels, automatic migration, startup diagnostics, and one-click repair.
- Preserved the established runtime for RTX 20/30/40 systems while applying hardware-specific setup only where required.

**Workflow and interface reliability**
- Added early validation for incompatible H3 media, prompt counts, durations, and sequence settings before expensive model loading begins.
- Fixed Omni reference uploads on iPhone and iPad so supported audio files are selectable even when iOS reports unusual file types.
- Generation cards now show active generation time in minutes and seconds, excluding time spent waiting in the queue or loading models.
- Expanded regression coverage for H3 continuation, reference packing, audio timing, frame injection, video editing, memory recommendations, RTX 50 setup, and the shared multi-window interface.

### v1.6.5 (2026-08-08)

**MiniMax H3 performance and lower-VRAM support**
- H3 Turbo now works with the recommended Pruned 20B models as well as the optional Full 33B models.
- Turbo now starts at six steps and LoRA strength 0.50, while keeping the LoRA visible and adjustable in Advanced settings.
- Reworked H3 model residency, activation chunking, and VRAM budgeting to reduce step-zero out-of-memory failures and excessive CPU offloading.
- Added resolution- and GPU-aware First / Last window recommendations, with clear warnings and a manual override for experimental combinations.
- Added an optional experimental First Block Cache for faster H3 generations, with selectable quality/speed thresholds.

**H3 resolutions and long-video planning**
- Added a faster model-aligned 720p tier using 1280x704 landscape output and matching portrait, square, and 4:3 canvases.
- Restored 1080p H3 generation with an experimental note and hardware-aware shorter-window recommendations.
- Hid the less efficient 768p preset from the main selector while retaining compatibility with existing saved settings and API requests.
- Added automatic H3 sliding-window storyboarding: one idea is expanded into a complete, editable prompt for every continuation window.
- Actions, dialogue, camera coverage, sound effects, ambience, and music are distributed across the timeline instead of being completed and repeated in the first window.
- Each exact window prompt is visible during generation in its own full-height editor, with the active window highlighted and no nested scrollbars.

**Director H3 workflow improvements**
- Director now uses the same H3 resolution, VRAM, and native-frame rules as Studio when planning shot lengths and execution profiles.
- Long scenes are divided before generation to fit the selected model, resolution, GPU, and Turbo configuration instead of being silently shortened at runtime.
- Added H3 Turbo controls and adjustable per-LoRA strengths directly to Director mode.
- Improved independent-shot context so recurring characters, wardrobe, locations, blocking, dialogue, and sound remain self-contained across prompt-only H3 shots.

**MiniMax LoRA discovery and compatibility**
- Added a MiniMax H3 filter to the CivitAI browser and routed downloaded H3 LoRAs into the correct shared H3 folder.
- Pasted Hugging Face MiniMax H3 LoRA URLs now use the same correct destination instead of defaulting to LTX.
- Added automatic H3 LoRA architecture conversion where required so compatible adapters can run on both Pruned and Full checkpoints.
- Added early validation, pinned support assets, and clearer recommendations for combinations that may exceed available VRAM.

### v1.6.1 (2026-08-06)

**MiniMax H3 Turbo mode**
- Added the H3 Turbo LoRA to the Full H3 model lists as a managed, first-use download.
- Added an experimental one-click Turbo mode for Full First & Last and Full Omni models.
- Turbo mode uses six inference steps and starts at LoRA strength 0.70.
- The active Turbo LoRA is shown in Advanced settings so its strength can be tuned per generation.
- User-adjusted Turbo strengths are preserved while duplicate Turbo adapters and incompatible Pruned-model combinations remain blocked.

### v1.6.0 (2026-08-06)

**MiniMax H3 Omni Reference**
- Added MiniMax H3 Omni for generating new video and synchronized audio from ordered image, video, voice, motion, and sound references.
- References can be reordered, labeled with their intended role, and used for identity, appearance, scene, motion, voice, performance, ambience, or music conditioning.
- Added both recommended Pruned 20B and optional Full 33B Omni models.
- Added Match Output reference preparation for consumer GPUs and an optional Maximum Detail mode for higher-memory systems.
- Improved reference-video memory use with output-aware sizing, chunked projections, dedicated attention workspace, and safer model re-profiling.

**Expanded H3 models and performance options**
- Simplified the model choices to First & Last and Omni, with clear Pruned 20B and Full 33B variants and concise explanations in the selector.
- Added Full 33B support for both workflows, including ConvRot checkpoint loading, fused projection handling, and memory-efficient streaming.
- Added selectable NVFP4-AWQ, GGUF Q2/Q4, Quanto INT8, and BF16 Qwen3-VL text encoders with hardware-aware recommendations.
- Added support for the MiniMax H3 Turbo LoRA on compatible Full 33B models with true 4, 6, and 8-evaluation schedules.
- Incompatible Turbo LoRA and Pruned-model combinations are rejected before loading instead of failing after a long generation.

**H3 Studio workflow and prompting**
- Omni generations are limited to the native 345-frame maximum: 14.375 seconds at 24 FPS, displayed as 14.4 seconds, with sliding-window controls automatically hidden.
- First & Last uses the same native 14.4-second maximum per window and can now generate longer videos by continuing each window from the preceding final frame.
- Long First & Last runs preserve the requested duration, remove continuation overlap, keep synchronized audio aligned, and apply an optional end image only to the final window.
- Fixed portrait and other selected aspect ratios being forced or decoded as 16:9.
- Improved H3 Prompt Enhance for exact dialogue retention, stable speaker IDs, voice-reference intent, opening ambience, silent intervals, and reduced gibberish or invented speech.

**MiniMax H3 in Director**
- Added model-aware Director workflows for both First & Last and Omni models.
- First & Last can create prompt-only shots or use optional generated start/end frames, while Omni can condition shots on character, location, voice, video, soundtrack, and other project references.
- Director no longer spends time writing or generating unused start images for H3 prompt-only workflows.
- H3 shot prompts now carry the project world, location, wardrobe, character blocking, screen position, dialogue, soundscape, and continuity needed by independently generated clips.
- Added stable project-wide speaker mapping, locked screenplay dialogue, duration-aware pacing, and multi-speaker exchanges with camera changes inside a single H3 clip.
- Incomplete or altered local-LLM shot plans are repaired deterministically without silently truncating, moving, duplicating, or rewriting approved dialogue.
- Dashboard repair and regeneration recreate the same H3 references and timing, including exact per-shot audio conditioning and one clean final soundtrack.

**Compatibility and reliability**
- Director model lists now show only image and video models that support the selected automated workflow.
- Native audio generation is distinguished from audio-reference input so incompatible models are no longer offered for audio-driven jobs.
- Reduced console noise by hiding successful system-stat polling while retaining failures and meaningful API requests.
- Interrupted saved Director jobs are now reported as interrupted instead of disappearing as missing projects.
- Expanded automated coverage for H3 checkpoints, quantization, Omni reference packing, Turbo LoRA, Studio continuation, Director compatibility, dialogue planning, memory behavior, and UI contracts.

### v1.5.5 (2026-08-04)

**MiniMax H3 local audio-video generation**
- Added native local MiniMax H3 Base FL2VA support with text-to-video, image-to-video, and first/last-frame video generation.
- H3 generates synchronized 32 kHz stereo audio together with the video instead of requiring a separate audio pass.
- Added approximately 5-15 second generation at 24 FPS with landscape, portrait, square, native 768p, and lower-VRAM resolution options.
- Added automatic, revision-pinned provisioning for the compact scaled-FP8 transformer, NVFP4 Qwen3-VL conditioner, video VAE, audio VAE, tokenizer, and processor assets.
- The initial integration focuses on H3 Base FL2VA; H3 Ref2VA reference-video/audio conditioning and hosted 2K regeneration are not yet included.

**H3 prompting and dialogue**
- Added an H3-specific Context-IR Prompt Enhance workflow using the model's native multimodal description, soundscape, music, speaker-ID, and dialogue-tag structure.
- Vague requests such as two characters discussing a subject can now be expanded into concise, meaningful dialogue sized to the selected duration.
- User-supplied dialogue is preserved verbatim, and remaining time is assigned to silent visible action to reduce invented speech and gibberish.
- Start-frame prompts now receive the correct H3 image-alignment instruction while raw prompting remains available by simply not using Prompt Enhance.
- H3 enhancement bypasses the incompatible generic cinematic enhancer and remains one native timeline instead of being divided into false sliding-window paragraphs.

**H3 compatibility, memory, and reliability**
- Corrected compact Qwen3-VL prompt conditioning so H3 follows the requested subject instead of producing unrelated repeated scenes.
- Added native row-scaled INT8 embedding support and corrected NVFP4 pre-quantization and combined-scale handling for Comfy-format checkpoints.
- Fixed mixed-dtype model profiling, keyframe CPU/CUDA device mismatches, and first-frame generation failures.
- Added activation chunking, explicit transformer working-memory reservation, and MMGP-friendly dtype locks so H3 can stream on consumer GPUs without starving the first denoising step.
- Added regression coverage for prompt conditioning, quantized checkpoint loading, keyframes, scheduler behavior, audio output, activation chunking, and H3 prompt structure.

**Multi-character Recast continuity**
- Improved SCAIL-2 Recast when a mapped character enters later within an otherwise continuous camera shot.
- Added hidden identity pre-roll conditioning so late-arriving characters can be introduced without publishing an artificial visible cut.
- Recast assembly now validates that all generated segments are present and that the final output retains the exact source timeline length.

### v1.5.0 (2026-08-02)

**SCAIL-2 Recast and multi-character replacement**
- Rebuilt Recast around SCAIL-2's native replacement conditioning for substantially stronger identity transfer and motion tracking.
- Added color-mapped character cards for replacing up to five people in one run.
- Added camera-shot detection and per-shot processing so characters remain correctly mapped when a video cuts between close-ups, wide shots, and group shots.
- Improved two-person and multi-person shots by conditioning each shot only on the characters visible in it.
- Added automatic reacquisition when a person first appears later, leaves the frame, or returns after a camera cut.
- Other people in the scene are now preserved automatically when bystanders are detected.
- References are automatically isolated from their backgrounds, aligned to the target, and supplemented with a face-detail view when useful.
- Added optional lighting and shadow matching using Z.ai's official SCAIL-2 Relighting LoRA, downloaded, verified, and converted automatically on first use.
- Added 480p, 512p, and 704p quality profiles with VRAM-aware window sizing; model steps remain independently adjustable.
- Fixed reference-image backgrounds, white bars, halos, false gray scenes, blurry identity starts, and reference stills appearing at the beginning of output videos.
- Fixed mismatched reference and control-video aspect ratios causing tensor errors or allowing the character image to control the output canvas.

**SCAIL-2 Repaint**
- Added Repaint as a first-class Edit mode for changing characters, objects, or the visual treatment of a video while retaining its motion and camera path.
- Repaint detects camera cuts, processes each shot independently, and rejoins the exact source timeline with one continuous audio track.
- Added multi-region and multi-character mapping with stable colors across shots.
- Repaint now shares Recast's 480p, 512p, and 704p resolution profiles and adaptive VRAM windows.
- Wired inference steps and applicable guidance controls to the generation pipeline while hiding advanced settings SCAIL-2 does not use.
- Simplified the Repaint and Recast interfaces, moved detailed guidance into tooltips, and ordered Edit modes as Retake, Edit Anything, Outpaint, Repaint, and Recast.

**LTX-2.3 Outpaint and Retake**
- Rebuilt Outpaint around LTX-2.3's official In/Outpainting IC-LoRA workflow with mask-preserving source conditioning.
- Added shot-aware Outpaint: multi-scene videos are split at camera cuts, processed independently, and reassembled at the exact original frame count with the source audio restored.
- Improved seams, detail, color-temperature matching, and removal of green/yellow marker spill without grading the protected source region.
- Source pixels remain protected while the full source frame stays available as visual context for newly generated areas.
- Output canvas dimensions now follow the selected quality preset and display the actual aligned pixel size before generation.
- Fixed Outpaint ignoring visible inference-step settings, using invalid schedules, or failing immediately on supported LTX models.
- Fixed Retake failing on LTX-2.3 distilled and two-stage pipelines.

**Krea 2 image generation and editing**
- Added Krea 2 RAW Identity Edit and Krea 2 Turbo Identity Edit using the current Krea 2 vision-conditioning pipeline and Identity Edit v1.2 LoRA.
- Added identity-preserving instruction edits, inpainting, outpainting, background removal, and support for up to two total reference images.
- Added automatic Qwen3-VL vision-encoder provisioning and accurate installed/readiness checks.
- Added compatibility with current Diffusers, Kohya, and GGUF Krea 2 weight formats.
- Added a dedicated Krea 2 filter to the CivitAI browser and My LoRAs view, with downloads routed to the correct Krea 2 library.
- Krea 2 RAW, Turbo, RAW Identity Edit, and Turbo Identity Edit are now enabled by default in Image mode for new and existing installations.

**Studio, models, and control video**
- Enabled-model choices now persist server-side across Maestro restarts and changing Pinokio ports.
- Newly downloaded CivitAI checkpoints appear in model selectors immediately without restarting Maestro.
- Control video and audio behavior are now independent in Frames mode: keep source audio, generate audio from the prompt, or use an uploaded soundtrack.
- Missing Temporal Depth assets for LTX control-video workflows are downloaded with progress, resume support, hash verification, and atomic installation.
- Voice Reference is now a standard feature, enabled by default and no longer hidden behind the in-development feature switch.
- Cleaned up Recast and Repaint Advanced Settings so only controls used by the selected SCAIL-2 pipeline are shown.

**Reliability and fixes**
- Director no longer creates a duplicate combined file when a run contains only one finished clip.
- Fixed SCAIL-2 relighting and user LoRAs failing validation when stale multi-phase weights were present.
- Fixed installed Maestro apps being hidden or blocked by an early Pinokio NVIDIA detection failure.
- Added broad regression coverage for SCAIL-2, Repaint, Outpaint, Retake, model visibility, temporal-depth downloads, and Krea 2 editing.

### v1.4 (2026-07-20)

**Storage and space optimization**
- Added a full Storage Manager with usage analytics and cleanup recommendations.
- Added safe deletion for workspaces, saved Director projects, models, and LoRAs.
- Added duplicate model and LoRA detection across linked installations.
- Added safe duplicate reclamation while preserving a verified copy.
- Added optional removal from linked installations through the Windows Recycle Bin.
- Improved storage accounting for shared weights, linked folders, junctions, symlinks, and hardlinks.

**LoRA management**
- Added LoRA file sizes, release dates, download dates, and compact age indicators.
- Added sorting by name, newest download, newest release, or file size.
- Added newest-first sorting to the Studio and Director LoRA selectors.
- Improved explanations for shared-weight, linked-only, and otherwise protected files.
- Added CivitAI response caching for faster browsing and fewer rate-limit problems.

**Director Dashboard and repair**
- Added a durable Check + Repair workflow for saved Director projects.
- Repair can regenerate missing images and videos, skip valid clips, and automatically rejoin the result.
- Repair continues when the browser is refreshed or closed.
- Interrupted repairs can be resumed without repeating completed clips.
- Fixed repair stopping after generating only one image or video.
- Fixed missing thumbnails, incorrect missing-clip counts, and incomplete clip tracking.
- Regenerating a start image now correctly marks its existing video for regeneration.
- Rejoin now rejects missing, invalid, or stale clips instead of creating an incomplete video.
- Dashboard operations now remain responsive while regeneration or repair runs in the background.

**Director character consistency**
- Director now generates an establishing character image when no reference image is supplied.
- The generated image becomes the shared reference for all subsequent start images.
- Character references and profiles are incorporated into the generated anchor.
- Generated start images are now correctly supplied to their corresponding video clips.
- The generated reference is retained for later Dashboard regeneration.

**Music-video timing and lip sync**
- Fixed Dashboard-regenerated clips becoming shorter than their original timeline slots.
- Regenerated clips now use the same FPS and frame schedule as a complete Director run.
- Fixed cumulative lip-sync drift after replacing one or more clips.
- Fixed rejoined videos using the wrong starting point in the source song.
- Rejoined videos continue to use one clean, continuous soundtrack without audible clip-boundary blips.
- Dashboard audio conditioning now matches the exact timeline segment assigned to each clip.

**Job cancellation and reliability**
- Significantly improved Stop and Cancel behavior across Director and Studio.
- Queued and actively generating child jobs are now canceled together.
- Late completion or failure can no longer overwrite a canceled job.
- Improved timeout handling and cleanup of partial outputs.
- Made Director state saving atomic to prevent damaged project files.
- Prevented delete, resume, repair, and regeneration operations from conflicting with one another.

**Downloads and model installation**
- Added clearer model and LoRA download progress, completion, failure, and retry states.
- Fixed inaccurate download percentages.
- Prevented concurrent downloads from writing to the same destination.
- Incomplete or corrupted downloads are no longer published as installed models.
- Hardened CivitAI archive extraction against unsafe paths and invalid files.
- Improved cleanup of failed and interrupted downloads.

**Safety, compatibility, and stability**
- Improved Director's minor-content safety checks.
- Improved detection across deeply nested planning data while reducing common false positives.
- Fixed sidebar crashes when changing models or generation modes.
- Improved NVIDIA GPU compatibility checks during Pinokio installation.
- Expanded automated regression testing for both dev and main.

### v1.3.3 (2026-07-17)

**Fixed**
- **Recast no longer crashes when the person leaves the scene.** If the target walked out of frame partway through the clip (or only appeared later in the video), the tracking step died with a cryptic "No points are provided" error and took the whole job with it. Tracking now locks on wherever the person first appears, works in both directions from there, and if it loses them mid-video it keeps everything tracked so far and picks them back up when they return. Frames where the person genuinely is not present simply keep the original footage, which is what replace mode should do. Both underlying bugs exist in upstream WanGP too; a keyword that matches nothing in the video now shows the friendly "could not find" message instead of a traceback.

### v1.3.2 (2026-07-17)

**New**
- **Models can be downloaded ahead of time.** In Settings -> System -> Enabled Models, the download icon next to each model is now a real button: click it and Maestro fetches everything that model needs (weights, text encoder, add-on modules, bundled LoRAs) in the background, with progress in the download banner. The row flips to a check mark when it finishes. Generating still auto-downloads on first use as before; this just lets you get the wait out of the way on your schedule.

**Fixed**
- **Recast no longer crashes on a fresh install.** The automatic masking step runs before the SCAIL-2 model loads, but its detector checkpoint only downloaded together with the model, so the very first Recast on a clean install failed with "SAM3.1 checkpoint was not found". The masking step now downloads the detector itself on first use.
- **The downloaded check marks tell the truth now.** Models that borrow their weights from a base model (SCAIL-2 14B Fast, the Z-Image ControlNets) always showed as not downloaded, even when they were ready to run. The check now follows those references and also requires add-on modules and bundled accelerator LoRAs, so a check mark means the model generates without downloading anything.
- Deleting a model now removes only the files that belong to it, so deleting a finetune leaves shared base weights in place for the models that still use them.
- SCAIL-2's image reference no longer fails when the detection phrase finds nothing in your character image; Maestro automatically falls back to broader phrases ("person", "woman", "man").

### v1.3.1 (2026-07-17)

**Fixed**
- **Model downloads no longer fail when your saved Hugging Face token has gone stale.** A stale or expired token made Hugging Face reject even public files with a misleading "Repository Not Found" (reported as the SCAIL-2 download failing, issue #20). Maestro now detects the rejection and retries the download anonymously, which covers everything Maestro ships. Valid tokens are still used first, so gated models keep working.
- Recast's Advanced Settings no longer show resolution and window controls that the generation ignores (Recast runs at SCAIL-2's native 480p with its 81-frame windows).

### v1.3.0 (2026-07-17)

**New: SCAIL-2 character animation.** Z.ai's follow-up to SCAIL Preview, integrated end to end. It transfers a performance from any video onto any character with no skeleton extraction, and it comes in two flavors: **SCAIL-2 14B** (the full native 40-step model) and **SCAIL-2 14B Fast** (bundled lightx2v distill, 6 steps, and no CFG for rapid animation). Fast is the recommended starting point for Recast, though results can vary by seed. Both are enabled by default. About 16.6 GB downloads on first use, plus a small detector model.

- **Animate (Video tab).** Pick SCAIL-2 in Frames mode, drop a character image as the Start Image and a performance clip on the new Control Video tile, generate. The character performs the clip's motion in their own scene. Output follows the source clip's frame rate (capped at 30fps) and keeps its audio.
- **Recast (Edit tab).** The headline: replace a person in an existing video with your character. Drop a video, type who to replace ("woman", "man in red"), preview the selection with the eye button, drop the character image, generate. Masking is fully automatic (SAM3 keyword tracking), and the scene, camera, and audio are preserved. The prompt is optional; describing the new character helps identity.
- **Use current frame as reference.** Gallery videos now have the same left-arrow button images have: scrub the preview to the moment you want and click to send that exact frame to the Reference tiles, which is the perfect way to pick a character out of an existing clip for Recast.

**Fixed**
- Sliding-window, frame-rate, and audio defaults now reach generations reliably (previously a 10s SCAIL-2 run could go out as one giant window and overflow VRAM, render at 16fps instead of the source rate, or come back silent).
- SCAIL-2's VRAM budget now accounts for its in-context conditioning (it carries the driving video as extra tokens), so 480p multi-window runs fit a 24 GB card with room to spare instead of spilling into system memory.
- "10 seconds" now means 10 seconds of your source clip regardless of its frame rate, and 60fps sources no longer double the generation work.
- Queued Recasts wait their turn for the GPU instead of running their detection passes on top of the active generation.

### v1.2.8 (2026-07-16)

**Fixed**
- **Linked LoRAs now show up in My LoRAs.** The library view only listed Maestro's own loras folder, even though guide generation and the Studio selectors already saw LoRAs from Linked Model Folders. My LoRAs now lists them too — with their names, previews, and generated guides — and marks them with a "Linked" badge so you can tell which library each one comes from.

### v1.2.7 (2026-07-16)

**Fixed**
- **LTX generation crash ("TypeError: not a string") on Linked Model Folder installs** — the follow-up to v1.2.6's text-encoder fix. That fix created Maestro's own Gemma folder to hold the downloaded weight, but the folder then hid the linked install's complete folder that has the tokenizer files, and the tokenizer loader crashed. Maestro now completes its own folder with the missing tokenizer files automatically (about 40 MB, once), and folder lookups skip folders that don't actually contain what's being looked for. Affected installs heal themselves on the next generation.

### v1.2.6 (2026-07-16)

**Fixed**
- **Endless re-downloading of text encoder models (Gemma, Qwen) on installs using Linked Model Folders.** When the text encoder's target folder didn't exist yet, the downloaded weight was silently renamed to the folder's own name instead of being placed inside it, so Maestro could never find it: every generation re-downloaded the full 13 GB and then crashed with "Loading Text Encoder 'None'". This only happened when a linked install (like an existing Wan2GP) already provided the folder's tokenizer files, which skipped the step that normally creates the folder. The fix also removes the misnamed leftover file automatically, so affected installs heal themselves on the next generation — just update and generate.

### v1.2.5 (2026-07-16)

**Fixed**
- **Black screen on launch for some Windows machines.** The UI's JavaScript was being served with a wrong MIME type on machines where a registry entry was hijacked (Python reads MIME types from the Windows registry), and browsers silently refuse to run module scripts served that way. Maestro now forces the correct types server-side no matter what the registry says. If the UI ever fails to start for any other reason, the black screen is replaced after 10 seconds with a diagnostic page listing recovery steps instead of leaving you guessing.
- The Classic UI link printed at startup was missing its trailing slash and returned a 404. Both the link and the bare /classic path work now.

### v1.2.4 (2026-07-15)

**Fixed**
- **Director now truly holds a stylized reference's art style.** Telling the image model to "preserve the art style" at the end of a prompt does nothing; what works is naming the medium at the very start. Director now looks at your reference once per run, names its style concretely ("black and white cartoon illustration"), and automatically leads every image prompt with "Maintain the same ... art style." Photographic references skip the prefix. Applies to start images, keyframes, the establishing shot, and per-clip reruns.
- Motion-blur and speed-line requests are stripped from start-frame prompts. The planner's music-video energy language was leaking into still images and the image model obliged with smeared backgrounds; start frames are now always sharp and motion stays in the video prompt where it belongs.
- The main performer is now anchored to the reference image in image prompts ("the singer from the reference image") instead of being described loosely, which made the image model invent a new character design for the star while giving the reference's look to background characters.

### v1.2.3 (2026-07-15)

**Added**
- **Uploads view in the workspace switcher.** Browse every image and video you've uploaded (start frames, reference photos) and send them straight back into the pipeline with the "use as input" arrow. Browse-only: generations keep saving to your real workspace.
- **Manual model unload.** A small power button in the System panel (bottom left, expanded view) unloads the resident generation model and LLM to free VRAM and RAM, with an inline confirm. Models still stay loaded between generations by default so retries start instantly.
- **Collapsible model families.** In Settings > Enabled Models, each family (Wan 2.1, Hunyuan, Flux 1, ...) can be collapsed — and stays collapsed across sessions — with a checkbox to enable or disable the whole family at once.

**Fixed**
- Director Stop now aborts the clip being generated within seconds. It used to only take effect between clips, so the current clip kept rendering (10+ minutes of GPU work on slower cards) and a stopped run could even be marked "completed". Finished clips are kept for the Dashboard.
- The Director text entry box grows upward as you type (up to ~11 lines) instead of staying two lines tall, and its scrollbar is actually visible.
- Director mode keeps the art style of your reference images. Hand-drawn, anime, watercolor and other stylized references now carry their medium into every image prompt instead of coming out photorealistic.
- Director no longer sneaks subjects from its internal instruction examples into your video (the recurring dragon), and a location you specify in your description is now binding — shot variety comes from camera angles, not invented places.
- Speaker identification during song analysis now actually runs. It was silently skipped on every install (the model never downloaded without a HuggingFace token); the checkpoints (~30 MB) now download automatically from an ungated mirror on first use. Its clustering is also tuned for singing now: a solo vocalist reads as one speaker and duets as two, instead of one singer splitting into six.
- The Load Settings pencil on songs restores everything: the Style / Music Caption (works retroactively on existing songs), the "Describe your song" text and Instrumental toggle (new songs), and it switches to the right Audio sub-tab — Speech, Music, or SFX — instead of leaving whichever was last open.

**Changed**
- A page refresh now starts clean: prompt fields empty, seed back to random, no LoRAs selected, and Advanced settings at the model's recommended defaults. Your mode, model selections, enabled models, and theme still persist, and switching between modes within a session still carries your work back and forth. (This reverses v1.2.0's restore-on-refresh behavior — stale text and seeds reappearing after a reload felt wrong.)

### v1.2.2 (2026-07-14)

**Fixed**
- Director Mode could get stuck at "Analyzing" forever after v1.2.0 on cards with less VRAM. Analysis runs right after the song renders, and the new default music model is much larger than the old one; on smaller GPUs the leftover model plus the vocal separator and Whisper overflowed VRAM, which Windows silently turns into an extreme slowdown instead of an error. The song model's VRAM is now released before analysis starts.
- Added an int8 version of the ACE-Step XL SFT transformer (5.5 GB instead of 10 GB). Cards using int8 quantization (what Auto-Tune selects below 24 GB) now download and load the smaller file automatically.

### v1.2.1 (2026-07-14)

**Fixed**
- Existing installs updating to v1.2.0 did not see the new ACE-Step XL SFT models enabled, and the music default stayed on Turbo. The curated default-model list is now versioned: entries added to it are merged into existing installs once (your own enable/disable choices are never overridden afterward), and installs still using the previous music default are moved to ACE-Step v1.5 XL SFT LM_4B with its recommended settings. Fresh installs were unaffected.

### v1.2.0 (2026-07-14)

**Added**
- **Light themes with a Dark / Light / Auto appearance mode.** Every theme family now has a daylight variant: Golden Hour pairs with warm paper and burnt orange, Classic with cool paper and blue, Onyx with light monochrome. Pick your style in Settings > System, then choose Dark, Light, or Auto; Auto follows your system's appearance and switches live when it changes. Warning banners, chips, gauges, and indicators were re-tuned to stay legible on light backgrounds, and video letterboxing stays dark on light themes to avoid glare.
- **ACE-Step v1.5 XL SFT, the premium music model.** The quality-focused CFG variant of the XL 4B DiT, now the default music model in Studio and Director (available with the 1.7B or 4B LM). Maestro implements the classifier-free guidance sampling path with Adaptive Projected Guidance this model requires, and unlocks the Steps and Guidance controls for it (defaults: 30 steps, guidance 7.0; raise steps toward 50 for maximum quality). Weights download on first use (about 10 GB).

**Fixed**
- The fast ACE-Step LM decoder (vllm engine) was silently disabled on every Windows install by a faulty runtime check, forcing song planning onto a slow fallback decoder. Planning is dramatically faster after this fix.
- ACE-Step's tuned LM sampling defaults (temperature 0.85, top-p 0.9, LM guidance 2.5) never reached the UI, so generations ran at temperature 1.0. Advanced Settings now loads the recommended values when you select a model.
- Director music-video planning crashed with a connection error when two reference images had the same dimensions (a llama-server bug in batched image encoding), sometimes with a false "lower VRAM headroom?" popup on a nearly empty GPU. Both fixed, and the LLM server's output is now saved to logs/llm for future diagnosis.
- Songs sometimes showed only 30-40 seconds in the gallery until a manual browser refresh. Audio files are now written atomically so a partially written file can never be picked up or cached.
- Field edits persist as you type: a page refresh restores exactly what you last had in every field, including the lyrics prompt (which previously always reset) and cleared fields (which previously came back).
- New ACE-Step models were filed under Text to Speech instead of Music in the model lists.

### v1.1.3 (2026-07-12)

**Fixed**
- Director-mode clips no longer show a broken start-image icon in the gallery, the info bar, or the sidebar after a Load Settings pencil restore. Director keyframes live in the output workspace rather than the uploads folder; the thumbnail lookup now finds them there. Existing clips are fixed retroactively.
- Two-phase LoRA weights (for example 0.75 for stage 1 and 0.50 for the refine stage on LTX-2 models) no longer fail generation with "there should be at most 1 phases". The weights were always supported by the pipeline; only the validation rejected them.
- Director mode's LoRA selector now shows the correct green dot and safe-zone color for CivitAI-recommended weights on all themes. Golden Hour remapped its green to amber, making every LoRA look like it had guessed defaults.

### v1.1.2 (2026-07-12)

**Fixed**
- Director dashboard Re-join now actually works end to end: it uses the real clip concatenation (previously it called a function that didn't exist) and lays the original song over the rejoined video, the same way the pipeline's final output does.
- Regenerated clips come back at their full planned length. Reruns were silently split into multiple sliding windows by a legacy default and only the first ~5 seconds was kept, which shifted every later clip in the rejoin and broke lip sync. Reruns now always generate the clip as a single window and record the completed file.
- The media gallery refreshes when a rerun clip or rejoined video is saved, no browser reload needed.

### v1.1.1 (2026-07-12)

**Fixed**
- Director music videos: regenerating a clip from the Pipeline Dashboard now keeps the song. Reruns are conditioned on the exact segment of the soundtrack the clip covers, instead of the model inventing its own audio.
- Director dashboard: complete multi-clip runs no longer show a bogus "Generate N missing" count, and the Re-join button works (and reports errors instead of silently doing nothing). Existing saved pipelines are repaired automatically on load.
- ACE-Step 1.5 with the song LM appeared to hang forever with a runaway progress counter (for example 96761/97200 and climbing). The generation was actually progressing; the counter now reads honestly (token n of 600 for a 2 minute song).
- Performance Auto-Tune assigned audio a memory profile meant for large video models, which silently locked the ACE-Step song LM to a slow fallback decoder on every card under 24 GB VRAM. Audio now gets its own profile: cards with 12 GB+ unlock the fast LM engine. Re-run Auto-Tune (Settings > System > Auto card) after updating to pick this up.

### v1.1.0 (2026-07-10)

**Added**
- **Linked Model Folders** (Settings > System): reuse checkpoints and LoRAs from other installs such as Wan2GP, with one-click scanning of your Pinokio apps. Linked folders are strictly read-only; new downloads always go to Maestro's own folder. AI LoRA guides work for linked LoRAs too and are stored in Maestro's directory.
- **Krea 2 image models** (Raw and Turbo), ported from upstream Wan2GP.
- **10Eros v1.4** model entry with the author's abliterated Gemma text encoder and the reference workflow's per-stage LoRA strengths.
- **Reference Pipeline toggle** for 10Eros models (on by default): runs the model author's published ComfyUI workflow config (9+3 steps on hand-tuned sigmas, per-step CFG and STG, rectified-flow ancestral sampling).
- Version number in the UI header and this Updates section.

**Fixed**
- LTX-2 Dev and 10Eros models producing blurry, over-saturated output (a leaked `euler_ancestral` sampler setting; the root cause of the "Dev models look bad" reports).
- Reference pipeline dissolving the start image on image-to-video runs.
- The Load Settings pencil losing inference steps, guidance, STG scale, and CFG rescale values.
- Near-unreadable muted text across all three themes ([#7](https://github.com/Blizaine/Maestro/issues/7)).
- The STG slider was a no-op; it now engages STG on the correct transformer blocks.

**Improved**
- Downloaded models always show bright in the Enabled Models list; mode groups start collapsed.
- NSFW filter toggles in the CivitAI browser and LoRA selector are remembered across sessions.
- Deleting models can never touch files inside linked installs.

### v1.0.0 (2026-07-08)

Initial public release. See [CHANGELOG.md](CHANGELOG.md) for the full feature rundown.

## Requirements

| | Minimum | Recommended |
|---|---|---|
| **OS** | Windows 10/11 or Linux | Windows 11 |
| **GPU** | NVIDIA, 6 GB VRAM | NVIDIA RTX 3090 / 4090 / 5090, 24 GB+ VRAM |
| **System RAM** | 16 GB | 32 GB+ |
| **Disk space** | **150 GB free** | **500 GB free** (for full model collection) |
| **Python** | Auto-installed by Pinokio | — |

**What to expect by GPU** (rough ballpark — varies with model, resolution, and length):

| Your card | First run | A short clip after models are cached |
|---|---|---|
| **24 GB** (3090 / 4090 / 5090) | smooth — everything runs | ~1–3 min |
| **12–16 GB** (3060 12GB / 4070 / 4080) | good — auto-tune picks an offload profile | ~4–10 min |
| **6–8 GB** | works, but expect heavy offloading | slow; stick to short/low-res clips |

The first video is always the slow one: install is ~10–20 min, then the first generation on each model downloads its weights (the default video model is ~18 GB). After that, weights are cached and only generation time applies. Maestro's auto-tune sizes the settings to your card on first launch so you don't have to.

> ⚠ **AMD GPUs and macOS are not currently supported.** The pipeline depends on CUDA and several NVIDIA-only kernels. MacOS support is in development.  

> ⚠ **Model downloads are large.** A typical install pulls **50–100 GB** of model weights on first launch. The full collection can exceed **300 GB**. Make sure you have headroom on the drive where Pinokio is installed. However, only models requested during generation will be downloaded. 

## Install

1. Install [Pinokio](https://pinokio.computer).
2. In Pinokio, open the **Discover** tab and search for *Maestro* — or click the **Download** button on the [Maestro repo page](https://github.com/Blizaine/Maestro) and paste the URL.
3. Click **Install**. The launcher will:
   - Create a Python virtual environment in `app/env/` (`app/env-rtx50/` on RTX 50-series GPUs)
   - Install all Python dependencies (torch, xformers, transformers, fastapi, …)
   - Build the React UI in `ui/`
4. When install finishes, click **Start**. The first generation in each model triggers a one-time weight download.

The install (without model downloads) typically takes **10–20 minutes** depending on internet speed. SAM 3.1 (used only for the experimental Inpaint feature) is **not installed by default** — install it on demand via Pinokio menu → "Install Inpaint Support (SAM 3.1)" if you want to use Inpaint.

RTX 50-series cards use a dedicated Python 3.11, PyTorch 2.10, CUDA 13 environment with native SageAttention and Lightx2v NVFP4 kernels. Existing RTX 50 installs migrate automatically the next time **Update** is run; the older environment is retained as a recovery point. Maestro prints a short runtime audit at startup. If that audit reports a missing RTX 50 kernel after Update completed, use **Advanced → Repair RTX 50 Runtime** in the Pinokio menu.

MiniMax H3 also offers an optional **Sol Engine (Experimental)** sparse-attention backend. RTX 50-series systems can enable it directly in H3 Advanced settings after Update. RTX 40-series systems keep Maestro's proven default runtime and can add a separate `app/env-sol/` environment through **Pinokio menu -> Install H3 Sol Engine Runtime**, then launch with **Start with H3 Sol Engine**. The first Sol generation compiles Triton kernels and can start more slowly. RTX 20/30-series GPUs remain on SageAttention because the optimized Sol kernels do not support their compute architecture. If Sol cannot handle a call, Maestro reports it once and falls back to the normal dense H3 attention path.

On one RTX 4090 test system, a 14.4-second 720p H3 generation at 30 steps measured 23m43s with neither optimization, 19m30s with Sol, 17m54s with First Block Cache, and 12m59s with both. Treat these as a directional example only: model choice, prompt/reference load, resolution, drivers, cache threshold, and hardware all affect speed and quality.

### Updating

Click **Update** in the launcher menu. This pulls the latest launcher scripts and app code, reinstalls any new Python dependencies, and rebuilds the React UI.

### Resetting

The optional `app/env-sol/` environment is removed along with the standard Maestro environments.

Click **Reset** to wipe the install and start over. Removes `app/env/`, `app/env-rtx50/`, `ui/node_modules/`, `ui/dist/`, and the SAM venv if installed. Model checkpoints in `app/ckpts/` are NOT removed by default — delete them manually if you want a true fresh start.

## Usage

After clicking **Start**, the launcher shows an **Open Web UI** button once the server is up.

- **Sidebar** — mode toggle (Studio / Director), model picker, prompt, LoRAs, advanced settings
- **Main feed** — generated outputs, dashboard, Director pipeline status
- **Settings drawer** (gear icon) — model visibility, performance auto-tune, services (LLM, API keys, NSFW, theme)
- **Pinokio menu** — Update, Reset, Install Inpaint Support, LoRA folder shortcuts

## Sharing on the local network

Maestro respects Pinokio's `PINOKIO_SHARE_LOCAL` environment variable. Set it to `false` (in the per-app or global ENVIRONMENT file) to bind the server to loopback only; set to `true` for LAN access. Pinokio's own daemon proxy is a separate concern that may also need to honor the variable depending on your setup.

## Credits

Maestro is built on top of, and indebted to, the following projects:

- [**Wan2GP / WanGP**](https://github.com/deepbeepmeep/Wan2GP) by [@deepbeepmeep](https://github.com/deepbeepmeep) — the entire generation pipeline. Maestro inherits WanGP's non-commercial license.
- [**LTX-Video**](https://github.com/Lightricks/LTX-Video) by Lightricks — LTX-2 and LTX-2.3 distilled models.
- [**MiniMax H3**](https://huggingface.co/MiniMaxAI/MiniMax-H3) by MiniMax — joint video-and-audio generation with text, first-frame, and first/last-frame conditioning.
- [**Wan 2.1 / 2.2**](https://github.com/Wan-Video/Wan2.1) by Alibaba — text-to-video and image-to-video.
- [**Flux**](https://github.com/black-forest-labs/flux) by Black Forest Labs — image generation.
- [**Qwen**](https://github.com/QwenLM/Qwen) by Alibaba — image generation and LLMs.
- [**Gemma**](https://ai.google.dev/gemma) by Google — Gemma 4 LLM (default for Director mode).
- [**SAM**](https://github.com/facebookresearch/sam2) by Meta — segmentation backbone for Inpaint.
- [**MMAudio**](https://github.com/hkchengrex/MMAudio) — automatic ambient audio generation.
- [**CivitAI**](https://civitai.com) — LoRA browser and weight recommendations.
- [**llama.cpp**](https://github.com/ggml-org/llama.cpp) — local LLM inference engine.
- [**Pinokio**](https://pinokio.computer) by [@cocktailpeanut](https://github.com/cocktailpeanut) — the launcher framework.
- The original Pinokio Wan2GP launcher by [@cocktailpeanut](https://github.com/cocktailpeanut), which Maestro forks and extends.

## License

Maestro is released under the **WanGP Non-Commercial Evaluation License 1.1**, inherited from the upstream Wan2GP project. See [LICENSE](LICENSE) for the summary and [app/LICENSE.txt](app/LICENSE.txt) for the full text.

**TL;DR**: free to use and modify for non-commercial purposes; the *outputs* you generate are yours to use commercially (with attribution); commercial use of the *software itself* (including hosted services and APIs) requires a separate commercial license from the WanGP licensor.

Third-party models, weights, and components keep their own licenses — review them before redistributing. MiniMax H3 weights remain subject to MiniMax's separate model terms and any authorization or waiver required for the user's location. Notably, the [seed-vc](https://github.com/Plachta/seed-vc) voice-conversion component is **GPL-3.0**, so it is distributed from its own repository ([Blizaine/maestro-seedvc](https://github.com/Blizaine/maestro-seedvc)) and cloned into `app/postprocessing/seedvc/` at install time rather than shipped in this tree. Other vendored components include BigVGAN (MIT), FlashVSR sparse-sage (Apache-2.0), and IndexTTS2 (bilibili model license).

## Issues

Bug reports and feature requests: [github.com/Blizaine/Maestro/issues](https://github.com/Blizaine/Maestro/issues).
