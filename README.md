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
- **Video** — LTX-2.3, Wan1/2, Hunyuan, and many more.
- **Image** — Flux 2 Klein 9B (default), Qwen Image Edit, and many more
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

### 🛠️ Edit Mode *(experimental)*
- **Retake** — re-roll a section of an existing video with a new prompt
- **Outpaint** — extend a video's frame in any direction
- **Edit Anything** — allows users to modify, add, or remove elements from existing videos using text prompts and In-Context LoRA (IC-LoRA) models

### 📂 Workspaces
Multiple isolated output directories with a quick switcher in the sidebar. Useful for separating client projects, NSFW vs SFW, or experiments. Pinned and favorited outputs are tracked per workspace.

### 🔒 Mature mode + experimental gate
- **NSFW mode** is opt-in with a disclaimer step. Disabled by default. Gates uncensored model variants, NSFW LoRAs in the CivitAI browser, and the Settings → Services NSFW toggle.
- **Experimental features gate** hides power-user toggles (external API keys, Voice Reference, Inpaint, Restyle, Wan2GP Enhancer) by default for a focused first-launch experience.

### 📊 Director Pipeline Dashboard
View all past Director runs with their full state — clip plans, generated images, generated clips, polish diffs. Re-run any clip without re-running the whole pipeline.

## Updates

The version you are running is shown next to the Maestro title in the UI. To update, use the launcher's Update button in Pinokio.

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
   - Create a Python virtual environment in `app/env/`
   - Install all Python dependencies (torch, xformers, transformers, fastapi, …)
   - Build the React UI in `ui/`
4. When install finishes, click **Start**. The first generation in each model triggers a one-time weight download.

The install (without model downloads) typically takes **10–20 minutes** depending on internet speed. SAM 3.1 (used only for the experimental Inpaint feature) is **not installed by default** — install it on demand via Pinokio menu → "Install Inpaint Support (SAM 3.1)" if you want to use Inpaint.

### Updating

Click **Update** in the launcher menu. This pulls the latest launcher scripts and app code, reinstalls any new Python dependencies, and rebuilds the React UI.

### Resetting

Click **Reset** to wipe the install and start over. Removes `app/env/`, `ui/node_modules/`, `ui/dist/`, and the SAM venv if installed. Model checkpoints in `app/ckpts/` are NOT removed by default — delete them manually if you want a true fresh start.

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

Third-party models, weights, and components keep their own licenses — review them before redistributing. Notably, the [seed-vc](https://github.com/Plachta/seed-vc) voice-conversion component is **GPL-3.0**, so it is distributed from its own repository ([Blizaine/maestro-seedvc](https://github.com/Blizaine/maestro-seedvc)) and cloned into `app/postprocessing/seedvc/` at install time rather than shipped in this tree. Other vendored components include BigVGAN (MIT), FlashVSR sparse-sage (Apache-2.0), and IndexTTS2 (bilibili model license).

## Issues

Bug reports and feature requests: [github.com/Blizaine/Maestro/issues](https://github.com/Blizaine/Maestro/issues).
