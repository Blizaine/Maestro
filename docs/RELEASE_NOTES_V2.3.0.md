# Maestro v2.3.0

September 20, 2026. All changes since v2.2.4.

## Qwen Image 2.1

- Add the unified **Qwen Image 2.1 7B** model for text-to-image generation,
  reference editing and combining up to ten images.
- Add dedicated generation/editing prompt-enhancement guides, transparent
  RGBA PNG output, BF16/INT8 ConvRot model downloads and separate storage for
  compatible 2.1 LoRAs. Older Qwen 20B models retain their own controls.
- Use MMGP offloading and tiled VAE processing to limit memory use. The model
  appears once in Model Visibility without replacing the selected image model.
- Fix the shared-loader argument mismatch and RGB/RGBA reference handoff found
  during the first live generation and editing tests.
- Start with 1024 × 1024, 40 steps and CFG 1. Weights download on first use.
  Qwen's Research License permits noncommercial research/evaluation; commercial
  use requires separate permission from Qwen. See the [image guide](Qwen-Image-2.1.md).

## My Music training — Experimental

- **Auto mode:** prepare songs, select suggested main-voice excerpts, train
  voice/sound and then train song style in one queued workflow. Choose targets
  up front; defaults are 100 voice/sound steps and 200 song-style steps.
  Training advances with the browser closed and can be stopped/resumed or
  opened later for additional training and comparisons.
- **Guided mode:** recordings → voice/sound → song style → test song makes the
  two training stages explicit. Carry the matched sound adaptation into song
  style training, reopen projects, and keep technical controls and alternate
  methods in Expert settings. Auto/Guided selection is visually clearer.
- **Full-song preparation:** vocal separation, timed lyric transcription,
  detected-voice previews and suggested phrase clips reduce manual preparation.
  Guided review covers speakers, lyrics, delivery, descriptions and boundaries.
  Auto-prepared excerpts remain marked unreviewed. Check-only recordings stay
  out of training; manual/reviewed clips survive rescanning.
- **Matched voice/sound adaptation:** train the real-audio tokenizer and decoder
  together against recordings, compare original/before/after reconstruction,
  then prepare new tokens for song-style training. New projects recommend v9;
  existing v4 projects retain their assets and saved styles remain intact.
- **Checkpoints and experiments:** resume paired or joint training, retain
  optimizer state, audition saved checkpoints and refine an existing style
  while retaining its original baseline. Resume preserves existing ranks,
  objectives and unfinished step targets.
- Improve long-song transcription with language-controlled rescanning, fix
  style ZIP imports on Python 3.10, and support the newer TorchAudio runtime
  through a shared compatibility layer.

Vocal likeness and music quality depend on the recordings, training, trigger,
style prompt and generation settings. These tools do not guarantee singer
cloning. See the [training guide](YuE2-music.md).

## Music LoRAs and generation

- Move music LoRAs into **Advanced → LoRAs & presets**, with searchable
  checkboxes, individual strength controls and an active-count badge.
- **Show in LoRA selector** lets users shortlist saved LoRAs without activating
  them. Search the full library, remove/restore entries, and retain trained
  weights, projects, queued jobs and previous songs.
- Apply and display training triggers automatically. Restoring a song's settings
  also restores its music LoRAs.
- Support experimental multi-LoRA combinations with independent strengths,
  matched sound companions and saved metadata. Known v4/v9 mismatches are
  rejected. Combining adapters does not guarantee different singers by section.
- Import combined AI-Toolkit YuE2 adapters without losing their musical or
  acoustic branches.
- **Instrumental** automatically selects the dedicated Mothersuperior
  instrumental AR LoRA, Melody and chords planning, and an instrumental section
  prompt. Artist LoRAs are paused for that job. Studio, queued jobs and Director
  share the same recipe, recorded in the output metadata.
- YuE2 songwriting and style enhancement retain the requested vocalist,
  delivery and training trigger instead of substituting generic vocal traits
  or production influences.

## Studio duration and notifications

- Add **Allow 30s clips · Experimental** in H3 Duration settings. Frames and
  References can use one 719-frame pass (29.96 seconds at 24 fps), including
  queued enhancement and restored jobs. Auto retains its GPU recommendations;
  Animate keeps its fixed window. Longer passes use more memory and may lose
  consistency.
- Stop replaying old Studio/Director completion and failure notifications on
  startup or reconnect. Completed history no longer starts status polling;
  active jobs still notify when they finish.

## Updating

Use **Update** in Pinokio, restart Maestro and refresh the browser. Existing
models, outputs, characters, recordings, projects and LoRAs stay in place.
Optional models and training/instrumental assets download when first used.
Individual models and tokenizers retain their upstream license terms.

Earlier enhancement, Director, gallery and runtime improvements remain
included. See the [full changelog](../CHANGELOG.md) for previous releases and
the [validation record](VALIDATION_V2.3.0.md) for checks and known limits.
