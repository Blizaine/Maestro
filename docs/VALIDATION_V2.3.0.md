# v2.3.0 validation

Validated locally on September 20, 2026, before publication.

- **Python:** 2,250 tests passed in 147 seconds using the Python 3.10
  compatibility environment with CUDA disabled; 10 optional/CUDA skips.
- **Static checks:** Python compilation and undefined-name checks passed for
  services, launch code, H3, YuE2 and Qwen Image 2.1. All five standalone JSON
  grammar regressions passed.
- **Frontend:** ESLint, TypeScript and the Vite production build passed.
  Existing bundle-size and mixed static/dynamic import warnings remain.
- **Browser tests:** ten isolated suites passed for Qwen 2.1 model selection,
  notification history, music startup, My Music, Auto training, dataset
  preparation, the LoRA library, paired adaptation, Guided training and Studio
  duration. Coverage includes desktop/mobile flows, saved settings, queue
  handoff, training resumption and relevant failure recovery. They mock backend
  operations and do not submit real training or generation jobs.
- **Live Qwen:** complete 1024px, 40-step generations and reference edits ran
  through Maestro's GPU job API on the maintainer's RTX 4090. The maintainer
  separately confirmed generation and editing work well before authorizing
  this release. The shared-loader mismatch and RGBA reference crash have
  regression coverage.
- **Live music:** the maintainer has tested voice/sound adaptation, song-style
  training, checkpoint auditions and single/multiple LoRA generation during
  development. Quality judgments remain subjective and vary by data/settings.

The automated checks verify behavior and compatibility, not universal image,
voice or music quality. Training likeness, multi-LoRA voice separation and
extended H3 duration remain experimental. The 30-second option does not imply
that a given GPU can fit a full-length pass; its duration plumbing is tested,
but this release check does not benchmark every GPU or model combination.

GitHub CI repeats the clean-source guard, Python suite, grammar regressions and
production UI build on Linux for each public branch. Runtime models, recordings,
training projects, outputs and local settings are excluded from publication.
