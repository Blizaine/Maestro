# Maestro v2.4.2 validation

Release validation, September 27, 2026. Public baseline before v2.4.2:
`d98af3516bcb8131d6a39c3175332798b17febeb` on both dev and main. Its
[GitHub CI run passed](https://github.com/Blizaine/Maestro/actions/runs/36245206098).
The checks below were completed during local preparation. Public release CI
runs are listed in [GitHub Actions](https://github.com/Blizaine/Maestro/actions/workflows/ci.yml);
the baseline result above is not v2.4.2 CI.

## Checks completed

- **46 focused Python tests passed** in one combined unittest run using the
  installed Python 3.11 runtime. Modules: `test_krea2_features`,
  `test_krea2_reference_boost`, `test_krea2_reference_preprocessing`,
  `test_gallery_media_trim`, and `test_studio_preference_persistence`.
- Krea coverage includes subject/scene boost behavior, neutral attention,
  padding, GQA, CFG/NAG, reference proportions, grounding limits, model gating,
  Turbo guidance and validated persistent settings.
- Trimming tests use synthetic media and real FFmpeg: audio rate/channel
  preservation, MP3 cover-art classification, fractional video trims with
  frame rate and audio, invalid ranges, traversal/workspace isolation,
  symlink paths, process startup failure, timeout cleanup and processing-slot
  reuse.
- **Four browser suites passed:** `gallery_inputs.cjs`,
  `gallery_audio_destinations.cjs`, `gallery_media_trim.cjs`, and
  `krea_identity_controls.cjs`. Requests that could upload, submit a job or
  mutate the app were mocked. The gallery-routing suite read only the live
  model catalogue and capability endpoints.
- Browser coverage includes existing image/video destinations, new audio
  roles, source folders, full/range selection, keyboard and pointer controls,
  mobile layout, preview playback, invalid ranges, failed request/retry,
  duplicate submission and destinations changed while a trim is pending.
  Krea coverage verifies independent model values, queue payloads, preference
  reload, output-settings restoration and unrelated-model isolation.
- Desktop/mobile screenshots were generated; mobile trim and Krea layouts
  were visually inspected. A missing filesystem import in the new audio
  destination test harness was corrected before its passing run.
- **Frontend ESLint, TypeScript and production Vite build passed.** Existing
  editor mixed-import and large-bundle warnings remain; neither blocks builds.
- **Python syntax checks passed** for services, Krea model code, shared image
  utilities, launch/generation entry points and scripts. Ruff F821/F823 passed
  for services, H3, Krea and `launch.py`.
- **Public-source boundary guard passed** with all 2,614 staged/tracked source
  files included. Release version, documentation links, modified JSON defaults
  and staged whitespace checks passed. Launcher scripts are unchanged.

## Scope and remaining limits

This is focused regression validation for the changes since v2.4.1, not a
fresh full-suite, clean-install or cross-platform certification. No models
were downloaded, no generations were submitted, and the running application
was not restarted or its settings changed during preparation.

The user previously exercised gallery reuse and Krea identity controls. The
final Krea reference-sizing correction has CPU/browser coverage, but its
visual likeness improvement still needs a controlled GPU comparison using
the same prompt, references and seed. Likeness at 4 is a starting point, not
a guaranteed match; high boosts can oppose edits. Grounding above 768 is
experimental.

Trim previews can be cancelled before submission. Once submitted, trimming
and transfer must finish or fail before the dialog can close. A disconnected
client does not cancel FFmpeg; processing has a 300-second timeout and cleans
up temporary output and releases the processing slot on failure. Physical
mobile Safari playback/gesture verification remains separate from the mocked
Chrome browser checks.

The earlier character-swap LoRA memory experiment is not changed by this
release. The already-public DLSS Linux test fix is part of the baseline,
not a new DLSS performance claim.

[Release notes](RELEASE_NOTES_V2.4.2.md)
