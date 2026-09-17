# v2.2.2 validation record

Prepared 16 September 2026. This distinguishes automated regression coverage
from rendered workflows exercised during development and tests still needed on
other hardware.

## v2.2.2 reliability update

- Release CPU discovery passed **2,069 tests**, with six CUDA-dependent skips,
  in 126.589 seconds. Syntax, undefined-name checks, five standalone JSON grammar
  checks, UI lint and production build also passed. The normal Vite bundle-size
  and mixed-import warnings remain informational. No LLM retests were run during
  release preparation.
- 869 focused Python regression tests passed during development, covering H3
  parsing, exact dialogue, timing and camera plans; Director; Studio enhancement
  and queue retries; adaptive enhancement; and the prompt bench.
- Six final live regression requests completed without review warnings or
  automated check failures: three prompts, once each with Gemma E4B and Qwen3.6
  27B. The three-window reported brief retained its exact line with the older man.
  These were enhancement-only, text-only Frames equivalents because the reporter's
  reference image was unavailable. Qwen3.8 was installed but omitted by an overly
  strict benchmark filename check; that detection is now corrected, with no new
  Qwen3.8 generation or enhancement run claimed.
- All 50 development/evaluation attempts were retained privately, including
  failures and rejected approaches. Two earlier evaluations exposed more parsing
  defects and became regression fixtures; the final six are not unseen holdouts.
- Text review found remaining limitations despite operational passes: one Gemma
  draft described boarding/sitting only in its closing state and paired a spoken
  turn with the other character's acting cue. Some simple scenes were stretched
  too slowly across three windows. Warning-free output does not certify fidelity.
  Final requests took 24.5-289 seconds, with up to eleven writer calls.
- Production UI build and mocked browser checks passed for reviewed-draft
  acceptance, rejected retries, immediate queue visibility without a history
  request, and mobile validation remaining visible. No test video jobs were sent.
- Director regressions cover silent/unknown vocal intervals, singers and
  instrumentalists, contradictory mouth cues, explicit expressions, wind
  instruments and persistence through final compilation. A newly rendered music
  video was not required or claimed for this release.

## v2.2.1 enhancement hotfix

- 497 H3, 33 adaptive-enhancement and 23 Studio/queue regression tests passed
  (553 total). Coverage includes numbered shot formats, Windows line endings,
  source offsets, local camera-setting retention and genuine omitted actions.
- Ten live local enhancements passed with no review warnings or fallback on
  the final candidate. Eight used the exact reported 28-second, eight-shot
  brief and its composition reference: three queued-path repeats each with
  installed Gemma E4B and Qwen3.6 27B, plus two direct Enhance-now endpoint
  replays with the existing user settings. Two single-window controls exercised
  exact dialogue ownership and a quiet scene without speech.
- The six repeated queued-path runs each used three LLM calls, without focused
  repair. All eight exact-brief plans retained eight ordered source events,
  two windows totaling 28 seconds, and the requested first-shot optical settings.
- An earlier candidate rejected two Gemma paraphrases because it checked every
  descriptive sentence independently. It was corrected before the complete
  repeat above; those failures are not counted as successful final checks.
- These checks validate enhancement and compiled prompts, not rendered-video
  fidelity. The complete feature release's existing coverage remains below.

## v2.2.0 feature-release checks

| Check | Result |
| --- | --- |
| Python discovery with CUDA hidden | 2,013 tests run; passed, with six CUDA-dependent skips |
| Standalone JSON grammar regression runner | Five checks passed |
| Python compilation | Services, H3/YuE2 modules, prompt bench, launch, WGP and scripts passed |
| Undefined-name checks (`F821`, `F823`) | Services, H3/YuE2 modules, prompt bench and launch passed |
| UI lint | Passed |
| TypeScript and production UI build | Passed |
| Integrated sidebar/Director/queue browser regressions | Passed |
| Six standalone browser regressions | Passed |

The release regression environment used Windows, Python 3.10 and Torch 2.7.1.
CUDA-specific training math and the live WGP prepared-settings integration also
passed in the earlier CUDA-enabled run. The final CPU run explicitly skips
those checks; it does not simulate GPU execution. The Linux CI workflow runs
the CPU suite, JSON grammar checks, syntax/undefined-name checks, public-source
boundary guard and UI build on pushes to dev and main.

The production build retains the existing large-bundle and mixed static/dynamic
import warnings. They do not prevent type checking or building.

## Browser coverage

Isolated browser checks exercise the built UI with controlled API fixtures.
They do not submit generation jobs or change the user's saved projects.

- Integrated coverage spans six themes at 1360, 767, 440, 390 and 320 pixels:
  sidebar scrolling, long prompts, simulated keyboard viewport changes,
  duration controls, anchored popovers, reference controls, Director settings,
  saved shot edits, gallery navigation, queued enhancement and completed history.
- Separate checks cover music startup/default migration and later model choices;
  My music playback, held-out data and audition controls; gallery refresh;
  image source/enhanced prompt details; Director GPU clip limits; and persisted
  Studio step settings.
- Backend regressions cover queued enhancement ownership/retry/cancellation,
  prompt provenance and writing repairs, music training/reconstruction,
  gallery identities, music timing/cues and the issue fixes listed in the
  [release notes](RELEASE_NOTES_V2.2.0.md).

## Workflows exercised during development

These are local acceptance checks, not a cross-hardware performance benchmark.

- **Director:** the maintainer reviewed H3 music-video generations with singer,
  guitarist, drummer and audience references. For a saved 119.999s analysis and
  H3's 14.375s native cap (shown as 14.4s), Cut Speed settings −2 through +2
  produce 9, 10, 13, 14 and 17 clips. At −2, clips span 12.29–14.21s and cover
  the complete 120s frame-aligned output. The maintainer confirmed the revised
  slower setting worked in the UI.
- **YuE2 generation:** Direct generation, planned songs, source-song covers and
  handoff were exercised. Saved WAV probes verified 48 kHz stereo and actual
  durations, including a generated song ending at 131.479s below its requested
  duration ceiling. These checks do not establish every score/backend combination.
- **My music:** real data preparation, training, checkpoints, stop/resume,
  auditions, reconstruction, strength changes and style import/export were
  exercised. Follow-up listening improved style/vocal resemblance in some cases
  but did not establish reliable identity cloning. The feature remains
  **Experimental**.
- **TaoMate:** rendered checks used pruned INT8 ConvRot H3, 864×480, 124 frames,
  three steps, with text and start-image input on a 24 GB RTX 4090. Internal
  token-refiner adapter targets were verified. See [TaoMate scope](TaoMate-H3.md).
- **Prompt writing:** bounded local-writer experiments and maintainer-reviewed
  conversation/action clips informed the changes. Text regressions protect
  source events, speakers, timing and continuity, but cannot certify that a
  video model will render each direction correctly.

## Remaining validation limits

- Native Linux CUDA compilation/loading for issue #135 needs confirmation on
  the affected installation. Build/runtime unit tests use controlled fixtures.
- The issue #131 residency correction has budget/cache regression coverage;
  RTX 5080 speed recovery has not been measured here. The #128/#130 kernel or
  compilation stalls are not claimed fixed.
- Physical iPhone keyboard behavior, every GPU/driver/model combination, very
  long productions and reliable singer identity transfer are not established
  by these automated checks.
- New model weights and optional tools retain their own licenses and download
  requirements. Model/training assets and private outputs are not release files.

See [release notes](RELEASE_NOTES_V2.2.0.md) for the feature scope and which
GitHub reports remain open for retesting or unfinished work.
