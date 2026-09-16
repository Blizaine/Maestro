# v2.2.0 validation record

Prepared 16 September 2026. This distinguishes automated regression coverage
from rendered workflows exercised during development and tests still needed on
other hardware.

## Release checks

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
