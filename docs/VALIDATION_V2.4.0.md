# v2.4.0 validation

Local release preparation on September 24, 2026. Publication remains a
separate step; no public branch, release or issue status was changed during
this preparation.

## Release checks

The checks use the existing Python 3.10 environment with CUDA disabled and
isolated browser fixtures; no real generation or training jobs are submitted.

- **Python suite:** 2,730 tests completed in 387.302 seconds: 2,720 passed,
  ten optional/CUDA checks skipped, no failures or errors.
- **Static Python checks:** compilation and Ruff undefined-name/unbound-local
  checks passed for services, launch code, H3 and Qwen Image 2.1. Script
  compilation and all five standalone JSON grammar regressions passed.
- **Frontend:** ESLint, TypeScript and the Vite production build passed.
  Existing bundle-size and mixed static/dynamic editor-import warnings remain.
- **Three isolated browser suites:** gallery viewing/comparison/zoom and
  staged video playback passed in desktop and phone-sized touch viewports.
  Prompt review passed full-job generation, targeted repair, rewriting,
  original-source use, editing and mobile-layout cases with mocked APIs.
- **Four JavaScript regression suites:** Singularity defaults and opt-out,
  API-key pending/failure/retry behavior, LoRA import routing and fused-H3
  LoRA settings round trips passed.
- **Release hygiene:** the clean-source guard passed over 2,572 staged/tracked
  files. Whitespace checks, the product version and local documentation links
  passed. Private artifacts and runtime assets remain excluded.

Release checks exposed older tests expecting fixed gallery labels and blanket
vocalist instructions, plus an AST test fixture missing a newly imported
helper. Assertions now cover the intended destination-aware and cast-scoped
behavior. Gallery refs were moved out of render, and playback regressions
passed after that React lifecycle correction.

## Prompt-quality evidence and remaining limits

Development included local Qwen3.8 and Gemma-4 enhancement attempts with Frames
and References, short concepts, detailed timelines, dialogue, silent action,
music-driven audio and multiple windows. Saved responses and source contracts
were reviewed separately from the automated acceptance status.

The last broad semantic review did **not** establish universal release-quality
prompt fidelity. It still found repeated door actions, missing or altered prop
placements, and movement extending past an intended supplied-audio silent tail.
For example, one final prompt placed a chart on a stone ledge or column top
instead of the requested table; another closed and locked the same door twice.
Some of these drafts passed automated checks. Successful completion and passing
unit tests therefore must not be reported as perfect source preservation or
guaranteed video quality. Later retry controls let users choose repair effort
and whether to proceed; they do not resolve every semantic weakness.

Keep automatic continuation off by default. Review important multi-window
drafts, especially exact object transfers, repeated actions and the end of a
music timeline. Further semantic testing and video feedback remain useful;
this packaging run does not repeat the earlier all-day LLM campaign.

## Scope of verification

- Browser tests exercise desktop and phone-sized touch viewports, synthetic
  media and mocked server routes. They do not replace a physical iPhone Safari
  / Home Screen check or verify every autoplay/fullscreen policy.
- Qwen memory, LoRA conversion and Director routing have regression coverage.
  This release preparation does not benchmark GPU throughput or establish the
  maximum number/size of references on every card.
- H3 Singularity remains experimental. Loader/layout, managed assets and UI
  defaults are checked; comparative generation quality remains to be tested.
- Remote LLM image and credential routing uses mocked providers in automated
  checks. No paid provider calls are made during release preparation.
- Recast's change bounds temporary mask-composition allocations. It does not
  make all memory used by a long video constant.

Runtime models, generated media, characters, recordings, local settings,
private benchmark reports and caches are excluded from the release source.
