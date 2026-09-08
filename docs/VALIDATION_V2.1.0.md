# Maestro v2.1.0 local release validation

Prepared on 7 September 2026. This is a local release candidate, not a published
release. See the [release notes](RELEASE_NOTES_V2.1.0.md) for the user-facing changes.

Updated on 8 September with the dialogue, shared-guide and Studio UI
follow-ups below.

## Release boundary

- Public `dev` and `main` were checked directly against GitHub with read-only
  `git ls-remote`. Both point to `a5dddd4faa53e8fa8d76ef528c1074935eded8c0`, whose
  application `VERSION` is `2.0.1`.
- The feature work before release preparation ended at `a48da84`, 15 commits
  after that public baseline. Release notes cover that complete difference and
  the cleanup performed during this release check.
- Root `VERSION` is now `2.1.0`. The isolated application version reader returns
  `2.1.0`; README, changelog and release notes agree. Pinokio's launcher schema
  version and the private frontend package version are separate values.
- The production UI was rebuilt locally. Build output remains ignored; normal
  installation/update builds it from source.
- The running backend was not restarted. It reads the version at startup and
  will display `2.1.0` after the next normal restart. Active projects and jobs
  were not reset or replaced.
- No push, remote branch update, release tag, or GitHub release was performed.

## Checks performed for this candidate

The Python run used the existing Python 3.11.13 environment with
`CUDA_VISIBLE_DEVICES=-1`. Browser checks used Chromium with isolated test
origins and intercepted writes/generation requests; they did not run GPU jobs.

| Check | Result |
| --- | --- |
| Full Python unittest discovery | 1,408 tests run: 1,407 passed and one CUDA-only test skipped, in 64.048 seconds. |
| Standalone JSON grammar regressions | All five checks passed. |
| Python compile check | Passed for application services, launch/runtime entry points, H3, Wan T5, DLSS, Face Refiner, RIFE and release scripts. |
| Frontend ESLint | Passed. |
| Production TypeScript/Vite build | Passed. Existing bundle-size and dynamic-import warnings remain. |
| Studio/Director browser suite | Passed; includes six theme variants and widths of 1360, 767, 440, 390 and 320 pixels. |
| Character, Viggle, LoRA and duration browser suites | All five additional suites passed. |
| JSON/configuration parsing | 348 strict JSON files and two TypeScript JSONC configurations passed. |
| JavaScript syntax | 28 complete scripts plus the assembled built-in Gradio script passed; none of the launchers were executed by this check. |
| PowerShell syntax | Both tracked scripts passed parsing; neither installer/helper was executed. |
| Repository boundary and release hygiene | Clean-repository guard, release documentation links and diff whitespace checks passed. User data, environments, downloaded models, runtime binaries and build artifacts remain outside the commit. |

The one skipped test is
`test_triton_temporal_convolution_matches_torch` in `test_h3_wan_port.py`.
It requires CUDA and was intentionally skipped in this CPU validation run.

The browser coverage includes prompt stability during typing and polling,
mobile keyboard viewport simulation, Animate-to-Image editing, Director
horizontal bounds, all resolution/aspect choices, duration convergence and
slider stability, character ordering/recovery, held queue payloads, saved
settings, enhancement review, direct Recipes/Browser actions and the H3 Extend
full-model transition.

Commands used, from the repository root unless noted:

```text
python -m unittest discover -s tests -p "test_*.py" -v
python tests/test_call_llm_json_grammar.py
python -m compileall -q app/services app/launch.py app/wgp.py app/models/minimax_h3 app/models/wan/modules/t5.py app/postprocessing/dlss5 app/postprocessing/h3_face_refiner app/postprocessing/rife scripts
python scripts/verify_clean_repo.py
npm run lint       (inside ui)
npm run build      (inside ui)
node tests/ui/sidebar_redesign.cjs <local-Maestro-URL>
node tests/ui/studio_duration.cjs <local-Maestro-URL>
node tests/ui/lora_url_import.cjs
node tests/ui/fused_h3_lora_roundtrip.cjs
node tests/ui/character_images.cjs
node tests/ui/viggle_characters.cjs
```

The final unittest and sidebar logs are retained locally as
`.codex-tmp/v2.1.0-unittest-final.log` and
`.codex-tmp/v2.1.0-sidebar-ui.log`; the grammar log is
`.codex-tmp/v2.1.0-grammar.log`. Browser screenshots are under
`.codex-tmp/sidebar-validation/`. These artifacts are intentionally not shipped.

## Cleanup prompted by these checks

- ETA-history SQLite connections now close after their transactions, resolving
  Windows file locks without changing the commit/rollback behavior.
- Wan/SCAIL's T5 encoder no longer selects CUDA at module import. An omitted
  device is resolved when constructing the encoder, with a CPU fallback.
- Shared Media Flow helpers were moved out of React component modules.
  Capability refreshes discard stale responses after unmount or a newer request.
- Existing source-contract assertions were updated for the intentional Studio
  redesign, the Viggle affine asset and retained Music3 defaults migration.
  The obsolete automatic prompt-growth assertion now reflects the fixed Studio
  composer; browser tests separately exercise its actual size and scrolling.
- Feature guides now use the current Characters, Duration and Advanced locations.
  The README no longer labels post-public features as part of v2.0.1.

## Creative dialogue follow-up — 8 September

- Replaced stale dialogue examples with duration-specific writing targets using
  the shared 2.8 words/second speech pace and 3 words/second maximum. Targets
  account for action, pauses, brief exchanges and explicit silence.
- Corrected Creative's quoted-line contract, speech detection, and complete-script
  timing guidance. Raised H3's two-turn planning cap to six turns per window.
  Added bounded retries for sparse dialogue in single prompts and H3 sequences;
  sequence retries retain source events, exact lines, and speaker ownership.
- Full CPU-only Python discovery: **1,423 tests run; 1,422 passed, one CUDA-only
  test skipped**, in 64.184 seconds. Log:
  `.codex-tmp/creative-dialogue-full-suite-final.log`.
- Added 15 neutral-content regression tests, including the real enhancement
  pipeline with mocked LLM transport, per-window word counts, exact quote/event
  ownership, unknown speakers, silence, and failure recovery. The focused H3 and
  dialogue selection contains 101 tests. After a final fallback-message wording
  correction, all 101 passed again. The five standalone JSON-grammar checks also passed.
- No frontend or launcher behavior changed in this follow-up. The earlier UI
  build/browser checks remain the applicable evidence. No live LLM authoring,
  GPU media generation, application restart, or public push was performed.
- Before release, review actual AI Creative outputs with the selected local LLM:
  a 14.4-second tutorial, a conversation over several H3 windows, and an action
  scene requesting brief dialogue. Mocked transport verifies scheduling and
  preservation; it does not establish every model's writing or rendered quality.

## Shared H3 content guide follow-up — 8 September

- Replaced H3's separate inline Mature-mode note with conditional loading of
  Studio's existing `enhance/nsfw_shared.md`. Both planning styles use it for
  story treatment, chapter planning, segment expansion and bounded retries.
  The guide's wording and the server-side eligibility checks are unchanged.
- Bumped the shared H3 planner version so old plans are invalidated. JSON output
  contracts, timing rules and reference constraints remain in the stage guides.
- **143 focused tests passed**: 104 shared-guide, Frames planner, story ledger
  and dialogue tests, plus 39 Reference sequence and prompt-budget tests. Logs:
  `.codex-tmp/h3-shared-guide-tests.log` and
  `.codex-tmp/h3-shared-guide-sequence-tests.log`.
- Three new tests and an updated Frames regression use neutral guide markers
  and mocked LLM calls to check on/off routing, both styles, long-form chapters,
  retries, empty-guide behavior and switching Mature mode off in the same
  process. No explicit-content authoring or live model quality testing was done.
- No frontend or launcher changes, application restart, or public push.
  Restart Maestro to load the backend change; restart again after editing a
  guide because guide text is cached. Re-enhance prompts to apply the change.

## Studio scrolling follow-up — 8 September

- Studio's media tabs, workflow selector, inputs and prompt now share one
  vertical scroller above the fixed settings and generation controls. The main
  prompt uses a hidden text mirror for intrinsic sizing, grows/shrinks with its
  contents, and retains its mounted editor and selection. Long scripts have no
  inner scrollbar. Caret-line tracking keeps typing visible on keyboard changes.
- Production TypeScript/Vite build and ESLint passed. The existing bundle-size
  and dynamic-import warnings remain. Build output is local and ignored.
- All **14 selected Python source regressions passed**. The full isolated
  sidebar browser suite passed, including six theme variants, duration/menu
  behavior, H3 prompt review, explicit enhancement, held queues and Director.
- New browser coverage exercises Frames, References, Image and Speech at
  1280x520, 1000x420, 390x844 and 320x568 with expanded hardware status. Real
  mouse-wheel input over the textarea moves the shared scroller while settings
  and Generate stay fixed. Long scripts grow, short scripts shrink, mode
  controls and enhancement tools remain reachable, and polling or simulated
  writing-extension overlays do not change the text layout.
- Updated mobile keyboard tests check the active caret near the end of a long
  Animate-to-Image prompt as viewport height/offset changes. The 390px and
  320px simulations also retain the return action and Animate appearance input.
  These Chromium checks do not replace a real iPhone/Safari keyboard pass.
- Logs: `.codex-tmp/composer-scroll-build.log`,
  `.codex-tmp/composer-scroll-lint.log`,
  `.codex-tmp/composer-scroll-source-tests.log`,
  `.codex-tmp/composer-scroll-ui.log` and
  `.codex-tmp/composer-scroll-full-ui.log`. Screenshots remain under
  `.codex-tmp/sidebar-validation/`. No backend restart, live generation,
  launcher changes or public push was performed. Refresh the browser to use
  the rebuilt UI.

## Clip settings follow-up — 8 September

- Resolution and Aspect menus now size to their option labels and open directly
  above their indicators, within the sidebar. All model-provided choices remain.
- Studio video duration now has Time/Window tabs with an Auto toggle. Auto dims
  and disables manual time controls; Time retains the model-step slider through
  five minutes and the 10m, 15m, 30m, 60m and Custom choices. Window Length stays
  visible with automatic sizing, GPU recommendations and manual overrides.
  Window overlap is collapsed by default in both tabs. The popup keeps a steady
  height while sliders move and scrolls internally when extra controls need room.
- Production TypeScript/Vite build and ESLint passed. The focused clip-settings
  browser checks and the full isolated sidebar suite passed at desktop/mobile
  widths and across all six themes. Checks cover menu size/alignment, Auto's
  disabled controls, stable slider position, overlap editing, Window mode,
  continuity, enhancement actions, queues and the existing shared scroller.
- The duration/state integration suite passed, including native increments,
  GPU caps, manual 14.4-second windows, 60-minute output, Extend restoration and
  Viggle source-duration planning. Existing duration calculations were retained.
- A simulated keyboard transition exposed stale popup positioning; measuring
  after the drawer's layout update and observing its size corrected it. The
  final keyboard viewport checks passed. These Chromium simulations still need
  the normal real iPhone/Safari check during user testing.
- Logs: `.codex-tmp/clip-settings-build.log`,
  `.codex-tmp/clip-settings-lint.log`, `.codex-tmp/clip-settings-ui.log`,
  `.codex-tmp/clip-settings-full-ui.log` and
  `.codex-tmp/clip-settings-duration-math.log`. Menu and Auto screenshots are in
  `.codex-tmp/sidebar-validation/`. No live generation, backend restart,
  launcher change or public push was performed. Refresh to use the rebuilt UI.

## Auto interaction follow-up — 8 September

- Auto and the current duration now stay visible above both Time and Window,
  preserving the tabs' vertical position. Enabling Auto from Window returns
  to the automatic Time recommendation.
- Auto's manual time controls remain dimmed but interactive. Grabbing the
  slider turns Auto off on pointer-down, and the same drag sets the duration.
  Keyboard adjustment, preset selection and custom time entry also take over
  directly. Model limits and genuinely unavailable controls stay enforced.
- Production TypeScript/Vite build and ESLint passed. The focused isolated
  clip-settings browser suite passed at 1360, 767, 440, 390 and 320 pixels,
  checking immediate activation, preserved drag position, keyboard native
  increments, preset/custom values, stable tabs and Auto from Window. Existing
  overlap, window sizing, continuation and keyboard viewport checks also passed.
- The duration/state integration suite passed, including GPU recommendations,
  manual 14.4-second limits, 60-minute presets, Extend and Viggle restoration.
  Logs: `.codex-tmp/auto-duration-build.log`,
  `.codex-tmp/auto-duration-lint.log`, `.codex-tmp/auto-duration-ui.log` and
  `.codex-tmp/auto-duration-state.log`. Browser actions used intercepted writes;
  no live generation, backend restart, launcher changes or public push occurred.
  Refresh Maestro to load the rebuilt UI.

## Prior model checks and remaining manual validation

The [H3/media port record](development/wan2gp-12-71-port-plan.md) records earlier
RTX 4090 queue tests for Full/Pruned VDN, 45-second and five-minute H3 audio,
audio-refinement video preservation, H3 outpainting and RIFE x3. Those checks
are historical evidence, not GPU runs repeated during this release preparation.

The following remain manual checks before describing the release as fully
validated across supported hardware:

- Fresh installation and an update from an existing public v2.0.1 installation.
  No reset, reinstall or dependency replacement was performed on the active
  development installation for this candidate.
- A final real iPhone/Safari keyboard and scrolling pass. Chromium viewport
  simulation passed; it does not reproduce every iOS keyboard animation.
- Native DLSS Neural Rendering and Frame Generation on supported Windows 11/RTX
  hardware. This Windows 10 machine cannot execute that path; synthetic worker,
  capability, transport and cancellation tests do not establish its image
  quality or performance. The optional installer was not run.
- Representative long/high-resolution VDN and multi-character/face-refinement
  quality checks for the intended hardware. No fixed speed improvement is
  claimed from this release validation.

Publication is deliberately held. The local maintainer checklist records the
remaining manual and publish steps; no public branch or tag should be updated
until the maintainer authorizes publication.
