# Maestro v2.1.0 local release validation

Prepared on 7 September 2026. This is a local release candidate, not a published
release. See the [release notes](RELEASE_NOTES_V2.1.0.md) for the user-facing changes.

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
