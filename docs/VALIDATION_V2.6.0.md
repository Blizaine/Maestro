# Maestro v2.6.0 validation

Release preparation, October 4, 2026. Public baseline:
`194cd36a40be631885d846f675d8d8dca1047cc1` on both dev and main, including
v2.5.0 and its regression/readme follow-ups. This record describes local
verification; public CI has not run for the unpublished release candidate.

## Release checks

The release metadata and public documentation use **2.6.0**. Pinokio's schema
version remains **8.0**; it is independent of Maestro's product version.

| Check | Result |
| --- | --- |
| UI production build (`npm run build` in `ui`) | Passed TypeScript and Vite. Existing large-chunk and mixed-import notices remain nonblocking. |
| Full frontend lint (`npm run lint` in `ui`) | Passed. |
| Selected UI regressions | All 23 suites passed; see the list below. |
| Full backend discovery (`python -m unittest discover -s tests -p "test_*.py"`) | Passed: 3,263 tests in 424.168 seconds, 11 skipped. |
| Explicit standalone regressions (`python -m pytest -q` with the seven CI modules) | Passed: 127 tests. Some unittest classes also appear in backend discovery; these counts are separate runs. |
| JSON grammar (`python tests/test_call_llm_json_grammar.py`) | All five functions passed. |
| Launcher startup (`node tests/startup_readiness.cjs`) | Normal and Sol readiness, stale addresses and invalid endpoints passed. |
| Python syntax (`compileall`) | Passed for services, launch, H3, LongCat, previews, quantization and scripts. |
| Undefined Python names (`ruff check --select F821,F823`) | Passed for services, H3 and launch. |
| Clean-repo guard (`python scripts/verify_clean_repo.py`) | Passed on all 2,750 tracked/staged files, including new tests and decoder licenses. |
| Release documentation and JSON | All 122 local Markdown references in the 14 changed Markdown files resolve to published files; all 12 changed JSON definitions parse. |
| CI configuration | YAML parses; Python environment, additional regressions and frontend lint are wired. |
| Release snapshot | Staged diff whitespace check passed; weights, caches, environments, uploads and output media are excluded. |

Backend checks used the installed Python 3.11 runtime with CUDA hidden. Pytest
8.4.2 was installed into an ignored test-tools directory; the application
environment was not changed. Optional hardware/runtime tests retain their skips.
Dependency deprecation, disabled CUDA autocast and CPU fallback notices were
nonblocking. Public CI uses its pinned Python 3.10/CPU PyTorch environment and
has not yet run on this candidate.

Release review fixed grouped camera coverage parsing: application-issued
ordering/enabler instructions remain in the semantic contract, while action
splitting uses the assigned source events. Unknown physical predicates still
require repair; ordering violations remain enforced. Focused regressions and
the full discovery run passed. Older isolated endpoint mocks and gallery source
assertions were updated to exercise the new quote validator, cancellable
download workers, source-owned camera cards, identity measurements and menus.

The browser suites were `avatar_inputs`, `longcat_routing`, `generation_preview`,
`dasiwa_models`, `h3_checkpoint_download_ui`, `h3_checkpoint_import`,
`h3_singularity`, `download_cancellation`, `status_polling`, `remote_model_memory`,
`reference_video_duration`, `sound_effect_reference`, `object_reference`,
`frames_video_input_posters`, `gallery_navigation`, `gallery_clip_actions`,
`h3_duration_guidance_unit`, `h3_duration_guidance`, `studio_duration`,
`h3_frames_timing`, `gallery_inputs`, `gallery_media_trim` and
`media_metadata_upload_delete`. They cover desktop and mobile layouts, including
the extended H3 duration slider and gallery
selection after metadata and pagination changes. `gallery_inputs` uses a
read-only gallery/options query; uploads and generation are intercepted.

The installed Pinokio runtime's `local.rm` API was inspected to confirm that
clearing readiness does not invoke URL sharing. Both launcher URL assignments
still consume the captured match through `local.set` and `input.event[1]`.
Install/update scripts read `app/requirements.txt`, whose MMGP 3.8.2 and GGUF
0.17.1 pins are also used by CI; updating rebuilds the frontend.

CI also runs frontend lint, compiles the changed runtime modules, installs
pytest and explicitly runs standalone preview, scheduler, source-camera,
Tiny VAE and Media Flow regressions. Those pytest functions would otherwise be
omitted by plain unittest discovery. The CPU test environment sets
`PYTHONPATH=app`, hides CUDA and disables xformers' optional Triton device probe.
This avoids a local Windows GPU-library import failure while testing with no GPU.

Browser runners accept `MAESTRO_PLAYWRIGHT` for an external Playwright installation
and `MAESTRO_CHROME` for the browser executable. Gallery media checks additionally
accept `MAESTRO_FFMPEG`; `gallery_inputs` requires the running local Maestro URL
as its first argument. Other selected suites use isolated fixtures.

## Scope and limits

Automated tests check routing, source fidelity, recipes, checkpoint structure,
quantized math, cancellation, timeline geometry and isolated browser behavior.
They do not certify every creator checkpoint, LoRA combination or generated
scene. “Verified import” means the file, layout, workflow, recipe and checksum
passed import checks, not that the model's creative output is guaranteed.

This preparation does not require a production generation, large weight download,
user preference change or app restart. Browser regressions use intercepted APIs
and synthetic media. Physical iPhone Safari, clean installs/updates, additional
GPU architectures and complete model renders remain separate validation.

DaSiWa and Singularity selections, approximate Sol attention, long experimental
H3 windows and unsupported-format restrictions retain their published caveats.
Preview decoding can add GPU work and is model-dependent; H3 has no Fast Frames
decoder. Download cancellation is offered only where the backend controls the
transfer; cancellation can wait for a connection timeout. LM Studio memory
controls require its native API, not just an OpenAI-compatible inference route.

Windows startup handling does not establish trust for a Python interpreter
blocked by Smart App Control/App Control policy (#164).

[Release notes](RELEASE_NOTES_V2.6.0.md)
