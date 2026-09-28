# Maestro v2.5.0 validation

Release preparation, September 28, 2026. Public baseline:
`3beefd4f3c4c851293f8f69ff6ad1f7d3cf55175` on both dev and main, including
v2.4.2 and its gallery-test correction. Baseline
[CI passed](https://github.com/Blizaine/Maestro/actions/runs/36329443028).
Release CI runs are available in [GitHub Actions](https://github.com/Blizaine/Maestro/actions/workflows/ci.yml).

## Regression checks

- **99 focused CPU tests passed during release preparation:** 22 LoRA metadata,
  8 LoRA guidance, 18 URL import, 5 H3 LoRA naming, 18 DLSS experimental,
  17 DLSS direct-worker, and 11 gallery-thumbnail tests. Thumbnail checks use
  real FFmpeg to verify requested 480/960/1920 dimensions and no enlargement
  of smaller sources. The gallery suite requires execution from the repository
  root; an initial invocation from `tests/` failed to import `app`, then passed
  from the correct directory.
- **54 H3 checks passed immediately before release preparation** on the same
  implementation: 8 new geometry checks, 39 window-planner tests and 7 selected
  existing H3 compatibility tests. Coverage includes normalized overlaps,
  automatic memory-based window sizing, single-pass versus joined timelines,
  clean-tail spans, saved settings, reviewed prompts and planning requests.
- The released handler reproduced the reported class of failure: total 336
  became 345 frames, producing four windows. The fixed handler retains 336
  frames and three windows with a 124-frame pass and 18-frame overlap.
- **Three browser suites passed during release preparation:** gallery viewer
  (desktop and phone), LoRA display names, and Frames image picker. Their API
  calls are intercepted, with synthetic media and isolated browser storage.
- **Two additional browser suites passed immediately before preparation** on
  unchanged source: H3 Frames Enhance-to-Generate and Studio duration. These
  cover preserved manual edits, same-model refresh, stale-boundary rejection,
  rolling totals near the native maximum, single-pass rounding, extended H3
  duration, References and Viggle duration behavior. No jobs were submitted to
  the running app.
- Python compilation and undefined-name checks (Ruff F821/F823) passed for the
  changed application areas. The modified PowerShell installer parsed without
  executing installation or downloading components.
- Frontend ESLint, TypeScript checking and the production Vite build passed.
  Existing editor mixed-import and bundle-size warnings remain non-blocking.
- The public-source boundary guard passed with all 2,631 staged/tracked files,
  including new source and tests. Models, generated guides, local aliases,
  user settings and media remain outside the public snapshot. The release uses
  the root `VERSION` file; launcher schema versions and launcher URLs are
  unchanged.

## Scope and limits

These checks validate routing, metadata, timing, file handling, CPU math and
browser behavior. They do not certify every third-party LoRA or guarantee that
a video model follows every supplied guide or trigger word. Incompatible model
architectures remain incompatible.

Release preparation did not download models, submit GPU generations, change
user preferences, restart the app or install native DLSS components. Windows 10
Frame Generation still requires its separate runtime, supported hardware,
HAGS, driver support and a successful native capability probe. CPU mocks do
not establish native support on every Windows/GPU combination.

The maintainer previously tested the gallery and LoRA improvements. Physical
mobile Safari behavior, clean installation/update and final GPU quality checks
remain distinct from isolated Chrome and CPU regressions. Full Python discovery
and standalone JSON-grammar tests run separately in Linux GitHub CI.

[Release notes](RELEASE_NOTES_V2.5.0.md)
