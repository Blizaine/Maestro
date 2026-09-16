# Maestro v2.1.6 release validation

Validated on 11 September 2026. See the [release notes](RELEASE_NOTES_V2.1.6.md).

## Release boundary

Public `dev` and `main` both pointed to
`11cd18e58d858003fd27d364e3525cf2a14276a4` before preparation. This release
adds transformer residency, production-heading parsing, compact reference
editing and per-model step persistence. The application version is read from
the root `VERSION` file.

Launcher scripts, dependencies and model weights are unchanged. Private
characters, local settings, research notes, logs and generated media remain
outside the published source. The previously saved unified-enhancement plan
is separate future work and is not implemented by this release.

## Automated checks

Checks used Windows, Python 3.11.13 and PyTorch 2.10.0+cu130. CUDA was hidden
for Python discovery so the suite did not compete with generation for the GPU.

| Check | Result |
| --- | --- |
| Full Python unittest discovery | 1,592 tests in 111.977 seconds; passed with two CUDA-only skips |
| Standalone JSON grammar checks | All five passed |
| Python compilation | Services, H3, launch, generation entry point and scripts passed |
| Undefined Python names | F821/F823 passed for services, H3 and launch |
| UI type check and production build | `tsc -b && vite build` passed; 1,858 modules transformed |
| UI lint | Passed |
| Step-preference browser regression | Passed with either model-options or defaults responding first |
| Compact reference-editor browser checks | Passed on desktop and 390/320px mobile layouts, including simulated keyboard changes |
| Full Studio/Director browser suite | Passed across six theme variants and 1360/767/440/390/320px widths |
| Public-repository boundary guard | Passed across 2,219 staged/tracked files |
| Release version and documentation | Isolated app reader returned `2.1.6`; all 60 checked local links resolved |
| Staged whitespace check | Passed |

The Python skips are the optional CUDA temporal-convolution comparison and
the explicitly enabled CUDA INT8 dispatch check. The UI build retains its
existing nonfatal bundle-size and mixed static/dynamic import warnings.

New or expanded coverage includes:

- Actual per-job budget calculations reaching MMGP, manual-preload precedence,
  tighter coefficients, unchanged encoder/VAE budgets, cached model reuse and
  reload, failure/cancellation cleanup and cross-model isolation.
- Production headings, quoted visual directions, Role A/B descriptions,
  titled ranges, flattened pastes, cast completeness and no-slow-motion
  constraints. The reported temple brief produces five timed action phases
  and zero dialogue entries; real speech and its timing limits remain tested.
- H3 Fused step bounds through 12, validation of invalid saved values, model
  switching, page reloads, changed browser ports, server preference failures,
  delayed responses, clamping and fixed-step recipes.
- Inline reference names/types, preview, image isolation, audio timeline and
  attachment controls, forward/backward drag ordering, keyboard ordering,
  linked character appearance/voice and prompt access during mobile typing.

Browser tests use isolated fixtures. The sidebar fixture reads the running
app's model catalogue/options; browser writes, uploads and generation requests
are intercepted. These checks do not submit GPU jobs or alter saved settings.

## Reported A4500 performance

Cocktail Peanut supplied measurements from an RTX A4500 with 28 GB system RAM,
using Viggle INT8 ConvRot at 864 x 480, 124-frame windows and three steps.

| Configuration | Reported denoising time per step |
| --- | ---: |
| After the INT8 execution fix shipped in v2.1.5 | Approximately 75 seconds |
| With the transformer residency fix | Approximately 60 seconds |
| WanGP on the same test machine | Approximately 63–71 seconds |

The report's four post-fix samples were 56.3, 63.3, 60.7 and 53.7 seconds per
step. Its MMGP log showed roughly 9.3 GB preloaded instead of the former
58 MB base. MMGP can allocate less than the requested budget because its own
working-memory limits still apply.

These are contributor measurements, not an independently repeated A4500
benchmark. The local integration preserves the reported transformer-only
approach and adds checks for tighter ceilings, cached-model reloads and
restoration. Speed and stability still depend on the GPU, RAM, checkpoint,
resolution, references and window size. Additional steps take proportionally
more work; the expanded step range is not a guarantee of better image quality.

## Local video reproduction

A complete 396-frame, 16.5-second Viggle replay on the current local code used
the same inputs and seed as a known-good earlier output. All decoded video
frames matched exactly, including the rapid cuts and ending. Two existing
outputs from a different seed also matched each other exactly while showing
the reported character drift. The maintainer subsequently confirmed another
test was working well.

This is a reproducibility check for the inspected configuration, not a quality
guarantee across seeds, clips or GPUs. Neither per-window seed behavior nor
camera-cut handling was changed for this release. The separate CPU comparison
of old/new control-frame preparation also preserved every conditioning pixel
at all four window boundaries of the inspected clip.

## Validation limits

No fresh-install or physical iPhone keyboard test was repeated during release
preparation. Mobile viewport/keyboard geometry was simulated in Chromium.
The lower-RAM A4500 case was assessed through the contributed report and CPU
integration tests; this checkout's video replay does not independently measure
that machine's residency speedup. No additional model or dependency download
is required by the release changes.
