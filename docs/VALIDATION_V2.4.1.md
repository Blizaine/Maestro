# v2.4.1 validation

Release preparation on September 25, 2026, against v2.4.0 (`4afe169`).

## Release checks

The release checks use the installed Python 3.10 environment with CUDA and
model-network access disabled, plus isolated browser fixtures. No live
generation, training, paid provider calls or model downloads are needed for
this release pass.

- **Python regressions:** 2,808 tests completed in 391.697 seconds: 2,797
  passed, 11 optional/CUDA checks skipped, no failures or errors.
- Python syntax compilation and Ruff undefined-name/unbound-local checks
  passed for services, launch code, the changed model integrations, DLSS
  workers/installer and optional kernel adapters.
- ESLint, TypeScript and the Vite production build passed. The existing
  large-bundle and mixed static/dynamic editor-import warnings remain.
- Qwen Image 2.1 browser checks passed visibility/selection persistence,
  reference capability, managed acceleration profiles, all six native 2K
  aspect payloads, Auto 2K, control-image-only requests, inpaint assets and
  options, and resolution fallback when switching to a model without 2K.
- Gallery browser checks passed folder-qualified identity, filters,
  pagination, stale-query protection, chronology during running-job refreshes,
  atomic selection, removals and overlapping polls.
- Media details/upload deletion browser checks passed timestamps, measured
  facts, processing metadata, older upscale records, and recovery after a
  refused deletion. Tests use mocked deletion; no user's uploads were removed.
- All five standalone JSON grammar regression checks passed.
- The public-source guard passed across 2,601 staged/tracked files. Version,
  local release-document links, the explicit 78-file release inventory and
  whitespace checks passed; private records and runtime/media assets are excluded.

Release review corrected a React Fast Refresh lint issue by moving the shared
timestamp helper out of the component module. It also replaced deletion of
LanPaint's captured cache locals with rebinding to `None`, retaining cache
release before VAE decode without the static undefined-name warning. A CPU
regression checks separate positive/negative CFG caches, extract/reuse modes
and uncached generation. This was not a reproduced generation failure.
Three older UI source assertions were updated for the Media Info label,
shared reference-source slot accounting, and the Qwen/H3 Turbo step lock.
Distribution review also added
the Sol-runtime interpreter to the Windows 10 DLSS instructions and documented
the exact community runtime archive sources and hashes. Optional Comfy Kitchen
call sites retain established implementations when acceleration is unavailable.

## Earlier local hardware and API validation

These are the recorded development checks for the code included in this
release; they were not repeated as additional GPU jobs during packaging.

- **Windows 10 DLSS:** Windows 10 22H2, RTX 4090 24 GB, driver 591.86. Normal
  queued still/video checks passed at 1x and 2x. Additional native still and
  four-frame tests passed at 1.5x, 1.724x and 3x after the alignment fix.
  Queued 36-frame 24 fps videos from a 608x352 source produced 912x528,
  1048x606 and 1824x1056 output at the latter scales. They retained all frames,
  1.5-second duration and byte-identical audio packets. A cancellation check
  stopped both workers and removed the partial output.
- **Frame-processing correction:** a four-frame 1280x704 to 2560x1408
  RAFT/half-depth test reduced processing from 15.58 to 2.77 seconds with
  identical output pixels. This excludes initialization, shutdown and encoding
  and is not a general real-time performance guarantee.
- **Media Info API:** an existing 704x1280, 24 fps, 1,624-frame DLSS 1x video
  returned matching input/output facts, 67.667-second duration and its recorded
  processing time. Upload details and the deletion route registration were
  verified without deleting user media.

## Remaining limits

- The Windows 10 bridge remains experimental community compatibility work.
  Other GPU/driver combinations and long-video quality require user testing.
  Its known shutdown hang is contained in disposable workers; valid frame and
  finish receipts are required before bounded cleanup can count as success.
- Qwen issue #153 lacks a reproduced example with complete settings. Wider
  VAE overlap and the base CFG recipe address plausible causes, not every
  possible source of image artifacts. Turbo quality for masks, transparency,
  many references and native 2K needs broader evaluation.
- Kernel eligibility and fallbacks are tested; upstream H3/LTX/VAE speed
  claims have not been established across Maestro users' hardware. Package
  installation has not been exercised on every supported operating system.
- Browser tests use desktop/phone-sized emulation and mocked requests, not a
  physical-device sweep. Old media can lack original finishing facts; the UI
  labels available dates and leaves unknown history unclaimed.
- This update does not make new claims about general LLM prompt fidelity;
  the semantic limits documented for [v2.4.0](VALIDATION_V2.4.0.md) still apply.
