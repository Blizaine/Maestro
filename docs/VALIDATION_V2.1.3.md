# Maestro v2.1.3 release validation

Validated on 9 September 2026. See the [release notes](RELEASE_NOTES_V2.1.3.md).

## Release boundary

Both public branches, `dev` and `main`, pointed to
`684e9783b187b21f2016978471fc40ec4a8b423e` (v2.1.2) before this patch.
The application version comes from the root `VERSION` file. No launcher,
dependency, model-weight, private character or local content-guide changes
are included.

## Completed checks

| Check | Result |
| --- | --- |
| Full Python suite | 1,475 tests passed in 85.222 seconds using Maestro's Windows Python 3.11 environment. |
| Director image checkpoint | All 69 Director cancellation/recovery tests passed after adding an assertion that a saved start image survives a later keyframe timeout, in memory and on disk. |
| JSON grammar runner | All five standalone checks passed. |
| Undefined names | Ruff `F821,F823` passed across services, H3, `app/launch.py`, NVFP4 and FlashVSR runtime. |
| Python syntax | Edited runtime modules and service/script directories compiled successfully. |
| UI type-check and build | Passed; existing mixed-import and bundle-size warnings remain. |
| Published-source boundary | Clean-repo guard passed across 2,199 tracked/staged files; staged whitespace checks passed. |

## Regression coverage

- **UTF-8:** actual Requests response objects with absent/incorrect charset
  headers, one-byte network chunks, Arabic, Cyrillic, Chinese, accented text,
  emoji and Unicode separators; local/OpenAI-compatible and Anthropic paths.
- **Director:** the section boundaries supplied in #117, without its prompts
  or private media; H3 and LTX native scheduling; idempotent reviewed timelines;
  prepared-plan subdivision with each source image assigned to its new shots;
  legacy Dashboard data preservation and image checkpoint recovery.
- **Memory:** exception-owned activation lifetime, cleanup after handled
  failure, successful cache retention, original exception preservation and
  release of FlashVSR plus registered post-processors.
- **NVFP4:** reported cuBLAS failure text, failed-shape caching, numerical
  fallback output with bias/dtype preservation, row padding and output trimming,
  and propagation of unrelated CUDA/OOM failures. Kernels are mocked and tensor
  checks run on CPU.
- **Linux runtime:** tar files containing shared-library alias chains and
  hardlinks, archive-path/link rejection, runtime search environment, automatic
  repair despite a current cache receipt, and unchanged Windows environment.
  Existing release-pointer/download tests also pass.

## Limits

The full suite imports production modules but mocks heavy model execution and
LLM responses. No full music-video render, listening/lip-sync assessment,
physical Blackwell kernel run or Linux Pinokio startup was performed. These
checks establish the repaired code paths, not universal hardware compatibility
or generated-media quality.

The running application was not restarted. It reads the new version and
backend changes on its next normal restart. Private test inputs, local logs,
characters and generated media remain outside the published source.
