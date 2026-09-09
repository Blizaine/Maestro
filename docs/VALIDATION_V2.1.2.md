# Maestro v2.1.2 release validation

Validated on 9 September 2026. See the [release notes](RELEASE_NOTES_V2.1.2.md)
for the user-facing changes.

## Release boundary

Both public branches, `dev` and `main`, pointed to
`2eaf4eca2f8d1842401e0de92f8ce8521975153f` before publication. This patch includes
the subsequent H3 prompt/reference audit fixes, undefined-name corrections,
regression tests, CI check and release documentation. The application version
comes from the root `VERSION` file.

## Completed audit checks

| Check | Result |
| --- | --- |
| Full Python suite | 1,457 tests passed in 75.388 seconds using Maestro's local Windows Python 3.11 environment. |
| New H3 regression tests | 15 methods passed, including subcases for three Omni variants and both enhancement styles. |
| Standalone JSON grammar runner | All five checks passed. |
| Undefined-name check | Ruff `F821,F823` passed across `app/services`, `app/models/minimax_h3` and `app/launch.py`. |
| Python syntax | The audited services, H3 implementation and `app/launch.py` compiled successfully. |
| Application version | Root `VERSION` and the isolated backend version reader report `2.1.2`. |
| UI type-check and production build | Passed. Existing bundle-size and mixed static/dynamic import warnings remain. |
| Release documentation and CI configuration | Release links exist and the CI workflow parses as YAML. |
| Published-source boundary | Clean-repo guard passed across 2,192 tracked/staged files; staged whitespace checks passed. |

## Reproductions and limits

The H3 regression tests import the actual service and reference compiler. They
cover enhancement without source dialogue, quoted character descriptions,
metadata, nested quotations, explicit music intent, silent windows, AI repair
and fallback, postposed attribution, named guests and independent Subject and
Speaker IDs. Existing H3, RefMod and story-planning tests pass as part of the
complete suite.

LLM responses are mocked and prompts are synthetic. The reporters' original
music-reference prompts were unavailable, so these tests confirm matching code
defects without claiming to reconstruct each exact request. Complete GPU
generation, listening tests and full Blend/inpaint workflows were not run.

The checks do not restart the user's application. The running backend displays
the new version after its next normal restart. Private characters, outputs,
local guides and verification logs remain outside the published source.
