# Maestro v2.1.1 release validation

Validated on 8 September 2026. See the [release notes](RELEASE_NOTES_V2.1.1.md)
for the user-facing changes.

## Release boundary

Both public branches, `dev` and `main`, pointed to
`f49915a5022fc364786d447a9f3c507f182c80b8` before publication. This patch includes
the subsequent dialogue, cast binding, planning feedback and log-documentation
fixes. Root `VERSION` and the isolated application version reader report `2.1.1`.

## Completed checks

| Check | Result |
|---|---|
| Complete Python suite, with CUDA disabled | 1,442 tests run: 1,441 passed and one CUDA-only test skipped, in 92.174 seconds. |
| Standalone JSON grammar runner | All five checks passed. |
| Python syntax | `app/services`, `app/launch.py` and `scripts` compiled successfully. |
| UI type-check, production build and ESLint | Passed. Existing bundle-size and dynamic-import warnings remain. |
| Studio prompt component in Chrome | At 390px and 1280px, warnings are visible, Refresh preserves Creative/Faithful, and there is no horizontal overflow or browser error. Uses isolated state and production styles. |
| Documentation | Issue-form and CI YAML parse; corrected log paths were verified against real files; release links and README anchors checked. |
| Source boundary | Clean-repo guard and staged whitespace checks passed before publication. |

## Live dialogue validation

The installed Qwen3.8 27B model was tested with the reported three-character,
six-topic, five-window conversation. Fresh planning produced 129 spoken words
across ten turns, compared with the original 53 words. All six requested topics
appeared in speech and every window stayed within its timing budget.

A separate run of the focused repair stage took the original sparse dialogue
to 111 words across ten turns. All five windows passed their word and topic
checks without remaining warnings. Regression tests also cover a failed window
preserving another window's successful repair, exact supplied lines, explicit
silence, brief speech, ambiguous RefMods and Subject/voice matching.

These live checks validate prompt planning. Generated video quality still needs
hardware and listening review. The publication checks run without interrupting
the user's active generation; the running backend displays the new application
version after its next normal restart.

Local verification logs use the `.codex-tmp/v2.1.1-` prefix. The earlier focused
browser and live LLM reports use `.codex-tmp/creative-`; these local artifacts
remain outside the published source.
