# H3 reference audio and writer draft recovery

Studio and Director share the audio-role controls beneath each H3 video
reference. The visual role (motion, character or scene) remains independent.
For a video with sound, or with an attached audio file, choose its audio use:

| Role | Behavior |
| --- | --- |
| Ignore source audio | Excludes this source audio from conditioning. It does not mute generated output. |
| Audio reference | Previous checkbox behavior: H3 can reproduce or reinterpret the source sounds and can generate additional audio. |
| Preserve soundtrack / performance timeline | Uses the source as the exact target soundtrack, with visuals synchronized to it. Only one exact soundtrack can own the output timeline. |
| Voice reference | Borrows vocal identity for newly requested dialogue; does not copy the source words or timeline. |
| Music / sound style | Borrows music or sound characteristics; generates a new waveform. |
| Sound-effect reference | Reuses the effect's characteristics for actions in the prompt; does not play the exact source waveform. |

Old saved references without an explicit audio role keep their checkbox
behavior. An externally attached soundtrack takes precedence over embedded
video audio. Maestro extracts embedded audio into a reusable local cache only
when a typed audio role needs it; original source files stay intact. A selected
role needs a readable audio track or an attached audio file.

Timeline audio advances through the matching source segment for each window;
voice and effect references remain reusable samples. Exact soundtrack duration
and visual window lengths still follow the model's supported frame geometry.
Generative audio-reference/style/effect modes cannot guarantee that H3 will
avoid new voices. Choose exact preservation when the original soundtrack must
remain authoritative.

## Inspect rejected drafts

Studio's **Job prompts** view includes **Writer drafts**, with the received
text, the validation stage and specific rejection reasons. Interactive **Enhance
now** also saves a local history and exposes a load-history control beside the
prompt. Director's Dashboard has the same load-history control under its LLM
planning log. Older jobs may have no history: the update cannot recover drafts
already overwritten by earlier versions.

Histories belong to the local job/project and survive normal retries and
restarts. Interrupted calls retain any received partial text. A received draft
is an intermediate response, not a quality approval. Uploaded image bytes and
API credentials are excluded from the diagnostic records. The ordinary API
status polls do not send the complete history; it is loaded when requested.

## Recover a multi-window plan

Retry flagged windows to retain the accepted prompts and repair only the
affected parts of the saved schedule. If a focused camera repair fails, Maestro
attempts to retain the usable cards from the existing draft before falling
back for missing events. Changed source inputs require a fresh plan.

Manual mode still requires one prompt per computed window. A single original
paragraph is not a general recovery plan for a multi-window story. Incomplete
or mismatched schedules remain blocked with a specific explanation; creative
quality warnings remain visible for review. These checks do not certify motion,
identity fidelity or lip sync in a rendered result.

## Read-only diagnostics API

- `GET /api/v1/jobs/{job_id}/enhancement`: queued Studio draft history.
- `GET /api/v1/llm/drafts/{history_id}`: interactive history; successful planning
  responses include `draft_history_id`, and errors include the
  `X-Maestro-Draft-History` response header.
- `GET /api/v1/director/pipeline/{pipeline_id}/llm-drafts`: live or saved Director
  history, including runs that failed before a final clip plan was created.

Diagnostic collection does not initiate generation or call another writer.
