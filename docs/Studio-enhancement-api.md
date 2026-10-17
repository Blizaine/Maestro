# Queued Studio enhancement

Use the running Maestro URL shown by your launcher. Submit the same generation settings you would use in Studio, plus `_enhance_on_generation: true`. The server captures the current writer selection and original inputs. Later settings changes do not rewrite the queued job.

`_queue_mode: "held"` waits for **Run queue**. `"now"` joins generation immediately and waits for the shared generation slot when it is busy. `_client_submission_id` is an optional unique identifier; reuse it when retrying an uncertain submission so a duplicate request returns the same job.

These examples submit a held image job. Replace `PORT` with your Maestro port and use an installed model's normal generation settings.

```javascript
const base = 'http://127.0.0.1:PORT';
const response = await fetch(`${base}/api/v1/generate`, {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    model_type: 'flux2_klein_9b', generation_mode: 'image',
    prompt: 'A mountain monastery at dawn.',
    _enhance_on_generation: true, _queue_mode: 'held',
    _client_submission_id: crypto.randomUUID(),
  }),
});
if (!response.ok) throw new Error(await response.text());
const {job_id} = await response.json();
```

```python
import uuid
import requests

base = 'http://127.0.0.1:PORT'
response = requests.post(f'{base}/api/v1/generate', json={
    'model_type': 'flux2_klein_9b', 'generation_mode': 'image',
    'prompt': 'A mountain monastery at dawn.',
    '_enhance_on_generation': True, '_queue_mode': 'held',
    '_client_submission_id': str(uuid.uuid4()),
})
response.raise_for_status()
job_id = response.json()['job_id']
```

Save the same JSON body as `job.json`, then submit it with curl:

```sh
curl -X POST "http://127.0.0.1:PORT/api/v1/generate" -H "Content-Type: application/json" --data-binary @job.json
```

| Endpoint | Purpose |
| --- | --- |
| `GET /api/v1/jobs` | Job status, phase and compact enhancement state. |
| `GET /api/v1/jobs/{job_id}/enhancement` | Original prompt/settings, completed draft and prepared window plan. Writer settings and credentials are not exposed here. |
| `POST /api/v1/jobs/queue/start` | Release all currently held Studio jobs in their captured order. |
| `POST /api/v1/cancel/{job_id}` | Cancel the whole job, including enhancement. |
| `POST /api/v1/jobs/{job_id}/retry` | Retry a terminal enhanced job; JSON body has `action` below. |
| `DELETE /api/v1/jobs/{job_id}` | Dismiss a terminal job's saved queue history. Generated media is retained. |

Retry actions are `retry` (reuse completed enhancement or repair flagged H3 windows when a resumable draft exists), `refresh` (rewrite every prompt from the original), `as_written` (generate the whole job from the original prompt with enhancement off), and `accept_draft` (generate the whole job using the saved draft, including flagged windows). Normal input and window validation still applies to `as_written`. Retries create a new job; an active retry is returned instead of being submitted twice. A successful writing retry continues to full-job generation; unresolved review warnings pause it again.

H3 requests with an unclosed quoted utterance return an input error before writing or generation. Close each spoken quote so dialogue ownership and timing include every line. Quotes used for visible labels, titles, or character descriptions are separate from spoken dialogue. Put nonverbal directions outside the quote, for example:

```text
Alex says in a frustrated voice, "You son of a..."
A sharp censor beep interrupts the unfinished insult. The audience laughs.
```

An annotation such as `[beep censor]` inside a spoken quote is not a reliable sound-effect instruction. Describe the beep and audience reaction separately. At the normal planning pace of 2.8 words per second, reserve additional time for entrances, pauses, sounds, and reactions; use another window when the complete scene needs it. A reference fallback keeps the source's entrance order instead of placing every referenced character in the opening frame.

H3 plans expose `retryable_windows` as one-based window numbers and retain a `camera_checkpoint`. Keep that checkpoint with the plan. A targeted retry retains the shared schedule, exact dialogue and neighbouring entry/exit states; passed window prompts are copied unchanged. `refresh` explicitly discards this checkpoint. Older drafts without one and failures in shared story planning require full re-enhancement.

For interactive planning, send the saved plan as `retry_plan` alongside the same inputs to `POST /api/v1/llm/plan-h3-windows` or `POST /api/v1/llm/plan-h3-sequence`. This updates the draft only. A changed prompt, model, references, timing or relevant settings invalidates targeted repair; the server returns an error rather than silently rewriting all windows. Manual prompt edits disable the old repair checkpoint because they may change continuity boundaries.

Enhancement states are `pending`, `enhancing`, `complete`, `review`, or `failed`. A cancelled job is identified by its job status. The server checkpoints the completed enhancement before generation. A writer failure or fallback requiring review does not silently proceed into generation. Other eligible jobs can continue.

Single-window H3 enhancement can make one additional advisory continuity call
when the native visual field contains multiple consecutive shots and fidelity
repairs are enabled. It reviews the final compacted prose for source event order,
persistent physical state across camera cuts, observable state cues and
source-supported relationships between object reference views. It preserves
exact dialogue, shot markers, timestamp values, explicit cuts and nonvisual
fields, and checks the candidate through the existing H3 validators. It does
not run for Director's custom writing instructions or multi-window planning.

The saved enhancement exposes `continuity_reviews`, with `status`, `checked`,
`repaired`, `issues`, `state_handoffs`, `reference_groups` and `diagnostic`.
These are writing diagnostics, not rendered-quality approval. Malformed or
rejected advisory responses retain the preceding valid prompt and do not add a
new review gate. Setting `enhance_fidelity_retries` to zero skips this extra
call. Cancellation still stops the job. A valid timeline does not guarantee
that H3 will execute every transition or avoid repetition, particularly in
experimental extended-duration generations.

Reference sequences retain the complete journey in planning context, while
native window prompts omit global setting and lighting itineraries that would
preview future locations. Stable shared settings, visual style and effect
mechanics remain available in every window.
Native continuation camera writing and compilation use accepted
`native_closing_state` snapshots for visual continuity. The camera writer
receives local shared context rather than the global location/lighting route;
authoritative action-completion receipts remain unchanged in the planning
checkpoint. Legacy receipts without a concrete snapshot use
the preceding visible composition rather than repeating an action recap.
Source-supported complementary object views can share a persistent object
relationship without creating new Subject IDs. Fallback coverage separates
explicit camera changes while preserving source action order and continuous
shot constraints. These controls improve the instructions; rendered
continuity still needs review.

After a server restart, interrupted enhanced jobs return as held. Resume them explicitly; Maestro does not automatically repeat a possibly completed generation. Browser closure does not stop server execution.
