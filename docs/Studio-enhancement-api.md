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

Retry actions are `retry` (reuse completed enhancement or retry an unfinished writer), `refresh` (write a new draft from the original), `as_written` (explicitly skip enhancement), and `accept_draft` (generate a reviewed fallback). Normal input and window validation still applies to `as_written`. Retries create a new job; an active retry is returned instead of being submitted twice.

Enhancement states are `pending`, `enhancing`, `complete`, `review`, or `failed`. A cancelled job is identified by its job status. The server checkpoints the completed enhancement before generation. A writer failure or fallback requiring review does not silently proceed into generation. Other eligible jobs can continue.

After a server restart, interrupted enhanced jobs return as held. Resume them explicitly; Maestro does not automatically repeat a possibly completed generation. Browser closure does not stop server execution.
