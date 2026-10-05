# Download controls

Download actions in **Settings → Performance → Enabled Models** sit at the right of each model row, separate from its enable checkbox. **Download** fetches the model's required assets without starting a generation. While the download is active, **Cancel** stops that model's download operation, including any remaining files it has not fetched yet.

The download banner also offers **Cancel** for each cancellable transfer. The model browser's download rows offer the same control for CivitAI checkpoints and LoRAs, and Hugging Face imports. Cancelling one operation leaves other downloads running.

**Cancelling** means the request was accepted and the worker is stopping. A stalled connection can take until its read timeout to unwind. **Cancelled** appears after cleanup; it is separate from a failed or unexpectedly interrupted transfer. Use **Retry** in Enabled Models, or start the download again in the browser. Downloads and cancellation state are recovered when the settings drawer or browser is reopened.

Completed model files remain available. A cancelled CivitAI transfer removes its private temporary file and never publishes incomplete weights or registers an incomplete checkpoint. Hugging Face can retain incomplete cache files to resume a later retry. Once verified weights have reached final installation, Cancel is unavailable until that operation completes. Opaque native Hugging Face transfers above the installed library's HTTP size limit also keep Cancel unavailable while that transfer runs.

## API

Resolve the running Maestro base URL through its Pinokio launcher. Cancellation uses an opaque ID supplied by the server, never a file path.

- `GET /api/v1/models/downloads/status` returns a model-keyed registry with `status`, `error`, `cancel_id`, and `cancellable`.
- `POST /api/v1/models/{model_type}/download/cancel` cancels a manually started model pre-download.
- `GET /api/v1/downloads/active` returns file progress, `cancel_id`, and `cancellable`.
- `GET /api/v1/civitai/downloads` returns browser import progress with the same cancellation fields.
- `POST /api/v1/downloads/cancel` accepts `{"download_id":"<cancel_id>"}` for one operation.

Statuses include `downloading`, `cancelling`, `cancelled`, `completed`, and `failed`. The active-file feed can also show its existing `incomplete` state. A successful cancellation request returns `cancelling`; poll until `cancelled`. Invalid IDs return 400, unknown IDs return 404, and a final installation that already won the cancellation race returns 409. Repeating cancellation for a finished operation preserves its terminal status.

JavaScript:

```javascript
const result = await fetch(`${base}/api/v1/downloads/cancel`, {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({download_id: activeDownload.cancel_id}),
}).then(response => response.json());
```

Python:

```python
result = requests.post(f"{base}/api/v1/downloads/cancel",
                       json={"download_id": active_download["cancel_id"]},
                       timeout=10).json()
```

Curl:

```sh
curl "$MAESTRO_URL/api/v1/downloads/cancel" \
  -H 'Content-Type: application/json' \
  --data '{"download_id":"<cancel_id>"}'
```

Cancellation is available only for explicitly started downloads with a server-provided control. Downloading dependencies automatically for an active generation keeps the existing generation/job controls.
