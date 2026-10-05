# Free LM Studio model memory

When an external LM Studio model shares your GPU with Maestro, its loaded
weights can leave too little VRAM for image or video generation. Maestro can
explicitly unload one language-model instance through LM Studio's native API.
This requires **LM Studio 0.4 or newer** with its server enabled.

In **Settings → Integrations**, choose **Remote OpenAI-Compatible**, configure
the server URL and its optional API key, then use **LM Studio memory**:

1. Click **Refresh loaded models**.
2. Choose the loaded model instance. A unique match for your configured writer
   is selected automatically; multiple instances require a choice.
3. Click **Unload selected model**. Maestro checks the native model list again
   before confirming that the selected instance is unloaded.

The next writer request can load that model again. Other clients using the
same instance may also need to reload it. Existing idle cleanup and provider
switching do not send an external unload request.

Maestro rejects this action while generation or its writer is busy. If the
server changes, authentication fails, or release cannot be verified, the panel
shows the error and requires a fresh model list. Other OpenAI-compatible
servers can still be used for inference, but this memory control requires the
LM Studio native API.

## Maestro API

Both routes use Maestro's saved remote URL and remote-provider API key. The
browser and callers do not supply an external target or credentials.

**GET /api/v1/llm/remote/loaded-models** returns a server_url and instances.
Each instance has an instance_id, model_key and display_name.

**POST /api/v1/llm/remote/unload** accepts:

    {"instance_id": "writer-instance", "server_url": "http://localhost:1234"}

Use the server_url and instance ID from the list response. The URL is a
consistency check against the saved configuration. If an API caller omits
instance_id, only an unambiguous configured writer can be unloaded.

The result is status "unloaded" after verified removal, or "not_loaded" when it
was already absent. Invalid selections return 400, a busy writer/generation or
changed server returns 409, and failed native API requests or verification
return 502.

### JavaScript

    const loaded = await fetch('/api/v1/llm/remote/loaded-models').then(r => r.json());
    // Choose the intended instance from loaded.instances.
    const selected = loaded.instances.find(item => item.instance_id === 'writer-instance');
    if (!selected) throw new Error('The intended model is not loaded');
    const response = await fetch('/api/v1/llm/remote/unload', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ instance_id: selected.instance_id, server_url: loaded.server_url }),
    });
    if (!response.ok) throw new Error((await response.json()).detail);
    console.log(await response.json());

### Python

    import requests

    maestro_url = "http://127.0.0.1:8000"  # Use your Maestro WebUI URL.
    response = requests.get(maestro_url + "/api/v1/llm/remote/loaded-models")
    response.raise_for_status()
    loaded = response.json()
    selected = next(item for item in loaded["instances"] if item["instance_id"] == "writer-instance")
    response = requests.post(maestro_url + "/api/v1/llm/remote/unload", json={
        "instance_id": selected["instance_id"], "server_url": loaded["server_url"],
    })
    response.raise_for_status()
    print(response.json())

### Curl

    curl http://127.0.0.1:8000/api/v1/llm/remote/loaded-models
    curl -X POST http://127.0.0.1:8000/api/v1/llm/remote/unload \
      -H 'Content-Type: application/json' \
      -d '{"instance_id":"writer-instance","server_url":"http://localhost:1234"}'

Use your Maestro WebUI URL and values from the list response. The external
contract is documented in LM Studio's [model list](https://lmstudio.ai/docs/developer/rest/list)
and [unload](https://lmstudio.ai/docs/developer/rest/unload) references.
