# ComfyUI workflows in Maestro

ComfyUI support is optional and off by default. Maestro connects to a ComfyUI
server that you install and run separately. Turning on the connection does not
install, start, stop, update, or download ComfyUI, its models, or custom nodes.

ComfyUI models and workflow graphs are flexible, but a file or graph is not
automatically compatible with every Maestro model route. Set up and verify one
workflow at a time, and keep the intended image, video, or audio kind explicit.

## Connect an existing ComfyUI server

1. Install and start ComfyUI using its own instructions. Confirm its server
   address from the ComfyUI launch or desktop app.
2. In Maestro, open **Settings → ComfyUI**, enable the connection, and enter
   that server address. The
   optional dedicated memory setting asks ComfyUI to free its idle models; it
   does not launch a separate process or reserve GPU memory.
3. Use **Test connection** to check that Maestro can reach the endpoint. This
   is a reachability probe; it does not queue a graph or prove that a model can
   generate successfully.
4. Discover configured model folders, scan the paths you want to reuse, and
   link the selected folders into Maestro's model inventory.

The ComfyUI process remains independently managed. A ComfyUI restart or update
is performed through its own installation. Keep the server address and model
directories available while Maestro workflows use them.

## Reuse model files

Discovery checks likely nearby ComfyUI installs and the paths named by
`extra_model_paths.yaml` or ComfyUI Desktop's `extra_models_config.yaml`. It
does not search every drive. The scanner reads configured directories and
model file metadata; it does not load tensor payloads or use the GPU. Linking
a model path records it for Maestro to reuse. It does not copy, move, rename,
delete, or rewrite files in ComfyUI or another application's model directory.

ComfyUI supports model-specific folders such as checkpoints, LoRAs, VAEs and
text encoders. Its extra-path YAML uses a `base_path` with category-specific
paths, and ComfyUI Desktop keeps its model configuration in its user config
area. Follow the official [ComfyUI model and extra-path guide](https://docs.comfy.org/basic-concepts/models)
when editing those files. Maestro reads the model paths; it does not edit them
or install custom nodes.

Model inventory labels are evidence about the observed path, not a universal
compatibility verdict. A known native match requires a byte or supported tensor
layout match. A filename alone does not establish compatibility. **ComfyUI
only** means a file is present in a ComfyUI model path; it may still be usable
through an explicitly validated graph. **Needs review** means Maestro could
not prove a native match. It is not a claim that ComfyUI cannot load the file.
No model file is downloaded automatically.

## Import and validate a workflow

Import a ComfyUI graph only when you trust its node types and the custom nodes
already installed in that ComfyUI environment. Import registers the graph in
Maestro; it does not install nodes or automatically submit the graph. A graph
can depend on node packages, model files, or custom behavior that is missing
from another ComfyUI installation.

Use a ComfyUI API-format prompt graph for import. A canvas workflow export may
need to be converted to the API format that ComfyUI accepts at its prompt
endpoint. For each graph:

1. Choose its kind: **image**, **video**, or **audio**.
2. Add typed bindings for the graph inputs Maestro should supply, such as
   prompts, model selections, dimensions, references, or audio.
3. Set defaults only for values that should remain fixed for that graph.
4. Run **Validate** with the actual selected inputs. Resolve missing node,
   model, and binding errors before adding a model route.
5. Generate a small result and inspect it in Maestro's normal gallery.

Validation is a preflight check, not an AI output test or a guarantee of
visual, audio, or temporal quality. If you use a model-free CPU diagnostic, it
checks basic service and request handling; it does not prove that an AI model
loaded or that the full workflow generated correctly. Successful generated
outputs are handled as normal Maestro jobs and appear in the normal gallery.

### HTTP API request shape

- `POST /api/v1/comfyui/workflows` registers `name`, `kind`, the API-format
  `workflow` graph, and its typed `bindings` list. `description` is optional.
- `POST /api/v1/comfyui/workflows/{id}/validate` accepts `{}` for defaults or
  `{"inputs": {...}}` to preflight actual selected values, including a
  checkpoint selection.
- `POST /api/v1/comfyui/workflows/{id}/run` takes `client_submission_id`
  (UUID), `inputs`, and `workspace`.
- `GET /api/v1/comfyui/jobs/by-submission/{uuid}` returns the receipt for that
  submission. Check it before retrying a request with an uncertain outcome.
- `POST /api/v1/comfyui/jobs/by-submission/{uuid}/resume` reconciles that
  submission's monitor state; it does not replay a prepared prompt.
- `GET /api/v1/comfyui/guide` returns the concise guide for MCP clients.

## Route Maestro models through ComfyUI

Maestro's native engine remains the default unless a caller explicitly selects
ComfyUI and an enabled route is configured for that model type.

Configure a route per Maestro model type after its graph has passed validation.
Select the registered workflow, match its image/video/audio kind, and map each
Maestro input to the correct graph field with **field bindings**. Use
**input defaults** for graph settings that should not vary with each request.
Keep the mapping specific to the verified graph; Maestro does not translate
unrelated ComfyUI graphs or infer that every file in a folder is compatible.

Enabled image routes require mappings for `prompt`, `seed`, `width`, `height`,
`num_inference_steps`, and `guidance_scale`. Video routes also require
`video_length` and `fps`. Only values explicitly mapped to the graph are sent;
do not assume an unmapped optional input is carried through.

Director can send individual clips through the same engine boundary when a
matching video route is enabled. Native Maestro LoRA handling, finishing
stages, and H3 multi-window continuation requests are rejected by the first
native mapper. Use Native for these requests, or run a workflow whose own
controls implement the operations. For audio-backed
video, a nonzero source offset must map to `audio_frame_offset` or
`audio_start_seconds` in the graph so the soundtrack begins at the intended
time.

## Agent Access

Agent Access uses the registered workflow ID and its validated bindings. A
caller needs a new `client_submission_id` UUID for each logical submission and
should pass uploaded Maestro asset references for media instead of local file
paths. Workflow IDs and model routes are explicit configuration; the API does
not expose an arbitrary ComfyUI server proxy. Observe access to jobs and
submission receipts remains subject to Maestro's backend ownership checks.

| Operation | Agent Access scope |
| --- | --- |
| Read status, workflow lists, or submission receipts | Observe |
| Test the connection or discover, scan, and link model paths | Manage |
| Run a registered workflow or reconcile its submission receipt | Generate |
| Validate workflows, change settings or routes, import or delete workflows, or call an unknown write | Admin |

Submission requests may have an ambiguous outcome if the connection drops.
Check the submission receipt by its UUID before deciding whether to submit
again. Keep private-intent receipt data private for review and resume; do not
turn private prompts or media context into a shared workflow default.

Cancellation deletes only the owned ComfyUI prompt while it is still queued.
For a running prompt, Maestro waits for terminal cleanup and discards cancelled
outputs. The cancellation request survives monitor recovery. It does not
interrupt a running job on a server shared by other users. ComfyUI
also exposes queue-management and interrupt routes at the server level; do not
use a global queue clear or interrupt as a substitute for per-job ownership
([ComfyUI server routes](https://docs.comfy.org/development/comfyui-server/comms_routes)).

## Official ComfyUI references

- [Models, folders, and extra model paths](https://docs.comfy.org/basic-concepts/models)
- [ComfyUI HTTP and WebSocket server routes](https://docs.comfy.org/development/comfyui-server/comms_routes)
