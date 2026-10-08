# WanGP H3 benchmark client

`app/scripts/benchmark_wangp.py` runs a bounded, serial H3 comparison through WanGP's documented external MCP v2 Streamable HTTP API. It uses Python's standard library and never starts or stops the app.

Start the intended WanGP instance separately with external MCP enabled. The API-v2 asynchronous server needs:

~~~text
--mcp --mcp-api-version 2 --mcp-async --mcp-transport streamable-http --mcp-host 127.0.0.1 --mcp-port <port>
~~~

Connect to `http://127.0.0.1:<port>/mcp`. The client checks that the generate schema permits `wait=false`, reads defaults/capabilities/profiles, and checks the connected generation queue before every job. It submits one job at a time and refuses any existing queued or running work, including UI work.

## Manifest

Root settings are copied to each named case; case settings override them. Case repeats run as separate serial jobs, with `repeat_generation: 1` keeping each job to one generated output. The client rejects H3's unsupported `generation_mode`, `guidance_scale`, and `negative_prompt` fields; H3 uses implicit video generation and CFG 1.0.

~~~json
{
  "name": "h3-pruned-comparison",
  "settings": {
    "model_type": "minimax_h3_fl2va_pruned",
    "config": "gguf_q2_k,int8_convrot",
    "prompt": "A quiet cinematic shot of a red kite over a field.",
    "resolution": "864x480",
    "video_length": 124,
    "num_inference_steps": 20,
    "flow_shift": 7.0,
    "seed": 424242,
    "repeat_generation": 1,
    "prompt_enhancer": "",
    "activated_loras": [],
    "image_prompt_type": "",
    "video_prompt_type": "",
    "audio_prompt_type": "",
    "sliding_window_size": 124,
    "sliding_window_overlap": 18,
    "sliding_window_discard_last_frames": 0,
    "override_attention": "sage2",
    "override_profile": 4
  },
  "cases": [
    {"name": "480p", "repeats": 2},
    {"name": "720p", "repeats": 2, "settings": {"resolution": "1280x720"}}
  ]
}
~~~

Before submitting anything, the client verifies that the connected H3 model defaults declare `video_length`, `num_inference_steps`, and `flow_shift`, and that supplied values have compatible types. It does not require every request field to appear in the pristine model defaults: fields such as model selection, prompt, resolution, seed, the per-model selector, and API-declared task controls can be valid request context outside those defaults. The report lists requested fields absent from the default map so they remain visible for review.

The `config` field is WanGP's per-model selector string. For H3, `system_configs` is the Text Encoder group and `system_configs2` is the Video VAE group, so `gguf_q2_k,int8_convrot` requests the exact Q2_K text encoder and INT8 ConvRot VAE. The H3 model definition maps these to `qwen3vl-32B-MiniMax-H3-Q2_K.gguf` and `minimax_h3/minimax_h3_video_vae_int8_convrot.safetensors` (plus the INT8 X2 VAE). The client compares `config` against the settings recorded in each output file.

WanGP's global settings are separate: `transformer_quantization: "int8"` selects `MiniMax-H3-FL2VA-pruned_rank8_int8_convrot.safetensors` for this model, and `attention_mode: "sage2"` forces SageAttention 2. These runtime values and the actual loaded file paths are not exposed by saved media settings; confirm the loaded paths from the app's model-load log.

The installed app's `generation_preview` global supports `rgb`, `tiny_vae_frames`, and `tiny_vae_video`; source validation rejects other values. There is no documented preview-off setting in this version, so the client does not send an invented `live_preview` field.

## Run and report

~~~powershell
python app/scripts/benchmark_wangp.py `
  --base-url http://127.0.0.1:42019 `
  --manifest path/to/h3-benchmark.json `
  --timeout 1800 `
  --poll-interval 2 `
  --output path/to/report.json
~~~

The JSON report contains model defaults/capabilities/profiles, queue snapshots, requested settings, job IDs, progress events, timestamps, elapsed time, results/files/errors, Gallery output metadata, and actual saved generation settings. Per-output comparison reports matched, mismatched, and unreported fields.

A timeout cancels only the `job_id` returned by the benchmark's own submission, then polls that job for a bounded cancellation grace period. Ambiguous submissions are not retried. If another task enters the queue between repeats, the next job is refused. Reports are atomically written, including partial-failure results.

Limits: 1 MB manifest, 12 cases, three repeats per case, and 24 total expanded jobs.
