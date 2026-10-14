# Local LLM CUDA runtime

Maestro's local writer uses a separate **llama-server** process. CUDA working in
PyTorch does not establish that this executable supports CUDA.

When **Settings → LLM → CUDA** is selected, device detection allows up to
120 seconds for llama-server startup. A timeout means CUDA availability could
not be verified. Maestro reports that timeout with the startup output it
captured, including any driver or library messages. Only a completed check
with actual CUDA device rows permits GPU loading; a partial device list from
a timed-out process is not accepted. Maestro keeps the selected device and
does not switch to CPU automatically.

## Underdog Saluki 27B

Choose **Underdog Saluki 27B IQ2-Mix Uncensored (Vision, Thinking)** in
**Settings → Integrations → LLM Model** with the Local provider to use
[Kujira's GGUF](https://huggingface.co/Kujira/Underdog-Saluki-27B-1.0-MTP-Abliterated-GGUF).
It is a compressed Qwen3.8-27B derivative of
[Underdog Saluki](https://huggingface.co/ConwayResearch/Underdog-Saluki-27B-1.0).
The selected file is `Underdog-Saluki-27B-1.0-IQ2-mix-MTP-abliterated.gguf`
(approximately 8.35 GB). Maestro also uses the original project's
`mmproj-Underdog-Saluki-27B-1.0-Q8_0.gguf` vision projector (0.63 GB).
Files download when the model is first loaded.

The profile uses a 32,768-token context, one server slot, Flash Attention and
Q8 K/V caches. Its **12 GB VRAM** picker hint estimates weights, projector and
KV cache; it is not a measured fit guarantee for every prompt or image set.
Runtime buffers, recurrent state and other applications also use memory.

Creative writing and prompt enhancement use the existing bounded Qwen thinking
controls and the publisher's thinking sampling settings. Explicit non-thinking
and grammar-constrained requests keep thinking off. The current default writer
is unchanged.

The file includes an MTP prediction head. Ordinary decoding is the default;
enable **MTP acceleration — Experimental** below the local LLM options to try
native speculative decoding. The toggle applies only to verified MTP-capable
weights, currently this Saluki file. The existing Qwen3.8 no-MTP quant and Gemma
models continue to use ordinary decoding. It also applies to a separate local
enhancement writer when that writer supports MTP.

The preference takes effect on the next writer load; saving it does not unload
or interrupt current work. **Loaded writer: MTP on/off** reports the actual
loaded mode. MTP drafts up to two tokens and may improve decoding speed, but
can use more memory or affect stability. A bounded runtime capability check
requires native MTP support before loading; an incompatible runtime produces
an update-or-disable error. Turn the option off to return to ordinary decoding.
The embedded model/template metadata and existing compatible llama.cpp loader
were checked; end-to-end quality and memory use still require a generation test.

## Windows startup checks

Windows uses the prebuilt llama.cpp CUDA executable and its bundled CUDA DLLs.
If device detection times out, retry the enhanced generation or Director request.
If it repeats, open
Maestro's Pinokio terminal and run these commands from the launcher folder:

```powershell
nvidia-smi
.\app\ckpts\llm\bin\llama-server.exe --list-devices
```

The second command should finish and list a device such as `CUDA0`. Include
both commands' output when reporting a persistent startup failure. A timeout
in this check happens before the writer model is loaded or H3 generation
starts; turning **Enhance prompt** off lets a Studio render proceed without
the writer. **CPU** remains available when explicitly selected in Settings.

## Dialogue enhancement checks

Once the console reports `Model loaded` and completed LLM responses, a dialogue
enhancement error concerns the draft returned by the writer. Maestro checks that
it has usable speaker/text pairs, honors any explicit turn count, and fits the
selected clip's spoken-word budget before asking for the H3 visual prompt.

Rejected attempts print `[Enhance dialogue]` with the specific reason. When a
valid AI-written exchange exceeds the clip's word budget, the next request
copyedits only its spoken text. Each line retains its original speaker and place
in the exchange; the writer receives a target and a hard maximum for each line.
If the edited exchange is still over budget, Maestro provides its measured
word counts and asks for shorter complete utterances for the overlong lines.
Up to two focused requests may follow the copyedit. This dialogue stage uses
at most four writer requests, including the initial draft.
The first copyedit uses structured text output with thinking disabled. Focused
repairs enable low-effort reasoning with 2,048 extra tokens reserved alongside
the 768-token answer allowance, so the writer can reconcile the measured word
counts with complete, meaningful lines. The total response allowance is bounded;
models that do not enforce a separate thinking limit may share it with reasoning.
Maestro checks the edited text against the same duration budget before continuing.
Exact user-supplied lines are never shortened by this stage.

If writing and repair fail, the saved job error includes the last rejection and
clip duration.
For an overlong AI-written exchange, increase the duration or request a more
concise exchange. A 124-frame H3 clip at 24 fps lasts about 5.17 seconds and permits
at most 15 spoken words across all speakers. A 243-frame clip lasts 10.125 seconds
and permits at most 30. A longer conversation needs more time. Malformed
speaker/text pairs or an incorrect turn count require a new draft
rather than a CUDA or memory-setting change.

## Linux CUDA build

The upstream
[Ubuntu llama.cpp archive](https://github.com/ggml-org/llama.cpp/releases/tag/b10964)
is CPU-only.

When **Settings → LLM → CUDA** is selected, Maestro checks `llama-server
--list-devices` before downloading/loading the model. An existing CPU-only Linux
runtime is replaced by a CUDA build of the compatible llama.cpp release. This is
a one-time build, cached under `app/ckpts/llm/bin`; it may take several minutes.
Progress and compiler errors are recorded in `llama-cuda-build.log` in that folder.
Maestro verifies a CUDA device before installing the new runtime and again after
relocation. A successful cached build is reused on later loads.

The build needs Git, CMake 3.24 or newer, Ninja or Make, a C++ compiler compatible
with the installed NVIDIA CUDA toolkit, and the toolkit's `nvcc` compiler. For
RTX 50-series cards use CUDA 12.8 or newer. Maestro searches `CUDACXX`,
`CUDA_HOME`/`CUDA_PATH`, PATH, and `/usr/local/cuda/bin/nvcc`. It does not install
system packages or change the generation environment. Follow the
[upstream CUDA build instructions](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#cuda)
if these prerequisites are missing, then retry the LLM request.

To use your own compatible CUDA build, set `MAESTRO_LLAMA_BIN` to its directory
before starting Maestro. Include its shared libraries beside the executable;
`llama-server --list-devices` must list a device such as `CUDA0`.

If a known CUDA build no longer sees a GPU, Maestro reports its driver/library
diagnostic instead of rebuilding repeatedly or silently using the CPU. Selecting
**CPU** explicitly remains supported. Startup and model-offload diagnostics are
saved in `logs/llm/llama-server.log`, including failed loads.

The Windows prebuilt CUDA download path is retained. Remote/API writers do not
use this local runtime.
