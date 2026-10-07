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
