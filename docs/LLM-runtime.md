# Local LLM runtime on Linux

Maestro's local writer uses a separate **llama-server** process. CUDA working in
PyTorch does not establish that this executable supports CUDA. The upstream
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
if these prerequisites are missing, then retry **Load** in Settings.

To use your own compatible CUDA build, set `MAESTRO_LLAMA_BIN` to its directory
before starting Maestro. Include its shared libraries beside the executable;
`llama-server --list-devices` must list a device such as `CUDA0`.

If a known CUDA build no longer sees a GPU, Maestro reports its driver/library
diagnostic instead of rebuilding repeatedly or silently using the CPU. Selecting
**CPU** explicitly remains supported. Startup and model-offload diagnostics are
saved in `logs/llm/llama-server.log`, including failed loads.

The Windows prebuilt CUDA download path is retained. Remote/API writers do not
use this local runtime.
