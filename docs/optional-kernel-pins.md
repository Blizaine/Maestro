# Optional CUDA kernel pins

The optional installers pin WGPKernels 1.5.0 and llama.cpp GGUF CUDA kernels 1.0.29. They select only the Maestro CUDA stacks below, update stale or mismatched builds, and keep a manually installed newer public release when its local build matches the active stack. The upstream release records are mutable, so the URLs carry the SHA-256 digest reported for each wheel by the release API; pip/uv verifies the bytes while installing. If a release asset is replaced, the optional install fails and the existing fallback remains available.

## WGPKernels 1.5.0

Source: [release v1.5.0](https://github.com/deepbeepmeep/WGPKernels/releases/tag/v1.5.0), [release API metadata](https://api.github.com/repos/deepbeepmeep/WGPKernels/releases/tags/v1.5.0), [v1.1.0 INT8 source](https://raw.githubusercontent.com/deepbeepmeep/WGPKernels/v1.1.0/wgpkernels/int8/__init__.py).

| Python / PyTorch / CUDA | Windows x86_64 SHA-256 | Linux x86_64 SHA-256 |
| --- | --- | --- |
| 3.10 / 2.7.1 / 12.8 | `3c471e4de8c6dc89653b4bcfaa0ee39cfa1b538e0303f733f5dc02cdb2078335` | `dff268568701dd2485971d5ffcdc29d61577689092f89a558ba557dd69dc1a17` |
| 3.11 / 2.10.0 / 13.0 | `013a406e47cceca81282e73027c602e2f8db9c6d4bfebba61afb39f7b7551893` | `5206cd03c1f9881c4e257f59b0f6d40f131b4cdd960d400e8869db6a70c41406` |

The Linux CUDA wheels require glibc 2.34 or newer. Maestro's INT8 dispatch requires an NVIDIA GPU with compute capability 8.0 or newer. The optional installer selects the CUDA wheels for the supported stacks above. WGPKernels 1.5.0 also publishes experimental ROCm wheels.

## GGUF kernels 1.0.29

Source: [release gguf-v1.0.29](https://github.com/deepbeepmeep/kernels/releases/tag/gguf-v1.0.29), [release API metadata](https://api.github.com/repos/deepbeepmeep/kernels/releases/tags/gguf-v1.0.29).

| Python / PyTorch / CUDA | Windows x86_64 SHA-256 | Linux x86_64 SHA-256 |
| --- | --- | --- |
| 3.10 / 2.7.1 / 12.8 | `42ed4fd96ee9756b89b1c27cb85836bbe1eec09501da6df3410c7db9df82cf4d` | `72da2794ff79bef5e7d2db41959382a1dfb5ad6a3797c58b5abeb5b6247b50f2` |
| 3.11 / 2.10.0 / 13.0 | `6ea255647d2b9ee6143b975af6b1f1abc6d7256622fd8e212000d46bb4ca641b` | `9509d38b737e9b825dd8d523e27a74d534eec2d30394b0c2c3fe7381011ea014` |

The Linux CUDA wheels require glibc 2.28 or newer. The release also has experimental ROCm wheels, but the Maestro updater retains its existing CUDA-only platform map.
