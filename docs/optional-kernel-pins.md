# Optional CUDA kernel pins

The optional installers pin WGPKernels 1.4.2 and llama.cpp GGUF CUDA kernels 1.0.29. They select only the Maestro CUDA stacks below, update stale or mismatched builds, and keep a manually installed newer public release when its local build matches the active stack. At verification on 2026-10-26, the GitHub API marked both release records as mutable, so the URLs carry the SHA-256 digest reported for each wheel by the release API; pip/uv verifies the bytes while installing. If a release asset is replaced, the optional install fails and the existing fallback remains available.

## WGPKernels 1.4.2

Source: [release v1.4.2](https://github.com/deepbeepmeep/WGPKernels/releases/tag/v1.4.2), [release API metadata](https://api.github.com/repos/deepbeepmeep/WGPKernels/releases/tags/v1.4.2), [v1.1.0 INT8 source](https://raw.githubusercontent.com/deepbeepmeep/WGPKernels/v1.1.0/wgpkernels/int8/__init__.py).

| Python / PyTorch / CUDA | Windows x86_64 SHA-256 | Linux x86_64 SHA-256 |
| --- | --- | --- |
| 3.10 / 2.7.1 / 12.8 | `9b9747595b3e89cec7c8bf74499bf3f36988f5dc23100797f643fcbd4e2feca7` | `6b63bd5434ddad45d45276782872f88fe8015fe9f1ac56b168ea5d48b5c7f5bb` |
| 3.11 / 2.10.0 / 13.0 | `2fa49576514987fd6725b21b6317cfae1e7856ef54fbbeb22e2cfe50e4fcf88b` | `a33354ea1326c57340172eea86dc4a5f3559ba24665f24746eca7d0ad7d9a3c9` |

The Linux CUDA wheels require glibc 2.34 or newer. The current Maestro INT8 dispatch still requires an NVIDIA SM 8.0 GPU, even though the 1.4.2 package also contains SM 7.5 INT8 code. Version 1.4.2's new FP8-weight/W8A16 API lives in `wgpkernels.fp8`; the current app path uses `wgpkernels.int8` and does not opt into that feature. The package's INT8 C ABI moved from 10 in 1.1.0 to 11 in 1.4.2; the package loader validates its paired library, and the public `linear`, `choose_path`, `prepare`, `linear_prepared`, and `selftest` calls used by Maestro remain present.

## GGUF kernels 1.0.29

Source: [release gguf-v1.0.29](https://github.com/deepbeepmeep/kernels/releases/tag/gguf-v1.0.29), [release API metadata](https://api.github.com/repos/deepbeepmeep/kernels/releases/tags/gguf-v1.0.29).

| Python / PyTorch / CUDA | Windows x86_64 SHA-256 | Linux x86_64 SHA-256 |
| --- | --- | --- |
| 3.10 / 2.7.1 / 12.8 | `42ed4fd96ee9756b89b1c27cb85836bbe1eec09501da6df3410c7db9df82cf4d` | `72da2794ff79bef5e7d2db41959382a1dfb5ad6a3797c58b5abeb5b6247b50f2` |
| 3.11 / 2.10.0 / 13.0 | `6ea255647d2b9ee6143b975af6b1f1abc6d7256622fd8e212000d46bb4ca641b` | `9509d38b737e9b825dd8d523e27a74d534eec2d30394b0c2c3fe7381011ea014` |

The Linux CUDA wheels require glibc 2.28 or newer. The release also has experimental ROCm wheels, but the Maestro updater retains its existing CUDA-only platform map.
