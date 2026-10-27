# MMGP source snapshot

- Repository: `deepbeepmeep/WanGP`
- Immutable commit: `9905f0df941e5ec7b0c80e32555d3a61fe415726`
- Upstream package metadata (`mmgp/pyproject.toml`): `4.3.0`
- Runtime module (`mmgp/__init__.py`): `4.4.0`
- License: WanGP Community License 2.0; see `LICENSE.txt` and `mmgp/LICENSE.md`.
- Declared dependencies: `torch>=2.1.0`, `optimum-quanto`, `accelerate`, `safetensors`, and `psutil`.

The two upstream version declarations disagree. Keep this path named `v43` for
the staged integration slot, but identify it in runtime settings as
`Experimental Profile0 (WanGP9905)` and verify the actual runtime module version
as 4.4.0. Do not infer compatibility from the package metadata alone.

The complete Python/C++ source tree and upstream notices are copied here. The
prebuilt `allocator/vmm_alloc_*.dll` and `allocator/vmm_alloc_*.so`,
`allocator/oom_error_*.dll` and `allocator/oom_error_*.so`, and
`allocator/ram_alloc_*.dll` and `allocator/ram_alloc_*.so` files are omitted.
No native binary was loaded or executed during this staging. Profile 0 remains
blocked until the platform's VMM allocator is built or separately validated,
and activated before any CUDA context is initialized.

The upstream `mmgp/allocator/build.py` documents the build toolchain. On
Windows the VMM allocator is a C++17 x64 DLL built with Visual Studio Build
Tools; no CUDA toolkit is required, but runtime support requires an NVIDIA
driver with CUDA 11.2+ and a device supporting virtual memory management and
memory pools. Its pluggable allocator hooks require PyTorch 2.3+. The optional
PyTorch OOM companion and RAM allocator link against PyTorch's `c10` ABI; the
source only enables their checked ABI range 2.6 through 2.15. OOM formatting and
the RAM allocator are optional to VMM Profile 0; `vmm_alloc_win_amd64.dll` is
mandatory for its automatic preload behavior.

Maestro's Fish expert adapter is not API-compatible with this snapshot. It is
guarded in `app/services/memory_runtime.py`; do not expose Fish as supported on
this runtime until the expert-format, residency, CPU, GPU, release, and graph
contracts have explicit parity coverage.

Maestro additionally stages its own Windows x64 VMM build from the pinned C++
source. `mmgp/allocator/VMM_BUILD.json` records the source, compiler, linker,
flags, artifact hash and static export check. It is isolated from the legacy
allocator; it is never activated by default. A subsequent local build supplies
the optional Windows x64 OOM companion from the exact pinned `oom_error.cpp`.
`mmgp/allocator/OOM_BUILD.json` records its source, binary and compiler hashes
and CPU cross-DLL exception checks on installed PyTorch 2.7.1 and 2.10.0.
Those checks do not allocate CUDA memory or certify GPU OOM recovery. Linux
and optional RAM companion binaries remain unbundled. GPU validation is
separate from this build provenance.

Maestro's publication checks additionally remove an unreachable return from
`offload.cudacontext` and explicitly initialize the tower-sequence tracker in
`offload._detect_main_towers`. These small Python source adaptations retain
the upstream license and do not change the native build sources or hashes.
