# Optional Fish CPU expert backend

This directory contains the routed CPU expert kernel from WanGP, pinned at
`de302eec64b692dc83949d687cadf7d60bd74091` (the Wan2GP v18.10 source used
for this port). The source and both binaries retain their upstream copyright
notice. The applicable WanGP Community License 2.0 is copied to
`LICENSE_WanGP.txt`.

| File | SHA-256 | Size |
| --- | --- | ---: |
| `experts.cpp` | `c0ec1e233ccc2218fed704430edaf5558e446004d5eab42710ce01b4a64c8d5c` | 20,293 bytes |
| `cpu_experts_win_amd64.dll` | `139669dddb8cba3eb2ab34de8f335367a40dc666c153ff2a1daf9a0daa973d1a` | 46,592 bytes |
| `cpu_experts_linux_x86_64.so` | `3e76fbd33d129e20da96063a7a4e2310896749d8ce003c79ab60671cb79f082f` | 57,600 bytes |

The libraries use the standalone C ABI in `experts.cpp`; they do not use the
PyTorch C++ ABI. The Windows build depends on the Microsoft Visual C++ and
OpenMP runtimes (`VCRUNTIME140`, `MSVCP140`, and `VCOMP140`). Linux requires
glibc and `libgomp`. If the appropriate library, runtime, CPU instruction set,
or platform is unavailable, Fish Audio uses its portable PyTorch CPU expert
implementation instead. The native path never compiles code during import or
inference.

The copied upstream build flags are `/O2 /EHsc /LD /MD /std:c++17 /arch:AVX2
/openmp` with MSVC on Windows, and `-O3 -std=c++17 -shared -fPIC
-fvisibility=hidden -mavx2 -mfma -mf16c -fopenmp` with g++ on Linux.

`../expert_triton.py` is adapted from the same pinned revision's
`shared/kernels/expert_triton.py`. Its SHA-256 is
`1f80054cad6b31d140263b05b99047d63de0f60435f5f175d88eb0af61b9b761`
(7,355 bytes); the upstream file SHA-256 is
`493cfb0c577cdc61ee1765ac856b844b1641ac9277fe9e8080be83ea84df555e`
(7,421 bytes). The only source change removes the generic WanGP Triton
compilation-logger import/installation; four inline lint annotations mark the
upstream expert-dimension parameter `I`. Maestro's Fish-local kernel module
does not require that shared logger. The source retains the WanGP copyright
header and is covered by the copied WanGP Community License 2.0.
