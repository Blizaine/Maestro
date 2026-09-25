# Third-Party Notices

This file distinguishes the project's MIT-licensed code from third-party technology and names referenced by the integration.

## NVIDIA

NVIDIA, DLSS, NGX, GeForce and related names/trademarks are property of NVIDIA Corporation and/or its affiliates.

This repository does **not** include NVIDIA proprietary runtime binaries or NVIDIA NGX SDK headers. In particular, it does not include `_nvngx.dll` or `nvngx_dlssnr.dll`.

The native bridge declares only the minimum ABI shapes/function signatures it needs to dynamically call an installed/user-supplied NGX runtime. Those interfaces and identifiers correspond to NVIDIA technology; this project's MIT license does not grant rights in NVIDIA software, APIs, trademarks or binaries.

NVIDIA's public DLSS repository marks NGX headers as NVIDIA proprietary. Users and redistributors should review the license terms applicable to the NVIDIA software they use.

Reference: https://github.com/NVIDIA/DLSS

## Zonnery/dlss5-nr-player

The project was informed by behavior documented/exposed by the experimental `Zonnery/dlss5-nr-player` repository, including use of NGX feature id 18 and Neural Rendering parameter names.

As of preparation of v0.2.0, that repository did not expose an explicit open-source LICENSE file in its repository root. This repository therefore does not vendor its source files or claim a license to redistribute them.

Reference: https://github.com/Zonnery/dlss5-nr-player

## ComfyUI

ComfyUI and Comfy Org are separate projects. This custom node is unofficial and is not endorsed by Comfy Org.

Reference: https://github.com/Comfy-Org/ComfyUI

## NVIDIA Optical Flow SDK / NVOFA

Temporal mode dynamically calls `nvofapi64.dll`, which is installed with NVIDIA's display driver. This repository does not redistribute that DLL and does not vendor NVIDIA Optical Flow SDK headers.

The minimum common ABI concepts used by the implementation (optical-flow mode, S10.5 flow vectors, grid sizes, input/reference frame semantics) are documented by NVIDIA's public Optical Flow SDK materials.

References:

- https://github.com/NVIDIA/NVIDIAOpticalFlowSDK
- https://docs.nvidia.com/video-technologies/optical-flow-sdk/nvofa-programming-guide/index.html

## NIGos/dlss5-bridge

The D3D11 NVOF function-table layout, current/reference direction validation, S10.5-to-normalized-UV conversion and nearest-grid reconstruction used by this project's experimental temporal implementation were informed by the MIT-licensed `NIGos/dlss5-bridge` project.

Reference: https://github.com/NIGos/dlss5-bridge

MIT License

Copyright (c) 2026 NIGos

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
