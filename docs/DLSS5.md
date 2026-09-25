# Optional DLSS finishing in Maestro

Maestro adapts Wan2GP v12.71's native DLSS worker interfaces for Studio
postprocessing, Tools → Upscale, and the batch Media Flow panel.

Neural Rendering supports native-resolution x1 refinement and x1.5, x1.724,
x2 and x3 enlargement, intensity from 0 to 2, depth precision and motion
estimation controls. Estimated depth and motion come from the recorded media;
lighting, material changes, fine detail and temporal stability are content-dependent.

DLSS Frame Generation supports x2–x4 on supported RTX 40/50 hardware and
x5/x6 where supported on RTX 50. Maestro exposes only factors reported by
the installed worker. RIFE 4.26 x2/x3/x4 uses the Python GPU runtime and does
not need these native binaries. Frame interpolation preserves clip duration
and retains the original soundtrack in the finished file.

## Requirements

- Windows 11, an up-to-date compatible NVIDIA driver, and DirectX 12.
- RTX 30 or newer for this Neural Rendering integration; RTX 40 or newer for
  Frame Generation. HAGS must be enabled for Frame Generation.
- Native components installed in `app/dlss5/`, separate from Python packages.
  Triton alone does not supply DLSS. First use downloads missing depth/flow
  model weights through Maestro's existing model paths.

An opt-in **experimental Windows 10** backend is also available, described
below. Installing it does not change the Windows 11 backend or enable DLSS
Frame Generation on Windows 10.

## Experimental Windows 10 enhancement and upscaling

This backend targets Windows 10 22H2 (build 19045), an RTX 30 or newer GPU,
and a compatible NVIDIA driver. It uses the pinned ComfyUI-DLSS5-NR v0.3.1
native bridge in a separate process; ComfyUI itself is not required.
The local compatibility machine is an RTX 4090 running driver 591.86.

- **DLSS 5 1x:** neural enhancement at the source resolution.
- **DLSS 5 1.5x, 1.724x, 2x and 3x:** actual DLSS Super Resolution in a separate, unhooked worker,
  followed by neural enhancement at the enlarged resolution. It does not
  substitute Lanczos resizing for DLSS.
- Video enhancement uses driver-provided NVIDIA optical flow (NVOFA), with
  history reset at scene cuts. An unavailable temporal backend fails clearly.
- Depth and DIS/RAFT options apply to the upscaling stage. Same-size direct
  enhancement does not download or load depth/RAFT models.
- Frame Generation remains Windows 11 only. Use RIFE for interpolation on
  Windows 10.

Output dimensions are rounded to even pixels. The pinned Windows 10 SR worker
needs aligned output rows on the tested runtime. Maestro extends the source's
right edge and its guides internally when needed, then removes only that added
border before neural enhancement. The original picture is not resized to meet
this constraint. If internal padding would exceed the native 7680x4320 size
limit, Maestro rejects the request with a size error.

From the project folder, explicitly install using Maestro's Python environment:

```powershell
.\app\env\Scripts\python.exe .\app\scripts\install_dlss5_direct.py
```

Use `app\env-sol\Scripts\python.exe` for the RTX 40 Sol runtime, or
`app\env-rtx50\Scripts\python.exe` for the RTX 50 runtime, when that is your
active environment. Use the same interpreter for `--disable` below.
Review the installer disclosure and type `I ACCEPT`, then restart
Maestro. In **Tools → Upscale**, choose an image or video and select the
**Win10 experimental** DLSS option. The same choices appear in Studio finishing
and Media Flow. No app install/update silently enables this backend.

To disable it without deleting files:

```powershell
.\app\env\Scripts\python.exe .\app\scripts\install_dlss5_direct.py --disable
```

Restart Maestro after changing installation/enablement. The pinned files live
under `app/dlss5/direct/`, separate from the existing Windows 11 runtime. The
installer verifies both archive and extracted-binary SHA-256 values and refuses
to overwrite differing binaries. For offline installation, repeated `--cache-dir`
arguments can point to folders holding the original verified ZIP archives.

**Known native limitation:** this bridge can hang during shutdown even after
returning valid pixels. Maestro contains it in a disposable worker with bounded
shutdown and cancellation. Complete frame and finish receipts are required before
forced cleanup can count as a successful render. A timeout during initialization
or rendering remains an error. Maestro's main process never loads the bridge.
This is community compatibility work, not official NVIDIA Windows 10 support.

Local validation on Windows 10 22H2 / RTX 4090 / driver 591.86 passed through
Maestro's normal queue: a 1216x704 still image, and a 36-frame, 24 fps video
at both 608x352 (1x) and 1216x704 (2x). Both videos retained all frames, their
1.5-second duration and byte-identical audio packets. Cancelling a second 2x
run after two frames stopped both native workers and removed the partial output.
Additional scale validation passed both still and four-frame native tests at
1.5x, 1.724x and 3x. Normal queued upscales of the same 608x352 source produced
912x528, 1048x606 and 1824x1056 videos respectively, retaining all 36 frames,
24 fps, the 1.5-second duration and byte-identical audio packets. Regression
tests cover edge padding, guide alignment, output cropping, native size limits,
worker cleanup and the unchanged 1x/2x paths.
These short compatibility tests do not establish long-video quality or support
on every GPU/driver; start with a short clip when testing another system.

Offline video finishing is not a real-time game integration. Games supply GPU
color, depth and motion buffers directly; Maestro decodes recorded frames,
estimates guides and transfers frames through separate SR/NR workers before
encoding. A performance correction removed redundant per-pixel Python scans
and full-frame copies while retaining vectorized output validation. In a
four-frame 1280x704 → 2560x1408 test with RAFT/half-depth, frame processing fell
from 15.58 to 2.77 seconds with identical output pixels. Warmed processing was
about 0.49 seconds/frame on the above machine, excluding initialization,
shutdown and final encoding. This remains below real-time video throughput.
The OpenCV DIS motion and quarter-depth options can reduce guide work at a
quality tradeoff; they do not remove frame-transfer or encoding costs.

## Install on a supported Windows machine

Close Maestro, then run in PowerShell from the Maestro project folder:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\app\scripts\install_dlss5.ps1
```

The installer displays the upstream third-party component disclosure and
requires typing `I ACCEPT`. The Neural Rendering bundle uses RenoDX/ReShade
and community-modified, unsigned NVIDIA-derived DLSSNR binaries outside
the official SDK. Read that disclosure before accepting. No native files
are installed automatically by normal Maestro installation or update.

Downloads and extracted binaries are checked against pinned SHA-256 values.
ReShade's installer is extracted, not executed. Differing existing files stop
installation; the optional `-Force` flag backs them up before replacement.
Do not use that flag without reviewing the conflict list. Restart Maestro,
then use **Refresh availability** in the finishing controls.

Runtime layout:

```text
app/dlss5/host/nr-depth-worker.exe
app/dlss5/host/dxgi.dll
app/dlss5/host/renodx-dlss5.addon64
app/dlss5/host/nvngx_dlssnr.dll
app/dlss5/dlss/nvngx_dlss.dll
app/dlss5/dlssg/dlssg-worker.exe
app/dlss5/dlssg/nvngx_dlssg.dll
```

## Use and troubleshoot

Start with one short clip and x1 Neural Rendering at intensity 1. Compare
faces and motion against the source before processing a collection.
For motion smoothing, select RIFE x3 or an available DLSS factor in
**Temporal upsampling**. In Media Flow, add files, choose finishing options,
and queue the batch. Each file has its own progress, result and cancellation.
Sources remain intact; the outputs receive a `_media_flow` suffix.

Missing files, failed native capability probes, unsupported hardware, HAGS,
worker errors and cancellation surface through the ordinary Maestro queue.
Availability can also be inspected at `GET /api/v1/media-flow/capabilities`.
The installer does not change your GPU driver, OS or HAGS setting.

Sources: [Wan2GP DLSS5 overview and installation guide](https://github.com/deepbeepmeep/Wan2GP/blob/1e1dd2757f24923f008593d9d4ec09062234be20/docs/DLSS5.md),
[ComfyUI-DLSS5-NR v0.3.1](https://github.com/lisitskyaa/ComfyUI-DLSS5-NR/tree/41dcdfa593cb61b6a98c65bb8ed27606260bb598).
Licenses and provenance: [Third-party notices](../THIRD_PARTY_NOTICES.md).
