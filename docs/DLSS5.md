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

The reviewed local development PC runs Windows 10 build 19045 with an RTX
4090 and driver 591.86. The native DLSS path has therefore not been executed
on that machine. Maestro reports it as unavailable; RIFE remains usable.

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

Source: [Wan2GP DLSS5 overview and installation guide](https://github.com/deepbeepmeep/Wan2GP/blob/1e1dd2757f24923f008593d9d4ec09062234be20/docs/DLSS5.md).
Licenses and provenance: [Third-party notices](../THIRD_PARTY_NOTICES.md).
