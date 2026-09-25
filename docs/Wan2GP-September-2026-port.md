# Wan2GP feature port — September 2026

The September 25 work uses Wan2GP revision
`2345ae148f82740f66e82c41292dbbdd592e713d` as its comparison baseline.
Existing Maestro integration, settings and model-specific safeguards are retained.

- [Qwen Image 2.1 controls and testing](Qwen-Image-2.1.md): masked editing,
  control-image transfer, LanPaint, outpainting, Turbo profiles, optional cache,
  and native 2K aspect presets.
- H3 automatically selects the existing INT8 ConvRot video VAE loader when the
  transformer setting is INT8. Downloads and loading resolve the same filename.
  Explicit VAE choices remain authoritative; other automatic transformer formats
  retain the FP16 video VAE. Audio VAE selection is unchanged.
- `comfy-kitchen==0.2.35` is pinned in the shared requirements used by install,
  update and runtime repair. Optional acceleration uses compatibility checks and
  falls back to the established implementation when unsupported. This port keeps
  upstream's SM120 (RTX 50 series) eligibility checks; older NVIDIA cards retain
  their established kernels. Upstream's `kernel_precision` configuration is
  honored at startup (`fast` by default; `strict` disables approximate paths).

Upstream speed figures are not Maestro benchmark results. Qwen issue #153 does
not provide a sample or full settings; wider VAE tiles/overlap and the base CFG
recipe address plausible differences, not a confirmed visual reproduction.
These changes ship in [Maestro v2.4.1](RELEASE_NOTES_V2.4.1.md).
The tiling changes alone do not establish that issue #153 is resolved.

## Source and license notices

Newly adapted Wan2GP-owned implementation in this port is from the revision
above, distributed under its [Community License 2.0](../app/LICENSES/WanGP-Community-2.0.txt).
That accompanying license is retained for the new material; this does not
replace the notices or licenses attached to the existing Maestro tree.
Qwen/Diffusers code and weights retain the notices in
`app/models/qwen21/THIRD_PARTY_NOTICES.md`. Separately installed Comfy Kitchen and
model adapters retain their own package/model licenses.

Sources: [Wan2GP](https://github.com/deepbeepmeep/Wan2GP/tree/2345ae148f82740f66e82c41292dbbdd592e713d),
[Comfy Kitchen](https://github.com/Comfy-Org/comfy-kitchen),
[Qwen Image 2.1](https://huggingface.co/Qwen/Qwen-Image-2.1),
[Viggle Turbo](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo).
