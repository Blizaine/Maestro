# Wan2GP 12.71 feature port into Maestro 2.0.1

Source reviewed: [Wan2GP commit 1e1dd2757f24923f008593d9d4ec09062234be20](https://github.com/deepbeepmeep/Wan2GP/tree/1e1dd2757f24923f008593d9d4ec09062234be20), retrieved 5 September 2026. The release announcement is [Wan2GP v12.71](https://github.com/deepbeepmeep/Wan2GP/blob/1e1dd2757f24923f008593d9d4ec09062234be20/README.md). Implementation is local in Maestro's `dev` working copy.

## Integration approach

Maestro has its own H3 model loader, reference compiler, dialogue planner, queue, React UI, and Editor. Upstream now splits H3 into a different pipeline. Port the requested algorithms and contracts into Maestro's existing paths, preserving saved settings and the current character/dialogue behavior. Do not replace Maestro's backend wholesale or merge upstream's application UI.

| Feature | Maestro integration | Required checks |
| --- | --- | --- |
| H3 VDN | Add the trained hybrid attention branch, checkpoint-module loading and dedicated FL2VA presets; retain ordinary H3 attention for existing models. | Tensor layouts and checkpoint key mapping, module/LoRA downloads, Triton capability, reference/control compatibility, small GPU execution. |
| DLSS 5 Neural Rendering | Add the native worker adapter to generation postprocessing and Tools; expose native-resolution refinement and supported enlargement factors, intensity, depth precision and motion estimator. | Windows/GPU/runtime capability, worker errors and cancellation, still-image/video handling, output geometry and audio retention. |
| Temporal upsampling | Extend RIFE v4.26 to direct x3 timesteps; add DLSS Frame Generation with runtime-reported factors. | Exact intermediate timesteps, clip joins, output frame count/FPS, unsupported factors, cancellation and audio synchronization. |
| Media Flow | Provide batch processing of uploaded/existing media through the same Maestro queue and postprocessing services; reuse these services for H3 outpainting. | Per-item progress/results/errors, cancellation, source preservation, queue serialization and repeatable settings. |
| H3 Voice Audio | Add an audio-only H3 Reference preset using a hidden 32x32 latent video; expose one/two voice references and up to 45s segments / 300s assembled audio in Studio Audio. | Correct checkpoint reuse, speaker/reference bindings, no video decode, Whisper boundary alignment, complete-script preservation, cancellation, 32 kHz stereo output and actual duration accounting. |
| H3 outpainting | Add H3 to the existing Outpaint workflow, using grouped-row masking and latent-aligned canvas margins. | Protect the original region, margin geometry, temporal continuity, source audio, invalid control inputs and batch operation. |
| H3 audio refinement | Add an optional six-step soundtrack pass at 0.5 denoising strength; freeze original video latents and disable LoRAs/references/control reinjection for that pass. | Video equality, adapter restoration on success/error/cancel, exact six steps, FL2VA soundtrack and PDD exclusions, output audio length. |

## Implementation order

1. Preserve the current working changes and record upstream provenance.
2. Implement shared postprocessing adapters, RIFE x3, DLSS capability reporting, and tool/queue integration.
3. Port H3 VDN and audio-only generation into the existing model classes.
4. Port grouped masking/outpainting and audio refinement, then connect their controls and settings.
5. Connect batch Media Flow to the same tested services and generation queue.
6. Run focused unit/integration checks, build the React UI, run feasible local media/GPU smoke tests, and reload through Pinokio after the queue is idle.
7. Record tested behavior, remaining runtime requirements, and a user test checklist in this document.

## Runtime and provenance

The local test machine has an RTX 4090 with 24 GB VRAM and NVIDIA driver 591.86. DLSS Frame Generation x5/x6 must remain unavailable on this card. VDN's speed improvement is an upstream claim; actual performance must be measured separately from functional correctness.

DLSS is an optional Windows native runtime, described in [upstream's installation guide](https://github.com/deepbeepmeep/Wan2GP/blob/1e1dd2757f24923f008593d9d4ec09062234be20/docs/DLSS5.md). Worker/runtime binaries and model weights must remain outside tracked source. Preserve upstream notices and the applicable license for imported code. Do not silently replace existing native runtime files.

## Validation and completion record

All seven requested integrations are implemented locally. Native DLSS execution remains unverified because this PC runs Windows 10 and has no DLSS worker bundle. The installer is provided separately and has not been executed. The other model and media paths have run through Maestro's live queue on the RTX 4090.

| Check | Observed result |
| --- | --- |
| Focused regressions | 576 tests passed across H3, dialogue planning, settings round trips, SCAIL-2, image workflows, job lifecycle and media finishing. Includes shipped Studio presets, adapter download routing, VDN tensor/adapter conversion, Triton convolution, masking permutations, refinement invariants, native-worker transport/cancellation and long H3 audio planning/assembly. |
| Frontend | TypeScript and Vite production build passed. Existing bundle-size/dynamic-import warnings remain. Browser checks confirmed v2.0.1, Voice Audio selection and duration presets/custom values, Media Flow, RIFE x3 and unavailable DLSS controls. |
| H3 VDN | Pruned and Full checkpoints generated playable 124-frame videos with the trained INT8 module, default adapter and dedicated eight-step adapter. Triton temporal convolution and SageAttention window kernels were active. A generated Pruned frame was visually inspected. |
| H3 Voice Audio | Initial text-only and two-reference smoke jobs produced 5.000 seconds of 32 kHz stereo PCM WAV. The duration extension generated exactly 45.000 seconds in one segment and 300.000 seconds in seven segments. A four-turn dialogue run produced 29.661 seconds with returning-speaker reference reuse and conservative Whisper trimming. Cancellation published no partial audio; the API rejected 301 seconds. See [the audio guide](../H3-Voice-Audio.md) for behavior and transcript limitations. |
| H3 Audio Refinement | Same-seed baseline and refined runs produced identical decoded video SHA-256 hashes, while their decoded audio hashes differed. The extra pass used six steps. |
| RIFE x3 batch | 123 source frames at 24 fps became 369 frames at 72 fps, retaining the 5.125-second video duration. Source/output compressed-audio SHA-256 hashes matched after stream-copy muxing. |
| Generation postprocessing | A newly generated 124-frame clip became 372 frames at 72 fps through generation's assembled-file postprocessing; the saved settings retained `rife3`. |
| Batch cancellation | A three-item batch completed two jobs and left the cancelled middle job with no output. Jobs used the same generation slot as Studio. Sources were retained. |
| Container handling | The shared finishing service also produced a 123-frame, 192x144 WebM with VP9/Opus output. |
| H3 Outpaint | Final two-window run retained the exact 896x448 canvas and all 230 source frames at 24 fps (9.583 seconds), with source audio restored. The protected 576x448 region was compared at frames 0, 100, 123, 124, 200 and 229, including the join and final frame; mean absolute pixel error was 4.43–5.66/255 after lossy encoding and Lanczos resizing. |
| Native DLSS | Capability guards, transport and cancellation were tested with synthetic workers. Proprietary Neural Rendering and Frame Generation binaries were not executed. x5/x6 stays unavailable on the RTX 4090. |

The initial small H3 smoke tests used four base steps to exercise integration, the extended Voice Audio tests used 20 steps per segment, and the VDN tests used its eight-step recipe. These are functional checks, not a visual-quality evaluation or a speed benchmark. Wan2GP's “at least 20% faster” VDN claim has not been independently measured here. Long 720p/1080p VDN runs and DLSS visual stability still need representative footage and supported runtime testing.

Validation outputs are in the **Wan-Port-Validation** workspace (`app/outputs/Wan-Port-Validation`). Existing dialogue work, Blaine prompts and GPU/window preferences were preserved. VERSION remains **2.0.1**; no release, commit or push was made.

The final verified Outpaint output is `2026-09-05-14h01m11s_seed607_Colorful geometric shapes extend naturally into th.mp4`. Earlier smoke-test iterations and cumulative window previews also remain in that validation workspace. For the audio-refinement comparison use the two seed603 clips; for VDN use seed604 (Pruned) and seed606 (Full); seed608 exercises generation plus RIFE x3.

## Where to test in Maestro

1. **VDN:** Studio → Video → Frames → model selector → **H3 VDN First / Last — Pruned** or **H3 VDN First / Last — Full**. In Advanced → Presets choose **VDN Turbo 8 Steps** for the accelerated adapter/sampling recipe. Start at 480p and one short window. The trained module and both adapters are already downloaded locally; the checkpoint loader verifies required components. VDN has its own attention path and does not combine with Sol/SLA or ordinary H3 Turbo controls.
2. **Temporal finishing:** In Advanced → Post Processing choose **RIFE 4.26 ×3**, or use Studio → Video → Upscale → **Original size (temporal only)**. The source duration and soundtrack should remain unchanged while the frame rate triples.
3. **Media Flow:** Expand **Media Flow — batch processing** in Upscale. Add videos or images, set finishing options, and queue the files. Monitor/cancel individual jobs in the normal queue. Outputs are new files with a `_media_flow` suffix and recoverable processing settings.
4. **H3 Outpaint:** Studio → Video → Outpaint; select ordinary **H3 First / Last — Pruned** or **Full**. Upload a video and expand the canvas. Margins snap to H3's 32-pixel latent grid. The original central region is restored in the final encode, and Preserve Source Audio defaults on. Expand **Media Flow — batch Outpaint** to apply the same proportional margins and prompt to multiple full videos. VDN, Reference, fused Turbo and image models are excluded from this H3 workflow.
5. **Voice Audio:** Studio → Audio → Speech → **H3 Voice Audio — Pruned**. Choose a maximum duration of 5–300 seconds (default 15). Individual generations stay within 45 seconds; longer scripts split automatically and assemble into one file. Advanced exposes inference steps. Plain text is spoken dialogue. For two voices, add two audio references and use `Speaker 1: ...` / `Speaker 2: ...` blocks; `[English, calm]` adds unspoken directions. For non-speech audio, start with `Sound:`. Whisper trims surplus speech at script boundaries; actual speech output may be shorter than the selected ceiling. See [the audio guide](../H3-Voice-Audio.md).
6. **Audio refinement:** On ordinary H3 First/Last or Reference generation, enable **Audio Refinement Extra Phase** in Advanced. It freezes video latents, partially re-noises audio at 0.5 and takes six extra steps without LoRAs or re-injecting reference/control media. The control is unavailable for fixed PDD/fused recipes and when a source soundtrack controls FL2VA. It can refine audio with supported VDN recipes as well.
7. **DLSS:** Follow [the optional installation guide](../DLSS5.md) on a supported Windows 11/RTX machine, then refresh availability. Neural Rendering is offered at x1, x1.5, x1.724, x2 and x3 with intensity 0–2; temporal factors come from the worker's hardware probe. Start with x1 and one short clip before a collection. No OS, driver or HAGS changes were made on this PC.

## API entry points

All paths use Maestro's existing local server and queue. API source paths refer to uploaded media or files accessible to the selected workspace.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/v1/media-flow/capabilities?refresh=true` | Re-probe native DLSS support and available factors. |
| `POST /api/v1/media-flow` | Queue image/video finishing, returning `batch_id` and `job_ids`. |
| `POST /api/v1/media-flow/outpaint` | Queue per-video Outpaint jobs, returning job IDs and per-file preparation errors. |
| `GET /api/v1/status/{job_id}` | Progress, completion, output filenames and errors. |
| `POST /api/v1/cancel/{job_id}` | Cancel a queued or running job. |

Example finishing request (upload or substitute a source path first):

```json
{
  "files": ["C:/path/to/source.mp4"],
  "workspace": "default",
  "spatial_upsampling": "",
  "temporal_upsampling": "rife3"
}
```

For Neural Rendering use `spatial_upsampling: "dlss5*1"` and optional `dlss_intensity: 1`, `dlss_depth: "half"`, `dlss_motion: "original"` (or `"raft"`). Temporal DLSS uses values such as `"dlssg*3"`. The server rejects unsupported combinations before queuing a finishing batch.

For Outpaint supply `files`, `model_type: "minimax_h3"` (or `"minimax_h3_full"`), `margins_percent: [0, 0, 25, 25]` in top/bottom/left/right order, `prompt`, `resolution_preset`, `preserve_source_audio`, and optional generation/window settings. Margins are calculated separately for each source.

## Source changes and runtime footprint

- The implementation stays in `app/` and `ui/`; Pinokio launchers and their URL-capture pattern are unchanged.
- H3 additions live beside the existing transformer and pipeline: `vdn_attention.py`, `lora_vdn.py`, `masked_edit.py`, `voice_audio.py` and `audio_refinement.py`. Existing Diffusers checkpoints, text encoders, LoRA conversion and Maestro prompt planning remain in use.
- Shared finishing lives in `services/media_processing.py`, `services/media_flow.py` and `postprocessing/dlss5/runtime.py`. Native workers process one continuous clip with persistent temporal history and estimated guide data. Batch processing streams frames instead of loading an entire collection into VRAM.
- New checkpoints: the VDN INT8 module is approximately 2.17 GB, its default adapter 334 MB and its Turbo adapter 851 MB. Their downloaded SHA-256 values matched the pinned Hugging Face file metadata. Existing local H3 base/conditioner/VAE files were reused.
- Upstream-derived code is covered by the preserved WanGP Community License and third-party notices. Optional native components are separate, checksum-pinned downloads with upstream's explicit installation disclosure.
