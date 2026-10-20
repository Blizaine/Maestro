# Maestro v3.2.0

20 October 2026. Public **Dev** preview for testing. Main remains v3.1.0.

## FastH3 V2 and OmniRef

- Enable **FastH3 V2 8-Step — Text to Video (Experimental)** in the Frames model picker for
  text-to-video with generated audio. It uses its own fixed eight-step DMD
  schedule and trained VSA-H3 attention.
- Enable **FastH3 OmniRef 8-Step (Experimental)** in References for ordered image,
  video and audio conditioning. At least one image or video reference is required.
  Its eight evaluations use 32 PDD output heads and a separate sparse-attention
  pattern. This is a different model and recipe from V2 and H3 Fused 4-Step.
- The normal OmniRef model offers official BF16, INT8 ConvRot and FP8 checkpoints
  through the Transformer Quantization setting. Existing verified INT8 downloads
  are reused because the official file is byte-identical to the earlier upload.
- A separate **FastH3 OmniRef QAD NVFP4 8-Step (Experimental)** option uses the
  QAD-trained Comfy export. It requires supported NVIDIA Blackwell hardware and
  a working native NVFP4 kernel; unsupported systems receive an actionable error.
- Checkpoints use immutable source revisions, exact size/SHA-256 verification,
  format-specific geometry checks and the matching grouped-QKV layout. Retained
  FP32 output heads and AdaLN projections keep their source precision.
- Start at **480p / 5.2 seconds**. OmniRef does not support Director, continuation,
  start/end frames, masks, additional LoRAs or ordinary H3 audio refinement.
  FastH3 uses its own attention instead of SLA or Veda. Existing H3 hardware
  duration tiers are not treated as verified FastH3 limits.

The prior INT8 OmniRef 4090 image-reference smoke test completed an audio/video
render in 86 seconds of active generation. The new format regressions passed,
and real FP8/NVFP4 compression gates executed on an RTX 5070 Laptop GPU, with
NVFP4 using its native kernel. Those layer checks do not verify a complete
BF16/FP8/NVFP4 render, output fidelity or consumer-GPU throughput.

FastVideo's published near-real-time OmniRef benchmarks used **eight GB200 GPUs**.
Its 5090 and 4090 launch recipes do not establish comparable speeds in Maestro.
Maestro uses BF16/FP16 VSA and does not yet include FastVideo's full FP4-attention
and NVFP4-decoder acceleration stack. See the [FastH3 guide](FastH3.md) for sources,
formats, hardware requirements and validation scope.

## Qwen Image 2.1 downloads and VAE decoding

- Repair processor assets after upstream repository moves. Generation and prompt
  enhancement use a verified replacement revision and canonical filenames;
  existing compatible local assets are retained.
- Omit unused temporal feature caches during single-image decoding to reduce
  final VAE memory use. Low memory use during numbered transformer steps does
  not guarantee that final decoding fits the same budget.
- Address thin tile-boundary lines using full-sized tiles at image edges,
  at least 512px decoder context in Medium/Aggressive modes, overlap and FP32
  CPU blending of reliable tile interiors before one final pixel conversion.
  Increased overlap can add decode work; it does not repeat denoising.
- In a decoder-only comparison at 1088×1920, Full peak allocated VRAM fell from
  6.85GiB to 3.26GiB on the 4090. An isolated 5070 Laptop decoder check also
  completed at 3.26GiB allocated and 4.22GiB reserved. These are decoder checks,
  not proof that every resolution, reference count or complete render fits 8GB.

See the [Qwen 2.1 guide](Qwen-Image-2.1.md) for workflow and measurement details.

## Updating the Dev build

1. Let current work finish, then stop Maestro and use **Update** for your Dev
   installation in Pinokio.
2. Restart Maestro, refresh the browser and confirm **v3.2.0** beside the title.
3. Enable the desired experimental FastH3 model through **Manage models**.
   Weights download when requested; saved model visibility and user preferences
   are retained.

This push updates public Dev only. It does not publish a stable GitHub release,
move Main or change existing user duration recommendations.
