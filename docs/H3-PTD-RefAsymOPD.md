# MiniMax H3 PTD RefAsymOPD

Maestro includes separate Full and Pruned Ref2VA PTD RefAsymOPD profiles. The
recipe uses eight independent step adapters plus the pinned PDD output-head
bank. Every denoising step activates only its matching hop adapter; a generic
single LoRA cannot reproduce the recipe.

## Fixed recipe

- Eight Euler evaluations, one guidance phase, guidance scale 1 (CFG 1).
- Video/audio flow shifts 12/3 and full-strength denoising.
- Ref2VA reference encoding uses a fixed 768px short edge for every window.
- Skip-step cache, a second pass, partial-schedule denoising, Audio Refinement,
  face refinement, and other Turbo/PDD accelerators are incompatible.
- Published examples focus on image references. Quality with video/audio
  references and other Ref2VA inputs remains unvalidated.

The default profile starts at 864×480 and 124 frames (about 5.2 seconds), with
the H3 transformer working budget set to 10 GiB. The Pruned profile is the
lower-memory choice; the Full profile uses the larger checkpoint. The PDD
head bank is CPU-resident between evaluations and its active heads follow the
same eight-step Euler schedule. This budget is MMGP's working-memory reserve,
not a guarantee that every GPU can run every resolution and reference set.

For a bounded low-resolution smoke, use a 512×864 output canvas and the default
124-frame window. PTD still upscales each reference to its fixed 768px short
edge, so reducing output size does not reduce reference-encoder input size.
The model limits do not change for the smoke run.

## Pinned assets

- The PTD adapter set and PDD head bank come from
  `DeepBeepMeep/MiniMax-H3` revision
  `de1028e84b0fd5e7f15b2684b704d8c7289b02be`.
- The H3 base checkpoint URLs point to
  `DeepBeepMeep/MiniMax-H3` revision
  `fec7846aef352e58a1cfb699455e3d104281e68b` (the Pruned profile also lists
  the pinned Comfy-Org FP8 asset).
- Each profile stores all eight adapter URLs and the one-hot hop schedule.
  The PDD head bank is queried separately by the H3 handler and is never
  confused with a ninth LoRA.

The profile keeps the upstream MiniMax H3 model license. PTD artifacts remain
attributed to their source repository and are fetched from its immutable
revision; users should retain the source repository's license and notice files
with those artifacts.
