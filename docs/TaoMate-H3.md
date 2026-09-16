# TaoMate H3 three-step preset

TaoMate is an optional acceleration adapter for the H3 **Frames / First–Last**
workflow. Select the compatible standard H3 model, enable Turbo, and choose
**TaoMate FL2VA — 3-step**. Maestro downloads the pinned adapter on first use and
applies three Euler steps, video/audio CFG 1, and adapter strength 1. Existing
fused four-step and PDD options keep their recipes.

The integrated adapter is the compressed BF16 rank-19 conversion distributed by
[DeepBeepMeep](https://huggingface.co/DeepBeepMeep/MiniMax-H3), derived from
[TaoLive's TaoMate-H3](https://huggingface.co/TaoLiveAIGC/TaoMate-H3). Its exact
revision and SHA-256 are recorded in `app/models/minimax_h3/turbo_presets.json`.
The linked full-rank and other conversions are variants of the adapter; they are
not additional required downloads.

The adapter includes transformer and internal token-refiner weights. These do
not create a separate second video-refinement pass. Maestro's existing face
refinement is a separate workflow, described in [the face-refiner guide](H3-Face-Refiner.md).
This preset does not enable TaoLive's multi-GPU streaming architecture.

Local generation checks cover text-only and start-image clips on the pruned INT8
ConvRot Frames checkpoint, at 864×480, 124 frames and three steps on an RTX 4090.
The adapter's 208 target pairs, including eight token-refiner pairs, match the
loaded architecture. References, Viggle, VDN and stacking another acceleration
adapter are not offered by this recipe. Quality on long sequences and other
checkpoint variants needs separate comparison; the preset remains experimental.
