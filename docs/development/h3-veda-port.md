# SM86 staged attention and experimental H3 Veda

This Dev integration adapts WanGP source at
[`af2ed29a83dbccc7f1c0571a73005648436b392e`](https://github.com/deepbeepmeep/WanGP/tree/af2ed29a83dbccc7f1c0571a73005648436b392e).
It keeps Maestro's existing models, samplers, conditioning, previews, LoRAs,
per-generation memory policy and cancellation behavior. It does not update
the running application or change saved performance settings.

## 3080 Ti memory path

SM86 SageAttention can now stage per-block INT8 Q/K and FP16 V directly into
the attention output buffer. This enables the existing Q/K/V lifetime and
head-split helpers on compatible 3080-class runtimes. The optional per-block
quantizer and Triton kernel must both be available; CPU and unsupported
devices retain their ordinary path. Existing SM89/SM120 paths are unchanged.
Sol, SLA and Veda own their own transformer attention; this change chiefly
benefits ordinary SageAttention jobs and compatible auxiliary attention.

No Auto-tune thresholds or duration recommendations are raised on the basis
of this source port. Memory savings and speed need matched hardware tests.

## Veda selection

1. Finish active tests, update Dev and restart Maestro when convenient.
2. Choose an ordinary **H3 Full** or **H3 Pruned** checkpoint in **Frames** or
   **References**. Full may need to be enabled through Manage models.
3. In Studio's H3 optimization controls or Director's H3 controls, enable
   **Veda Sparse Attention — Experimental**. Selecting Sol or SLA disables
   Veda, and selecting Veda disables those choices.
4. Keep the same prompt, seed, model, resolution, duration, steps, references,
   LoRAs and preview settings for comparisons. Review the actual motion,
   identity and audio as well as time, VRAM, RAM and failures.

Veda is off by default. Studio and Director save independent selections.
It requires BF16 activations on NVIDIA SM80+ with Triton 3.0 or newer. First
use compiles kernels and downloads the matching learned predictor. An
unsupported explicit Veda request fails clearly before loading model
weights, rather than silently producing a dense benchmark. Fused Frames and
References offer Veda experimentally for local comparisons; SLA remains their
default. Fused output quality with Veda has not been validated. VDN, audio-only,
Singularity, DaSiWa and imported specialty profiles are excluded.
Grouped Rows masked editing requires ordinary dense attention.

Veda uses learned per-layer/per-head tile selection, while preserving dense
access to non-video tokens and the predictor's reference-attention budget.
Absolute head identities survive projection splitting. Predictor modules
are registered under each attention layer so MMGP streams and unloads them
with the transformer. Geometry buffers are released after denoising,
including cancellation or failure. Normal LoRA/module forwards remain the
fallback when grouped INT8 projection is ineligible.

## Assets and provenance

Predictors come from [DeepBeepMeep/MiniMax-H3](https://huggingface.co/DeepBeepMeep/MiniMax-H3/tree/49fe56ae78d4f8a410274d1e5f8c4d2b6f6e1b33/minimax_h3),
pinned at `49fe56ae78d4f8a410274d1e5f8c4d2b6f6e1b33`.
Frames uses the T2VA predictor and References uses the R2VA predictor; Full
and Pruned share the appropriate workflow predictor. ConvRot transformers
use INT8 ConvRot predictors (about 281 MB), and other eligible transformers
use BF16 predictors (about 551 MB). Exact filenames, sizes and hashes are
recorded in `app/models/minimax_h3/veda_assets.py`.

Files are fetched only on explicit first use. Verified files in linked
model roots can be reused. Downloads go to the primary checkpoint root,
resume through a partial file and are atomically published after verification.
Invalid linked files are left untouched. No predictor or model weights are
bundled in Git.

The vendored Veda directory retains upstream notices and license texts,
including WanGP Community License 2.0, MIT and SageAttention notices.
The SM86 adaptation and grouped INT8/tuner additions are attributed to the
same pinned WanGP revision; its Community License text is included at
`app/models/minimax_h3/veda/LICENSE.txt`. Maestro's existing root engine
license file is unchanged. Predictor use remains subject to its source
model terms.

## Validation and next hardware comparisons

CPU-only tests exercise SM86 kernel call contracts and strided outputs,
Full/Pruned capability gates, attention head identity and dense parity seams,
predictor metadata/plans, reference and continuation geometry, grouped INT8
fallbacks, hash verification and download reuse. The intercepted browser
regression covers Studio/Director exclusivity, persistence, unsupported
choices and model switching without contacting a running Maestro instance.

No GPU render, kernel probe, restart, dependency install or test-machine
update was performed during this port because user tests were active.
Consequently this is an experimental Dev feature, with no measured Maestro
Veda speedup or quality claim. A useful next comparison on both machines is
Pruned Frames and References at 720p/10.1s, then a longer matched case,
comparing SageAttention, SLA and Veda serially. Check a normal LoRA and
multi-window references before broadening to Full or native 1080p. Count
first-use downloads/compilation separately from warmed generation time.
