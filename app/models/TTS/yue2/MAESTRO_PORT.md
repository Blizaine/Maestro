# YuE2 runtime provenance

Ported from Wan2GP commit `5c40db6500cc8a142a15ed93720174955587babe`
(September 14, 2026), `models/TTS/yue2` and `shared/llm_engines/nanovllm`.
The decoder engine is namespaced under `_engine` so its cache and sampling
changes do not alter Maestro's other models. Unused Qwen3.5 speculative code
is not included. Original YuE2 and third-party license notices are retained.

Maestro changes: pinned downloads, optional score-model acquisition, handler
signatures, cancellation/progress compatible with MMGP 3.7.12, audio/score
provenance, and release of retained acoustic latents after each song.

Optimized checkpoints: DeepBeepMeep/TTS revision
`864a479cbf3e810e1b2c1993b438510750e383b2`. Weights use CC BY-NC 4.0;
see MODEL_LICENSE. This is separate from the code licenses.

Integration and hardware validation are tracked in the v2.2.0 work log.

Optional training additions follow Mothersuperior real-audio v4 revision
`f2278a2e005dc4ecc421c53a0929f62b3aeb2280`: AR lyric-cursor supervision and
fixed-tokenizer NAR flow adaptation. The audio encoder is ported from
`m-a-p/YuE2-Vae` revision `9a94e1d0ea9f8087e98f77fa88df4a4068104d2a`.
Existing Oobleck/Snake MIT notices in `THIRD_PARTY_NOTICES.md` apply.
`acoustic_regularizer.json` pins a small disjoint adaptation training/control
subset of Mothersuperior's minted corpus. The inference-time model architecture
and style ZIP layout remain unchanged. Cursor heads and optimizer state stay in
private resume files; personal NAR exports include matching AR conditioning.

The optional v9 comparison pins Mothersuperior revision
`430084f7c8eeeb5fc6947ea31b3f2f25f7602def`, using
`tokenizer_head_joint_v9.safetensors` with `nar_lora_joint_v9.safetensors`.
Their sizes and SHA-256 hashes are recorded in `music_assets.py`. Existing v4
projects keep their original token dialect, cache identity and resume contract.

Combined-adapter compatibility was checked against AI-Toolkit's YuE2 source at
`cc98c3b690421ebc5649972a1d2b46027df7ca64`. Version-2 style manifests distinguish
joint AR/NAR LoRAs from separate AR styles with a fixed NAR/I/O companion;
version-1 bundles remain supported. Fused QKV/MLP deltas are converted without
merging base weights. External tokenizer provenance stays unknown when absent.
The experimental native joint trainer uses detached AR conditioning, token CE
and latent flow losses, with shared A matrices for the fused projection groups.
It does not implement the v9 author's decoded-audio loss or assert reproduction
of an external checkpoint's undocumented training configuration.

Joint refinement can initialize from a verified saved style with the same token
dialect and trigger. Such runs retain independent A matrices, preserve frozen
decoder I/O when present, and export the source adapter mode's strength semantics.
The resume contract records source hashes and ranks; a step-zero pair is saved
before optimization. Base-initialized joint runs keep their original shared-A
parameterization and resume contracts.

Author head/NAR adaptation is a separate workflow ported from Mothersuperior's
`scripts/joint_v6.py`, revision `e2e63d859f3af879baf1b4d4e9f22d1eeda6fde5`.
It includes straight-through head gradients through frozen AR, full decoder I/O,
soft-neighbor minted anchors and differentiable decoded-waveform supervision.
`pair_regularizer.json` pins complete audio for the same bounded 32/4 anchor
split used by acoustic experiments. See docs/YuE2-music.md for differences from
the historical research run, including initialization, windows and scheduling.
Adapted head/NAR pairs are immutable and require fresh downstream AR tokens;
existing static-pair and legacy joint workflows retain their original semantics.

Instrumental generation uses Mothersuperior's `YuE2-instrumental-cot-full-loras`
at revision `947f2f4b28978b2b6c3e316e6a87925c76bf3c4b`. Its original named BF16
AR adapter is mapped to the same unfused targets as music-training adapters,
then applied as additive forward hooks at strength 1. CUDA graphs are released
before applying and removing hooks. Full score planning and a clean section
caption follow the author's recommended path; the stock NAR is an explicitly
supported decoder option. No tokenizer head is required for text generation.
Vocal artist bundles are preserved in settings but not activated in this mode.
