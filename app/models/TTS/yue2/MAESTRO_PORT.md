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
