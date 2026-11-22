# ZONOS3 Mini notices

This package ports the pinned WanGP ZONOS3 Mini integration. Preserve the
WanGP Community License notice for its integration code in `LICENSE_WANPG.txt`.
The Zyphra reference-code license is in `LICENSE_ZONOS3`; the AntSpeaker MECT
implementation has its Apache-2.0 text in `mect/LICENSE_APACHE_2.0`.

The upstream Zyphra model card lists Apache-2.0 for model weights and MIT for
its remote code. It describes the SuperBPE tokenizer and 27 included voices as
part of the model package. Maestro downloads the converted checkpoint and
runtime assets from `DeepBeepMeep/TTS` at revision
`87b4ea98d666a4f97eaaef4bfeb4e34975d7a109`, rather than bundling them here.
The pinned BF16 checkpoint is 1,772,658,392 bytes with SHA-256
`5a6f61a9ceb0d2c07a8cb9cd6178980e8fde7151883ae579dcdb8704bc12bedd`:

- `ZONOS3_Mini_bf16.safetensors` — language model checkpoint
- `ZAC1_keep2_decoder.safetensors` — audio decoder
- `MECT_ownmect_s2.safetensors` — speaker encoder
- `voices.safetensors` — built-in voice references and embeddings
- `superbpe8192.json` — SuperBPE tokenizer vocabulary/configuration
- `steering.safetensors` — voice steering vectors

The model remains experimental. A short Hindi built-in-voice generation with
Arjun completed at 44.1 kHz, and a second generation used that WAV as a voice
reference for an English line. Lower-VRAM profiles remain unverified.
The runtime language table is retained in `zonos3_mini.json`; its 44 presets
map to 43 languages because US and UK English are separate presets. Hindi (`hi`)
is included. The supported-language claim and licensing summary match
Zyphra's public `ZONOS3-Mini-hf` model card.
