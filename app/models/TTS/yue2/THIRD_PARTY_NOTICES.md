# Third-party code notices

The Oobleck decoder and SnakeBeta implementation in `vae.py` is derived
from stable-audio-tools commit `a6ae0cdf8b2eb1567a4b42ceadddec3712d99d45`.
The decoder retains the original activation equations and module hierarchy.
Weight normalization is folded into ordinary convolution weights during repacking;
the unused encoder and training paths are omitted.

- Oobleck / stable-audio-tools: Copyright (c) 2023 Stability AI, MIT.
  Full text: `licenses/stable-audio-tools-MIT.txt`.
- SnakeBeta / BigVGAN: Copyright (c) 2022 NVIDIA CORPORATION, MIT.
  Full text: `licenses/SnakeBeta-NVIDIA-MIT.txt`.

These notices cover the identified source code and retain its original licenses.
The YuE2 model checkpoint weights are separately licensed under CC BY-NC 4.0;
see MODEL_LICENSE for the scope and full terms. This does not relicense third-party code.

The optional real-audio tokenizer and adaptation objective are adapted from
Mothersuperior's `yue2-mothersuperior-realaudio-tokenizer-v4`, including
`scripts/joint_v6.py` at `e2e63d859f3af879baf1b4d4e9f22d1eeda6fde5`.
The repository identifies its tokenizer/decoder assets as CC BY-NC 4.0. Source
URLs, pinned asset hashes and differences from the research training recipe are
documented in `MAESTRO_PORT.md`, `music_assets.py` and `docs/YuE2-music.md`.

The optional instrumental AR LoRA is from
https://huggingface.co/Mothersuperior/YuE2-instrumental-cot-full-loras
at revision `947f2f4b28978b2b6c3e316e6a87925c76bf3c4b`, licensed CC BY-NC 4.0.
Maestro downloads the original BF16 safetensors file on demand and applies its
rank-64 additive AR matrices at strength 1, with full score planning and the
stock acoustic decoder. The adapter is removed after each job.
