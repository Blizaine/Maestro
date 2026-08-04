# MiniMax H3 upstream components

The video VAE, audio VAE, and scheduler in this directory are derived from
the Hugging Face Diffusers MiniMax H3 implementation at commit
`abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc`.

Those files retain their upstream Apache-2.0 copyright and license headers.
Maestro-specific model loading, packing, memory management, and Studio
integration are implemented separately in this directory.

The default runtime stack is pinned to:

- `MiniMaxAI/MiniMax-H3` commit `5d9b308a59ab12e67147f191e184baf704185bd1`
  for the official processor and text-encoder configuration.
- `Comfy-Org/MiniMax-H3` commit `0543966fbdce5ba05709a8f2031c94bdba629b4a`
  for the scaled-FP8 FL2VA transformer, NVFP4-AWQ conditioner, and compact
  video/audio VAE checkpoints.

Those model weights are downloaded at runtime and are not distributed in the
Maestro repository. They remain governed by their respective model terms and
any authorization or waiver required for the user's location.
