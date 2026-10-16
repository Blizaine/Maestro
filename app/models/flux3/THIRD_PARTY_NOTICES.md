# Third-party notices

The FLUX 3 DiT, autoencoder, sampling, and reference-preparation code in this
directory is adapted from Black Forest Labs' FLUX 3 Image Klein reference
implementation at commit
`45e57c9fb06f9751cdced29a65c5d2171003be05`. The reference repository states
that its code is released under the Apache License, Version 2.0. The applicable
license text is included as [LICENSE](LICENSE).

The adapted implementation removes the standalone checkpoint/pipeline loader,
uses Maestro's MMGP component lifecycle, exposes step and cancellation hooks,
and lets PyTorch select an available scaled-dot-product-attention backend
instead of forcing cuDNN attention. The FLUX 3 Image Klein model weights have a
separate FLUX Kommunity License and are not included here.

Qwen3-VL model code is provided by the installed Transformers package and is
not vendored in this directory. The Qwen3-VL checkpoint remains a separately
licensed upstream asset.
