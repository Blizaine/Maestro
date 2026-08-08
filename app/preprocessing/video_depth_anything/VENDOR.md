# Video Depth Anything

This directory vendors the inference implementation from:

- Project: https://github.com/DepthAnything/Video-Depth-Anything
- Commit: `4f5ae23172ba60fd7bc11ef671cca678842c7072`
- Source license: Apache License 2.0 (see `LICENSE`)

Maestro adds only Python package marker files (`__init__.py`) so imports are
unambiguous inside the application. The model weights are not redistributed.
They are downloaded from the official `depth-anything` Hugging Face
repositories on first use and retain their upstream licenses:

- Small: Apache License 2.0
- Base and Large: CC-BY-NC-4.0

