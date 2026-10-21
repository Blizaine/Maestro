# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Small compatibility layer for MMGP runtime helpers.

Maestro's bundled MMGP predates the optional ``mmgp.runtime`` module. Keep the
device lookup local to MOSS so newer MMGP builds can provide their own helper,
while older builds retain the same forced-device behavior as the other local
model adapters.
"""

from __future__ import annotations

import torch


try:
    from mmgp import runtime as runtime
except ImportError:  # pragma: no cover - used with Maestro's bundled MMGP
    class _RuntimeFallback:
        @staticmethod
        def execution_device(module):
            forced = getattr(module, "_force_device", None)
            if forced is not None:
                return torch.device(forced)
            for tensor in (*module.parameters(), *module.buffers()):
                if not tensor.is_meta:
                    return tensor.device
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runtime = _RuntimeFallback()
