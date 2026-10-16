# Adapted from shared/attention_kit.py at the pinned Wan2GP integration
# revision de302ee. Kept Fish-local because Maestro's shared helper does not
# expose rows_ yet; qkv attention itself continues to use the shared module.
from __future__ import annotations

from collections.abc import Callable

from torch import Tensor

_ROW_ELEMENTS = 1 << 23


def rows_(
    tensor: Tensor,
    fn: Callable[[Tensor, int, int], Tensor],
    start: int = 0,
    stop: int | None = None,
    *,
    row_elements: int = _ROW_ELEMENTS,
) -> None:
    """Apply ``fn`` to bounded token-row chunks and copy results in place.

    ``fn`` receives ``(chunk, first, last)`` where row indices are absolute
    in dimension 1. Chunking bounds the temporary used by normalization/RoPE
    for long prompts while retaining the upstream attention helper contract.
    """
    stop = tensor.shape[1] if stop is None else stop
    if start < 0 or stop < start or stop > tensor.shape[1]:
        raise ValueError("row range must satisfy 0 <= start <= stop <= tensor.shape[1]")
    if row_elements < 1:
        raise ValueError("row_elements must be positive")
    step = max(1, row_elements // max(1, tensor[:, :1].numel()))
    for first in range(start, stop, step):
        last = min(first + step, stop)
        part = tensor[:, first:last]
        part.copy_(fn(part, first, last))
