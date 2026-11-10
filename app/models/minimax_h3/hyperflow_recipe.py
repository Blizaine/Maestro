"""HyperFlow Ref2VA scheduling and the published AdaLN offset table.

Schedule and (t, r) conditioning from Video Rebirth's HyperFlow (Apache-2.0).
Uses the Ref2VA offset format in WanGP 5b63ae9 (Community License 2.0).
"""

from functools import lru_cache
from pathlib import Path

import torch
from safetensors import safe_open

from .packing import build_row_timesteps


HYPERFLOW_PRESET = "hyperflow-ref2va-8step"
_RAW_SIGMAS = (1.0, 0.931506, 0.839236, 0.703462, 0.5, 0.296538, 0.160764, 0.068494)
_OFFSET_PREFIX = "hyperflow_adaln.ref2va."


def hyperflow_sigmas(shift: float, *, tail: int = 8) -> tuple[float, ...]:
    """Return the requested evaluation levels followed by the clean endpoint."""
    if not 1 <= tail <= 8:
        raise ValueError("A HyperFlow schedule must contain 1 to 8 evaluations.")
    return (*[shift * s / (1.0 + (shift - 1.0) * s) for s in _RAW_SIGMAS[-tail:]], 0.0)


@lru_cache(maxsize=1)
def _read_offsets(path: str, mtime_ns: int, size: int):
    del mtime_ns, size
    with safe_open(path, framework="pt", device="cpu") as file:
        try:
            pairs = file.get_tensor(_OFFSET_PREFIX + "pairs").double()
            blocks = file.get_tensor(_OFFSET_PREFIX + "blocks")
            final = file.get_tensor(_OFFSET_PREFIX + "final")
        except (KeyError, RuntimeError) as exc:
            raise ValueError("HyperFlow adapter lacks its Ref2VA AdaLN offset table.") from exc
    if (
        pairs.ndim != 2 or pairs.shape[1] != 2 or not pairs.shape[0]
        or not bool(torch.isfinite(pairs).all())
        or blocks.ndim != 3 or blocks.shape[1] != pairs.shape[0]
        or final.ndim != 2 or final.shape[0] != pairs.shape[0]
    ):
        raise ValueError("HyperFlow adapter has an invalid Ref2VA offset table.")
    return pairs, blocks, final


def load_hyperflow_offsets(path: str, block_count: int, hidden_size: int):
    source = Path(path).resolve(strict=True)
    stat = source.stat()
    tables = _read_offsets(str(source), stat.st_mtime_ns, stat.st_size)
    pairs, blocks, final = tables
    if blocks.shape != (block_count, len(pairs), 18 * hidden_size) or final.shape != (len(pairs), 2 * hidden_size):
        raise ValueError("HyperFlow offsets do not match the selected Ref2VA transformer.")
    return tables


def step_conditioning(
    tables, layout, timesteps, row_indices,
    *, next_video: float, next_audio: float, condition_video: float,
):
    # Pair the existing row's t with its endpoint r. Text follows video;
    # reference and exact-drive rows remain fixed.
    endpoints, endpoint_indices = build_row_timesteps(
        layout, next_video, next_audio, condition_video, 1.0,
    )
    endpoints = endpoints.to(timesteps.device)
    endpoint_indices = endpoint_indices.to(row_indices.device)
    pair_count = len(endpoints)
    combinations, remap = torch.unique(
        row_indices * pair_count + endpoint_indices, sorted=True, return_inverse=True,
    )
    t = timesteps[combinations // pair_count]
    r = endpoints[combinations % pair_count]
    wanted = torch.stack((t, r), dim=1).to(device="cpu", dtype=torch.float64)
    pairs, blocks, final = tables
    distances, match = (wanted[:, None] - pairs[None]).abs().amax(dim=2).min(dim=1)
    if bool((distances > 0.0001).any()):
        bad = wanted[int(distances.argmax())].tolist()
        raise ValueError(f"HyperFlow has no AdaLN offsets for timestep pair {bad}.")
    return t, remap, blocks.index_select(1, match).to(t.device), final.index_select(0, match).to(t.device)
