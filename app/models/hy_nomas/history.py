# Adapted from Hy-Nomas (https://github.com/Tencent-Hunyuan/Hy-Nomas, hymm/models/dit/modules/uniform_history_cache.py), Copyright (C)
# 2026 Tencent, Apache License 2.0. Modified for WanGP.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""History of a Hy-Nomas world: normalized and rotated keys and values from finished segments are archived in CPU RAM as Q8 tensors
(int8 per token/head and 32 channels, FP16 scales). The next segment sees at most HISTORY_FRAMES latent frames selected by the source's
temporal_context sampling (power-spaced back from the newest frame, always including the first frame) plus each selected segment's visible
text. The selected working set is gathered from the archive by block; the local MMGP 4.x adapter transfers only the current block to the
active device for attention. A history write copies one block's keys and values into CPU RAM after its forward has finished.

The archive is bounded (ARCHIVE_BYTES, or twice the frames seen, long explorations): beyond it, it keeps the frames a temporal_context
sampling of ARCHIVE_THINNING of the bound would choose (the newest ones, the first one, and power-spaced ones between), and a selected
frame that was dropped is replaced by the nearest kept one."""
import itertools
from dataclasses import dataclass, field

import numpy as np
import torch
from .attention import Q8_GROUP_SIZE, quantize_heads

HISTORY_FRAMES, HISTORY_GAMMA = 18, 2.0  # past latent frames a segment sees (the release's single GPU preset)
ARCHIVE_BYTES = 6 << 30  # RAM of the archive (about 66 latent frames, 11 s, at 832x480)
ARCHIVE_THINNING = 0.75  # an archive over its bound keeps this share of it
MEMORY_CHOICES = (9, 18, 36, 72, 144)  # history frames of the explorer's Memory menu
_idents = itertools.count()
GROUP = Q8_GROUP_SIZE


def frame_bytes(tokens, blocks, heads, head_dim):
    """Bytes of `tokens` history tokens of every block (q8 keys and values with their FP16 scales)."""
    return tokens * blocks * heads * 2 * (head_dim + head_dim // GROUP * 2)


def memory_choices(tokens_per_frame, blocks, heads, head_dim, frame_seconds, preload):
    """(frames, seconds, needs) of the explorer's Memory menu: how much video the frames seen amount to, the VRAM the history adds (the
    working set of the blocks the state ring holds on the GPU: MMGP's prefetch window + 1, 3 blocks when the memory profile preloads blocks
    (preload), 2 otherwise; and one block's keys and values dequantized to BF16 for its attention; with preload, up to the whole working set
    kept in VRAM where the preload leaves room) and its RAM (the working set, and the bound of the archive: all the frames that can come back
    into the sample, at least twice those seen), for a world of tokens_per_frame image tokens per latent frame (both stored as int8 with a
    scale per 32 values; the text of the segments adds a little)."""
    archived = frame_bytes(tokens_per_frame, blocks, heads, head_dim)
    attention = tokens_per_frame * heads * head_dim * 4
    streamed, kept = archived * (3 if preload else 2) / blocks + attention, archived + attention

    def needs(frames):
        vram = f"{frames * streamed / 2**30:.1f}" + (f"-{frames * kept / 2**30:.1f}" if preload else "")
        return f"{vram} GB VRAM · up to {(frames * archived + max(2 * frames * archived, ARCHIVE_BYTES)) / 2**30:.0f} GB RAM"
    return [(frames, f"{frames * frame_seconds:g}", needs(frames)) for frames in MEMORY_CHOICES]


def temporal_context_frames(total, past, gamma=HISTORY_GAMMA):
    """Indices of the history frames to attend to, oldest first: distances d_i = i + rint((total - count) * (i / (count - 1)) ** gamma)
    from the newest frame, count = min(past, total)."""
    count = min(total, past)
    if count == total:
        return list(range(total))
    if count == 1:
        return [total - 1]
    slack = total - count
    offsets = [0, *(min(slack, max(0, int(value))) for value in np.rint(slack * (np.arange(1, count - 1) / (count - 1)) ** gamma)), slack]
    return [total - 1 - (index + offset) for index, offset in reversed(list(enumerate(offsets)))]


@dataclass
class _Segment:
    frame_ids: list  # the latent frames of the clip kept for this segment, in order
    per_frame: int  # image tokens per latent frame
    text_tokens: int  # visible text tokens, after the image tokens of the kept frames
    keys: torch.Tensor  # (blocks, tokens, heads, head_dim) int8, CPU
    key_scales: torch.Tensor  # (blocks, tokens, heads, head_dim // 32) FP16, CPU
    values: torch.Tensor
    value_scales: torch.Tensor
    ident: int = field(default_factory=lambda: next(_idents))

    @property
    def image_tokens(self):
        return len(self.frame_ids) * self.per_frame

    def frame_rows(self, frame):
        start = self.frame_ids.index(frame) * self.per_frame
        return slice(start, start + self.per_frame)

    def text_rows(self):
        return slice(self.image_tokens, self.image_tokens + self.text_tokens)

    def tensors(self):
        return self.keys, self.key_scales, self.values, self.value_scales


class WorldHistory:
    def __init__(self, modules, heads, head_dim, device, past_frames=HISTORY_FRAMES):
        """modules: the DiT's blocks (a ModuleList called one by one), whose forwards fetch their part of the working set."""
        self.modules, self.blocks, self.heads, self.head_dim, self.device, self.past_frames = modules, len(modules), heads, head_dim, device, past_frames
        self.archive, self.length, self.writing, self.total = [], 0, None, 0
        # The archive and one CPU working-set view per block. MMGP 4.0 has no
        # public state_ring; each block's Q8 view is transferred on demand by
        # the local PrefixCache adapter, never as one all-block GPU tensor.
        self.working = None
        self.keys = []  # (segment ident, frame or "text") of the working set's rows, in order
        self._written = None  # the GPU rows of the segment written by the block that runs (store), copied to its record by flush

    def set_past_frames(self, past_frames):
        """How many history frames the next segments see (the working set is gathered again from the archive)."""
        if past_frames != self.past_frames:
            self.past_frames, self.working = past_frames, None

    def block_kv(self, block):
        """One block's quantized working set (key data, key scales, value data, value scales), None before the first segment: the list of
        GPU tensors the state ring rebinds (the block's slot, which its forward fills, or its tensors kept in VRAM)."""
        return None if self.working is None or self.length == 0 else self.working[block]

    def write_from(self, frame_start, frames, image_tokens, tokens):
        """Opens the record of a finished segment: its first `tokens` keys and values (image tokens, then the text later segments see) are
        stored by the blocks of the next forward."""
        shape, scale_shape = (self.blocks, tokens, self.heads, self.head_dim), (self.blocks, tokens, self.heads, self.head_dim // GROUP)
        self.writing = _Segment(list(range(frame_start, frame_start + frames)), image_tokens // frames, tokens - image_tokens,
                                *(torch.empty(s, dtype=d, device="cpu") for s, d in ((shape, torch.int8), (scale_shape, torch.float16)) * 2))

    def store(self, block, part, heads, values=False):
        """Quantizes the keys (or values) of one block for the heads `heads` (part: (1, tokens, heads, head_dim))."""
        if self._written is None:
            self._written = tuple(torch.empty(tensor.shape[1:], dtype=tensor.dtype, device=self.device) for tensor in self.writing.tensors())
        data, scales = quantize_heads(part[:, :self.writing.image_tokens + self.writing.text_tokens])
        target, target_scales = self._written[2:] if values else self._written[:2]
        target[:, heads.start:heads.stop].copy_(data)
        target_scales[:, heads.start:heads.stop].copy_(scales)

    def flush(self, block):
        """After the block's forward of a history write: its keys and values of the segment go to the segment's record (a blocking copy to
        RAM); its GPU rows serve the next block."""
        for archived, rows in zip(self.writing.tensors(), self._written):
            archived[block].copy_(rows)

    def commit(self):
        """Archives the written segment, bounds the archive and gathers the working set of the frames now selected."""
        self.archive.append(self.writing)
        self.total = self.writing.frame_ids[-1] + 1
        self.writing = self._written = None
        self._bound_archive()
        self.working = None
        self.load_working_set()

    def _bound_archive(self):
        kept = sum(len(record.frame_ids) for record in self.archive)
        limit = max(2 * self.past_frames, ARCHIVE_BYTES // frame_bytes(self.archive[-1].per_frame, self.blocks, self.heads, self.head_dim))
        if kept <= limit:
            return
        # thinned below the bound, so that the next segments add frames without copying the archive again at each one
        target = set(temporal_context_frames(self.total, int(limit * ARCHIVE_THINNING)))
        archive = []
        for record in self.archive:
            frames = [frame for frame in record.frame_ids if frame in target]
            if not frames:
                continue
            if len(frames) < len(record.frame_ids):
                rows = torch.cat([torch.arange(record.frame_rows(frame).start, record.frame_rows(frame).stop, device="cpu") for frame in frames]
                                 + [torch.arange(record.text_rows().start, record.text_rows().stop, device="cpu")])
                record = _Segment(frames, record.per_frame, record.text_tokens, *(tensor.index_select(1, rows) for tensor in record.tensors()),
                                  ident=record.ident)
            archive.append(record)
        self.archive = archive

    def release_working_set(self):
        """Frees the working set and its state ring (its RAM, its GPU slots and its hooks on the blocks): between generations, or when
        the world is closed or its generation interrupted; the archive stays in RAM."""
        self.working, self.keys = None, []

    def load_working_set(self):
        """Builds the working set in the state ring: block by block, the selected frames of each archived segment (a dropped frame
        replaced by the nearest kept one) and its visible text, gathered from the archive into the block's rows and stored in the ring
        (RAM, fetched to the GPU when the block runs, or kept in VRAM), the ring's memory serving again."""
        if self.working is not None:
            return
        if not self.archive:
            self.keys, self.length, self.working = [], 0, [None] * self.blocks
            return
        kept = np.array([frame for record in self.archive for frame in record.frame_ids])
        selected = {int(kept[np.abs(kept - frame).argmin()]) for frame in temporal_context_frames(self.total, self.past_frames)}
        self.keys, pieces = [], []  # (archive record, rows)
        for record in self.archive:
            frames = [frame for frame in record.frame_ids if frame in selected]
            for key, rows in [((record.ident, frame), record.frame_rows(frame)) for frame in frames] + ([((record.ident, "text"), record.text_rows())] if frames else []):
                if pieces and pieces[-1][0] is record and pieces[-1][1].stop == rows.start:  # adjacent rows of one record: one piece
                    pieces[-1] = (record, slice(pieces[-1][1].start, rows.stop))
                else:
                    pieces.append((record, rows))
                self.keys.append(key)
        self.length = sum(rows.stop - rows.start for _record, rows in pieces)
        working = []
        for block in range(self.blocks):
            block_tensors = []
            for index in range(4):
                rows = [record.tensors()[index][block, selection] for record, selection in pieces]
                if rows:
                    block_tensors.append(torch.cat(rows, dim=0).contiguous())
                else:
                    exemplar = self.archive[0].tensors()[index][block, :0]
                    block_tensors.append(exemplar.contiguous())
            working.append(tuple(block_tensors))
        self.working = working

    @property
    def seen_frames(self):
        """Frames of the working set (what the next segment sees)."""
        return sum(1 for key in self.keys if key[1] != "text")

    @property
    def stored_frames(self):
        """Frames kept in the archive."""
        return sum(len(record.frame_ids) for record in self.archive)

    def archive_rows(self, key):
        record = next(record for record in self.archive if record.ident == key[0])
        return record.text_tokens if key[1] == "text" else record.per_frame
