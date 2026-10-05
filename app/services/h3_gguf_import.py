"""Bounded, weight-free GGUF tensor index reader for MiniMax H3 imports.

This module parses only the GGUF header, metadata, and tensor descriptors. It
does not import the optional ``gguf`` package and never reads tensor payloads.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any, BinaryIO

from .h3_checkpoint_import import H3CheckpointError


_MAGIC = b"GGUF"
_SUPPORTED_VERSIONS = {2, 3}
_MAX_READ_BYTES = 8 * 1024 * 1024
_MAX_STRING_BYTES = 1024 * 1024
_MAX_METADATA_ENTRIES = 65_536
_MAX_TENSORS = 65_536
_MAX_U64 = (1 << 64) - 1
_DEFAULT_ALIGNMENT = 32
_MAX_TENSOR_DIMS = 4

# Values and block sizes mirror the app's installed gguf.GGML_QUANT_SIZES.
# This intentionally contains only types decoded by shared/qtypes/gguf.py.
_GGUF_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    30: ("BF16", 1, 2),
}
_UNQUANTIZED_TYPES = {"F32", "F16", "BF16"}

# GGUF metadata value types from the GGUF v2/v3 format.
_META_UINT8 = 0
_META_INT8 = 1
_META_UINT16 = 2
_META_INT16 = 3
_META_UINT32 = 4
_META_INT32 = 5
_META_FLOAT32 = 6
_META_BOOL = 7
_META_STRING = 8
_META_ARRAY = 9
_META_UINT64 = 10
_META_INT64 = 11
_META_FLOAT64 = 12
_SCALAR_STRUCTS: dict[int, tuple[str, int]] = {
    _META_UINT8: ("<B", 1),
    _META_INT8: ("<b", 1),
    _META_UINT16: ("<H", 2),
    _META_INT16: ("<h", 2),
    _META_UINT32: ("<I", 4),
    _META_INT32: ("<i", 4),
    _META_FLOAT32: ("<f", 4),
    _META_UINT64: ("<Q", 8),
    _META_INT64: ("<q", 8),
    _META_FLOAT64: ("<d", 8),
}


@dataclass(frozen=True)
class GGUFIndex:
    """A SafeTensor-like header and offsets for a GGUF file."""

    header: dict[str, Any]
    payload_start: int
    quant_types: set[str]


class _Reader:
    def __init__(self, stream: BinaryIO, file_size: int, *, start_position: int = 0):
        self.stream = stream
        self.file_size = file_size
        self.position = start_position
        self.bytes_read = 0

    def tell(self) -> int:
        return self.position

    def read_exact(self, size: int, what: str) -> bytes:
        if size < 0:
            raise H3CheckpointError(f"Invalid negative read size for {what}.", code="invalid_gguf")
        if self.bytes_read + size > _MAX_READ_BYTES:
            raise H3CheckpointError(
                "GGUF header exceeds the 8 MiB index read budget.",
                code="gguf_index_too_large",
                budget_bytes=_MAX_READ_BYTES,
            )
        start = self.tell()
        if start < 0 or size > self.file_size - start:
            raise H3CheckpointError(
                f"GGUF file is truncated while reading {what}.",
                code="gguf_truncated",
                offset=start,
                requested_bytes=size,
            )
        chunks = bytearray()
        remaining = size
        try:
            while remaining:
                chunk = self.stream.read(remaining)
                if not chunk:
                    raise H3CheckpointError(
                        f"GGUF file is truncated while reading {what}.",
                        code="gguf_truncated",
                        offset=start + len(chunks),
                        requested_bytes=remaining,
                    )
                if len(chunk) > remaining:
                    raise H3CheckpointError(
                        f"GGUF stream returned too many bytes while reading {what}.",
                        code="gguf_read_failed",
                    )
                chunks.extend(chunk)
                remaining -= len(chunk)
                self.position += len(chunk)
                self.bytes_read += len(chunk)
        except H3CheckpointError:
            raise
        except Exception as exc:
            raise H3CheckpointError(
                f"Unable to read GGUF {what}.", code="gguf_read_failed"
            ) from exc
        return bytes(chunks)

    def unpack(self, fmt: str, what: str):
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.read_exact(size, what))[0]

    def string(self, what: str) -> str:
        length = self.unpack("<Q", f"{what} length")
        if length > _MAX_STRING_BYTES:
            raise H3CheckpointError(
                f"GGUF {what} exceeds the 1 MiB string limit.",
                code="gguf_string_too_large",
                string_bytes=length,
            )
        raw = self.read_exact(length, what)
        try:
            value = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise H3CheckpointError(
                f"GGUF {what} is not valid UTF-8.", code="gguf_invalid_utf8"
            ) from exc
        if "\x00" in value:
            raise H3CheckpointError(
                f"GGUF {what} contains a NUL character.", code="invalid_gguf_string"
            )
        return value

    def count(self, count: int, *, name: str, minimum_entry_bytes: int, maximum: int) -> None:
        if count > maximum:
            raise H3CheckpointError(
                f"GGUF {name} count exceeds the supported limit.",
                code="gguf_count_too_large",
                count=count,
                maximum=maximum,
            )
        remaining_file_bytes = self.file_size - self.tell()
        if count > remaining_file_bytes // minimum_entry_bytes:
            raise H3CheckpointError(
                f"GGUF {name} count cannot fit in the remaining file.",
                code="gguf_invalid_count",
                count=count,
            )
        if count > (_MAX_READ_BYTES - self.bytes_read) // minimum_entry_bytes:
            raise H3CheckpointError(
                f"GGUF {name} count cannot fit in the remaining index read budget.",
                code="gguf_index_too_large",
                count=count,
                budget_bytes=_MAX_READ_BYTES,
            )


def _metadata_value(reader: _Reader, value_type: int, *, allow_array: bool = True):
    if value_type == _META_STRING:
        return reader.string("metadata string"), value_type, None
    if value_type == _META_BOOL:
        raw = reader.unpack("<B", "metadata boolean")
        if raw not in (0, 1):
            raise H3CheckpointError(
                "GGUF boolean metadata must be encoded as 0 or 1.",
                code="invalid_gguf_metadata",
            )
        return bool(raw), value_type, None
    if value_type == _META_ARRAY:
        if not allow_array:
            raise H3CheckpointError(
                "Nested GGUF metadata arrays are not supported.",
                code="nested_gguf_metadata_array",
            )
        element_type = reader.unpack("<I", "metadata array element type")
        if element_type == _META_ARRAY:
            raise H3CheckpointError(
                "Nested GGUF metadata arrays are not supported.",
                code="nested_gguf_metadata_array",
            )
        if element_type not in _SCALAR_STRUCTS and element_type not in (_META_STRING, _META_BOOL):
            raise H3CheckpointError(
                f"Unsupported GGUF metadata array element type {element_type}.",
                code="unsupported_gguf_metadata_type",
                value_type=element_type,
            )
        count = reader.unpack("<Q", "metadata array length")
        if element_type == _META_STRING:
            item_size = 8
        elif element_type == _META_BOOL:
            item_size = 1
        else:
            item_size = _SCALAR_STRUCTS[element_type][1]
        reader.count(
            count,
            name="metadata array element",
            minimum_entry_bytes=item_size,
            maximum=_MAX_READ_BYTES,
        )
        values = []
        for _ in range(count):
            value, _, _ = _metadata_value(reader, element_type, allow_array=False)
            values.append(value)
        return values, value_type, element_type
    scalar = _SCALAR_STRUCTS.get(value_type)
    if scalar is None:
        raise H3CheckpointError(
            f"Unsupported GGUF metadata value type {value_type}.",
            code="unsupported_gguf_metadata_type",
            value_type=value_type,
        )
    value = reader.unpack(scalar[0], "metadata value")
    if isinstance(value, float) and not math.isfinite(value):
        raise H3CheckpointError(
            "GGUF floating point metadata must be finite.", code="invalid_gguf_metadata"
        )
    return value, value_type, None


def _read_metadata(reader: _Reader, count: int) -> tuple[dict[str, Any], dict[str, tuple[int, int | None]]]:
    reader.count(
        count,
        name="metadata entry",
        minimum_entry_bytes=13,
        maximum=_MAX_METADATA_ENTRIES,
    )
    metadata: dict[str, Any] = {}
    metadata_types: dict[str, tuple[int, int | None]] = {}
    for _ in range(count):
        key = reader.string("metadata key")
        if not key:
            raise H3CheckpointError("GGUF metadata keys cannot be empty.", code="invalid_gguf_metadata")
        if key in metadata:
            raise H3CheckpointError(
                f"GGUF metadata key {key!r} occurs more than once.",
                code="duplicate_gguf_metadata",
            )
        value_type = reader.unpack("<I", "metadata value type")
        value, actual_type, element_type = _metadata_value(reader, value_type)
        metadata[key] = value
        metadata_types[key] = (actual_type, element_type)
    return metadata, metadata_types


def _numel(shape: tuple[int, ...] | list[int], *, tensor_name: str) -> int:
    if not shape:
        raise H3CheckpointError(
            f"GGUF tensor {tensor_name!r} cannot have an empty shape.",
            code="invalid_tensor_dimensions",
        )
    product = 1
    for dimension in shape:
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
            raise H3CheckpointError(
                f"GGUF tensor {tensor_name!r} has an empty or invalid dimension.",
                code="invalid_tensor_dimensions",
                shape=list(shape),
            )
        if product > _MAX_U64 // dimension:
            raise H3CheckpointError(
                f"GGUF tensor {tensor_name!r} has an overflowing shape.",
                code="invalid_tensor_dimensions",
                shape=list(shape),
            )
        product *= dimension
    return product


def _read_tensors(reader: _Reader, count: int):
    reader.count(
        count,
        name="tensor",
        minimum_entry_bytes=33,
        maximum=_MAX_TENSORS,
    )
    tensors: dict[str, dict[str, Any]] = {}
    extents: list[tuple[int, int, str]] = []
    quant_types: set[str] = set()
    for _ in range(count):
        name = reader.string("tensor name")
        if not name:
            raise H3CheckpointError("GGUF tensor names cannot be empty.", code="invalid_tensor_name")
        if name in tensors:
            raise H3CheckpointError(
                f"GGUF tensor {name!r} occurs more than once.", code="duplicate_tensor"
            )
        ndim = reader.unpack("<I", f"tensor {name!r} dimension count")
        if ndim < 1 or ndim > _MAX_TENSOR_DIMS:
            raise H3CheckpointError(
                f"GGUF tensor {name!r} has unsupported rank {ndim}.",
                code="invalid_tensor_dimensions",
                rank=ndim,
            )
        gguf_dims = tuple(reader.unpack("<Q", f"tensor {name!r} dimension") for _ in range(ndim))
        if any(dimension == 0 for dimension in gguf_dims):
            raise H3CheckpointError(
                f"GGUF tensor {name!r} has an empty dimension.",
                code="invalid_tensor_dimensions",
                gguf_shape=list(gguf_dims),
            )
        logical_shape = tuple(reversed(gguf_dims))
        numel = _numel(logical_shape, tensor_name=name)
        gguf_type = reader.unpack("<I", f"tensor {name!r} ggml type")
        type_info = _GGUF_TYPES.get(gguf_type)
        if type_info is None:
            raise H3CheckpointError(
                f"GGUF tensor {name!r} uses unsupported GGML type {gguf_type}.",
                code="unsupported_gguf_type",
                gguf_type=gguf_type,
            )
        type_name, block_size, type_size = type_info
        if numel % block_size:
            raise H3CheckpointError(
                f"GGUF tensor {name!r} has {numel} elements, which is not divisible by the {type_name} block size {block_size}.",
                code="gguf_tensor_size_mismatch",
                gguf_type=gguf_type,
                elements=numel,
                block_size=block_size,
            )
        if type_name not in _UNQUANTIZED_TYPES and logical_shape[-1] % block_size:
            raise H3CheckpointError(
                f"GGUF tensor {name!r} row width {logical_shape[-1]} is not divisible by the {type_name} block size {block_size}.",
                code="gguf_tensor_size_mismatch",
                gguf_type=gguf_type,
                row_width=logical_shape[-1],
                block_size=block_size,
            )
        data_offset = reader.unpack("<Q", f"tensor {name!r} data offset")
        tensor_bytes = (numel // block_size) * type_size
        data_end = data_offset + tensor_bytes
        if data_end > _MAX_U64 or tensor_bytes <= 0:
            raise H3CheckpointError(
                f"GGUF tensor {name!r} has an invalid byte size.", code="gguf_tensor_size_mismatch"
            )
        dtype = type_name if type_name in _UNQUANTIZED_TYPES else f"GGUF_{type_name}"
        descriptor = {
            "dtype": dtype,
            "shape": list(logical_shape),
            "data_offsets": [data_offset, data_end],
            "gguf_type": gguf_type,
            "gguf_type_name": type_name,
        }
        tensors[name] = descriptor
        extents.append((data_offset, data_end, name))
        if type_name not in _UNQUANTIZED_TYPES:
            quant_types.add(type_name)
    return tensors, extents, quant_types


def _apply_original_shapes(
    tensors: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
    metadata_types: dict[str, tuple[int, int | None]],
) -> None:
    prefix = "comfy.gguf.orig_shape."
    for key, shape_value in metadata.items():
        if not key.startswith(prefix):
            continue
        tensor_name = key[len(prefix) :]
        if not tensor_name or tensor_name not in tensors:
            raise H3CheckpointError(
                f"GGUF original-shape metadata references missing tensor {tensor_name!r}.",
                code="invalid_original_shape_metadata",
                metadata_key=key,
            )
        if metadata_types.get(key) != (_META_ARRAY, _META_INT32) or not isinstance(shape_value, list):
            raise H3CheckpointError(
                f"GGUF original-shape metadata for {tensor_name!r} must be an ARRAY INT32.",
                code="invalid_original_shape_metadata",
                metadata_key=key,
            )
        original_shape = tuple(shape_value)
        if not 1 <= len(original_shape) <= _MAX_TENSOR_DIMS:
            raise H3CheckpointError(
                f"GGUF original shape for {tensor_name!r} must have rank 1 through {_MAX_TENSOR_DIMS}.",
                code="invalid_original_shape_metadata",
                original_shape=list(original_shape),
            )
        original_numel = _numel(original_shape, tensor_name=tensor_name)
        current_shape = tensors[tensor_name]["shape"]
        current_numel = _numel(current_shape, tensor_name=tensor_name)
        if original_numel != current_numel:
            raise H3CheckpointError(
                f"GGUF original shape for {tensor_name!r} has {original_numel} elements; the stored tensor has {current_numel}.",
                code="original_shape_numel_mismatch",
                original_shape=list(original_shape),
                tensor_shape=current_shape,
            )
        gguf_type = tensors[tensor_name]["gguf_type"]
        type_name, block_size, _ = _GGUF_TYPES[gguf_type]
        if type_name not in _UNQUANTIZED_TYPES and original_shape[-1] % block_size:
            raise H3CheckpointError(
                f"GGUF original row width for {tensor_name!r} is not divisible by the {type_name} block size {block_size}.",
                code="original_shape_block_mismatch",
                original_shape=list(original_shape),
                block_size=block_size,
            )
        tensors[tensor_name]["shape"] = list(original_shape)


def read_gguf_index(stream: BinaryIO, file_size: int) -> GGUFIndex:
    """Read a bounded GGUF v2/v3 index without reading tensor payload bytes.

    ``file_size`` is the full GGUF file size, even when ``stream`` is a
    bounded, nonseekable range response containing only the index bytes. The
    returned descriptor offsets are relative to ``payload_start`` and shapes
    use logical Torch order (GGUF's serialized dimensions are reversed).
    """

    if isinstance(file_size, bool) or not isinstance(file_size, int) or file_size < 24:
        raise H3CheckpointError("GGUF file size is invalid.", code="invalid_gguf_file_size")
    # ``tell`` is useful when available, but streamed HTTP range bodies often
    # expose neither it nor ``seek``. Never probe or seek those bodies.
    start_position: int | None = None
    tell_method = getattr(stream, "tell", None)
    if callable(tell_method):
        try:
            start_position = int(tell_method())
        except Exception:
            start_position = None
    if start_position is not None and start_position != 0:
        raise H3CheckpointError(
            "GGUF index stream must start at byte zero.", code="gguf_stream_not_at_start"
        )

    seekable = False
    seekable_method = getattr(stream, "seekable", None)
    if callable(seekable_method):
        try:
            seekable = bool(seekable_method())
        except Exception:
            seekable = False
    if seekable:
        if start_position is None:
            try:
                start_position = int(stream.tell())
            except Exception as exc:
                raise H3CheckpointError(
                    "Seekable GGUF stream does not expose its starting position.",
                    code="gguf_stream_not_seekable",
                ) from exc
        if start_position != 0:
            raise H3CheckpointError(
                "GGUF index stream must start at byte zero.", code="gguf_stream_not_at_start"
            )
        try:
            stream.seek(0, 2)
            actual_size = int(stream.tell())
            stream.seek(0)
        except Exception as exc:
            raise H3CheckpointError(
                "Seekable GGUF stream could not be checked and reset.",
                code="gguf_stream_not_seekable",
            ) from exc
        if actual_size != file_size:
            raise H3CheckpointError(
                "The supplied GGUF file size does not match the stream.",
                code="gguf_file_size_mismatch",
                expected=file_size,
                actual=actual_size,
            )

    reader = _Reader(stream, file_size)
    magic = reader.read_exact(4, "magic")
    if magic != _MAGIC:
        raise H3CheckpointError("File does not begin with GGUF magic.", code="invalid_gguf_magic")
    version = reader.unpack("<I", "version")
    if version not in _SUPPORTED_VERSIONS:
        raise H3CheckpointError(
            f"GGUF version {version} is unsupported; only v2 and v3 are accepted.",
            code="unsupported_gguf_version",
            version=version,
        )
    tensor_count = reader.unpack("<Q", "tensor count")
    metadata_count = reader.unpack("<Q", "metadata count")
    if tensor_count == 0:
        raise H3CheckpointError("GGUF index contains no tensors.", code="invalid_tensor_count")
    metadata, metadata_types = _read_metadata(reader, metadata_count)
    tensors, extents, quant_types = _read_tensors(reader, tensor_count)

    alignment = _DEFAULT_ALIGNMENT
    if "general.alignment" in metadata:
        if metadata_types["general.alignment"] != (_META_UINT32, None):
            raise H3CheckpointError(
                "GGUF general.alignment must be a UINT32 metadata value.",
                code="invalid_gguf_alignment",
            )
        alignment = metadata["general.alignment"]
    if alignment <= 0 or alignment & (alignment - 1):
        raise H3CheckpointError(
            "GGUF general.alignment must be a positive power of two.",
            code="invalid_gguf_alignment",
            alignment=alignment,
        )

    unaligned_payload_start = reader.tell()
    payload_start = (unaligned_payload_start + alignment - 1) & ~(alignment - 1)
    if payload_start > file_size:
        raise H3CheckpointError(
            "GGUF tensor payload begins beyond the end of the file.",
            code="gguf_tensor_out_of_bounds",
            payload_start=payload_start,
        )
    data_size = file_size - payload_start
    previous_end = -1
    for data_offset, data_end, tensor_name in sorted(extents):
        if data_offset % alignment:
            raise H3CheckpointError(
                f"GGUF tensor {tensor_name!r} offset is not aligned to {alignment} bytes.",
                code="gguf_tensor_offset_alignment",
                tensor=tensor_name,
                data_offset=data_offset,
                alignment=alignment,
            )
        if data_end > data_size:
            raise H3CheckpointError(
                f"GGUF tensor {tensor_name!r} extends beyond the file payload.",
                code="gguf_tensor_out_of_bounds",
                tensor=tensor_name,
                data_end=data_end,
                payload_bytes=data_size,
            )
        if data_offset < previous_end:
            raise H3CheckpointError(
                f"GGUF tensor {tensor_name!r} overlaps another tensor.",
                code="gguf_tensor_overlap",
                tensor=tensor_name,
            )
        previous_end = data_end

    _apply_original_shapes(tensors, metadata, metadata_types)
    header: dict[str, Any] = {"__metadata__": metadata}
    header.update(tensors)
    return GGUFIndex(header=header, payload_start=payload_start, quant_types=quant_types)


__all__ = ["GGUFIndex", "read_gguf_index"]
