"""Independent binary-fixture tests for the bounded H3 GGUF index reader."""

from __future__ import annotations

import io
import struct
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from services.h3_checkpoint_import import H3CheckpointError  # noqa: E402
from services.h3_gguf_import import read_gguf_index  # noqa: E402


MAGIC = b"GGUF"
META = {
    "uint8": 0,
    "int8": 1,
    "uint16": 2,
    "int16": 3,
    "uint32": 4,
    "int32": 5,
    "float32": 6,
    "bool": 7,
    "string": 8,
    "array": 9,
    "uint64": 10,
    "int64": 11,
    "float64": 12,
}
META_STRUCT = {
    META["uint8"]: "<B",
    META["int8"]: "<b",
    META["uint16"]: "<H",
    META["int16"]: "<h",
    META["uint32"]: "<I",
    META["int32"]: "<i",
    META["float32"]: "<f",
    META["uint64"]: "<Q",
    META["int64"]: "<q",
    META["float64"]: "<d",
}
# Numeric GGML type IDs and block sizes are written independently of the reader.
QTYPE = {
    "F32": (0, 1, 4),
    "F16": (1, 1, 2),
    "Q4_0": (2, 32, 18),
    "Q4_1": (3, 32, 20),
    "Q5_0": (6, 32, 22),
    "Q5_1": (7, 32, 24),
    "Q8_0": (8, 32, 34),
    "Q2_K": (10, 256, 84),
    "Q3_K": (11, 256, 110),
    "Q4_K": (12, 256, 144),
    "Q5_K": (13, 256, 176),
    "Q6_K": (14, 256, 210),
    "BF16": (30, 1, 2),
}


def _string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _metadata(key: str, value_type: int, value, element_type: int | None = None) -> bytes:
    result = bytearray(_string(key))
    result += struct.pack("<I", value_type)
    if value_type == META["string"]:
        result += _string(value)
    elif value_type == META["array"]:
        result += struct.pack("<IQ", element_type, len(value))
        if element_type == META["string"]:
            for item in value:
                result += _string(item)
        elif element_type == META["bool"]:
            result += bytes(int(item) for item in value)
        elif element_type == META["array"]:
            # Deliberately invalid nested-array fixture.
            for item in value:
                result += struct.pack("<I", item)
        else:
            fmt = META_STRUCT[element_type]
            for item in value:
                result += struct.pack(fmt, item)
    elif value_type == META["bool"]:
        result += struct.pack("<B", value)
    elif value_type in META_STRUCT:
        result += struct.pack(META_STRUCT[value_type], value)
    return bytes(result)


def make_gguf(
    tensors,
    *,
    metadata=(),
    version=3,
    alignment=32,
    data_offsets=None,
    tensor_count=None,
    metadata_count=None,
    trailing_payload=0,
):
    """Write a minimal GGUF fixture from the public on-disk field ordering."""
    metadata = list(metadata)
    tensors = list(tensors)
    metadata_blob = b"".join(_metadata(*item) for item in metadata)
    if metadata_count is None:
        metadata_count = len(metadata)
    if tensor_count is None:
        tensor_count = len(tensors)

    def write_tensor_table(offsets):
        table = bytearray()
        for tensor, offset in zip(tensors, offsets):
            name, logical_shape, type_id = tensor
            stored_shape = tuple(reversed(logical_shape))
            table += _string(name)
            table += struct.pack("<I", len(stored_shape))
            table += b"".join(struct.pack("<Q", dim) for dim in stored_shape)
            table += struct.pack("<IQ", type_id, offset)
        return bytes(table)

    requested_alignment = alignment
    table_with_zero_offsets = write_tensor_table([0] * len(tensors))
    payload_start = (24 + len(metadata_blob) + len(table_with_zero_offsets) + requested_alignment - 1) & ~(requested_alignment - 1)
    if data_offsets is None:
        offsets = []
        cursor = 0
        for _, shape, type_id in tensors:
            type_info = next((entry for entry in QTYPE.values() if entry[0] == type_id), None)
            if type_info is None:
                byte_size = 4
            else:
                _, block_size, type_size = type_info
                numel = 1
                for dim in shape:
                    numel *= dim
                byte_size = max(1, (numel * type_size + block_size - 1) // block_size)
            cursor = (cursor + requested_alignment - 1) & ~(requested_alignment - 1)
            offsets.append(cursor)
            cursor += byte_size
    else:
        offsets = list(data_offsets)

    tensor_blob = write_tensor_table(offsets)
    base = bytearray(MAGIC + struct.pack("<IQQ", version, tensor_count, metadata_count))
    base += metadata_blob + tensor_blob
    actual_payload_start = (len(base) + requested_alignment - 1) & ~(requested_alignment - 1)
    assert actual_payload_start == payload_start
    data_end = trailing_payload
    for tensor, offset in zip(tensors, offsets):
        _, shape, type_id = tensor
        type_info = next((entry for entry in QTYPE.values() if entry[0] == type_id), None)
        if type_info is None:
            byte_size = 4
        else:
            _, block_size, type_size = type_info
            numel = 1
            for dim in shape:
                numel *= dim
            byte_size = max(1, (numel * type_size + block_size - 1) // block_size)
        data_end = max(data_end, offset + byte_size)
    payload = bytes(data_end)
    blob = bytes(base) + bytes(actual_payload_start - len(base)) + payload
    return blob, actual_payload_start


class ReadGuard:
    """Virtual file stream that rejects any read of the payload region."""

    def __init__(self, header_and_file: bytes, payload_start: int, *, logical_size=None):
        self.data = header_and_file
        self.payload_start = payload_start
        self.logical_size = len(header_and_file) if logical_size is None else logical_size
        self.position = 0
        self.read_bytes = 0
        self.read_ranges = []

    def tell(self):
        return self.position

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        if whence == 2:
            self.position = self.logical_size + offset
        elif whence == 1:
            self.position += offset
        else:
            self.position = offset
        return self.position

    def read(self, size=-1):
        if size < 0:
            raise AssertionError("reader must request a bounded read")
        end = self.position + size
        if self.position < self.payload_start < end or self.position >= self.payload_start:
            raise AssertionError(f"reader touched payload bytes at {self.position}:{end}")
        self.read_ranges.append((self.position, end))
        self.read_bytes += size
        chunk = self.data[self.position : end]
        self.position += len(chunk)
        return chunk


class HeaderRangeStream:
    """Nonseekable range body that exposes only header bytes in short reads."""

    def __init__(self, header_bytes: bytes, payload_start: int, *, chunk_size=7):
        self.data = header_bytes
        self.payload_start = payload_start
        self.chunk_size = chunk_size
        self.position = 0
        self.read_bytes = 0
        self.read_ranges = []

    def read(self, size=-1):
        if size < 0:
            raise AssertionError("reader must request a bounded read")
        end = min(self.position + size, self.position + self.chunk_size, len(self.data))
        if end > self.payload_start:
            raise AssertionError(f"reader touched payload bytes at {self.position}:{end}")
        self.read_ranges.append((self.position, end))
        chunk = self.data[self.position:end]
        self.position = end
        self.read_bytes += len(chunk)
        return chunk


class H3GGUFImportTests(unittest.TestCase):
    def parse(self, blob):
        return read_gguf_index(io.BytesIO(blob), len(blob))

    def assert_code(self, blob, expected_code):
        with self.assertRaises(H3CheckpointError) as caught:
            self.parse(blob)
        self.assertEqual(caught.exception.code, expected_code)

    def test_indexes_plain_and_packed_tensors_in_logical_torch_order(self):
        blob, payload_start = make_gguf(
            [
                ("float.weight", (2, 3), QTYPE["F32"][0]),
                ("half.weight", (2, 4), QTYPE["F16"][0]),
                ("bfloat.weight", (8,), QTYPE["BF16"][0]),
                ("quant.weight", (2, 256), QTYPE["Q4_K"][0]),
            ],
            metadata=[
                ("qkv_layout", META["string"], "grouped"),
                ("modelspec.title", META["string"], "MiniMax H3 test"),
                ("general.architecture", META["string"], "minimax_h3"),
            ],
        )
        index = self.parse(blob)
        self.assertEqual(index.payload_start, payload_start)
        self.assertEqual(index.header["float.weight"]["shape"], [2, 3])
        self.assertEqual(index.header["float.weight"]["dtype"], "F32")
        self.assertEqual(index.header["float.weight"]["gguf_type"], 0)
        self.assertEqual(index.header["float.weight"]["gguf_type_name"], "F32")
        self.assertEqual(index.header["quant.weight"]["shape"], [2, 256])
        self.assertEqual(index.header["quant.weight"]["dtype"], "GGUF_Q4_K")
        self.assertEqual(index.header["quant.weight"]["gguf_type"], 12)
        self.assertEqual(index.header["quant.weight"]["gguf_type_name"], "Q4_K")
        self.assertEqual(index.header["quant.weight"]["data_offsets"][1] - index.header["quant.weight"]["data_offsets"][0], 288)
        self.assertEqual(index.quant_types, {"Q4_K"})
        self.assertEqual(index.header["__metadata__"]["qkv_layout"], "grouped")
        self.assertEqual(index.header["__metadata__"]["modelspec.title"], "MiniMax H3 test")
        self.assertEqual(index.header["__metadata__"]["general.architecture"], "minimax_h3")
        no_layout_blob, _ = make_gguf([("x", (4,), QTYPE["F32"][0])])
        self.assertNotIn("qkv_layout", self.parse(no_layout_blob).header["__metadata__"])

    def test_supported_ggml_types_use_primary_block_sizes(self):
        for type_name, (type_id, block_size, type_size) in QTYPE.items():
            with self.subTest(type_name=type_name):
                blob, _ = make_gguf([("weight", (block_size,), type_id)])
                descriptor = self.parse(blob).header["weight"]
                self.assertEqual(descriptor["gguf_type"], type_id)
                self.assertEqual(descriptor["gguf_type_name"], type_name)
                self.assertEqual(descriptor["data_offsets"][1] - descriptor["data_offsets"][0], type_size)
                expected_dtype = type_name if type_name in {"F32", "F16", "BF16"} else f"GGUF_{type_name}"
                self.assertEqual(descriptor["dtype"], expected_dtype)

    def test_validated_original_shape_is_applied_without_changing_numel(self):
        blob, _ = make_gguf(
            [("linear.weight", (2, 32), QTYPE["Q4_0"][0])],
            metadata=[("comfy.gguf.orig_shape.linear.weight", META["array"], [1, 64], META["int32"])],
        )
        index = self.parse(blob)
        self.assertEqual(index.header["linear.weight"]["shape"], [1, 64])
        self.assertEqual(index.header["__metadata__"]["comfy.gguf.orig_shape.linear.weight"], [1, 64])

    def test_preserves_supported_scalar_metadata_and_array_types(self):
        blob, _ = make_gguf(
            [("weight", (4,), QTYPE["F32"][0])],
            metadata=[
                ("general.alignment", META["uint32"], 16),
                ("modelspec.title", META["string"], "A model"),
                ("general.quantized_by", META["string"], "creator"),
                ("general.test_bool", META["bool"], 1),
                ("test.array", META["array"], [1, 2, 3], META["int32"]),
                ("test.bool_array", META["array"], [True, False], META["bool"]),
            ],
            alignment=16,
        )
        index = self.parse(blob)
        self.assertEqual(index.header["__metadata__"]["general.alignment"], 16)
        self.assertEqual(index.header["__metadata__"]["general.test_bool"], True)
        self.assertEqual(index.header["__metadata__"]["test.array"], [1, 2, 3])
        self.assertEqual(index.header["__metadata__"]["test.bool_array"], [True, False])

    def test_reader_stays_within_header_and_eight_mib_read_budget(self):
        blob, payload_start = make_gguf(
            [("weight", (256,), QTYPE["Q2_K"][0])], trailing_payload=2 * 1024 * 1024
        )
        guarded = ReadGuard(blob, payload_start)
        index = read_gguf_index(guarded, len(blob))
        self.assertEqual(index.payload_start, payload_start)
        self.assertGreater(guarded.read_bytes, 0)
        self.assertLessEqual(guarded.read_bytes, 8 * 1024 * 1024)
        self.assertTrue(all(end <= payload_start for _, end in guarded.read_ranges))

    def test_nonseekable_range_stream_reads_short_chunks_without_touching_weights(self):
        blob, payload_start = make_gguf(
            [("weight", (256,), QTYPE["Q2_K"][0])], trailing_payload=2 * 1024 * 1024
        )
        header_range = blob[:payload_start]
        advertised_file_size = payload_start + 5 * 1024 * 1024 * 1024
        stream = HeaderRangeStream(header_range, payload_start, chunk_size=3)
        index = read_gguf_index(stream, advertised_file_size)
        self.assertEqual(index.payload_start, payload_start)
        self.assertEqual(index.header["weight"]["shape"], [256])
        self.assertGreater(stream.read_bytes, 0)
        self.assertLessEqual(stream.read_bytes, 8 * 1024 * 1024)
        self.assertTrue(all(end <= payload_start for _, end in stream.read_ranges))

    def test_nonseekable_short_read_truncation_is_reported(self):
        blob, payload_start = make_gguf([("weight", (4,), QTYPE["F32"][0])])
        truncated_range = blob[:payload_start - 7]
        stream = HeaderRangeStream(truncated_range, payload_start, chunk_size=5)
        with self.assertRaises(H3CheckpointError) as caught:
            read_gguf_index(stream, payload_start + 1024 * 1024)
        self.assertEqual(caught.exception.code, "gguf_truncated")

    def test_inspectable_nonzero_start_and_seekable_size_mismatch_are_rejected(self):
        blob, _ = make_gguf([("weight", (4,), QTYPE["F32"][0])])
        offset_stream = io.BytesIO(blob)
        offset_stream.seek(1)
        with self.assertRaises(H3CheckpointError) as caught:
            read_gguf_index(offset_stream, len(blob))
        self.assertEqual(caught.exception.code, "gguf_stream_not_at_start")

        with self.assertRaises(H3CheckpointError) as caught:
            read_gguf_index(io.BytesIO(blob), len(blob) + 1)
        self.assertEqual(caught.exception.code, "gguf_file_size_mismatch")

    def test_rejects_header_that_exceeds_the_cumulative_read_budget(self):
        large_fields = [
            (f"large.{i}", META["string"], "x" * 900_000)
            for i in range(10)
        ]
        blob, _ = make_gguf([("weight", (4,), QTYPE["F32"][0])], metadata=large_fields)
        self.assert_code(blob, "gguf_index_too_large")

    def test_rejects_unsupported_versions_and_implausible_counts(self):
        blob, _ = make_gguf([("weight", (4,), QTYPE["F32"][0])], version=1)
        self.assert_code(blob, "unsupported_gguf_version")
        blob, _ = make_gguf([("weight", (4,), QTYPE["F32"][0])], version=4)
        self.assert_code(blob, "unsupported_gguf_version")
        blob, _ = make_gguf(
            [("weight", (4,), QTYPE["F32"][0])], tensor_count=2**64 - 1
        )
        self.assert_code(blob, "gguf_count_too_large")
        blob, _ = make_gguf(
            [("weight", (4,), QTYPE["F32"][0])], metadata_count=2**64 - 1
        )
        self.assert_code(blob, "gguf_count_too_large")

    def test_rejects_bad_magic_truncation_and_metadata_lengths(self):
        blob, _ = make_gguf([("weight", (4,), QTYPE["F32"][0])])
        self.assert_code(b"NOPE" + blob[4:], "invalid_gguf_magic")
        self.assert_code(blob[:-1], "gguf_tensor_out_of_bounds")
        self.assert_code(blob[:57], "gguf_truncated")

        oversized_string = bytearray(blob)
        # First tensor name starts after the 24-byte header and contains its own
        # uint64 string length. Set the length over the parser's 1 MiB cap.
        struct.pack_into("<Q", oversized_string, 24, 1024 * 1024 + 1)
        self.assert_code(bytes(oversized_string), "gguf_string_too_large")

    def test_rejects_unknown_types_and_nested_metadata_arrays(self):
        blob, _ = make_gguf([("weight", (4,), 31)])
        self.assert_code(blob, "unsupported_gguf_type")
        blob, _ = make_gguf(
            [("weight", (4,), QTYPE["F32"][0])],
            metadata=[("nested", META["array"], [], META["array"])],
        )
        self.assert_code(blob, "nested_gguf_metadata_array")
        blob, _ = make_gguf(
            [("weight", (4,), QTYPE["F32"][0])],
            metadata=[("comfy.gguf.orig_shape.weight", META["array"], [4], META["uint32"])],
        )
        self.assert_code(blob, "invalid_original_shape_metadata")
        blob, _ = make_gguf(
            [("weight", (4,), QTYPE["F32"][0])],
            metadata=[("comfy.gguf.orig_shape.weight", META["array"], [-1, 4], META["int32"])],
        )
        self.assert_code(blob, "invalid_tensor_dimensions")
        blob, _ = make_gguf(
            [("weight", (4,), QTYPE["F32"][0])],
            metadata=[("comfy.gguf.orig_shape.weight", META["array"], [1, 1, 1, 1, 4], META["int32"])],
        )
        self.assert_code(blob, "invalid_original_shape_metadata")
        blob, _ = make_gguf(
            [("weight", (2, 32), QTYPE["Q4_0"][0])],
            metadata=[("comfy.gguf.orig_shape.weight", META["array"], [4, 16], META["int32"])],
        )
        self.assert_code(blob, "original_shape_block_mismatch")

    def test_rejects_empty_shapes_and_quant_block_size_mismatch(self):
        blob, _ = make_gguf([("weight", (0, 32), QTYPE["Q4_0"][0])])
        self.assert_code(blob, "invalid_tensor_dimensions")
        for shape in ((31,), (2, 16)):
            with self.subTest(shape=shape):
                blob, _ = make_gguf([("weight", shape, QTYPE["Q4_0"][0])])
                self.assert_code(blob, "gguf_tensor_size_mismatch")

    def test_rejects_invalid_alignment_and_tensor_offsets(self):
        blob, _ = make_gguf(
            [("weight", (4,), QTYPE["F32"][0])],
            metadata=[("general.alignment", META["uint32"], 3)],
            alignment=3,
        )
        self.assert_code(blob, "invalid_gguf_alignment")
        blob, _ = make_gguf(
            [("weight", (4,), QTYPE["F32"][0])], data_offsets=[1]
        )
        self.assert_code(blob, "gguf_tensor_offset_alignment")
        blob, _ = make_gguf(
            [("one", (4,), QTYPE["F32"][0]), ("two", (4,), QTYPE["F32"][0])],
            data_offsets=[0, 0],
        )
        self.assert_code(blob, "gguf_tensor_overlap")
        blob, payload_start = make_gguf(
            [("weight", (4,), QTYPE["F32"][0])], data_offsets=[4096]
        )
        blob = blob[: payload_start + 16]
        self.assert_code(blob, "gguf_tensor_out_of_bounds")

    def test_rejects_duplicate_names_and_conflicting_original_shapes(self):
        blob, _ = make_gguf(
            [("weight", (4,), QTYPE["F32"][0]), ("weight", (4,), QTYPE["F32"][0])]
        )
        self.assert_code(blob, "duplicate_tensor")
        blob, _ = make_gguf(
            [("weight", (2, 2), QTYPE["F32"][0])],
            metadata=[("comfy.gguf.orig_shape.weight", META["array"], [3, 2], META["int32"])],
        )
        self.assert_code(blob, "original_shape_numel_mismatch")
        blob, _ = make_gguf(
            [("weight", (2, 2), QTYPE["F32"][0])],
            metadata=[("comfy.gguf.orig_shape.unknown", META["array"], [2, 2], META["int32"])],
        )
        self.assert_code(blob, "invalid_original_shape_metadata")


if __name__ == "__main__":
    unittest.main()
