"""CPU-only tests of H3 transport, provenance, atomic import and registration."""
from __future__ import annotations

import ast
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
from contextlib import nullcontext
from functools import partial
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from services.civitai_checkpoints import (
    CivitaiCheckpointError, RemoteH3Header, build_h3_definitions,
    resolve_h3_source, verify_h3_digest,
)
from models.minimax_h3.imported import normalize_imported_h3_request, normalize_imported_h3_steps
from models.minimax_h3.minimax_h3_handler import family_handler
from shared.checkpoint_downloads import CheckpointDownloadError, download_named_checkpoint


class Response:
    def __init__(self, body=b"", *, status=200, headers=None, data=None):
        self.raw = io.BytesIO(body)
        self.body, self.status_code, self.headers, self.data = body, status, headers or {}, data
        self.closed = False
    def close(self):
        self.closed = True
    def json(self):
        return self.data
    def iter_content(self, chunk_size):
        yield self.body


def source(**overrides):
    result = {"modelId": 12, "versionId": 23, "fileId": 34, "name": "Community H3",
              "versionName": "Hybrid", "filename": "community.safetensors", "size_bytes": 10000,
              "sha256": "a" * 64, "url": "https://civitai.com/api/download/models/23?fileId=34"}
    return {**result, **overrides}


def profile(**overrides):
    return {"status": "verified", "architectures": ["minimax_h3", "minimax_h3_ref2va"],
            "compressed_modulation": True, "qkv_layout": "grouped", "native_workflow": "ref2va",
            "sampling_profile": "turbo", "quantization_format": "int8_tensorwise", "convrot": True,
            "sampler": "euler", "video_shift": 9, "audio_shift": 4, "default_steps": 8,
            "min_steps": 4, "max_steps": 8, "baked_turbo": True, "fused_turbo": False, **overrides}


class TestCivitaiH3Transport(unittest.TestCase):
    def api_model(self, **file_overrides):
        return {"id": 12, "type": "Checkpoint", "name": "Community H3", "nsfw": False,
                "modelVersions": [{"id": 23, "baseModel": "MiniMax H3", "name": "Hybrid",
                    "files": [{"id": 34, "name": "community.safetensors", "sizeKB": 10,
                               "metadata": {"fp": "int8"}, "hashes": {"SHA256": "A" * 64}, **file_overrides}]}]}

    def resolve(self, data, **body):
        response = Response(data=data)
        resolved = resolve_h3_source({"model_id": 12, "version_id": 23, "file_id": 34, **body},
                                     get=lambda *args, **kwargs: response)
        self.assertTrue(response.closed)
        return resolved

    def test_canonical_file_id_and_server_metadata_replace_client_claims(self):
        actual = self.resolve(self.api_model(), filename="spoof.safetensors", download_url="http://127.0.0.1/private", sha256="b" * 64)
        self.assertEqual(actual["filename"], "community.safetensors")
        self.assertEqual(actual["sha256"], "a" * 64)
        self.assertEqual(actual["size_bytes"], 10240)
        self.assertEqual(actual["url"], "https://civitai.com/api/download/models/23?fileId=34")

    def test_unsupported_files_and_missing_checksums_fail_before_weight_download(self):
        for changes in ({"metadata": {"fp": "nvfp4"}}, {"name": "workflow.json"},
                        {"name": "weights.zip"}, {"hashes": {}}, {"name": "../escape.safetensors"}):
            with self.subTest(changes=changes), self.assertRaises(CivitaiCheckpointError):
                self.resolve(self.api_model(**changes))

    def test_int4_and_gguf_are_candidates_requiring_real_header_verification(self):
        for changes in ({"metadata": {"fp": "int4"}}, {"name": "weights.gguf"}):
            resolved = self.resolve(self.api_model(**changes))
            self.assertEqual(resolved["fileId"], 34)

    def test_model_version_and_file_must_match_actual_civitai_record(self):
        data = self.api_model()
        data["modelVersions"][0]["baseModel"] = "SDXL 1.0"
        with self.assertRaisesRegex(CivitaiCheckpointError, "not a MiniMax H3"):
            self.resolve(data)
        with self.assertRaises(CivitaiCheckpointError):
            self.resolve(self.api_model(), file_id=999)

    def test_auth_error_is_actionable_and_does_not_expose_tokens(self):
        response = Response(status=403)
        with self.assertRaisesRegex(CivitaiCheckpointError, "API Key") as context:
            resolve_h3_source({"model_id": 12, "version_id": 23, "file_id": 34}, api_key="secret-token", get=lambda *a, **k: response)
        self.assertNotIn("secret-token", str(context.exception))
        self.assertTrue(response.closed)

    def test_header_reads_only_declared_bytes_even_when_server_ignores_range(self):
        header = {"weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}
        encoded = json.dumps(header).encode()
        payload = struct.pack("<Q", len(encoded)) + encoded + b"\0" * 1000
        response = Response(payload)
        reader = RemoteH3Header(source(size_bytes=len(payload)), get=lambda *a, **k: response)
        self.assertEqual(reader.header, header)
        self.assertEqual(response.raw.tell(), 8 + len(encoded))
        self.assertTrue(response.closed)
        with self.assertRaisesRegex(CivitaiCheckpointError, "Only small"):
            reader.tensor_reader("weight")

    def test_gguf_preflight_caches_first_curve_without_reading_weights(self):
        from test_h3_gguf_import import make_gguf
        blob, payload_start = make_gguf([
            ("adaln_t_table", (1025, 8), 0),
            ("blocks.0.attn.qkv_proj.weight", (3, 256), 12),
        ])
        response = Response(blob)
        get = Mock(return_value=response)
        reader = RemoteH3Header(source(filename="community.gguf", size_bytes=len(blob)), get=get)
        self.assertEqual(reader.file_format, "gguf")
        self.assertEqual(reader.header["blocks.0.attn.qkv_proj.weight"]["dtype"], "GGUF_Q4_K")
        self.assertEqual(reader.tensor_reader("adaln_t_table"), bytes(1025 * 8 * 4))
        self.assertEqual(response.raw.tell(), payload_start + 1025 * 8 * 4)
        get.assert_called_once()
        self.assertTrue(response.closed)
        with self.assertRaisesRegex(CivitaiCheckpointError, "Only small"):
            reader.tensor_reader("blocks.0.attn.qkv_proj.weight")

    def test_gguf_download_requires_validator_and_publishes_only_after_it_passes(self):
        from test_h3_gguf_import import make_gguf
        blob, _ = make_gguf([("linear.weight", (3, 256), 12)])
        expected = source(filename="community.gguf", size_bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "community.gguf"
            path.write_bytes(b"existing verified file")
            with patch("requests.get") as get, self.assertRaisesRegex(CheckpointDownloadError, "architecture-specific"):
                download_named_checkpoint(expected, path, file_format="gguf")
            get.assert_not_called()
            def reject(candidate):
                self.assertEqual(path.read_bytes(), b"existing verified file")
                self.assertEqual(Path(candidate).read_bytes(), blob)
                raise CheckpointDownloadError("Wrong H3 layout")
            with patch("requests.get", return_value=Response(blob)), self.assertRaisesRegex(CheckpointDownloadError, "Wrong H3"):
                download_named_checkpoint(expected, path, file_format="gguf", validate_checkpoint=reject)
            self.assertEqual(path.read_bytes(), b"existing verified file")
            self.assertEqual(len(list(Path(directory).iterdir())), 1)
            with patch("requests.get", return_value=Response(blob)):
                result = download_named_checkpoint(expected, path, file_format="gguf", validate_checkpoint=lambda _: None)
            self.assertEqual(Path(result).read_bytes(), blob)

    def test_nonzero_ranges_require_matching_content_range(self):
        encoded = b'{"x":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}'
        reader = RemoteH3Header(source(size_bytes=8 + len(encoded) + 2),
                                get=lambda *a, **k: Response(struct.pack("<Q", len(encoded)) + encoded))
        with self.assertRaisesRegex(CivitaiCheckpointError, "bounded reads"):
            reader._open_range(30, 40)
        reader.get = lambda *a, **k: Response(status=206, headers={"Content-Range": "bytes 31-40/10000"})
        with self.assertRaisesRegex(CivitaiCheckpointError, "inconsistent"):
            reader._open_range(30, 40)

    def test_interrupted_header_read_has_a_sanitized_error(self):
        response = Response()
        response.raw = Mock()
        response.raw.read.side_effect = OSError("request to https://cdn.example/?token=secret-token failed")
        with self.assertRaisesRegex(CivitaiCheckpointError, "interrupted") as context:
            RemoteH3Header(source(), get=lambda *a, **k: response)
        self.assertNotIn("secret-token", str(context.exception))
        self.assertTrue(response.closed)

    def test_marker_reads_batch_nearby_descriptors_and_bound_distant_ranges(self):
        marker = b'{"format":"scaled_fp8"}'
        starts = (0, len(marker), 200000, 200000 + len(marker))
        header = {f"blocks.{index}.linear.comfy_quant":
                  {"dtype": "U8", "shape": [len(marker)], "data_offsets": [start, start + len(marker)]}
                  for index, start in enumerate(starts)}
        encoded = json.dumps(header).encode()
        payload_start = 8 + len(encoded)
        payload = bytearray(struct.pack("<Q", len(encoded)) + encoded + bytes(starts[-1] + len(marker)))
        for start in starts:
            payload[payload_start + start:payload_start + start + len(marker)] = marker
        ranges = []
        def get(_url, **kwargs):
            left, right = map(int, kwargs["headers"]["Range"].removeprefix("bytes=").split("-"))
            ranges.append((left, right))
            if left == 0:
                return Response(bytes(payload))
            return Response(bytes(payload[left:right + 1]), status=206,
                            headers={"Content-Range": f"bytes {left}-{right}/{len(payload)}"})
        reader = RemoteH3Header(source(size_bytes=len(payload)), get=get)
        for name in header:
            self.assertEqual(reader.tensor_reader(name), marker)
        self.assertEqual(len(ranges), 3, "one header and two marker spans; subsequent lookups use cached bytes")
        self.assertTrue(all(right - left + 1 <= 128 * 1024 for left, right in ranges if left > 0))

    def test_only_identical_existing_weights_are_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.safetensors"
            path.write_bytes(b"real content")
            expected = source(size_bytes=12, sha256=hashlib.sha256(b"real content").hexdigest())
            self.assertTrue(verify_h3_digest(path, expected))
            self.assertFalse(verify_h3_digest(path, {**expected, "sha256": "a" * 64}))
            self.assertFalse(verify_h3_digest(path, {**expected, "size_bytes": 999}))

    def test_architecture_validator_runs_before_atomic_publish(self):
        encoded = b'{"weight":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}'
        payload = struct.pack("<Q", len(encoded)) + encoded + b"ok"
        expected = source(size_bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.safetensors"
            path.write_bytes(b"previous valid checkpoint")
            def reject(candidate):
                self.assertEqual(Path(candidate).read_bytes(), payload)
                self.assertEqual(path.read_bytes(), b"previous valid checkpoint")
                raise CheckpointDownloadError("Wrong H3 architecture")
            with patch("requests.get", return_value=Response(payload)), self.assertRaisesRegex(CheckpointDownloadError, "Wrong H3"):
                download_named_checkpoint(expected, path, validate_checkpoint=reject)
            self.assertEqual(path.read_bytes(), b"previous valid checkpoint")
            self.assertEqual([item.name for item in Path(directory).iterdir()], [path.name])


class TestH3RegistrationAndRuntime(unittest.TestCase):
    def loader(self, metadata):
        transformer = Mock()
        transformer.eval.return_value.requires_grad_.return_value = transformer
        factory, offload = Mock(return_value=transformer), Mock()
        tree = ast.parse((ROOT / "app/models/minimax_h3/minimax_h3_main.py").read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_load_transformer")
        scope = {"__package__": "models.minimax_h3", "probe_h3_checkpoint": lambda _: metadata,
                 "_first_path": lambda value: value, "init_empty_weights": lambda **_: nullcontext(),
                 "MiniMaxH3Transformer": factory, "get_linear_split_map": lambda *a, **k: {"qkv": "split"},
                 "offload": offload, "partial": partial, "_strip_transformer_wrappers": lambda *a, **k: None}
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), "loader", "exec"), scope)
        return scope["_load_transformer"], factory, offload, transformer

    def test_loader_reverifies_profile_before_allocating_and_uses_native_lora_basis(self):
        metadata = profile(adaln_curve_grid=1025, time_embed_dim=8, convrot_group_size=256)
        loader, factory, offload, transformer = self.loader(metadata)
        with patch("services.civitai_checkpoints.inspect_local_h3", return_value=metadata):
            result = loader("local.safetensors", "bf16", qkv_layout="grouped", import_profile=metadata)
        self.assertEqual(result.h3_lora_model_type, "minimax_h3_ref2va")
        offload.split_linear_modules.assert_called_once()
        with patch("services.civitai_checkpoints.inspect_local_h3", return_value={**metadata, "time_embed_dim": 16}), self.assertRaisesRegex(ValueError, "no longer matches"):
            loader("local.safetensors", "bf16", qkv_layout="grouped", import_profile=metadata)
        self.assertEqual(factory.call_count, 1, "mismatch rejected before transformer allocation")

    def test_scaled_fp8_preserves_fused_grouped_projection(self):
        metadata = profile(adaln_curve_grid=1025, time_embed_dim=8, convrot_group_size=None, convrot=False, quantization_format="scaled_fp8")
        loader, _, offload, _ = self.loader(metadata)
        with patch("services.civitai_checkpoints.inspect_local_h3", return_value=metadata):
            loader("fp8.safetensors", "bf16", qkv_layout="grouped", import_profile=metadata)
        offload.split_linear_modules.assert_not_called()
        self.assertIsNone(offload.load_model_data.call_args.kwargs["fused_split_map"])

    def test_gguf_never_splits_packed_projection_rows_and_passes_selected_layout(self):
        for layout in ("grouped", "interleaved"):
            metadata = profile(adaln_curve_grid=1025, time_embed_dim=8, convrot=False,
                               convrot_group_size=None, quantization_format="gguf",
                               gguf_quant_types=["Q4_K"], qkv_layout=layout, qkv_layout_selection=layout)
            loader, _, offload, transformer = self.loader(metadata)
            with patch("services.civitai_checkpoints.inspect_local_h3", return_value=metadata) as inspector:
                loader("h3.gguf", "bf16", qkv_layout=layout, import_profile=metadata)
            offload.split_linear_modules.assert_not_called()
            self.assertIsNone(offload.load_model_data.call_args.kwargs["fused_split_map"])
            transformer.set_qkv_layout.assert_called_once_with(layout)
            self.assertEqual(inspector.call_args.kwargs["qkv_layout"], layout)

    def test_native_reference_checkpoint_gets_paired_workflows_and_no_stock_substitution(self):
        entries = build_h3_definitions(source(), profile(), "local.safetensors", ROOT / "app/defaults", auto_quantize=True)
        self.assertEqual(len(entries), 2)
        for slug, definition in entries.items():
            model = definition["model"]
            self.assertEqual(model["URLs"], ["local.safetensors"])
            self.assertFalse(model["auto_quantize"])
            self.assertEqual(model["compatible_model_paths"], {})
            self.assertEqual(model["minimax_h3_lora_workflow"], "ref2va")
            self.assertEqual(model["minimax_h3_model_id"], slug)
            self.assertEqual(set(model["h3_companion_models"].values()), set(entries))
            self.assertEqual(definition["num_inference_steps"], 8)
            self.assertEqual(definition["override_attention"], "")
            model_def = {**family_handler.query_model_def(model["architecture"], model), **model}
            self.assertTrue(model_def["sol_attention"])
            self.assertFalse(model_def["sla_attention"])
            self.assertFalse(model_def["sla_attention_default"])
            self.assertNotIn("token", json.dumps(definition))

    def test_version_and_file_ids_keep_standard_turbo_and_quants_distinct(self):
        standard = build_h3_definitions(source(), profile(), "local.safetensors", ROOT / "app/defaults")
        turbo = build_h3_definitions(source(versionId=24), profile(), "local.safetensors", ROOT / "app/defaults")
        quant = build_h3_definitions(source(fileId=35), profile(), "local.safetensors", ROOT / "app/defaults")
        self.assertFalse(set(standard) & set(turbo))
        self.assertFalse(set(standard) & set(quant))

    def test_full_fl2va_import_is_not_mislabeled_as_pruned_or_reference_capable(self):
        entries = build_h3_definitions(source(), profile(architectures=["minimax_h3_full"], compressed_modulation=False,
                    native_workflow="fl2va", qkv_layout="interleaved", quantization_format="none", convrot=False,
                    sampling_profile="standard", baked_turbo=False), "full.safetensors", ROOT / "app/defaults", auto_quantize=True)
        self.assertEqual(len(entries), 1)
        model = next(iter(entries.values()))["model"]
        self.assertEqual(model["architecture"], "minimax_h3_full")
        self.assertTrue(model["minimax_h3_full_checkpoint"])
        self.assertTrue(model["auto_quantize"])
        self.assertNotIn("references", model["h3_companion_models"])

    def test_baked_recipe_enforced_and_double_acceleration_rejected(self):
        body = {"num_inference_steps": 6, "minimax_h3_turbo_mode": True, "flow_shift": 99,
                "override_attention": "sol", "skip_steps_cache_type": "first_block"}
        normalize_imported_h3_request(body, profile())
        self.assertEqual((body["flow_shift"], body["audio_flow_shift"]), (9, 4))
        self.assertFalse(body["minimax_h3_turbo_mode"])
        self.assertEqual(body["override_attention"], "sol")
        self.assertEqual(body["skip_steps_cache_type"], "")
        with self.assertRaisesRegex(ValueError, "already includes acceleration"):
            normalize_imported_h3_request({"activated_loras": ["h3_turbo_lora.safetensors"]}, profile())
        for value in (True, 3, 9, 4.5, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_imported_h3_steps(value, profile())

    def test_imported_baked_attention_preserves_dense_modes_and_sol_but_clears_sla(self):
        for attention in ("", "auto", "sdpa", "sage", "sage2", "flash", "sol"):
            with self.subTest(attention=attention):
                body = {"num_inference_steps": 6, "override_attention": attention,
                        "skip_steps_cache_type": "first_block"}
                normalize_imported_h3_request(body, profile())
                self.assertEqual(body["override_attention"], attention)
                self.assertEqual(body["num_inference_steps"], 6)
                self.assertEqual(body["skip_steps_cache_type"], "")
        body = {"num_inference_steps": 6, "override_attention": "sla"}
        normalize_imported_h3_request(body, profile())
        self.assertEqual(body["override_attention"], "")

    def test_imported_settings_migration_preserves_sol_and_clears_stale_sla(self):
        entry = next(iter(build_h3_definitions(
            source(), profile(), "local.safetensors", ROOT / "app/defaults"
        ).values()))
        model = entry["model"]
        model_def = {**family_handler.query_model_def(model["architecture"], model), **model}
        for attention, expected in (("", ""), ("auto", "auto"), ("sdpa", "sdpa"),
                                    ("sage", "sage"), ("sage2", "sage2"),
                                    ("flash", "flash"), ("sol", "sol"), ("sla", "")):
            with self.subTest(attention=attention):
                settings = dict(entry)
                settings["num_inference_steps"] = 6
                settings["override_attention"] = attention
                settings["skip_steps_cache_type"] = "first_block"
                family_handler.fix_settings(model["architecture"], 2.58, model_def, settings)
                self.assertEqual(settings["override_attention"], expected)
                self.assertEqual(settings["num_inference_steps"], 6)
                self.assertEqual(settings["skip_steps_cache_type"], "")

    def test_imported_generation_validation_preserves_sol_and_clears_sla(self):
        entry = next(iter(build_h3_definitions(
            source(), profile(), "local.safetensors", ROOT / "app/defaults"
        ).values()))
        model = entry["model"]
        model_def = {**family_handler.query_model_def(model["architecture"], model), **model}
        for attention, expected in (("", ""), ("auto", "auto"), ("sdpa", "sdpa"),
                                    ("sol", "sol"), ("sla", "")):
            with self.subTest(attention=attention):
                request = {
                    "num_inference_steps": 6,
                    "override_attention": attention,
                    "minimax_h3_turbo_mode": True,
                    "minimax_h3_turbo_preset": "legacy",
                    "skip_steps_cache_type": "first_block",
                    "video_length": 65,
                    "sliding_window_size": 65,
                    "resolution": "1280x704",
                }
                error = family_handler.validate_generative_settings(
                    model["architecture"], model_def, request
                )
                self.assertIsNone(error)
                self.assertEqual(request["override_attention"], expected)
                self.assertEqual(request["num_inference_steps"], 6)
                self.assertFalse(request["minimax_h3_turbo_mode"])
                self.assertEqual(request["minimax_h3_turbo_preset"], "")
                self.assertEqual(request["skip_steps_cache_type"], "")


if __name__ == "__main__":
    unittest.main()
