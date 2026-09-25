import ast
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
import wave
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / "app"))
from PIL import Image
from services.media_info import enrich_metadata, probe_media, processing_record, record_upload


class MediaInfoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.png"
        self.output = self.root / "output.png"
        Image.new("RGB", (60, 40)).save(self.source)
        Image.new("RGB", (120, 80)).save(self.output)

    def test_reads_actual_dimensions_and_refreshes_after_replace(self):
        self.assertEqual(probe_media(self.output)["width"], 120)
        Image.new("RGB", (180, 120)).save(self.output)
        self.assertEqual(probe_media(self.output)["width"], 180)
        self.assertEqual(probe_media(self.output)["size_bytes"], self.output.stat().st_size)

    def test_audio_and_unsupported_files_keep_facts(self):
        audio = self.root / "sample.wav"
        with wave.open(str(audio), "wb") as stream:
            stream.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
            stream.writeframes(b"\x00\x00" * 8000)
        facts = probe_media(audio)
        self.assertAlmostEqual(facts["duration_seconds"], 1)
        self.assertNotIn("width", facts)
        self.output.write_bytes(b"not an image yet")
        self.assertEqual(probe_media(self.output), {"size_bytes": len(b"not an image yet")})

    def test_processing_records_exact_measured_geometry_and_relevant_options(self):
        record = processing_record(spatial="dlss5*1.724", temporal="rife2",
            before={"width": 610, "height": 352, "fps": 24},
            after={"width": 1050, "height": 606, "fps": 48}, elapsed=3.51234,
            options={"dlss_intensity": 0.7, "prompt": "do not store", "dlss_depth": "half"})
        self.assertEqual(record["multiplier"], 1.724)
        self.assertEqual(record["output"]["width"], 1050)
        self.assertEqual(record["frame_multiplier"], 2)
        self.assertEqual(record["elapsed_seconds"], 3.512)
        self.assertNotIn("prompt", record["options"])
        self.assertEqual(processing_record(spatial="dlss5*1")["multiplier"], 1)
        self.assertEqual(processing_record(spatial="flashvsr2pass4")["multiplier"], 4)

    def legacy(self):
        return {"tool": "media_flow", "params": {"spatial_upsampling": "dlss5*2", "media_path": str(self.source)},
                "tool_source": self.source.name, "created_at": 1700000000, "generation_time": 25}

    def test_legacy_tools_recovers_source_and_marks_processed_time(self):
        info = enrich_metadata(self.output, self.legacy(), source_roots=(str(self.root),))
        self.assertEqual(info["processing"]["input"]["width"], 60)
        self.assertEqual(info["processing"]["output"]["width"], 120)
        self.assertEqual(info["timestamp"], {"value": 1700000000, "kind": "processed"})
        self.source.unlink()
        missing = enrich_metadata(self.output, self.legacy(), source_roots=(str(self.root),))
        self.assertNotIn("input", missing["processing"])
        self.assertEqual(missing["processing"]["output"]["width"], 120)

    def test_does_not_probe_unrelated_source_or_claim_requested_finishing_succeeded(self):
        legacy = self.legacy()
        info = enrich_metadata(self.output, legacy, source_roots=(str(self.root / "elsewhere"),))
        self.assertNotIn("input", info["processing"])
        generated = enrich_metadata(self.output, {"params": {"spatial_upsampling": "dlss5*2"}})
        self.assertNotIn("processing", generated)
        self.assertEqual(generated["timestamp"]["kind"], "file_modified")

    def test_timestamp_provenance_and_upload_date_reuse(self):
        record_upload(str(self.source), "my-photo.png")
        sidecar = self.source.with_suffix(".meta.json")
        first = json.loads(sidecar.read_text())
        record_upload(str(self.source), "different-name.png")
        self.assertEqual(json.loads(sidecar.read_text()), first)
        info = enrich_metadata(self.source, first)
        self.assertEqual(info["timestamp"]["kind"], "uploaded")
        self.assertEqual(info["timestamp"]["value"], first["uploaded_at"])
        info = enrich_metadata(self.output, {"params": {"creation_timestamp": 1700000001}})
        self.assertEqual(info["timestamp"], {"value": 1700000001, "kind": "generated"})

    def test_tools_sidecar_survives_source_deletion(self):
        tree = ast.parse((Path(__file__).parents[1] / "app/launch.py").read_text(encoding="utf-8"))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_write_tool_sidecar")
        namespace = {"os": os, "time": time, "json": json}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "launch.py", "exec"), namespace)
        namespace["_write_tool_sidecar"](str(self.root), self.output.name, source_name=self.source.name,
            source_path=str(self.source), tool="upscale", params={"method": "lanczos2"}, elapsed=1.25, job_id="test", media_type="image")
        metadata = json.loads(self.output.with_suffix(".meta.json").read_text())
        self.source.unlink()
        enriched = enrich_metadata(self.output, metadata)
        self.assertEqual(enriched["processing"]["input"]["width"], 60)
        self.assertEqual(enriched["processing"]["output"]["width"], 120)
        self.assertEqual(enriched["processing"]["elapsed_seconds"], 1.25)

    def test_upload_timestamp_sidecar_preserves_imported_generation_metadata(self):
        record_upload(str(self.output), "export.png")
        tree = ast.parse((Path(__file__).parents[1] / "app/launch.py").read_text(encoding="utf-8"))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_output_metadata")
        function.decorator_list = []
        namespace = {"os": os, "json": json, "_gallery_directory": lambda ws: str(self.root),
                     "_safe_join": os.path.join, "_extract_output_seed": lambda name: None}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "launch.py", "exec"), namespace)
        embedded = {"prompt": "original description", "seed": 17, "creation_timestamp": 1600000000}
        with patch.dict(sys.modules, {"shared.utils.audio_video": SimpleNamespace(read_image_metadata=lambda path: embedded)}):
            info = namespace["get_output_metadata"](self.output.name, "__uploads__")
        self.assertEqual(info["params"], embedded)
        self.assertEqual(info["timestamp"]["kind"], "uploaded")
        self.assertGreater(info["timestamp"]["value"], 1600000000)

    def test_processing_timestamp_takes_precedence_over_inherited_source_date(self):
        info = enrich_metadata(self.output, {"params": {"creation_timestamp": 1600000000,
            "processing": {"method": "lanczos2", "completed_at": 1700000000}}})
        self.assertEqual(info["timestamp"], {"value": 1700000000, "kind": "processed"})


if __name__ == "__main__":
    unittest.main()
