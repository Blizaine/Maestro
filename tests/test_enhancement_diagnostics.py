"""Writer evidence survives repairs/failures without inference or credentials."""
import json
import ast
import asyncio
from functools import wraps
from pathlib import Path
import sys
import tempfile
import threading
import time
import uuid
import unittest
from types import SimpleNamespace
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from promptbench.trace import record_payload, record_response, traced
from services.enhancement_diagnostics import DraftHistoryArchive, capture_drafts, record_advisory, record_rejection
from services.studio_enhancement import (
    StudioJobArchive, enhancement_context, new_enhancement, public_enhancement,
)


class DraftHistoryTests(unittest.TestCase):
    def test_auxiliary_review_is_saved_without_rejecting_usable_raw_reply(self):
        @traced
        def writer():
            return '{"edits":[]}'
        with tempfile.TemporaryDirectory() as directory:
            archive = DraftHistoryArchive(directory)
            with capture_drafts(on_change=lambda drafts: archive.save("abc123", {"drafts": drafts})):
                writer()
                details = {"applied": 8, "skipped_reasons": ["extra suggestion"]}
                record_advisory("fish_performance", details)
                details["skipped_reasons"].append("later mutation")
            draft = archive.load("abc123")["drafts"][0]
        self.assertEqual(draft["status"], "received")
        self.assertEqual(draft["response_text"], '{"edits":[]}')
        self.assertEqual(draft["advisories"]["fish_performance"]["skipped_reasons"], ["extra suggestion"])

    def test_rejected_draft_is_not_overwritten_by_repair_and_survives_reload(self):
        @traced
        def stream(text):
            record_payload({"messages": [{"role": "user", "content": [
                {"type": "text", "text": "A source-directed brief"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,PRIVATE"}},
            ]}], "api_key": "SECRET"}, model_id="local-writer")
            record_response(text, "reasoning", complete=True)
            return text

        @traced
        def generate(text):
            return stream(text)

        with tempfile.TemporaryDirectory() as directory:
            archive = DraftHistoryArchive(directory)
            with capture_drafts(on_change=lambda drafts: archive.save("abc123", {"drafts": drafts})) as drafts:
                generate("First draft")
                record_rejection(["Unknown reference <Picture 9>."], stage="reference validation")
                generate("Repaired draft")
            loaded = DraftHistoryArchive(directory).load("abc123")["drafts"]
            self.assertEqual(len(loaded), 2)  # generate->stream delegation is one call.
            self.assertEqual(loaded[0]["response_text"], "First draft")
            self.assertEqual(loaded[0]["status"], "rejected")
            self.assertEqual(loaded[0]["rejections"][0]["reasons"], ["Unknown reference <Picture 9>."])
            self.assertEqual(loaded[1]["response_text"], "Repaired draft")
            self.assertEqual(loaded[1]["status"], "received")
            self.assertEqual(drafts, loaded)
            serialized = json.dumps(loaded)
            self.assertNotIn("SECRET", serialized)
            self.assertNotIn("PRIVATE", serialized)
            self.assertIn("source-directed brief", serialized)

    def test_interrupted_response_is_saved_with_error_and_partial_text(self):
        @traced
        def stream():
            record_response("partial draft", complete=False)
            raise InterruptedError("Writer cancelled")
        with capture_drafts() as drafts:
            with self.assertRaises(InterruptedError):
                stream()
        self.assertEqual(drafts[0]["response_text"], "partial draft")
        self.assertFalse(drafts[0]["response_complete"])
        self.assertEqual(drafts[0]["status"], "error")
        self.assertEqual(drafts[0]["error"], "Writer cancelled")

    def test_job_checkpoint_and_public_detail_keep_rejected_drafts(self):
        @traced
        def writer():
            return "A draft with an unmapped subject"
        with tempfile.TemporaryDirectory() as directory:
            archive = StudioJobArchive(directory)
            record = new_enhancement({"prompt": "Original brief"}, {})
            job = {"id": "abc123", "status": "failed", "enhancement": record}
            def save(drafts):
                record["drafts"] = drafts
                archive.save(job)
            with enhancement_context({}, lambda: False, on_drafts=save):
                writer()
                record_rejection(["Subject 2 is not supplied."], stage="reference validation")
            restored = archive.recover()["abc123"]["enhancement"]
            self.assertEqual(restored["drafts"][0]["status"], "rejected")
            self.assertNotIn("drafts", public_enhancement(restored, summary=True))
            self.assertEqual(public_enhancement(restored, include_drafts=True)["drafts"], restored["drafts"])

    def test_contexts_do_not_mix_jobs_and_resume_keeps_prior_attempts(self):
        @traced
        def writer(text):
            return text
        with capture_drafts() as first:
            writer("First job")
        with capture_drafts() as second:
            writer("Second job")
        with capture_drafts(first) as resumed:
            writer("First job retry")
        self.assertEqual([x["response_text"] for x in first], ["First job"])
        self.assertEqual([x["response_text"] for x in second], ["Second job"])
        self.assertEqual([x["index"] for x in resumed], [1, 2])

    def test_archive_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = DraftHistoryArchive(directory)
            for value in ("../outside", "", "C:/outside", "a/b"):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    archive.load(value)

    def test_interactive_failure_keeps_drafts_and_releases_resource_lock(self):
        source = Path(__file__).resolve().parents[1] / "app" / "launch.py"
        node = next(item for item in ast.parse(source.read_text(encoding="utf-8")).body
                    if isinstance(item, ast.FunctionDef) and item.name == "_interactive_enhancement_slot")
        with tempfile.TemporaryDirectory() as directory:
            lease = threading.Lock()
            namespace = {"wraps": wraps, "Request": object, "asyncio": asyncio, "Path": Path,
                         "uuid": uuid, "time": time, "HTTPException": HTTPException,
                         "_app_dir": directory, "_gen_lock": lease,
                         "_guard_interactive_llm_against_generation": lambda: None,
                         "_enhancement_settings_snapshot": lambda: {},
                         "enhancement_context": enhancement_context,
                         "wgp": SimpleNamespace(wan_model=None, offloadobj=None)}
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
            @traced
            def writer():
                record_response("Rejected response from the writer")
                return "Rejected response from the writer"
            async def endpoint(_request):
                writer()
                raise HTTPException(400, "Reference <Picture 8> is not supplied.")
            async def request_json():
                return {"prompt": "A source-directed scene"}
            wrapped = namespace["_interactive_enhancement_slot"](endpoint)
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(wrapped(SimpleNamespace(json=request_json)))
            self.assertEqual(raised.exception.status_code, 400)
            history_id = raised.exception.headers["X-Maestro-Draft-History"]
            saved = DraftHistoryArchive(Path(directory) / "settings" / "llm_drafts").load(history_id)
            self.assertEqual(saved["state"], "failed")
            self.assertEqual(saved["drafts"][0]["response_text"], "Rejected response from the writer")
            self.assertIn("Picture 8", saved["drafts"][0]["rejections"][0]["reasons"][0])
            self.assertFalse(lease.locked())

    def test_director_history_read_uses_saved_snapshot_after_restart(self):
        source = Path(__file__).resolve().parents[1] / "app" / "services" / "director_pipeline.py"
        node = next(item for item in ast.parse(source.read_text(encoding="utf-8")).body
                    if isinstance(item, ast.FunctionDef) and item.name == "get_pipeline_drafts")
        saved = {"llm_drafts": [{"response_text": "A saved rejected draft"}]}
        namespace = {"Optional": __import__('typing').Optional, "copy": __import__('copy'),
                     "get_pipeline": lambda _pid: None,
                     "load_pipeline_state": lambda _root, _pid: saved}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        result = namespace["get_pipeline_drafts"]("abc123", "outputs")
        self.assertEqual(result, saved["llm_drafts"])
        result.clear()
        self.assertEqual(len(saved["llm_drafts"]), 1)


if __name__ == "__main__":
    unittest.main()
