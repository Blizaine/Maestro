"""Durability, isolation, and recovery contracts for agent access jobs."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest


_APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(_APP))

from services.agent_access.jobs import (  # noqa: E402
    JobRegistry,
    JournalError,
    SubmissionJournal,
)
from services.agent_access.context import submission_context  # noqa: E402


class SubmissionJournalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.now = 1000.0

    def journal(self, **kwargs):
        return SubmissionJournal(self.directory, clock=lambda: self.now, **kwargs)

    def reserve(self, journal, *, principal="actor-a", key="key-a", payload=None):
        return journal.reserve(
            principal, key, "POST", "/api/agent/do", payload or {"prompt": "go"}, "mode=fast"
        )

    def test_parallel_reserve_has_one_new_winner(self):
        journal = self.journal()

        def attempt(_):
            return self.reserve(journal)

        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(attempt, range(32)))
        self.assertEqual(sum(item["state"] == "new" for item in results), 1)
        self.assertEqual(len({item["reservation_id"] for item in results}), 1)
        self.assertEqual(sum(item["state"] == "pending" for item in results), 31)

    def test_reusing_key_with_changed_arguments_conflicts(self):
        journal = self.journal()
        self.reserve(journal)
        with self.assertRaises(JournalError) as raised:
            self.reserve(journal, payload={"prompt": "different"})
        self.assertEqual(raised.exception.status, 409)

    def test_reservations_and_lookups_are_principal_scoped(self):
        journal = self.journal()
        first = self.reserve(journal, principal="actor-a")
        second = self.reserve(journal, principal="actor-b")
        self.assertNotEqual(first["reservation_id"], second["reservation_id"])
        self.assertEqual(first["state"], "new")
        self.assertEqual(second["state"], "new")
        self.assertIsNone(journal.lookup("actor-b", first["reservation_id"]))
        self.assertEqual(len(journal.list_reservations("actor-a")), 1)
        self.assertEqual(len(journal.list_reservations("actor-b")), 1)

    def test_completed_receipt_replays_after_restart_with_only_safe_headers(self):
        journal = self.journal()
        reserved = self.reserve(journal)
        receipt = journal.complete(
            reserved["reservation_id"],
            201,
            b'{"job_id":"job-7"}',
            [
                ("Content-Type", "application/json"),
                ("Cache-Control", "no-store"),
                ("Set-Cookie", "session=private"),
            ],
        )
        self.assertEqual(receipt["body"], b'{"job_id":"job-7"}')
        self.assertEqual(receipt["headers"], [
            ["cache-control", "no-store"],
            ["content-type", "application/json"],
        ])

        restarted = self.journal()
        replay = self.reserve(restarted)
        self.assertEqual(replay["state"], "complete")
        self.assertEqual(replay["reservation_id"], reserved["reservation_id"])
        self.assertEqual(replay["status_code"], 201)
        self.assertEqual(replay["body"], b'{"job_id":"job-7"}')
        self.assertEqual(replay["headers"], receipt["headers"])

    def test_find_submission_returns_json_receipt_without_request_or_secret_data(self):
        journal = self.journal()
        request_id = "client-request-secret"
        reserved = journal.reserve(
            "actor-a", request_id, "POST", "/api/agent/do",
            {"prompt": "private prompt", "api_key": "request-secret"},
        )
        journal.complete(
            reserved["reservation_id"], 200,
            b'{"job_id":"job-7","status":"queued"}',
            [("Content-Type", "application/json"), ("Set-Cookie", "private-cookie")],
        )

        found = self.journal().find_submission("actor-a", request_id)
        self.assertEqual(found["reservation_id"], reserved["reservation_id"])
        self.assertEqual(found["state"], "complete")
        self.assertEqual(found["status_code"], 200)
        self.assertEqual(found["result"], {"job_id": "job-7", "status": "queued"})
        self.assertTrue(found["result_available"])
        self.assertEqual(found["headers"], [["content-type", "application/json"]])
        self.assertNotIn("body", found)
        self.assertNotIn("request_hash", found)
        self.assertNotIn("key_hash", found)
        self.assertNotIn("private prompt", str(found))
        self.assertNotIn("request-secret", str(found))
        self.assertNotIn("client-request-secret", str(found))
        self.assertIsNone(self.journal().find_submission("actor-b", request_id))

    def test_find_submission_keeps_pending_and_uncertain_non_dispatchable(self):
        journal = self.journal()
        reserved = self.reserve(journal, key="recover-me")
        self.assertEqual(journal.find_submission("actor-a", "recover-me")["state"], "pending")
        uncertain = journal.fail_uncertain(reserved["reservation_id"], "dispatch outcome unknown")
        found = journal.find_submission("actor-a", "recover-me")
        self.assertEqual(found["state"], "uncertain")
        self.assertEqual(found["reason"], uncertain["reason"])
        self.assertNotIn("result", found)
        with self.assertRaises(JournalError) as too_long:
            journal.find_submission("actor-a", "x" * 8193)
        self.assertEqual(too_long.exception.status, 400)

    def test_find_submission_never_returns_non_json_cached_bytes(self):
        journal = self.journal()
        reserved = self.reserve(journal, key="binary-response")
        journal.complete(reserved["reservation_id"], 200, b"private binary payload")
        found = journal.find_submission("actor-a", "binary-response")
        self.assertEqual(found["state"], "complete")
        self.assertFalse(found["result_available"])
        self.assertNotIn("result", found)
        self.assertNotIn("body", found)
        self.assertNotIn("private binary payload", str(found))

    def test_uncertain_submission_recovers_owned_job_after_restart_without_redispatch(self):
        journal = self.journal()
        key = "lost-http-response"
        reserved = self.reserve(journal, key=key)
        dispatch_count = 0

        def fake_generation():
            nonlocal dispatch_count
            dispatch_count += 1
            journal.record_job(
                {"id": "job-after-lost-response", "status": "queued"},
                "actor-a", "session-a",
            )
            raise ConnectionError("simulated client connection failure")

        context_token = submission_context.set(reserved["reservation_id"])
        try:
            with self.assertRaises(ConnectionError):
                fake_generation()
        finally:
            submission_context.reset(context_token)
        journal.fail_uncertain(reserved["reservation_id"], "HTTP response was lost")

        restarted = self.journal()
        receipt = restarted.find_submission("actor-a", key)
        self.assertEqual(receipt["state"], "uncertain")
        self.assertEqual(receipt["resources"], [{"kind": "job", "id": "job-after-lost-response"}])
        self.assertEqual(
            restarted.get_job("job-after-lost-response", "actor-a")["snapshot"]["status"],
            "queued",
        )
        self.assertIsNone(restarted.get_job("job-after-lost-response", "actor-b"))

        retry = restarted.reserve(
            "actor-a", key, "POST", "/api/agent/do", {"prompt": "go"}, "mode=fast"
        )
        if retry["state"] == "new":
            fake_generation()
        self.assertEqual(retry["state"], "uncertain")
        self.assertEqual(dispatch_count, 1)

    def test_foreign_submission_context_cannot_link_or_reveal_resources(self):
        journal = self.journal()
        foreign = journal.reserve(
            "actor-b", "actor-b-key", "POST", "/api/agent/do", {"prompt": "b"}
        )
        context_token = submission_context.set(foreign["reservation_id"])
        try:
            journal.record_job({"id": "actor-a-job", "status": "running"}, "actor-a", "session-a")
            journal.register_resource_owner("pipeline", "actor-a-pipeline", "actor-a", "session-a")
        finally:
            submission_context.reset(context_token)

        found = journal.find_submission("actor-b", "actor-b-key")
        self.assertEqual(found["resources"], [])
        self.assertIsNone(journal.find_submission("actor-a", "actor-b-key"))
        self.assertIsNone(journal.get_job("actor-a-job", "actor-b"))
        self.assertIsNotNone(journal.get_job("actor-a-job", "actor-a"))

    def test_resource_links_are_idempotent_and_keep_the_first_submission(self):
        journal = self.journal()
        first = self.reserve(journal, key="first-resource-request")
        second = self.reserve(journal, key="second-resource-request")
        journal.record_job({"id": "preexisting-job", "status": "queued"}, "actor-a", "session-a")

        context_token = submission_context.set(first["reservation_id"])
        try:
            journal.record_job({"id": "stable-job", "status": "queued"}, "actor-a", "session-a")
            journal.register_resource_owner("pipeline", "stable-pipeline", "actor-a", "session-a")
        finally:
            submission_context.reset(context_token)

        context_token = submission_context.set(second["reservation_id"])
        try:
            journal.record_job({"id": "stable-job", "status": "running"}, "actor-a", "session-a")
            journal.record_job({"id": "preexisting-job", "status": "running"}, "actor-a", "session-a")
            journal.register_resource_owner("pipeline", "stable-pipeline", "actor-a", "session-a")
        finally:
            submission_context.reset(context_token)

        first_found = journal.find_submission("actor-a", "first-resource-request")
        second_found = journal.find_submission("actor-a", "second-resource-request")
        self.assertEqual(first_found["resources"], [
            {"kind": "job", "id": "stable-job"},
            {"kind": "pipeline", "id": "stable-pipeline"},
        ])
        self.assertEqual(second_found["resources"], [])

    def test_pending_and_uncertain_survive_restart_without_a_second_dispatch(self):
        journal = self.journal()
        dispatches = 0
        first = self.reserve(journal)
        if first["state"] == "new":
            dispatches += 1

        restarted = self.journal()
        pending = self.reserve(restarted)
        if pending["state"] == "new":
            dispatches += 1
        self.assertEqual(pending["state"], "pending")
        self.assertEqual(dispatches, 1)

        uncertain = restarted.fail_uncertain(
            pending["reservation_id"], "backend result was not observed"
        )
        self.assertEqual(uncertain["state"], "uncertain")
        again = self.reserve(self.journal())
        if again["state"] == "new":
            dispatches += 1
        self.assertEqual(again["state"], "uncertain")
        self.assertEqual(dispatches, 1)

    def test_request_body_and_key_are_never_stored_as_plaintext(self):
        journal = self.journal()
        secret_body = "private-prompt-should-not-be-persisted"
        secret_key = "secret-idempotency-key"
        self.reserve(journal, key=secret_key, payload={"prompt": secret_body})

        connection = sqlite3.connect(journal.database_path)
        try:
            row = connection.execute(
                "SELECT key_hash, request_hash FROM submissions"
            ).fetchone()
        finally:
            connection.close()
        self.assertTrue(all(len(value) == 64 for value in row))
        for path in self.directory.iterdir():
            if path.is_file():
                contents = path.read_bytes()
                self.assertNotIn(secret_body.encode(), contents)
                self.assertNotIn(secret_key.encode(), contents)

    def test_key_and_receipt_bounds_and_entry_quota(self):
        journal = self.journal(max_entries=1, max_key_bytes=8, max_response_bytes=4)
        with self.assertRaises(JournalError) as key_error:
            self.reserve(journal, key="too-long-key")
        self.assertEqual(key_error.exception.status, 400)

        first = self.reserve(journal, key="short")
        with self.assertRaises(JournalError) as body_error:
            journal.complete(first["reservation_id"], 200, b"12345")
        self.assertEqual(body_error.exception.status, 413)

        with self.assertRaises(JournalError) as quota_error:
            self.reserve(journal, key="other")
        self.assertEqual(quota_error.exception.status, 429)


class JobSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.now = 1000.0
        self.journal = SubmissionJournal(
            Path(self.temp.name), clock=lambda: self.now
        )

    def test_metadata_is_whitelisted_json_safe_and_owner_is_immutable(self):
        job = {
            "id": "job-1",
            "status": "running",
            "progress": 0.5,
            "output_files": ["result.mp4"],
            "output_metadata": {
                "filename": "result.mp4",
                "tensor": object(),
                "weight_path": "private/model.bin",
            },
            "params": {"api_key": "request-secret"},
            "model_state": object(),
            "principal_id": "forged-owner",
            "session_id": "forged-session",
        }
        record = self.journal.record_job(job, "actor-a", "session-a")
        snapshot = record["snapshot"]
        self.assertEqual(record["principal_id"], "actor-a")
        self.assertEqual(record["session_id"], "session-a")
        self.assertEqual(snapshot["output_files"], ["result.mp4"])
        self.assertEqual(snapshot["output_metadata"], {"filename": "result.mp4"})
        self.assertNotIn("params", snapshot)
        self.assertNotIn("model_state", snapshot)
        self.assertEqual(self.journal.get_job("job-1", "actor-a"), record)
        self.assertIsNone(self.journal.get_job("job-1", "actor-b"))

        updated = dict(job, progress=0.9, session_id="attacker-session")
        updated_record = self.journal.record_job(updated, "actor-a", "session-a")
        self.assertEqual(updated_record["principal_id"], "actor-a")
        self.assertEqual(updated_record["session_id"], "session-a")
        self.assertEqual(updated_record["snapshot"]["progress"], 0.9)

        for principal, session in (("actor-b", "session-a"), ("actor-a", "session-b")):
            with self.subTest(principal=principal, session=session):
                with self.assertRaises(JournalError) as raised:
                    self.journal.record_job(updated, principal, session)
                self.assertEqual(raised.exception.status, 409)

    def test_job_metadata_size_is_bounded(self):
        journal = SubmissionJournal(
            Path(self.journal.directory), max_job_snapshot_bytes=64
        )
        with self.assertRaises(JournalError) as raised:
            journal.record_job(
                {"id": "large", "message": "x" * 4096}, "actor", "session"
            )
        self.assertEqual(raised.exception.status, 413)

    def test_job_records_share_the_persistent_entry_quota(self):
        journal = SubmissionJournal(Path(self.journal.directory), max_entries=1)
        journal.record_job({"id": "job-1", "status": "queued"}, "actor", "session")
        with self.assertRaises(JournalError) as raised:
            journal.reserve("actor", "key", "POST", "/submit", {}, "")
        self.assertEqual(raised.exception.status, 429)

    def test_verified_shutdown_interrupts_only_nonterminal_jobs(self):
        active = self.journal.record_job(
            {"id": "active", "status": "running", "message": "working"},
            "actor-a",
            "session-a",
        )
        terminal = self.journal.record_job(
            {"id": "done", "status": "completed", "output_files": ["done.mp4"]},
            "actor-a",
            "session-a",
        )
        with self.assertRaises(JournalError):
            self.journal.mark_session_interrupted("session-a")
        self.now = 2000.0
        self.assertEqual(
            self.journal.mark_session_interrupted("session-a", verified_stopped=True), 1
        )
        recovered = self.journal.get_job("active", "actor-a")
        unchanged = self.journal.get_job("done", "actor-a")
        self.assertEqual(recovered["snapshot"]["status"], "interrupted")
        self.assertEqual(recovered["snapshot"]["message"], "Backend stopped before job completion")
        self.assertEqual(unchanged, terminal)
        self.assertEqual(active["snapshot"]["status"], "running")

    def test_registry_captures_owner_once_and_persists_only_explicitly(self):
        current = {"principal_id": "actor-a", "session_id": "session-a"}
        registry = JobRegistry(
            current_principal=lambda: current["principal_id"],
            current_session=lambda: current["session_id"],
        )
        registry["job-1"] = {"id": "job-1", "status": "queued"}
        current.update(principal_id="actor-b", session_id="session-b")
        registry["job-1"] = {"id": "job-1", "status": "running"}
        registry["job-2"] = {"id": "job-2", "status": "queued"}

        self.assertEqual(registry.ownership_for("job-1"), {
            "principal_id": "actor-a", "session_id": "session-a"
        })
        self.assertEqual(registry.ownership_for("job-2"), {
            "principal_id": "actor-b", "session_id": "session-b"
        })
        records = registry.record_all(self.journal)
        self.assertEqual(len(records), 2)
        self.assertEqual(self.journal.get_job("job-1", "actor-a")["session_id"], "session-a")


if __name__ == "__main__":
    unittest.main()
