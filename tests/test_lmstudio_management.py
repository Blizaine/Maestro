"""External memory release uses LM Studio's native contract, never unload-all."""

from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import requests
from services import llm_service, lmstudio_management as management


def model_list(*identifiers, key="writer"):
    return {"models": [
        {"type": "llm", "key": key, "display_name": "Writer",
         "loaded_instances": [{"id": identifier} for identifier in identifiers]},
        {"type": "embedding", "key": "embedding", "loaded_instances": [{"id": "embedding-instance"}]},
    ]}


def response(payload, status=200):
    result = mock.Mock(status_code=status)
    result.json.return_value = payload
    return result


class LmStudioManagementTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(requests, "request")
        self.request = patcher.start()
        self.addCleanup(patcher.stop)

    def test_unloads_only_selected_instance_and_verifies_native_state(self):
        self.request.side_effect = [
            response(model_list("first", "second")),
            response({"instance_id": "second"}),
            response(model_list("first")),
        ]
        result = management.unload_model(
            "http://localhost:1234/proxy/v1/", "remote-secret", instance_id="second",
        )
        self.assertEqual(result["status"], "unloaded")
        self.assertEqual(result["instance_id"], "second")
        self.assertEqual(result["server_url"], "http://localhost:1234/proxy")
        calls = self.request.call_args_list
        self.assertEqual([call.args[:2] for call in calls], [
            ("GET", "http://localhost:1234/proxy/api/v1/models"),
            ("POST", "http://localhost:1234/proxy/api/v1/models/unload"),
            ("GET", "http://localhost:1234/proxy/api/v1/models"),
        ])
        self.assertEqual(calls[1].kwargs["json"], {"instance_id": "second"})
        for call in calls:
            self.assertEqual(call.kwargs["headers"], {"Authorization": "Bearer remote-secret"})
            self.assertFalse(call.kwargs["allow_redirects"])
            self.assertEqual(call.kwargs["timeout"], (5, 30))

    def test_ambiguous_or_unknown_model_never_posts(self):
        for key in ("writer", "unknown"):
            with self.subTest(key=key):
                self.request.reset_mock()
                self.request.return_value = response(model_list("first", "second"))
                with self.assertRaises(management.ModelSelectionError):
                    management.unload_model("http://localhost:1234", model_id=key)
                self.assertEqual(self.request.call_count, 1)

    def test_configured_model_can_resolve_one_instance(self):
        self.request.side_effect = [
            response(model_list("selected")), response({"instance_id": "selected"}),
            response(model_list()),
        ]
        self.assertEqual(management.unload_model("http://localhost:1234", model_id="writer")["status"], "unloaded")

    def test_already_unloaded_is_idempotent_and_does_not_claim_another_model(self):
        self.request.return_value = response(model_list("other"))
        result = management.unload_model("http://localhost:1234", instance_id="gone", model_id="writer")
        self.assertEqual(result["status"], "not_loaded")
        self.assertIsNone(result["model_key"])
        self.assertEqual(self.request.call_count, 1)
        self.request.return_value = response(model_list())
        self.assertEqual(management.unload_model("http://localhost:1234", model_id="writer")["model_key"], "writer")

    def test_invalid_target_cannot_make_a_management_request(self):
        for url in ("", "file:///tmp/server", "http://user:secret@localhost:1234",
                    "http://localhost:1234?token=secret", "http://localhost:1234#secret",
                    "http://[bad", "http://localhost:99999"):
            with self.subTest(url=url):
                with self.assertRaises(management.ModelSelectionError):
                    management.loaded_models(url)
        for identifier in ("", " ", 42):
            with self.subTest(identifier=identifier):
                with self.assertRaises(management.ModelSelectionError):
                    management.unload_model("http://localhost:1234", instance_id=identifier)
        with self.assertRaises(management.ModelSelectionError):
            management.unload_model("http://localhost:1234")
        self.request.assert_not_called()

    def test_authentication_and_unsupported_api_errors_are_truthful(self):
        for status, message in ((401, "authentication"), (404, "native"), (405, "native"), (500, "HTTP 500")):
            with self.subTest(status=status):
                self.request.return_value = response({"error": "remote-secret must not be echoed"}, status)
                with self.assertRaisesRegex(management.ModelManagementError, message) as caught:
                    management.loaded_models("http://localhost:1234", "remote-secret")
                self.assertNotIn("remote-secret", str(caught.exception))
        self.request.side_effect = requests.ConnectionError("http://secret.example/token")
        with self.assertRaisesRegex(management.ModelManagementError, "Could not reach") as caught:
            management.loaded_models("http://localhost:1234")
        self.assertNotIn("secret.example", str(caught.exception))

    def test_unexpected_native_model_metadata_is_rejected(self):
        for payload in ({"data": []}, model_list("duplicate", "duplicate"),
                        {"models": [{"type": "llm", "key": "writer", "loaded_instances": None}]}):
            with self.subTest(payload=payload):
                self.request.return_value = response(payload)
                with self.assertRaises(management.ModelManagementError):
                    management.loaded_models("http://localhost:1234")

    def test_acceptance_is_not_reported_as_verified_release(self):
        cases = [
            [response(model_list("target")), response({"instance_id": "wrong"})],
            [response(model_list("target")), response({"instance_id": "target"}), response(model_list("target"))],
            [response(model_list("target")), response({"instance_id": "target"}), response({}, 503)],
        ]
        for replies in cases:
            with self.subTest(replies=len(replies)):
                self.request.side_effect = replies
                with self.assertRaises(management.ModelManagementError):
                    management.unload_model("http://localhost:1234", instance_id="target")


class RemoteWriterLifecycleTests(unittest.TestCase):
    def test_busy_writer_never_sends_external_unload(self):
        for uses, done in ((1, True), (0, False)):
            with self.subTest(uses=uses, done=done), \
                    mock.patch.object(llm_service, "_active_uses", uses), \
                    mock.patch.object(llm_service, "_stream_done", done), \
                    mock.patch.object(management, "unload_model") as unload:
                with self.assertRaises(management.ModelBusyError):
                    llm_service.unload_remote_instance("http://localhost:1234", instance_id="target")
                unload.assert_not_called()

    def test_only_matching_remote_connection_is_cleared(self):
        for provider, active_url, active_id, clear in (
            ("remote", "http://localhost:1234/v1", "writer", True),
            ("remote", "http://localhost:1234", "target", True),
            ("remote", "http://another:1234", "writer", False),
            ("remote", "http://localhost:1234", "other-writer", False),
            ("local", "", "writer", False),
        ):
            with self.subTest(provider=provider, url=active_url, model=active_id), \
                    mock.patch.object(llm_service, "_active_uses", 0), \
                    mock.patch.object(llm_service, "_stream_done", True), \
                    mock.patch.object(llm_service, "_provider", provider), \
                    mock.patch.object(llm_service, "_remote_url", active_url), \
                    mock.patch.object(llm_service, "_model_id", active_id), \
                    mock.patch.object(llm_service, "_unload_inner") as cleanup, \
                    mock.patch.object(management, "unload_model", return_value={
                        "status": "unloaded", "server_url": "http://localhost:1234",
                        "instance_id": "target", "model_key": "writer",
                    }):
                llm_service.unload_remote_instance("http://localhost:1234", instance_id="target")
                self.assertEqual(cleanup.call_count, int(clear))

    def test_regular_cleanup_never_unloads_the_external_model(self):
        with mock.patch.object(llm_service, "_unload_inner") as cleanup, \
                mock.patch.object(management, "unload_model") as external:
            llm_service.unload_model()
        cleanup.assert_called_once()
        external.assert_not_called()


if __name__ == "__main__":
    unittest.main()
