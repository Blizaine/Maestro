"""LLM idle-unload races, with fake HTTP responses and a controlled clock."""

from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import sys
import threading
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import requests
from services import llm_service


class FakeTimer:
    def __init__(self, interval, function, args=None, kwargs=None):
        self.function = function
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.cancelled = False

    def start(self):
        pass

    def cancel(self):
        self.cancelled = True

    def fire(self):
        # A callback already queued by the timer thread can run after cancel.
        self.function(*self.args, **self.kwargs)


class LlmLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.timers = []
        self.loaded = True
        for name, value in (
            ("_idle_timer", None), ("_idle_generation", 0), ("_active_uses", 0),
            ("_provider", "local"), ("_model_id", "lifecycle-test"),
        ):
            patcher = mock.patch.object(llm_service, name, value, create=True)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.patch(llm_service, "is_loaded", side_effect=lambda: self.loaded)
        self.patch(llm_service.threading, "Timer", side_effect=self.timer)
        self.patch(llm_service, "_count_local_tokens", return_value=None)
        self.unload = self.patch(llm_service, "_unload_inner", side_effect=self.mark_unloaded)
        self.post = self.patch(requests, "post")

    def patch(self, target, name, **kwargs):
        patcher = mock.patch.object(target, name, **kwargs)
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    def timer(self, *args, **kwargs):
        timer = FakeTimer(*args, **kwargs)
        self.timers.append(timer)
        return timer

    def mark_unloaded(self):
        self.loaded = False
        llm_service._cancel_idle_timer()

    def response(self, *, streaming=False):
        response = requests.Response()
        response.status_code = 200
        response.encoding = "utf-8"
        if streaming:
            response.raw = io.BytesIO(
                b'data: {"choices":[{"delta":{"content":"done"}}]}\n\n'
                b'data: [DONE]\n\n'
            )
        else:
            response._content = json.dumps({
                "choices": [{"message": {"content": "done"}, "finish_reason": "stop"}],
            }).encode()
        return response

    def test_completed_request_cannot_unload_another_request_still_running(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                entered = threading.Event()
                release = threading.Event()

                def post(_url, **kwargs):
                    if kwargs["json"]["messages"][-1]["content"] == "slow":
                        entered.set()
                        if not release.wait(5):
                            raise TimeoutError("Test did not release slow request")
                    return self.response(streaming=kwargs["json"].get("stream", False))

                self.post.side_effect = post
                self.timers.clear()
                generate = llm_service.generate_streaming if streaming else llm_service.generate
                with ThreadPoolExecutor(max_workers=1) as pool:
                    slow = pool.submit(generate, "slow", enable_thinking=False)
                    try:
                        self.assertTrue(entered.wait(5))
                        self.assertEqual(llm_service.generate("quick", enable_thinking=False), "done")
                        for timer in list(self.timers):
                            if not timer.cancelled:
                                timer.fire()
                        self.unload.assert_not_called()
                    finally:
                        release.set()
                    self.assertEqual(slow.result(timeout=5), "done")
                self.assertIsNotNone(llm_service._idle_timer)

    def test_cancelled_callback_does_not_unload_a_newly_idle_model(self):
        self.post.side_effect = lambda *args, **kwargs: self.response()
        llm_service.generate("first", enable_thinking=False)
        stale = self.timers[-1]
        llm_service.generate("second", enable_thinking=False)
        current = self.timers[-1]
        self.assertTrue(stale.cancelled)
        stale.fire()
        self.unload.assert_not_called()
        self.assertIs(llm_service._idle_timer, current)
        current.fire()
        self.unload.assert_called_once()

    def test_planning_session_stays_loaded_between_calls_then_releases(self):
        self.post.side_effect = lambda *args, **kwargs: self.response()
        llm_service._reset_idle_timer()
        stale = self.timers[-1]
        with llm_service.keep_loaded():
            self.assertEqual(llm_service.generate("treatment", enable_thinking=False), "done")
            stale.fire()
            llm_service._reset_idle_timer()
            self.assertIsNone(llm_service._idle_timer)
            self.unload.assert_not_called()
            self.assertEqual(llm_service.generate("camera", enable_thinking=False), "done")
        self.assertEqual(llm_service._active_uses, 0)
        self.assertIsNotNone(llm_service._idle_timer)
        self.timers[-1].fire()
        self.unload.assert_called_once()

    def test_request_failure_still_releases_idle_protection(self):
        for generate in (llm_service.generate, llm_service.generate_streaming):
            with self.subTest(generate=generate.__name__):
                self.post.side_effect = ValueError("invalid response")
                with self.assertRaisesRegex(ValueError, "invalid response"):
                    generate("test", enable_thinking=False)
                self.assertEqual(llm_service._active_uses, 0)
                self.assertIsNotNone(llm_service._idle_timer)

    def test_h3_faithful_keeps_model_for_treatment_and_both_window_plans(self):
        from services.h3_story_ledger import plan_h3_story_segments

        prompt = (Path(__file__).parent / "fixtures/h3_silent_wuxia_prompt.txt").read_text(encoding="utf-8")
        durations = [14.375, 13.625]
        calls = []
        llm_service._reset_idle_timer()
        stale = self.timers[-1]

        def generate(**kwargs):
            calls.append(kwargs)
            # Model activity is tracked across the whole planner, even when
            # there is no HTTP call in progress during validation/compilation.
            llm_service._reset_idle_timer()
            stale.fire()
            self.assertIsNone(llm_service._idle_timer)
            self.unload.assert_not_called()
            schema = kwargs["json_schema"]["properties"]
            if "setting_continuity" in schema:
                self.assertNotIn("beats", schema)
                return json.dumps({
                    "setting_continuity": "High-mountain stone ruins",
                    "visual_continuity": "Live-action realism",
                    "editing_style": "Anticipation followed by explosive impacts",
                    "ambient_audio": "Wind and tumbling stone",
                })
            number = schema["segment"]["minimum"]
            duration = durations[number - 1]
            return json.dumps({"segment": number, "shots": [{
                "end_seconds": duration * (index + 1) / 4,
                "framing": "Wide view of the two fighters",
                "camera": "a deliberate tracking shot",
            } for index in range(4)]})

        result = plan_h3_story_segments(
            prompt, segment_durations=durations, mode="sliding_window",
            camera_coverage="multi_shot", planning_style="faithful", llm_generate=generate,
        )
        self.assertEqual(len(calls), 3)
        self.assertEqual(result["planned_by"], "llm")
        self.assertEqual(result["planning_warnings"], [])
        self.assertEqual(result["planning_diagnostics"], [])
        self.assertEqual(llm_service._active_uses, 0)
        self.assertIsNotNone(llm_service._idle_timer)
        self.timers[-1].fire()
        self.unload.assert_called_once()

    def test_session_exception_and_explicit_unload_do_not_leak_protection(self):
        with self.assertRaisesRegex(ValueError, "bad plan"):
            with llm_service.keep_loaded():
                raise ValueError("bad plan")
        self.assertEqual(llm_service._active_uses, 0)
        self.assertIsNotNone(llm_service._idle_timer)
        with llm_service.keep_loaded():
            llm_service.unload_model()
        self.unload.assert_called_once()
        self.assertEqual(llm_service._active_uses, 0)
        self.assertIsNone(llm_service._idle_timer)


if __name__ == "__main__":
    unittest.main()
