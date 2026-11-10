"""The local LLM must look at an image, or say that it cannot.

`describe_image` used to take the image's *file name*, hand it to a text
model, and return whatever that model invented. The reply was fluent,
confident, and about nothing: asked to describe a frame from a comparison
tool, it described a comparison tool. Nothing downstream could tell that
apart from a real description, which is what makes it worth a test rather
than a comment.

The behaviour is checked directly where that is cheap, and asserted as
wiring in launch.py, which is over 31,000 lines and tracks an upstream that
rewrites it often.
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
_APP = str(_ROOT / "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)

from services import llm_service

LAUNCH = Path(_APP) / "launch.py"


class DescribeImageTests(unittest.TestCase):
    """The image itself has to reach the model."""

    def setUp(self):
        # A real file, because describe_image rejects a missing path first.
        self.image = Path(__file__).with_name("test_llm_vision_sample.png")
        self.image.write_bytes(b"\x89PNG\r\n\x1a\n not a real png")
        self.addCleanup(self.image.unlink)

    def _loaded(self, vision: bool):
        return mock.patch.multiple(
            llm_service,
            is_loaded=mock.Mock(return_value=True),
            _vision_available=vision,
            _model_id="some-model",
        )

    def test_the_image_path_is_handed_to_the_model(self):
        with self._loaded(vision=True), \
                mock.patch.object(llm_service, "generate", return_value="seen") as generate:
            result = llm_service.describe_image(str(self.image))

        self.assertEqual(result, "seen")
        self.assertEqual(generate.call_args.kwargs["image_paths"], [str(self.image)])

    def test_the_prompt_is_not_rewritten_to_mention_a_filename(self):
        """Naming the file instead of showing it is the whole original bug."""
        with self._loaded(vision=True), \
                mock.patch.object(llm_service, "generate", return_value="seen") as generate:
            llm_service.describe_image(str(self.image), prompt="What colour is the sky?")

        prompt = generate.call_args.kwargs["prompt"]
        self.assertEqual(prompt, "What colour is the sky?")
        self.assertNotIn(self.image.name, prompt)

    def test_a_text_only_model_refuses_instead_of_inventing(self):
        with self._loaded(vision=False), \
                mock.patch.object(llm_service, "generate") as generate:
            with self.assertRaises(RuntimeError) as caught:
                llm_service.describe_image(str(self.image))

        generate.assert_not_called()
        message = str(caught.exception)
        self.assertIn("some-model", message)
        self.assertIn("cannot accept images", message)

    def test_a_missing_image_is_still_reported_first(self):
        with self.assertRaises(FileNotFoundError):
            llm_service.describe_image(str(self.image.with_name("absent.png")))


class StatusTests(unittest.TestCase):
    """Callers outside Maestro decide whether to ask, so they must be told."""

    def test_vision_is_reported_when_a_projector_is_loaded(self):
        with mock.patch.multiple(
            llm_service,
            is_loaded=mock.Mock(return_value=True),
            _vision_available=True,
        ):
            self.assertTrue(llm_service.get_status()["vision"])

    def test_vision_is_false_for_a_text_only_model(self):
        with mock.patch.multiple(
            llm_service,
            is_loaded=mock.Mock(return_value=True),
            _vision_available=False,
        ):
            self.assertFalse(llm_service.get_status()["vision"])

    def test_vision_is_false_when_nothing_is_loaded(self):
        """A stale flag from a previous model would invite a doomed request."""
        with mock.patch.multiple(
            llm_service,
            is_loaded=mock.Mock(return_value=False),
            _vision_available=True,
        ):
            self.assertFalse(llm_service.get_status()["vision"])


class GenerateEndpointWiringTests(unittest.TestCase):
    """`/api/v1/llm/generate` is the only route that can carry an image.

    Without the pass-through, the endpoint accepts `image_paths`, ignores
    them, and answers from the prompt alone — the same silent failure in a
    new place. Asserted as wiring because importing launch.py means loading
    torch.
    """

    @classmethod
    def setUpClass(cls):
        tree = ast.parse(LAUNCH.read_text(encoding="utf-8"))
        cls.endpoint = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "llm_generate"
        )
        cls.source = ast.unparse(cls.endpoint)

    def test_images_are_forwarded_to_the_service(self):
        call = next(
            node
            for node in ast.walk(self.endpoint)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "generate"
        )
        self.assertIn("image_paths", [kw.arg for kw in call.keywords])

    def test_a_blind_model_is_refused_rather_than_asked(self):
        self.assertIn("vision", self.source)
        self.assertIn("cannot accept images", self.source)

    def test_an_unreadable_path_is_rejected_before_the_model_loads(self):
        self.assertIn("Image not found", self.source)


if __name__ == "__main__":
    unittest.main()
