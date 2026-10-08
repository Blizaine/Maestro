"""CPU-only tests for bounded MiniMax H3 plain-text prompt caching."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from models.minimax_h3.prompt_cache import (  # noqa: E402
    H3PromptEncodingAborted,
    MiniMaxH3PromptCache,
)


class FakeConditioner:
    def __init__(self, value: float = 1.0, on_call=None):
        self.value = value
        self.on_call = on_call
        self.calls: list[tuple[str, object]] = []

    def __call__(self, prompt, device, keyframes=None):
        self.calls.append((prompt, keyframes))
        if self.on_call is not None:
            self.on_call()
        value = self.value + len(prompt)
        embeddings = torch.full((1, 4, 8), value, dtype=torch.float32, device=device)
        tags = torch.tensor([1, 1, 1, 1], dtype=torch.int32, device="cpu")
        return embeddings, tags


class TestMiniMaxH3PromptCache(unittest.TestCase):
    def test_repeated_plain_prompt_hit_matches_fresh_dtype_and_is_mutation_safe(self):
        cache = MiniMaxH3PromptCache()
        conditioner = FakeConditioner()

        (first, first_tags), first_status = cache.condition(
            "same prompt",
            conditioner=conditioner,
            encoder_variant="gguf_q2_k",
            dtype=torch.float16,
            device="cpu",
        )
        first.fill_(0)
        first_tags.fill_(0)
        (cached, cached_tags), second_status = cache.condition(
            "same prompt",
            conditioner=conditioner,
            encoder_variant="gguf_q2_k",
            dtype=torch.float16,
            device="cpu",
        )
        fresh, fresh_tags = FakeConditioner()("same prompt", "cpu", None)

        self.assertEqual((first_status, second_status), ("miss", "hit"))
        self.assertEqual(len(conditioner.calls), 1)
        self.assertTrue(torch.equal(cached, fresh))
        self.assertTrue(torch.equal(cached_tags, fresh_tags))
        # Keep the actual encoder dtype; dtype still participates in the key.
        self.assertEqual(cached.dtype, torch.float32)
        self.assertEqual(cached_tags.dtype, torch.int32)
        self.assertEqual(cached_tags.device.type, "cpu")

    def test_prompt_conditioner_variant_and_requested_dtype_are_cache_keyed(self):
        cache = MiniMaxH3PromptCache()
        conditioner = FakeConditioner()
        statuses = []
        for prompt, active_conditioner, variant, dtype in (
            ("first", conditioner, "gguf_q2_k", torch.float32),
            ("first", conditioner, "gguf_q2_k", torch.float32),
            ("changed", conditioner, "gguf_q2_k", torch.float32),
            ("changed", FakeConditioner(), "gguf_q2_k", torch.float32),
            ("changed", conditioner, "other", torch.float32),
            ("changed", conditioner, "other", torch.float16),
        ):
            _, status = cache.condition(
                prompt,
                conditioner=active_conditioner,
                encoder_variant=variant,
                dtype=dtype,
                device="cpu",
            )
            statuses.append(status)

        self.assertEqual(statuses, ["miss", "hit", "miss", "miss", "miss", "miss"])

    def test_keyframes_and_viggle_bypass_cache(self):
        cache = MiniMaxH3PromptCache()
        conditioner = FakeConditioner()
        keyframes = [object()]

        (_, tags), status = cache.condition(
            "same prompt",
            conditioner=conditioner,
            encoder_variant="gguf_q2_k",
            dtype=torch.float32,
            device="cpu",
            keyframes=keyframes,
        )
        self.assertIsNone(status)
        self.assertEqual(tags.device.type, "cpu")
        self.assertIs(conditioner.calls[0][1], keyframes)

        (_, tags), status = cache.condition(
            "same prompt",
            conditioner=conditioner,
            encoder_variant="gguf_q2_k",
            dtype=torch.float32,
            device="cpu",
            viggle=True,
        )
        self.assertIsNone(status)
        self.assertEqual(tags.device.type, "cpu")
        self.assertIsNone(conditioner.calls[1][1])
        self.assertEqual(len(cache._cache._entries), 0)

    def test_encoder_failure_interrupt_and_missing_result_are_not_cached(self):
        def fail():
            raise RuntimeError("encoder failure")

        cache = MiniMaxH3PromptCache()
        with self.assertRaisesRegex(RuntimeError, "encoder failure"):
            cache.condition(
                "failure",
                conditioner=FakeConditioner(on_call=fail),
                encoder_variant="gguf_q2_k",
                dtype=torch.float32,
                device="cpu",
            )
        self.assertEqual(len(cache._cache._entries), 0)

        aborted = False

        def interrupt_during_encode():
            nonlocal aborted
            aborted = True

        with self.assertRaises(H3PromptEncodingAborted):
            cache.condition(
                "aborted",
                conditioner=FakeConditioner(on_call=interrupt_during_encode),
                encoder_variant="gguf_q2_k",
                dtype=torch.float32,
                device="cpu",
                interrupted=lambda: aborted,
            )
        self.assertEqual(len(cache._cache._entries), 0)

        class MissingConditioner:
            def __call__(self, _prompt, _device, _keyframes=None):
                return None, None

        with self.assertRaises(H3PromptEncodingAborted):
            cache.condition(
                "missing result",
                conditioner=MissingConditioner(),
                encoder_variant="gguf_q2_k",
                dtype=torch.float32,
                device="cpu",
            )
        self.assertEqual(len(cache._cache._entries), 0)

    def test_interrupt_does_not_replace_or_evict_a_valid_cached_result(self):
        cache = MiniMaxH3PromptCache()
        conditioner = FakeConditioner()
        kwargs = dict(
            conditioner=conditioner,
            encoder_variant="gguf_q2_k",
            dtype=torch.float32,
            device="cpu",
        )
        _, status = cache.condition("valid", **kwargs)
        self.assertEqual(status, "miss")
        self.assertEqual(len(cache._cache._entries), 1)

        with self.assertRaises(H3PromptEncodingAborted):
            cache.condition("valid", interrupted=lambda: True, **kwargs)
        self.assertEqual(len(cache._cache._entries), 1)

        _, status = cache.condition("valid", **kwargs)
        self.assertEqual(status, "hit")
        self.assertEqual(len(conditioner.calls), 1)

    def test_cache_is_cpu_only_bounded_and_does_not_retain_prompt_text(self):
        # Each record is 4*8*float32 + 4*int32 = 144 bytes; retain only one.
        cache = MiniMaxH3PromptCache(max_size_mb=180 / (1024 * 1024))
        conditioner = FakeConditioner()
        for prompt in ("first private prompt", "second private prompt"):
            cache.condition(
                prompt,
                conditioner=conditioner,
                encoder_variant="gguf_q2_k",
                dtype=torch.float32,
                device="cpu",
            )

        entries = cache._cache._entries
        self.assertLessEqual(cache._cache._size_bytes, cache._cache.max_size_bytes)
        self.assertEqual(len(entries), 1)
        for entry in entries.values():
            self.assertEqual(entry.value["embeddings"].device.type, "cpu")
            self.assertEqual(entry.value["tags"].device.type, "cpu")
        key = next(iter(entries))
        self.assertNotIn("first private prompt", key)
        self.assertNotIn("second private prompt", key)


if __name__ == "__main__":
    unittest.main()