"""Small CPU-only cache for plain-text MiniMax H3 prompt embeddings."""

from __future__ import annotations

import hashlib
from threading import RLock
from typing import Callable

import torch

from shared.utils.text_encoder_cache import TextEncoderCache


class H3PromptEncodingAborted(RuntimeError):
    """Raised when H3 prompt conditioning is interrupted or returns no result."""


def _not_interrupted() -> bool:
    return False


class MiniMaxH3PromptCache:
    """Cache repeated plain-text H3 conditioning without retaining device tensors."""

    def __init__(self, max_size_mb: float = 32.0) -> None:
        self._cache = TextEncoderCache(max_size_mb=max_size_mb)
        self._lock = RLock()

    def condition(
        self,
        prompt: str,
        *,
        conditioner,
        encoder_variant: str,
        dtype: torch.dtype,
        device: torch.device | str,
        keyframes=None,
        viggle: bool = False,
        interrupted: Callable[[], bool] | None = None,
    ) -> tuple[tuple[torch.Tensor | None, torch.Tensor | None], str | None]:
        """Condition a plain prompt, bypassing the cache for multimodal paths.

        The cache key uses a prompt digest rather than retaining prompt text.
        Cached tensors stay on CPU; both outputs return on the requested device
        with their original dtypes and are cloned to isolate entries from mutation.
        """

        if viggle or (keyframes is not None and len(keyframes) > 0):
            return conditioner(prompt, device, keyframes or None), None

        if interrupted is None:
            interrupted = _not_interrupted
        if interrupted():
            raise H3PromptEncodingAborted

        key = self._cache_key(prompt, dtype, conditioner, encoder_variant)
        with self._lock:
            if interrupted():
                raise H3PromptEncodingAborted
            was_hit = key in self._cache._entries

            def encode_one(prompts: list[str]) -> list[dict[str, torch.Tensor]]:
                values = []
                for item in prompts:
                    if interrupted():
                        raise H3PromptEncodingAborted
                    embeddings, tags = conditioner(item, device, None)
                    if interrupted() or embeddings is None or tags is None:
                        raise H3PromptEncodingAborted
                    if not torch.is_tensor(embeddings) or not torch.is_tensor(tags):
                        raise TypeError("MiniMax H3 conditioner must return tensor embeddings and tags.")
                    values.append({"embeddings": embeddings, "tags": tags})
                return values

            value = self._cache.encode(
                encode_one,
                prompt,
                device=None,
                cache_keys=key,
            )[0]

            if interrupted():
                if not was_hit:
                    self._discard(key)
                raise H3PromptEncodingAborted

            embeddings = value["embeddings"].detach().to(device=device).clone()
            tags = value["tags"].detach().to(device=device).clone()
            return (embeddings, tags), ("hit" if was_hit else "miss")

    @staticmethod
    def _cache_key(
        prompt: str,
        dtype: torch.dtype,
        conditioner,
        encoder_variant: str,
    ) -> tuple[str, int, str, str, bytes]:
        prompt_digest = hashlib.sha256(prompt.encode("utf-8", errors="surrogatepass")).digest()
        return (
            "minimax_h3_plain_v1",
            id(conditioner),
            str(encoder_variant),
            str(dtype),
            prompt_digest,
        )

    def _discard(self, key: tuple) -> None:
        """Remove an entry if an interrupt races with the final cache insert."""

        entry = self._cache._entries.pop(key, None)
        if entry is not None:
            self._cache._size_bytes -= entry.size_bytes