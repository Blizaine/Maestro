# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Maestro's method and asset bridge for the RelayVSR postprocessor."""
from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Callable


METHOD = "relayvsr"
WEIGHTS_REPO = "DeepBeepMeep/Wan2.1"
WEIGHTS_REVISION = "176bbae94db7b26f00e5b698d95d4f578debc51c"
FOLDER = "RelayVSR"
FILE_PREFIXES = {4: "RelayVSR", 2: "RelayVSR_X2"}
SUPPORTED_SCALES = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
ASSET_METADATA = {
    "RelayVSR_X2_keyframes_int8_convrot.safetensors": {
        "size": 5_165_689_310,
        "sha256": "3b145b6bc8e1c9c23a63facef641c39cacf1f5be2f6e1663cc573e9d13227c2f",
    },
    "RelayVSR_X2_decoder_bf16.safetensors": {
        "size": 62_316_152,
        "sha256": "b80ce84591e6e6e158de93b9649d4811b6440178c760ba0908cde5b759c4705c",
    },
    "RelayVSR_keyframes_int8_convrot.safetensors": {
        "size": 5_166_574_045,
        "sha256": "9a1113a71e67a66e9a0755e4f470b7a0efae3a00e59e817a999bacd4ae29b425",
    },
    "RelayVSR_decoder_bf16.safetensors": {
        "size": 62_316_152,
        "sha256": "1b008590f63fe7b90889bba85b074f8f4f6dc0c7c4f5e182c378a6663ff30508",
    },
}
_VERIFIED_ASSETS: set[tuple[str, int, int]] = set()


class RelayVSRBridge:
    """Expose RelayVSR methods through Maestro's existing spatial upscaler hooks."""

    X2_MODEL_CHOICES = (("RelayVSR X2", "x2"), ("RelayVSR x4 resized down", "x4"))

    def __init__(self, server_config: dict[str, Any], files_locator):
        self.server_config = server_config
        self.files_locator = files_locator

    def config(self) -> dict[str, Any]:
        services = self.server_config.get("services")
        value = services.get("relayvsr_x2_model") if isinstance(services, dict) else None
        if value is None:
            value = self.server_config.get("relayvsr_x2_model", "x2")
        return {"x2_model": value if value in {"x2", "x4"} else "x2"}

    @classmethod
    def format_ratio(cls, scale: float) -> str:
        return str(int(scale)) if float(scale).is_integer() else f"{float(scale):g}"

    @classmethod
    def upsampling_value(cls, scale: float) -> str:
        return f"{METHOD}{cls.format_ratio(scale)}"

    @classmethod
    def scale_for_upsampling(cls, method: Any) -> float | None:
        match = re.fullmatch(r"relayvsr(1(?:\.5)?|2(?:\.5)?|3(?:\.5)?|4)", str(method or "").strip().lower())
        if not match:
            return None
        scale = float(match.group(1))
        return scale if scale in SUPPORTED_SCALES else None

    def is_upsampling(self, method: Any) -> bool:
        return self.scale_for_upsampling(method) is not None

    def validate_upsampling(self, method: Any, image_mode: int = 0) -> str:
        if self.is_upsampling(method):
            return ""
        return "RelayVSR supports x1.0 through x4.0 in 0.5 increments"

    def native_scale(self, method: Any) -> int:
        scale = self.scale_for_upsampling(method)
        if scale is None:
            raise ValueError(f"Unknown RelayVSR upsampling method: {method}")
        return 2 if scale <= 2 and self.config()["x2_model"] == "x2" else 4

    @classmethod
    def filenames(cls, native: int) -> tuple[str, str]:
        prefix = FILE_PREFIXES[native]
        return f"{prefix}_keyframes_int8_convrot.safetensors", f"{prefix}_decoder_bf16.safetensors"

    @classmethod
    def download_def(cls, native: int) -> dict[str, Any]:
        return {
            "repoId": WEIGHTS_REPO,
            "revision": WEIGHTS_REVISION,
            "sourceFolderList": [FOLDER],
            "fileList": [list(cls.filenames(native))],
        }

    @classmethod
    def _verify_asset(cls, path: str, filename: str) -> None:
        expected = ASSET_METADATA.get(filename)
        if expected is None:
            raise ValueError(f"No pinned RelayVSR asset metadata for {filename}")
        stat = os.stat(path)
        if stat.st_size != expected["size"]:
            raise RuntimeError(
                f"RelayVSR checkpoint size mismatch for {filename}: "
                f"expected {expected['size']} bytes, got {stat.st_size}"
            )
        cache_key = (os.path.abspath(path), stat.st_size, stat.st_mtime_ns)
        if cache_key in _VERIFIED_ASSETS:
            return
        digest = hashlib.sha256()
        with open(path, "rb") as asset:
            for chunk in iter(lambda: asset.read(8 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected["sha256"]:
            raise RuntimeError(f"RelayVSR checkpoint SHA-256 mismatch for {filename}")
        _VERIFIED_ASSETS.add(cache_key)

    def query_download_def(self, enabled_only: bool = True, method: str | None = None):
        if method is not None:
            return self.download_def(self.native_scale(method))
        scales = [4, 2] if self.config()["x2_model"] == "x2" else [4]
        return [self.download_def(scale) for scale in scales]

    def _paths(self, native: int) -> tuple[str, str]:
        result = []
        for filename in self.filenames(native):
            path = self.files_locator.locate_file(os.path.join(FOLDER, filename), error_if_none=False)
            if not path or not os.path.isfile(path) or os.path.getsize(path) <= 0:
                raise FileNotFoundError(f"Pinned RelayVSR checkpoint is missing or empty: {filename}")
            self._verify_asset(path, filename)
            result.append(path)
        return tuple(result)

    def download(self, process_files: Callable[..., Any], method: str) -> bool:
        native = self.native_scale(method)
        try:
            self._paths(native)
            return False
        except FileNotFoundError:
            process_files(**self.download_def(native))
            self._paths(native)  # fail before model construction if download did not produce both expected files
            return True

    def upscale(
        self,
        sample,
        method,
        *,
        seed=0,
        frame_offset=0,
        continue_cache=None,
        return_continue_cache=False,
        process_files: Callable[..., Any],
        init_pipe: Callable[..., int],
        profile,
        abort_callback=None,
        progress_callback=None,
        **kwargs,
    ):
        scale = self.scale_for_upsampling(method)
        if scale is None:
            raise ValueError(f"Unknown RelayVSR upsampling method: {method}")
        native = self.native_scale(method)
        self.download(process_files, method)
        from .runtime import RelayVSRPaths, load_models, upscale_video
        keyframes, decoder = self._paths(native)
        load_models(RelayVSRPaths(keyframes=keyframes, decoder=decoder, scale=native), init_pipe=init_pipe, profile=profile)
        return upscale_video(
            sample, scale, seed=seed, frame_offset=frame_offset,
            continue_cache=continue_cache, return_continue_cache=return_continue_cache,
            abort_callback=abort_callback, progress_callback=progress_callback,
        )

    def release_vram(self) -> None:
        from .runtime import release_models
        release_models()


__all__ = ["RelayVSRBridge", "METHOD", "SUPPORTED_SCALES", "WEIGHTS_REVISION", "ASSET_METADATA"]
