from __future__ import annotations

import hashlib
import os
import tempfile
import threading
import urllib.request
from typing import Any, Callable

from postprocessing.flashvsr.sparse_backend_config import (
    SPARSE_BACKEND_AUTO,
    SPARSE_BACKEND_SPARGE,
    SPARSE_BACKEND_TRITON_SPARSE,
    normalize_sparse_backend,
)


_PLUS_PROMPT_DOWNLOAD_LOCK = threading.Lock()
_PLUS_ASSET_VERIFY_LOCK = threading.Lock()
_PLUS_VERIFIED_ASSETS: dict[tuple[str, str, str], tuple[int, int, int, int]] = {}


def _file_fingerprint(path: str) -> tuple[int, int, int, int]:
    stat = os.stat(path)
    return (
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
        getattr(stat, "st_ino", 0),
    )


class FlashVSRBridge:
    MODE_OFF = 0
    MODE_TINY = 1
    MODE_FULL = 2
    MODE_TINY_LONG = 3
    PERSIST_UNLOAD = 1
    PERSIST_RAM = 2
    BACKEND_AUTO = SPARSE_BACKEND_AUTO
    BACKEND_TRITON_SPARSE = SPARSE_BACKEND_TRITON_SPARSE
    BACKEND_SPARGE = SPARSE_BACKEND_SPARGE
    TOPK_RATIO_DEFAULT = 0.0
    TOPK_RATIO_MAX = 4.0
    UPSAMPLING_VALUE_PREFIX = "flashvsr"
    UPSAMPLING_TWO_PASS_VALUE_PREFIX = "flashvsr2pass"
    UPSAMPLING_PLUS_VALUE_PREFIX = "flashvsrplus"
    UPSAMPLING_PLUS_TWO_PASS_VALUE_PREFIX = "flashvsrplus2pass"
    UPSAMPLING_RATIOS = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)

    MODEL_V11 = "v1_1"
    MODEL_PLUS_TURBO = "plus_turbo"
    MODEL_CHOICES = (
        ("FlashVSR v1.1", MODEL_V11),
        ("FlashVSR+ Turbo", MODEL_PLUS_TURBO),
    )

    TRANSFORMER_FILENAME = "FlashVSR_v1.1_transformer_bf16.safetensors"
    LQ_PROJ_FILENAME = "FlashVSR_v1.1_lq_proj_bf16.safetensors"
    TCDECODER_FILENAME = "FlashVSR_v1.1_tcdecoder_bf16.safetensors"
    POSI_PROMPT_FILENAME = "FlashVSR_v1.1_posi_prompt_bf16.safetensors"
    VAE_FILENAME = "Wan2.1_VAE.safetensors"

    # Plus assets are pinned to immutable upstream revisions and use local names
    # that cannot collide with v1.1 or with a future revision of Plus Turbo.
    PLUS_REPO_ID = "ruchangyao/FlashVSR_plus"
    PLUS_REPO_REVISION = "46b79520946ac67dec94d1566490690953c7debc"
    PLUS_GITHUB_REVISION = "b548a2cc2c929d3d4676f2973abe27876ec4dee7"
    PLUS_GITHUB_PROMPT_BLOB_SHA1 = "95c840b20b8d6aa0d16504ee04e49f1c01a7f209"
    PLUS_GITHUB_PROMPT_SIZE = 4_195_504
    PLUS_TRANSFORMER_REMOTE_FILENAME = "dit.safetensors"
    PLUS_LQ_PROJ_REMOTE_FILENAME = "lq_proj_in.safetensors"
    PLUS_TCDECODER_REMOTE_FILENAME = "TCDecoder.ckpt"
    PLUS_VAE_REMOTE_FILENAME = "Wan2.1_VAE.pth"
    PLUS_TRANSFORMER_FILENAME = "FlashVSRPlus_Turbo_46b79520_dit.safetensors"
    PLUS_LQ_PROJ_FILENAME = "FlashVSRPlus_Turbo_46b79520_lq_proj_in.safetensors"
    PLUS_TCDECODER_FILENAME = "FlashVSRPlus_46b79520_TCDecoder.ckpt"
    PLUS_POSI_PROMPT_FILENAME = "FlashVSRPlus_b548a2cc_posi_prompt.pth"
    PLUS_VAE_FILENAME = "FlashVSRPlus_46b79520_Wan2.1_VAE.pth"
    PLUS_ASSET_METADATA = {
        PLUS_TRANSFORMER_FILENAME: {
            "remote_path": "turbo/dit.safetensors",
            "size": 5_676_070_392,
            "sha256": "7862b92178176fa3aebc34c6c071daca6b56ab1dddb8e7ff9a01c47f5226ec8b",
        },
        PLUS_LQ_PROJ_FILENAME: {
            "remote_path": "turbo/lq_proj_in.safetensors",
            "size": 1_151_384_272,
            "sha256": "e77b378ae12342eefbb58f924256f362a151b93ce73dd32808fcdaca9c051496",
        },
        PLUS_TCDECODER_FILENAME: {
            "remote_path": "TCDecoder.ckpt",
            "size": 189_018_333,
            "sha256": "e224bdcf2f52745cbf4d393ff5374c2ba09e90285d5d19062d2bf63b915b6161",
        },
        PLUS_VAE_FILENAME: {
            "remote_path": "Wan2.1_VAE.pth",
            "size": 507_609_880,
            "sha256": "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981",
        },
    }

    @classmethod
    def _verify_plus_hf_asset(cls, path: str, filename: str) -> None:
        metadata = cls.PLUS_ASSET_METADATA[filename]
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"Pinned FlashVSR+ asset is missing: {path or filename}")
        path = os.path.abspath(path)
        expected_size = int(metadata["size"])
        fingerprint = _file_fingerprint(path)
        if fingerprint[0] != expected_size:
            raise RuntimeError(
                f"FlashVSR+ asset size mismatch for {path}: expected {expected_size}, got {fingerprint[0]}"
            )
        cache_key = (path, "sha256", str(metadata["sha256"]))
        with _PLUS_ASSET_VERIFY_LOCK:
            if _PLUS_VERIFIED_ASSETS.get(cache_key) == fingerprint:
                return
            digest = hashlib.sha256()
            with open(path, "rb") as source:
                while block := source.read(4 * 1024 * 1024):
                    digest.update(block)
            actual = digest.hexdigest()
            if actual != metadata["sha256"]:
                raise RuntimeError(
                    f"FlashVSR+ SHA-256 mismatch for {path}: expected {metadata['sha256']}, got {actual}"
                )
            _PLUS_VERIFIED_ASSETS[cache_key] = fingerprint

    @classmethod
    def verify_plus_assets(cls, paths, variant: str) -> None:
        """Verify selected Plus checkpoints and prompt against immutable pins before loading."""
        selected = [
            (paths.transformer, cls.PLUS_TRANSFORMER_FILENAME),
            (paths.lq_proj, cls.PLUS_LQ_PROJ_FILENAME),
        ]
        if variant == "full":
            selected.append((paths.vae, cls.PLUS_VAE_FILENAME))
        else:
            selected.append((paths.tcdecoder, cls.PLUS_TCDECODER_FILENAME))
        for path, filename in selected:
            cls._verify_plus_hf_asset(path, filename)

        prompt_path = paths.posi_prompt
        if not prompt_path or not os.path.isfile(prompt_path):
            raise FileNotFoundError(f"Pinned FlashVSR+ prompt tensor is missing: {prompt_path or cls.PLUS_POSI_PROMPT_FILENAME}")
        actual_sha1, actual_size = cls._plus_prompt_git_blob_sha1(prompt_path)
        if actual_size != cls.PLUS_GITHUB_PROMPT_SIZE or actual_sha1 != cls.PLUS_GITHUB_PROMPT_BLOB_SHA1:
            raise RuntimeError(
                "FlashVSR+ prompt integrity check failed for the pinned official prompt: "
                f"{prompt_path} (size={actual_size}, git blob={actual_sha1})"
            )

    _VARIANTS = {
        MODE_TINY: "tiny",
        MODE_FULL: "full",
        MODE_TINY_LONG: "tiny-long",
    }

    def __init__(self, server_config: dict[str, Any], files_locator):
        self.server_config = server_config
        self.files_locator = files_locator

    @classmethod
    def normalize_topk_ratio(cls, value: Any) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = cls.TOPK_RATIO_DEFAULT
        return max(0.0, min(cls.TOPK_RATIO_MAX, value))

    @classmethod
    def normalize_backend(cls, value: Any) -> str:
        return normalize_sparse_backend(value)

    def normalize_config(self, config: dict[str, Any] | None = None) -> tuple[int, int]:
        config = self.server_config if config is None else config
        mode = config.get("flashvsr_mode", self.MODE_OFF)
        persistence = config.get("flashvsr_persistence", self.PERSIST_UNLOAD)
        try:
            mode = int(mode)
        except (TypeError, ValueError):
            mode = self.MODE_OFF
        try:
            persistence = int(persistence)
        except (TypeError, ValueError):
            persistence = self.PERSIST_UNLOAD
        if mode not in self._VARIANTS and mode != self.MODE_OFF:
            mode = self.MODE_OFF
        if persistence not in (self.PERSIST_UNLOAD, self.PERSIST_RAM):
            persistence = self.PERSIST_UNLOAD
        config["flashvsr_mode"] = mode
        config["flashvsr_persistence"] = persistence
        config["flashvsr_model"] = self.normalize_model(config.get("flashvsr_model", self.MODEL_V11))
        config["flashvsr_backend"] = self.normalize_backend(config.get("flashvsr_backend", self.BACKEND_AUTO))
        config["flashvsr_topk_ratio"] = self.normalize_topk_ratio(config.get("flashvsr_topk_ratio", self.TOPK_RATIO_DEFAULT))
        return mode, persistence

    def settings(self, config: dict[str, Any] | None = None) -> tuple[bool, str | None, int]:
        mode, persistence = self.normalize_config(config)
        return mode != self.MODE_OFF, self._VARIANTS.get(mode), persistence

    @classmethod
    def normalize_model(cls, value: Any) -> str:
        value = str(value or cls.MODEL_V11).strip().lower()
        return value if value in (cls.MODEL_V11, cls.MODEL_PLUS_TURBO) else cls.MODEL_V11

    def model(self, config: dict[str, Any] | None = None) -> str:
        config = self.server_config if config is None else config
        self.normalize_config(config)
        return config["flashvsr_model"]

    def model_for_upsampling(self, spatial_upsampling: Any) -> str:
        value = str(spatial_upsampling or "").strip().lower()
        if value.startswith((self.UPSAMPLING_PLUS_VALUE_PREFIX, self.UPSAMPLING_PLUS_TWO_PASS_VALUE_PREFIX)):
            return self.MODEL_PLUS_TURBO
        if value.startswith((self.UPSAMPLING_VALUE_PREFIX, self.UPSAMPLING_TWO_PASS_VALUE_PREFIX)):
            return self.MODEL_V11
        return self.model()

    @classmethod
    def model_label(cls, model: Any) -> str:
        model = cls.normalize_model(model)
        return "FlashVSR+ Turbo" if model == cls.MODEL_PLUS_TURBO else "FlashVSR v1.1"

    def topk_ratio(self) -> float:
        return self.normalize_topk_ratio(self.server_config.get("flashvsr_topk_ratio", self.TOPK_RATIO_DEFAULT))

    def backend(self) -> str:
        return self.normalize_backend(self.server_config.get("flashvsr_backend", self.BACKEND_AUTO))

    def enabled(self) -> bool:
        return self.settings()[0]

    @classmethod
    def format_ratio(cls, scale: float) -> str:
        scale = float(scale)
        return str(int(scale)) if scale.is_integer() else f"{scale:g}"

    @classmethod
    def format_ratio_label(cls, scale: float) -> str:
        return f"{float(scale):.1f}"

    @classmethod
    def upsampling_value(cls, scale: float) -> str:
        return f"{cls.UPSAMPLING_VALUE_PREFIX}{cls.format_ratio(scale)}"

    @classmethod
    def upsampling_two_pass_value(cls, scale: float) -> str:
        return f"{cls.UPSAMPLING_TWO_PASS_VALUE_PREFIX}{cls.format_ratio(scale)}"

    @classmethod
    def upsampling_plus_value(cls, scale: float) -> str:
        return f"{cls.UPSAMPLING_PLUS_VALUE_PREFIX}{cls.format_ratio(scale)}"

    @classmethod
    def upsampling_plus_two_pass_value(cls, scale: float) -> str:
        return f"{cls.UPSAMPLING_PLUS_TWO_PASS_VALUE_PREFIX}{cls.format_ratio(scale)}"

    @classmethod
    def upsampling_choices(cls, include_name: bool = True, include_two_pass: bool = False) -> list[tuple[str, str]]:
        prefix = "FlashVSR " if include_name else ""
        choices = [(f"{prefix}x{cls.format_ratio_label(scale)}", cls.upsampling_value(scale)) for scale in cls.UPSAMPLING_RATIOS]
        return choices + ([(f"{prefix}Two Pass x{cls.format_ratio_label(scale)}", cls.upsampling_two_pass_value(scale)) for scale in cls.UPSAMPLING_RATIOS] if include_two_pass else [])

    @classmethod
    def plus_upsampling_choices(cls, include_two_pass: bool = True) -> list[tuple[str, str]]:
        prefix = "FlashVSR+ Turbo "
        choices = [(f"{prefix}x{cls.format_ratio_label(scale)}", cls.upsampling_plus_value(scale)) for scale in cls.UPSAMPLING_RATIOS]
        return choices + ([(f"{prefix}Two Pass x{cls.format_ratio_label(scale)}", cls.upsampling_plus_two_pass_value(scale)) for scale in cls.UPSAMPLING_RATIOS] if include_two_pass else [])

    @classmethod
    def all_upsampling_choices(cls, include_name: bool = True, include_two_pass: bool = True) -> list[tuple[str, str]]:
        return cls.upsampling_choices(include_name=include_name, include_two_pass=include_two_pass) + cls.plus_upsampling_choices(include_two_pass=include_two_pass)

    @classmethod
    def scale_for_upsampling(cls, spatial_upsampling) -> float | None:
        text = str(spatial_upsampling or "").strip().lower()
        prefixes = (
            cls.UPSAMPLING_PLUS_TWO_PASS_VALUE_PREFIX,
            cls.UPSAMPLING_PLUS_VALUE_PREFIX,
            cls.UPSAMPLING_TWO_PASS_VALUE_PREFIX,
            cls.UPSAMPLING_VALUE_PREFIX,
        )
        prefix = next((candidate for candidate in prefixes if text.startswith(candidate)), None)
        if prefix is None:
            return None
        try:
            scale = float(text[len(prefix):])
        except ValueError:
            return None
        return scale if scale in cls.UPSAMPLING_RATIOS else None

    @classmethod
    def is_two_pass_upsampling(cls, spatial_upsampling) -> bool:
        text = str(spatial_upsampling or "").strip().lower()
        return text.startswith((cls.UPSAMPLING_TWO_PASS_VALUE_PREFIX, cls.UPSAMPLING_PLUS_TWO_PASS_VALUE_PREFIX))

    @classmethod
    def query_edit_mode_def(cls, include_name: bool = True) -> dict[str, Any]:
        return {
            "name": "FlashVSR",
            "spatial_upsampling_choices": cls.all_upsampling_choices(include_name=include_name, include_two_pass=True),
            "default_spatial_upsampling": cls.upsampling_value(2.0),
        }

    def is_upsampling(self, spatial_upsampling) -> bool:
        return self.scale_for_upsampling(spatial_upsampling) is not None

    def validate_upsampling(self, spatial_upsampling, image_mode: int) -> str:
        if not self.is_upsampling(spatial_upsampling):
            return ""
        if not self.enabled():
            return "FlashVSR Spatial Upsampling is disabled in Configuration > Extensions"
        return ""

    def query_download_def(self, enabled_only: bool = True, model: str | None = None, variant: str | None = None) -> dict[str, Any] | None:
        if enabled_only and not self.enabled():
            return None
        model = self.normalize_model(model if model is not None else self.model())
        if variant is None:
            _, variant, _ = self.settings()
        if model == self.MODEL_PLUS_TURBO:
            decoder = self.PLUS_VAE_FILENAME if variant == "full" else self.PLUS_TCDECODER_FILENAME
            remote_decoder = self.PLUS_VAE_REMOTE_FILENAME if variant == "full" else self.PLUS_TCDECODER_REMOTE_FILENAME
            return {
                "repoId": self.PLUS_REPO_ID,
                "revision": self.PLUS_REPO_REVISION,
                "sourceFolderList": ["turbo", ""],
                "targetFolderList": ["FlashVSR", "FlashVSR"],
                "fileList": [
                    [self.PLUS_TRANSFORMER_FILENAME, self.PLUS_LQ_PROJ_FILENAME],
                    [decoder],
                ],
                "fileNameMap": {
                    self.PLUS_TRANSFORMER_FILENAME: self.PLUS_TRANSFORMER_REMOTE_FILENAME,
                    self.PLUS_LQ_PROJ_FILENAME: self.PLUS_LQ_PROJ_REMOTE_FILENAME,
                    decoder: remote_decoder,
                },
            }
        return {
            "repoId": "DeepBeepMeep/Wan2.1",
            "sourceFolderList": ["FlashVSR", ""],
            "fileList": [[self.TRANSFORMER_FILENAME, self.LQ_PROJ_FILENAME, self.TCDECODER_FILENAME, self.POSI_PROMPT_FILENAME], [self.VAE_FILENAME]],
        }

    def _locate_flashvsr_file(self, filename: str) -> str:
        return self.files_locator.locate_file(os.path.join("FlashVSR", filename))

    def _locate_plus_file(self, filename: str, *, turbo: bool = False, error_if_none: bool = True) -> str | None:
        relative = os.path.join("FlashVSR", "turbo", filename) if turbo else os.path.join("FlashVSR", filename)
        return self.files_locator.locate_file(relative, error_if_none=error_if_none)

    def paths(self, variant: str, model: str | None = None):
        from postprocessing.flashvsr.runtime import FlashVSRPaths
        model = self.normalize_model(model if model is not None else self.model())
        if model == self.MODEL_PLUS_TURBO:
            return FlashVSRPaths(
                transformer=self._locate_plus_file(self.PLUS_TRANSFORMER_FILENAME, turbo=True),
                lq_proj=self._locate_plus_file(self.PLUS_LQ_PROJ_FILENAME, turbo=True),
                posi_prompt=self._locate_plus_file(self.PLUS_POSI_PROMPT_FILENAME, turbo=True),
                tcdecoder=None if variant == "full" else self._locate_plus_file(self.PLUS_TCDECODER_FILENAME),
                vae=self._locate_plus_file(self.PLUS_VAE_FILENAME) if variant == "full" else None,
            )
        return FlashVSRPaths(
            transformer=self._locate_flashvsr_file(self.TRANSFORMER_FILENAME),
            lq_proj=self._locate_flashvsr_file(self.LQ_PROJ_FILENAME),
            posi_prompt=self._locate_flashvsr_file(self.POSI_PROMPT_FILENAME),
            tcdecoder=None if variant == "full" else self._locate_flashvsr_file(self.TCDECODER_FILENAME),
            vae=self.files_locator.locate_file(self.VAE_FILENAME) if variant == "full" else None,
        )

    def vae_tile_size(self, vae_config: int, output_height: int | None = None, output_width: int | None = None) -> int:
        import torch
        from models.wan.modules.vae import WanVAE

        device_mem_capacity = torch.cuda.get_device_properties(0).total_memory / 1048576 if torch.cuda.is_available() else 0
        mixed_precision = self.server_config.get("vae_precision", "16") == "32"
        # Maestro's fork of WanVAE.get_VAE_tile_size predates upstream's optional
        # output_height/output_width args. Use them when available (better
        # output-resolution-aware tiling), fall back to the 3-arg signature.
        try:
            return WanVAE.get_VAE_tile_size(vae_config, device_mem_capacity, mixed_precision, output_height=output_height, output_width=output_width)
        except TypeError:
            return WanVAE.get_VAE_tile_size(vae_config, device_mem_capacity, mixed_precision)

    def _plus_prompt_path(self, *, create: bool = False) -> str | None:
        relative = os.path.join("FlashVSR", "turbo", self.PLUS_POSI_PROMPT_FILENAME)
        path = self.files_locator.locate_file(relative, error_if_none=False)
        if path is not None or not create:
            return path
        smart_location = getattr(self.files_locator, "get_smart_download_location", None)
        if callable(smart_location):
            return smart_location(os.path.join("turbo", self.PLUS_POSI_PROMPT_FILENAME), force_path="FlashVSR")
        return self.files_locator.locate_file(relative, create_path_if_none=True)

    @classmethod
    def _plus_prompt_git_blob_sha1(cls, path: str) -> tuple[str, int]:
        digest = hashlib.sha1()
        digest.update(f"blob {cls.PLUS_GITHUB_PROMPT_SIZE}\0".encode("ascii"))
        size = 0
        with open(path, "rb") as source:
            while block := source.read(1024 * 1024):
                size += len(block)
                digest.update(block)
        return digest.hexdigest(), size

    def _download_plus_prompt(self) -> bool:
        with _PLUS_PROMPT_DOWNLOAD_LOCK:
            existing = self._plus_prompt_path()
            if existing is not None:
                actual_sha1, actual_size = self._plus_prompt_git_blob_sha1(existing)
                if actual_sha1 != self.PLUS_GITHUB_PROMPT_BLOB_SHA1 or actual_size != self.PLUS_GITHUB_PROMPT_SIZE:
                    raise RuntimeError(
                        "FlashVSR+ prompt integrity check failed for the pinned official prompt: "
                        f"{existing} (size={actual_size}, git blob={actual_sha1})"
                    )
                return False

            destination = self._plus_prompt_path(create=True)
            if not destination:
                raise RuntimeError("Unable to choose a checkpoint path for the FlashVSR+ prompt tensor")
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            url = (
                "https://raw.githubusercontent.com/OpenImagingLab/FlashVSR_Plus/"
                f"{self.PLUS_GITHUB_REVISION}/prompt_tensor/posi_prompt.pth"
            )
            request = urllib.request.Request(url, headers={"User-Agent": "Maestro FlashVSR+ model downloader"})
            fd, temporary = tempfile.mkstemp(prefix=".flashvsr-plus-prompt-", suffix=".part", dir=os.path.dirname(destination))
            digest = hashlib.sha1()
            digest.update(f"blob {self.PLUS_GITHUB_PROMPT_SIZE}\0".encode("ascii"))
            total_size = 0
            try:
                with os.fdopen(fd, "wb") as target, urllib.request.urlopen(request, timeout=60) as response:
                    while block := response.read(1024 * 1024):
                        target.write(block)
                        digest.update(block)
                        total_size += len(block)
                    target.flush()
                    os.fsync(target.fileno())
                actual_sha1 = digest.hexdigest()
                if total_size != self.PLUS_GITHUB_PROMPT_SIZE or actual_sha1 != self.PLUS_GITHUB_PROMPT_BLOB_SHA1:
                    raise RuntimeError(
                        "FlashVSR+ prompt integrity check failed for the pinned official prompt: "
                        f"size={total_size}, git blob={actual_sha1}"
                    )
                os.replace(temporary, destination)
            finally:
                if os.path.exists(temporary):
                    os.remove(temporary)
            return True

    def download(self, process_files: Callable[..., Any], send_cmd=None, status_text: str | None = None, *, model: str | None = None, variant: str | None = None) -> bool:
        model = self.normalize_model(model if model is not None else self.model())
        if variant is None:
            _, variant, _ = self.settings()
        flashvsr_def = self.query_download_def(model=model, variant=variant)
        if flashvsr_def is None:
            return False
        if model == self.MODEL_PLUS_TURBO:
            required = [
                self._locate_plus_file(self.PLUS_TRANSFORMER_FILENAME, turbo=True, error_if_none=False),
                self._locate_plus_file(self.PLUS_LQ_PROJ_FILENAME, turbo=True, error_if_none=False),
            ]
            if variant == "full":
                required.append(self._locate_plus_file(self.PLUS_VAE_FILENAME, error_if_none=False))
            else:
                required.append(self._locate_plus_file(self.PLUS_TCDECODER_FILENAME, error_if_none=False))
            missing = any(path is None for path in required)
            prompt_path = self._plus_prompt_path()
            if not missing and prompt_path is not None:
                prompt_downloaded = self._download_plus_prompt()
                self.verify_plus_assets(self.paths(variant, model=model), variant)
                return prompt_downloaded
            from shared.utils.download import send_download_status

            send_download_status(send_cmd, status_text)
            if missing:
                process_files(**flashvsr_def)
            prompt_downloaded = self._download_plus_prompt()
            self.verify_plus_assets(self.paths(variant, model=model), variant)
            return missing or prompt_downloaded

        required = [os.path.join("FlashVSR", self.TRANSFORMER_FILENAME), os.path.join("FlashVSR", self.LQ_PROJ_FILENAME), os.path.join("FlashVSR", self.POSI_PROMPT_FILENAME)]
        required.append(self.VAE_FILENAME if variant == "full" else os.path.join("FlashVSR", self.TCDECODER_FILENAME))
        if all(self.files_locator.locate_file(path, error_if_none=False) is not None for path in required):
            return False
        from shared.utils.download import send_download_status

        send_download_status(send_cmd, status_text)
        process_files(**flashvsr_def)
        return True

    def upscale(self, sample, spatial_upsampling, *, seed=0, continue_cache=None, return_continue_cache=False, vae_tile_size=None, process_files: Callable[..., Any], vae_config: int, init_pipe: Callable[..., int], profile, still_image=False, abort_callback=None, progress_callback=None):
        scale = self.scale_for_upsampling(spatial_upsampling)
        if scale is None:
            raise ValueError(f"Unknown FlashVSR upsampling mode: {spatial_upsampling}")
        enabled, variant, persistence = self.settings()
        if not enabled:
            raise RuntimeError("FlashVSR spatial upsampling is disabled in Configuration > Extensions.")
        model = self.model_for_upsampling(spatial_upsampling)
        # The caller's profile is tuned for the MAIN diffusion model (often a
        # low-VRAM profile for 20B+ models). MMGP's streaming budgets can be
        # smaller than FlashVSR's 2.7 GB of weights (notably profiles 4/5),
        # requiring repeated RAM-to-GPU copies. The whole FlashVSR pipe is
        # ~3.4 GB: when free VRAM comfortably fits it, promote to
        # profile 3 (HighVRAM, "*" budget 70%) so weights load once and stay
        # resident. Low-VRAM cards keep the caller's profile untouched.
        if profile in (2, 4, 5):
            try:
                import torch
                if torch.cuda.is_available():
                    free_bytes, _total = torch.cuda.mem_get_info()
                    if free_bytes > 8 * (1 << 30):
                        print(f"[FlashVSR] {free_bytes / (1 << 30):.1f} GB VRAM free — promoting offload profile {profile} -> 3 (weights stay VRAM-resident)")
                        profile = 3
            except Exception:
                pass
        self.download(process_files, model=model, variant=variant)
        from postprocessing.flashvsr.attention_backend import set_sparse_backend
        set_sparse_backend(self.backend())
        from postprocessing.flashvsr.runtime import upscale_video

        output_height = int(sample.shape[-2] * scale)
        output_width = int(sample.shape[-1] * scale)
        flashvsr_tile_size = self.vae_tile_size(vae_config, output_height, output_width)
        return upscale_video(
            sample,
            scale,
            self.paths(variant, model=model),
            variant=variant,
            model=model,
            seed=seed,
            continue_cache=continue_cache,
            return_continue_cache=return_continue_cache,
            persistent_models=persistence == self.PERSIST_RAM,
            vae_tile_size=flashvsr_tile_size,
            topk_ratio=self.topk_ratio(),
            init_pipe=init_pipe,
            profile=profile,
            still_image=still_image,
            two_pass=self.is_two_pass_upsampling(spatial_upsampling),
            abort_callback=abort_callback,
            progress_callback=progress_callback,
        )

    def release_vram(self) -> None:
        from postprocessing.flashvsr.runtime import release_models
        release_models()
