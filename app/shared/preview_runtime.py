"""Optional preview setup and cadence, separate from the render's lifecycle."""
from __future__ import annotations

import time

from services.generation_preview import preview_mode


def preview_context(gen, mode="rgb"):
    return {
        "serial": time.monotonic_ns(), "mode": mode,
        "window": max(1, int(gen.get("window_no", 1) or 1)),
        "total_windows": max(1, int(gen.get("total_windows", 1) or 1)),
        "clip": max(1, int(gen.get("prompt_no", 1) or 1)),
        "total_clips": max(1, int(gen.get("prompts_max", 1) or 1)),
    }


def begin_preview_window(send_cmd, gen, mode):
    """Clear the last window before preparation, without optional failures."""
    if preview_mode(mode) == "off":
        return
    try:
        send_cmd("preview_context", preview_context(gen, mode))
    except Exception as error:
        print(f"[Preview] Could not clear the previous window: {error}")


def decoder_key(mode, architecture, model_def):
    if preview_mode(mode) in ("off", "rgb") or model_def.get("external_runtime") or model_def.get("audio_only"):
        return None
    from shared.tinyvae.decoder import decoder_for
    # Maestro's H3 names predate WanGP's FL2VA/pruned naming. They expose the
    # same normalized 24-channel preview latents as the upstream H3 decoder.
    architecture = {
        "minimax_h3": "minimax_h3_fl2va_pruned",
        "minimax_h3_full": "minimax_h3_fl2va",
        "minimax_h3_ref2va_full": "minimax_h3_ref2va",
        "ltx2": "ltx2_19B",
        "ltx2_25": "ltx2_25_22B",
    }.get(architecture, architecture)
    return decoder_for(architecture, model_def)


def rgb_preview_supported(model_handler):
    """Use the same registered RGB preview contract for setup and the catalog."""
    return hasattr(model_handler, "get_rgb_factors") or callable(
        getattr(model_handler, "preview_latents", None)
    )


def model_preview_support(architecture, model_def, model_handler):
    """Describe native preview support without loading or downloading a decoder.

    Image jobs use the Tiny VAE video preference as still previews. This reports
    registered model support; optional decoder setup can still fail at runtime.
    """
    if model_def.get("audio_only") or model_def.get("external_runtime"):
        return {"rgb": False, "tiny_vae_frames": False, "tiny_vae_video": False}
    tiny_supported = decoder_key("tiny_vae_frames", architecture, model_def) is not None
    return {
        "rgb": rgb_preview_supported(model_handler),
        "tiny_vae_frames": tiny_supported,
        "tiny_vae_video": tiny_supported,
    }


def prepare_preview_decoder(mode, architecture, model_def, gen=None):
    """Return (decoder, requested key, notice); optional failures stay local."""
    if preview_mode(mode) in ("off", "rgb") or model_def.get("audio_only"):
        return None, None, None
    key = None
    try:
        key = decoder_key(mode, architecture, model_def)
        if key is None:
            return None, None, "Clearer previews are unavailable for this model. Using fast frames where supported."
        import torch
        from mmgp import offload
        from shared.tinyvae.decoder import prepare_decoder, load_decoder
        if not getattr(offload.offload, "supports_cotenant_wildcards", False):
            raise RuntimeError("MMGP lacks wildcard cotenant support")
        path = prepare_decoder(key, gen=gen)
        # The preview must not change the seed used by the actual renderer.
        with torch.random.fork_rng(devices=[]):
            decoder = load_decoder(key, path)
        return decoder, key, None
    except Exception as error:
        print(f"[Preview] Tiny VAE could not start: {error}")
        return None, key, "Clearer previews could not start. Using fast frames where supported."


class RGBPreviewSession:
    """Sample the existing RGB preview contract without copying every step."""

    def __init__(self, capture, send_cmd, gen):
        self._capture, self.send_cmd, self.gen = capture, send_cmd, gen
        self.context = None
        self.last_step = -1
        self.failed = False

    def wants_capture(self, step, total, pass_no=-1):
        if self.failed or self.gen.get("abort", False) or step < 0:
            return False
        context = (pass_no, total)
        last = self.last_step if context == self.context and step >= self.last_step else -1
        return last < 0 or step == total - 1 or step - last >= max(1, (total + 4) // 6)

    def capture(self, latent, step, total, pass_no=-1):
        if not self.wants_capture(step, total, pass_no):
            return
        self.context, self.last_step = (pass_no, total), step
        try:
            self._capture(latent)
        except Exception as error:
            self.failed = True
            print(f"[Preview] Fast preview disabled: {error}")
            try:
                self.send_cmd("preview_notice", "Live previews are unavailable for this render. Generation will continue.")
            except Exception as notice_error:
                print(f"[Preview] Could not publish preview notice: {notice_error}")

    def close(self, cancel=False):
        self.failed = True
