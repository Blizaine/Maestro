"""Shared Maestro finishing contracts for generation, Tools and Media Flow.

Native DLSS implementation: Wan2GP 1e1dd2757f24923f008593d9d4ec09062234be20.
See THIRD_PARTY_NOTICES.md and app/LICENSES/WanGP-Community-2.0.txt.
"""
from __future__ import annotations

import math

NR_SCALES = (1.0, 1.5, 1.724, 2.0, 3.0)
TEMPORAL_METHODS = {"": 1, "rife2": 2, "rife3": 3, "rife4": 4,
                    **{f"dlssg*{n}": n for n in range(2, 7)}}


def dlss_scale(method):
    if not str(method).startswith("dlss5*"):
        return None
    try:
        scale = float(str(method).split("*", 1)[1])
    except ValueError as error:
        raise ValueError("Invalid DLSS Neural Rendering scale") from error
    if scale not in NR_SCALES:
        raise ValueError("DLSS Neural Rendering supports x1, x1.5, x1.724, x2 and x3")
    return scale


def normalize_options(options=None):
    options = dict(options or {})
    intensity = float(options.get("dlss_intensity", 1.0))
    if not math.isfinite(intensity) or not 0 <= intensity <= 2:
        raise ValueError("DLSS intensity must be between 0 and 2")
    depth = options.get("dlss_depth", "half")
    motion = options.get("dlss_motion", "original")
    if depth not in ("full", "half", "quarter"):
        raise ValueError("DLSS depth precision must be full, half or quarter")
    if motion not in ("original", "raft"):
        raise ValueError("DLSS motion estimator must be original or raft")
    return {"dlss_intensity": intensity, "dlss_depth": depth, "dlss_motion": motion}


def capabilities(refresh=False):
    from postprocessing.dlss5 import runtime
    if refresh:
        runtime.dlssg_capabilities.cache_clear()
        runtime._gpu_series.cache_clear()
        runtime._hags_enabled.cache_clear()
    nr_reason = runtime.unavailable_reason(temporal=False)
    fg_reason = runtime.unavailable_reason(temporal=True)
    probe = runtime.dlssg_capabilities() if not fg_reason else {}
    maximum = min(6 if runtime.is_rtx_50_series() else 4,
                  int(probe.get("multi_frame_count_max", 0)) + 1)
    factors = [n for n in range(2, maximum + 1)
               if n == 2 or int(probe.get("worker_version", 0)) >= 2] if not fg_reason else []
    return {
        "rife": {"available": True, "factors": [2, 3, 4], "version": "4.26"},
        "neural_rendering": {"available": not nr_reason, "reason": nr_reason,
                             "scales": list(NR_SCALES)},
        "frame_generation": {"available": not fg_reason, "reason": fg_reason,
                             "factors": factors},
        "guide": "docs/DLSS5.md",
    }


def validate_methods(spatial="", temporal="", *, image=False, options=None, check_runtime=True):
    scale = dlss_scale(spatial)
    if spatial and scale is None and spatial not in (
        "lanczos1.5", "lanczos2", "lanczos3", "lanczos4", "vae1", "vae2",
        "flashvsr2", "flashvsr3", "flashvsr4", "flashvsr2pass2", "flashvsr2pass4",
    ):
        raise ValueError(f"Unknown spatial processor: {spatial}")
    if temporal not in TEMPORAL_METHODS:
        raise ValueError(f"Unknown temporal processor: {temporal}")
    if image and temporal:
        raise ValueError("Temporal upsampling requires a video")
    normalized = normalize_options(options)
    if check_runtime and (scale is not None or temporal.startswith("dlssg")):
        caps = capabilities()
        if scale is not None and not caps["neural_rendering"]["available"]:
            raise ValueError("DLSS Neural Rendering: " + caps["neural_rendering"]["reason"] + ". See docs/DLSS5.md.")
        if temporal.startswith("dlssg"):
            fg = caps["frame_generation"]
            if not fg["available"]:
                raise ValueError("DLSS Frame Generation: " + fg["reason"] + ". See docs/DLSS5.md.")
            if TEMPORAL_METHODS[temporal] not in fg["factors"]:
                raise ValueError(f"This GPU/runtime supports DLSS factors {fg['factors']}; x5/x6 requires supported RTX 50 hardware")
    return normalized


def prepare_dlss(options, *, neural=False):
    """Reuse the same guide weights as Maestro's depth and motion controls."""
    import wgp
    from shared.utils import files_locator as fl
    from postprocessing.dlss5 import runtime
    runtime.configure_depth_estimator(wgp.server_config)
    files = []
    if neural:
        variant = runtime.DEPTH_MODEL_VARIANT
        if variant not in ("vitl", "vitb"):
            variant = runtime.DEPTH_MODEL_VARIANT = "vitl"
        files.append(("depth", f"depth_anything_v2_{variant}.pth"))
    if options["dlss_motion"] == "raft":
        files.append(("flow", "raft-things.pth"))
    for folder, filename in files:
        if not fl.locate_file(f"{folder}/{filename}", error_if_none=False):
            wgp.process_files_def(repoId="DeepBeepMeep/Wan2.1", sourceFolderList=[folder],
                                  fileList=[[filename]])


def neural_render(sample, method, *, options=None, still_image=False,
                  abort_callback=None, progress_callback=None):
    from postprocessing.dlss5 import runtime
    normalized = validate_methods(method, image=still_image, options=options)
    prepare_dlss(normalized, neural=True)
    return runtime.neural_render(sample, dlss_scale(method), still_image=still_image,
        depth_resolution=normalized["dlss_depth"], motion_vector=normalized["dlss_motion"],
        intensity=normalized["dlss_intensity"], abort_callback=abort_callback,
        progress_callback=progress_callback)


def temporal_upsample(sample, previous_last_frame, method, fps, *, options=None,
                      abort_callback=None, progress_callback=None):
    import torch
    import wgp
    from shared.utils import files_locator as fl
    normalized = validate_methods(temporal=method, options=options)
    if not method:
        return sample, previous_last_frame, fps
    factor = TEMPORAL_METHODS[method]
    if method.startswith("dlssg"):
        from postprocessing.dlss5 import runtime
        prepare_dlss(normalized)
        return runtime.frame_generate(sample, previous_last_frame, fps, factor,
            motion_vector=normalized["dlss_motion"], abort_callback=abort_callback,
            progress_callback=progress_callback)
    from postprocessing.rife.inference import temporal_interpolation
    version = "v4" if factor == 3 else wgp.server_config.get("rife_version", "v4")
    filename = wgp.RIFE_V4_FILENAME if version == "v4" else wgp.RIFE_V3_FILENAME
    path = fl.locate_file(filename, error_if_none=False)
    if path is None:
        wgp.process_files_def(repoId="DeepBeepMeep/Wan2.1", sourceFolderList=[""], fileList=[[filename]])
        path = fl.locate_file(filename)
    last_frame = sample[:, -1:].clone()
    if previous_last_frame is not None:
        if previous_last_frame.dtype != sample.dtype:
            previous_last_frame = (previous_last_frame.add(1).mul(127.5).clamp(0, 255).to(torch.uint8)
                if sample.dtype == torch.uint8 else previous_last_frame.to(sample.dtype).div(127.5).sub(1))
        sample = torch.cat([previous_last_frame, sample], dim=1)
    result = temporal_interpolation(path, sample, device=wgp.processing_device, rife_version=version,
        multiplier=factor, abort_callback=abort_callback, progress_callback=progress_callback)
    if result is not None and previous_last_frame is not None:
        result = result[:, 1:]
    return result, last_frame, fps * factor
