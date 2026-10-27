"""Per-generation placement and private, hardware-scoped performance evidence.

No GPU allocations, remote telemetry, prompts, or automatic benchmark jobs.
The existing model workspace guard remains authoritative over learned budgets.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import statistics
import sys
import threading
import time
from pathlib import Path
from contextlib import contextmanager

from services.h3_performance_policy import h3_performance_advice

# Revision 4 fingerprints the effective per-job H3 attention head split.
REVISION = 4


def _number(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _auto_preload_runtime_identity():
    """Fingerprint the selected MMGP planner source without importing it.

    Legacy keeps the previous raw-source hash, so its hardware-history key is
    unchanged. Experimental/unknown imports also include package identity and
    version to prevent sharing adaptive-preload evidence across runtimes.
    """
    app_dir = Path(__file__).resolve().parents[1]
    package = sys.modules.get("mmgp")
    if package is None:
        # Keep the historical path and digest when called before MMGP startup.
        path = app_dir / "mmgp" / "auto_preload.py"
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return "unavailable"

    planner = sys.modules.get("mmgp.auto_preload")
    origin = getattr(planner, "__file__", None) if planner is not None else None
    package_origin = getattr(package, "__file__", None)
    package_dir = Path(package_origin).resolve().parent if package_origin else None
    path = Path(origin).resolve() if origin else (
        package_dir / "auto_preload.py" if package_dir is not None else None
    )
    try:
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest() if path is not None else "unavailable"
    except OSError:
        source_hash = "unavailable"

    from services.memory_runtime import active_runtime_selection
    selection = active_runtime_selection(app_dir, getattr(sys.modules.get("wgp"), "memory_runtime_selection", None))
    backend = selection.backend if selection is not None else "unknown"
    if backend == "legacy":
        return source_hash
    version = str(getattr(package, "__version__", "unknown"))
    return _digest({"backend": backend, "version": version, "source": source_hash})


def auto_owns(config, key):
    services = config.get("services", {})
    defaults = services.get("auto_performance_defaults", {})
    return bool(services.get("auto_performance", True) and key in defaults
                and config.get(key) == defaults[key])


def _blackwell_16gb_h3_video_class(hardware, model_def, params, output_type):
    """Match the reported Windows RTX 50 16-GB / 64-GB H3 compatibility class."""
    architecture = str(model_def.get("architecture") or "").strip().lower()
    if not architecture.startswith("minimax_h3") or str(output_type or "video").lower() != "video":
        return False
    excluded_tokens = ("fasth3", "viggle", "voice_audio", "audio_only", "image")
    if any(token in architecture for token in excluded_tokens):
        return False

    excluded_flags = (
        "minimax_h3_viggle",
        "minimax_h3_fasth3",
        "minimax_h3_fasth3_omniref",
        "audio_only",
        "minimax_h3_audio_only",
        "image_only",
        "minimax_h3_image_only",
    )
    if any(bool(model_def.get(key)) for key in excluded_flags):
        return False

    def enabled(value):
        if isinstance(value, str):
            return value.strip().lower() not in ("", "0", "false", "no", "none", "off")
        return bool(value)

    model_type = str(params.get("model_type") or "").strip().lower()
    if (enabled(params.get("audio_only")) or enabled(params.get("image_mode"))
            or any(token in model_type for token in excluded_tokens)):
        return False

    if str(hardware.get("platform") or "").lower() != "win32":
        return False
    vram = _number(hardware.get("gpu_vram_gb"))
    ram = _number(hardware.get("ram_gb"))
    if not (15 <= vram <= 16.5 and 63.5 <= ram < 64.5):
        return False

    capability_tuple = hardware.get("gpu_capability_tuple")
    if isinstance(capability_tuple, (tuple, list)) and len(capability_tuple) >= 2:
        return capability_tuple[0] == 12 and capability_tuple[1] == 0

    capability = str(hardware.get("gpu_capability") or "").strip().lower().replace("_", "")
    if capability:
        return capability in ("sm120", "120", "12.0")

    # Some hardware snapshots omit CUDA capability. Fall back only to an
    # unambiguous GeForce RTX 50-series name. RTX 5000 Ada is a different
    # product generation and must not match this fallback.
    name = str(hardware.get("gpu_name") or "")
    return bool(re.search(r"\bgeforce\s+rtx[\s-]*50[1-9]\d\b", name, flags=re.IGNORECASE))


def hardware_key(hardware):
    return _digest({key: hardware.get(key) for key in (
        "gpu_name", "gpu_vram_gb", "ram_gb", "gpu_capability", "driver_version",
        "torch_version", "runtime_version", "platform", "machine", "allocator_active", "runtime_signature", "ram_allocator_active",
    )} | {
        "revision": REVISION,
        "auto_preload_runtime": _auto_preload_runtime_identity(),
    })


def workload_key(config, model_def, params, cli=None, *, effective_attention_head_split=None):
    # Hash file identities/stat revisions; never persist local filenames.
    def file_stamp(value):
        if isinstance(value, (list, tuple)):
            return [file_stamp(item) for item in value]
        if isinstance(value, dict):
            return {key: file_stamp(item) for key, item in value.items()}
        if isinstance(value, str):
            try:
                info = Path(value).stat()
                return (value, info.st_size, info.st_mtime_ns)
            except OSError:
                pass
        return value

    keys = ("model_type", "resolution", "image_mode", "num_inference_steps",
            "activated_loras", "loras_multipliers", "loras_multipliers2",
            "guidance_scale", "flow_shift", "sampler", "cache_type", "tea_cache",
            "single_stage_pipeline", "progressive_pipeline", "duration_seconds")
    workload = {key: file_stamp(params.get(key)) for key in keys}
    frames = _number(params.get("video_length"), 1)
    window = _number(params.get("sliding_window_size"), frames)
    workload["frames"] = min(frames, window) if window > 0 else frames
    workload["total_frames"] = frames
    workload["repeats"] = params.get("repeat_generation", params.get("num_repeats", 1))
    workload["window_prompts"] = [len(str(item)) for item in params.get("window_prompts", [])]
    workload["prompt_lengths"] = [len(item) for item in str(params.get("prompt", "")).split("\n")]
    # Reference identity is hashed, including duration/trimming/type metadata.
    # Exact matches are deliberately conservative: no extrapolation to heavier jobs.
    workload["references"] = {key: file_stamp(params.get(key)) for key in (
        "image_refs", "scene_references", "video_guide", "audio_guide", "video_source",
        "minimax_h3_references", "audio_guide2", "audio_guide3", "image_start", "image_end",
    )}
    # Include remaining generation options (e.g. per-request attention, cache
    # and sliding-window overlap). Placement overrides are comparison axes.
    excluded = {"override_profile", "workspace", "seed", "client_submission_id",
                "show_in_gallery", "performance_plan", "enhancement"}
    workload["generation_options"] = {
        key: file_stamp(value) for key, value in params.items()
        if key not in excluded and not key.startswith("_") and "prompt" not in key
    }
    workload["model"] = file_stamp(model_def)
    runtime = {key: config.get(key) for key in (
        "transformer_quantization", "attention_mode", "int8_kernels", "compile",
        "attention_head_split", "vae_config", "smart_memory_pinning", "read_ahead",
        "vram_safety_coefficient", "generation_preview", "generation_preview_mode",
    )}
    if effective_attention_head_split is not None:
        runtime["attention_head_split"] = effective_attention_head_split
    # Preload settings change the way MMGP moves model weights, not just a
    # placement candidate. Keep each output family's selected mode and amount
    # in the workload fingerprint so manual/dynamic evidence cannot train an
    # automatic/default plan for an otherwise identical generation.
    try:
        legacy_preload_mb = max(0, int(config.get("preload_in_VRAM", 0) or 0))
    except (TypeError, ValueError, OverflowError):
        legacy_preload_mb = 0
    preload_settings = {}
    for kind in ("video", "image", "audio"):
        mode = config.get(f"{kind}_preload_mode")
        if mode is None:
            mode = "manual" if legacy_preload_mb else "default"
        if mode not in ("default", "dynamic", "manual"):
            mode = "default"
        try:
            requested_mb = max(
                0,
                int(config.get(f"{kind}_preload_in_VRAM", legacy_preload_mb) or 0),
            )
        except (TypeError, ValueError, OverflowError):
            requested_mb = 0
        preload_settings[kind] = {"mode": mode, "requested_mb": requested_mb}
    try:
        cli_preload_mb = max(0, int((cli or {}).get("preload", 0) or 0))
    except (TypeError, ValueError, OverflowError):
        cli_preload_mb = 0
    if cli_preload_mb:
        # The CLI amount overrides the saved mode/amount for every output.
        for settings in preload_settings.values():
            settings.update(mode="manual", requested_mb=cli_preload_mb)
    runtime["preload"] = preload_settings
    workload["runtime"] = runtime
    return _digest(workload)


def build_performance_plan(hardware, config, model_def, params, adjustment=None,
                           *, output_type="video", profile=1, cli=None,
                           available_ram_gb=None, own_pinned_gb=0, external_vram_gb=0, history=None):
    """Plan temporary settings without altering saved config or job quality."""
    adjustment = adjustment or {}
    cli = cli or {}
    advice_params = dict(params or {})
    advice_params["_performance_output_type"] = output_type
    h3_advice = h3_performance_advice(hardware, config, model_def, advice_params)
    saved_head_split = _number(config.get("attention_head_split"), 0)
    # Preserve supported manual levels, including High (3), outside the
    # bounded Off/Low/Medium per-job advice policy.
    saved_head_split = (
        int(saved_head_split)
        if saved_head_split is not None and saved_head_split >= 0 and saved_head_split.is_integer()
        else 0
    )
    override_head_split = bool(
        hardware.get("cuda_available")
        and h3_advice is not None
        and h3_advice.get("attention_head_split") in (0, 1, 2)
        and auto_owns(config, "attention_head_split")
    )
    effective_head_split = (
        int(h3_advice["attention_head_split"]) if override_head_split else saved_head_split
    )
    vram = _number(hardware.get("gpu_vram_gb"))
    ram = _number(hardware.get("ram_gb"))
    default_ram_fraction = .4 if hardware.get("platform") == "win32" else .5
    mode = config.get(f"{output_type}_preload_mode", "default")
    explicit_profile = _number(params.get("override_profile"), -1) >= 0
    applied = bool(hardware.get("cuda_available") and vram > 0
                   and auto_owns(config, f"{output_type}_profile")
                   and mode == "default" and not explicit_profile
                   and _number(cli.get("profile"), -1) < 0
                   and not any(_number(cli.get(key)) > 0 for key in ("preload", "transformer_budget")))
    plan = {"applied": applied, "output_type": output_type, "profile": profile,
            "transformer_budget_mb": int(cli.get("transformer_budget") or 0),
            "hardware_key": hardware_key(hardware),
            "workload_key": workload_key(
                config, model_def, params, cli=cli,
                effective_attention_head_split=effective_head_split,
            ),
            "reasons": [], "warnings": [], "matching_renders": 0,
            "attention_head_split": effective_head_split,
            "override_attention_head_split": override_head_split,
            "h3_workload_advice": h3_advice,
            "source": "hardware" if applied else "manual",
            "override_read_ahead": False, "override_reserved_ram": False,
            "read_ahead": bool(config.get("read_ahead", False)),
            "reserved_ram_fraction": _number(cli.get("perc_reserved_mem_max")) or _number(config.get("perc_reserved_mem_max")) / 100 or default_ram_fraction}
    if h3_advice is not None:
        plan["reasons"].extend(h3_advice.get("reasons") or ())
        plan["warnings"].extend(h3_advice.get("warnings") or ())
        if h3_advice.get("attention_head_split") is None:
            plan["reasons"].append(
                f"H3 head-split advice is unavailable for this geometry; the saved level {saved_head_split} remains in effect."
            )
        elif override_head_split:
            plan["reasons"].append(
                f"Auto applies the recommended {h3_advice.get('split_label')} head-split level for this job; saved settings remain unchanged."
            )
        else:
            plan["reasons"].append(
                f"Advice only: H3 recommends {h3_advice.get('split_label')} head splitting; saved level {saved_head_split} remains in effect."
            )
    if (model_def.get("minimax_h3_full_checkpoint") and ram >= 96
            and _number(cli.get("perc_reserved_mem_max")) <= 0 and not config.get("perc_reserved_mem_max")):
        plan["reserved_ram_fraction"] = round(min(.65, max(.50, 60 / ram)), 3)
    if not applied:
        if explicit_profile and override_head_split:
            plan["reasons"].append(
                "The explicit profile keeps placement in manual control; Auto still applies its H3 head-split advice per job."
            )
        elif h3_advice is None:
            plan["reasons"].append("Saved settings or explicit overrides remain in control.")
        return plan
    is_h3 = str(model_def.get("architecture", "")).startswith("minimax_h3") or "h3_weight_budget_gb" in adjustment
    clamped = adjustment.get("h3_activation_reserve_clamped", False)
    # The previous guard exports this policy when the requested workspace cannot fit.
    clamped = clamped or adjustment.get("h3_residency_policy") == "profile_default"
    cap_gb = min(_number(adjustment.get("h3_weight_budget_gb"), vram),
                 _number(adjustment.get("effective_coef"), .7) * vram) * .97
    # A cold worker can see memory occupied by other applications. A warm
    # worker passes zero here because its own model residency is reusable.
    cap_gb = max(0, cap_gb - max(0, _number(external_vram_gb) - .75))
    if external_vram_gb > 2:
        plan["warnings"].append("Other GPU activity is using VRAM before model loading; close other GPU applications if memory is tight.")
    allowed_profiles = [profile]
    blackwell_16gb_h3_video = is_h3 and _blackwell_16gb_h3_video_class(
        hardware, model_def, params, output_type
    )
    if is_h3:
        if blackwell_16gb_h3_video:
            # The previous RAM-tier boundary classified a 63.7-GB host as
            # low-RAM and used Profile 4. Keep this narrow Auto-owned class
            # on transformer-only pinning; local evidence may choose only
            # the similarly conservative profiles below.
            plan["profile"] = 4
            allowed_profiles = [4, 4.5, 5]
            plan["reasons"].append(
                "H3 Auto uses Profile 4 as the baseline on Windows 16 GB Blackwell / 64 GB hosts."
            )
        elif ram >= 96 and vram >= 20:
            plan["profile"] = 2
            allowed_profiles = [2, 5]
            plan["reasons"].append("H3 streams transformer weights while keeping host models pinned (Profile 2).")
        elif ram < 64:
            # Extra host pinning is only learned from repeated safe evidence;
            # it is never the untested starting point on a smaller RAM host.
            allowed_profiles = list(dict.fromkeys([profile, 4.5, 5])) if ram >= 24 else [profile, 5]
            plan["reasons"].append("Conservative host placement protects the operating system on a smaller RAM machine.")
        if clamped:
            plan["transformer_budget_mb"] = 0
            plan["warnings"].append("The requested H3 workspace is tight for this GPU. Profile defaults are retained; a shorter window or lower resolution may be needed.")
        else:
            target = vram / 3 if ram >= 96 else min(3, vram / 5)
            fallback_budget_mb = max(0, int(min(target, cap_gb) * 1024) // 256 * 256)
            raw_guard_candidate_mb = adjustment.get("h3_residency_candidate_mb")
            guard_candidate_mb = _number(raw_guard_candidate_mb, 0)
            has_guard_candidate = (
                not isinstance(raw_guard_candidate_mb, bool)
                and guard_candidate_mb > 0
                and guard_candidate_mb.is_integer()
            )
            if blackwell_16gb_h3_video and has_guard_candidate:
                # The pre-load H3 guard already sized residency against the
                # packed-sequence workspace. Reuse its candidate, while
                # honoring any additional VRAM occupied by other processes.
                plan["transformer_budget_mb"] = min(
                    int(guard_candidate_mb), max(0, int(cap_gb * 1024))
                )
                plan["reasons"].append(
                    "H3 Auto preserves the pre-load workspace guard's residency allowance, bounded by currently available VRAM."
                )
            else:
                plan["transformer_budget_mb"] = fallback_budget_mb
                plan["reasons"].append(f"Transformer allowance {plan['transformer_budget_mb'] / 1024:g} GB; activation and reference reserves remain protected.")
    # Available RAM includes reclaimable MMGP-owned pinned buffers. Do not
    # mistake our warm model cache for an unrelated application's memory.
    if available_ram_gb is not None and ram > 0:
        spare = max(0, _number(available_ram_gb) + _number(own_pinned_gb) - max(4, ram * .10))
        if auto_owns(config, "read_ahead"):
            plan["override_read_ahead"] = True
            plan["read_ahead"] = bool(config.get("read_ahead") and spare >= 32)
        if auto_owns(config, "perc_reserved_mem_max") and _number(cli.get("perc_reserved_mem_max")) <= 0:
            plan["override_reserved_ram"] = True
            desired = max(.50, 60 / ram) if is_h3 and model_def.get("minimax_h3_full_checkpoint") and ram >= 96 else default_ram_fraction
            plan["reserved_ram_fraction"] = round(max(.01, min(.65, desired, spare / ram)), 4)
            if spare < 4 and is_h3:
                plan["profile"] = 5
                allowed_profiles = [5]
                plan["warnings"].append("Very little host RAM is available; full model pinning is avoided (Profile 5).")
        if spare < 8:
            plan["warnings"].append("Low available host RAM can cause paging; close other memory-heavy applications before generating.")
    if history is not None:
        choice, count = history.choose(plan, vram, ram, allowed_profiles,
                                       max_budget_mb=int(cap_gb * 1024), clamped=clamped)
        plan["matching_renders"] = count
        if choice:
            plan.update(profile=choice["profile"], transformer_budget_mb=choice["budget"], source="local_history")
            plan["reasons"].append("Placement selected from repeated matching local renders with memory headroom.")
    if blackwell_16gb_h3_video:
        plan["reasons"].append(
            f"H3 Auto selected Profile {plan['profile']:g} for this Windows 16 GB Blackwell / 64 GB host."
        )
    return plan


class PerformanceHistory:
    """Bounded local SQLite evidence. Connections are short-lived/thread-safe."""
    def __init__(self, path, limit=500):
        self.path = Path(path)
        self.limit = limit
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS renders (id TEXT PRIMARY KEY, hardware TEXT, workload TEXT, created REAL, data TEXT)")

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(str(self.path), timeout=3)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def record(self, job_id, plan, measurements, *, completed=False):
        # Cancelled/failed generations and missing device telemetry are not evidence.
        if not completed or not plan or measurements.get("samples", 0) < 2:
            return False
        if _number(measurements.get("denoising_seconds")) <= 0 or _number(measurements.get("peak_vram_gb")) <= 0:
            return False
        data = {"profile": plan["profile"], "budget": plan["transformer_budget_mb"],
                "reserved": plan.get("reserved_ram_fraction"), "read_ahead": plan.get("read_ahead"),
                **measurements}
        with self._connect() as db:
            db.execute("INSERT OR IGNORE INTO renders VALUES (?, ?, ?, ?, ?)",
                       (str(job_id), plan["hardware_key"], plan["workload_key"], time.time(), json.dumps(data)))
            db.execute("DELETE FROM renders WHERE id NOT IN (SELECT id FROM renders ORDER BY created DESC LIMIT ?)", (self.limit,))
        return True

    def choose(self, plan, vram, ram, allowed_profiles, *, max_budget_mb, clamped=False):
        with self._connect() as db:
            rows = db.execute("SELECT data FROM renders WHERE hardware=? AND workload=? ORDER BY created DESC LIMIT 100",
                              (plan["hardware_key"], plan["workload_key"])).fetchall()
        groups = {}
        for row in rows:
            data = json.loads(row[0])
            # A new RAM-pressure plan must not reuse evidence from looser budgets.
            if data.get("reserved") != plan.get("reserved_ram_fraction") or data.get("read_ahead") != plan.get("read_ahead"):
                continue
            profile, budget = data["profile"], data["budget"]
            if profile not in allowed_profiles or budget > max_budget_mb or (clamped and budget):
                continue
            if data["peak_vram_gb"] > vram - max(.5, vram * .03) or _number(data.get("min_available_ram_gb")) < max(3, ram * .05):
                continue
            groups.setdefault((profile, budget), []).append(data)
        candidates = []
        for (profile, budget), samples in groups.items():
            if len(samples) >= 2:
                candidates.append({"profile": profile, "budget": budget,
                                   "seconds": statistics.median(item["denoising_seconds"] for item in samples),
                                   "vram": statistics.median(item["peak_vram_gb"] for item in samples)})
        # A single candidate has no measured comparison; don't call it a winner.
        if len(candidates) < 2:
            return None, len(rows)
        fastest = min(item["seconds"] for item in candidates)
        return min((item for item in candidates if item["seconds"] <= fastest * 1.03), key=lambda item: item["vram"]), len(rows)

    def summary(self, hw_key):
        with self._connect() as db:
            count, workloads = db.execute("SELECT count(*), count(DISTINCT workload) FROM renders WHERE hardware=?", (hw_key,)).fetchone()
        return {"completed_renders": count, "workloads": workloads,
                "minimum_samples": 2, "scope": "This hardware, runtime, checkpoint and matching workload"}

    def reset(self, hw_key):
        with self._connect() as db:
            db.execute("DELETE FROM renders WHERE hardware=?", (hw_key,))


class GenerationPerformanceMonitor:
    """Sample physical device memory, not PyTorch's custom-allocator counters.

    NVML is device-wide and polling can miss short spikes. Only denoising
    time drives comparisons, avoiding cold-load/encoding timing confounds.
    """
    def __init__(self, stats, phase, interval=1.0, clock=time.monotonic):
        self.stats, self.phase, self.interval, self.clock = stats, phase, interval, clock
        self.stop_event = threading.Event()
        self.thread = None
        self.started = clock()
        self.last = self.started
        self.last_denoising = False
        self.result = {"samples": 0, "peak_vram_gb": 0, "min_available_ram_gb": None,
                       "denoising_seconds": 0, "elapsed_seconds": 0,
                       "memory_scope": "Device-wide NVML samples; brief peaks may be missed"}

    def sample(self):
        now = self.clock()
        if self.last_denoising:
            self.result["denoising_seconds"] += now - self.last
        self.last = now
        phase = str(self.phase()).lower()
        self.last_denoising = "denois" in phase and "decod" not in phase
        try:
            stats = self.stats()
            gpu, ram = stats.get("gpu", {}), stats.get("ram", {})
            if gpu.get("available") and _number(gpu.get("vram_used_gb")) > 0:
                self.result["samples"] += 1
                self.result["peak_vram_gb"] = max(self.result["peak_vram_gb"], gpu["vram_used_gb"])
            available = ram.get("available_gb")
            if available is not None and _number(ram.get("total_gb")) > 0:
                previous = self.result["min_available_ram_gb"]
                self.result["min_available_ram_gb"] = min(previous, available) if previous is not None else available
        except Exception:
            pass

    def start(self):
        self.sample()
        def run():
            while not self.stop_event.wait(self.interval):
                self.sample()
        self.thread = threading.Thread(target=run, name="maestro-performance", daemon=True)
        self.thread.start()
        return self

    def finish(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=3)
        self.sample()
        self.result["elapsed_seconds"] = self.clock() - self.started
        return {key: round(value, 3) if isinstance(value, float) else value
                for key, value in self.result.items()}
