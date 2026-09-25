"""Maestro registration for the unified Qwen Image 2.1 architecture."""

import os
from urllib.parse import urlsplit

from shared.utils.hf import build_hf_url


MODEL_TYPE = "qwen_image_21_7B"
REPO = "DeepBeepMeep/Qwen_image_2"
ENCODER_FOLDER = "Qwen3-VL-8B-Instruct"
# Complete Transformers 4.57-compatible processor export before the upstream
# folder was removed and its legacy tokenizer files renamed in Ideogram4.
PROCESSOR_REVISION = "55ab7995df218d8f759c6c283ef4c3be66111345"

_VIGGLE_PROFILES = {
    "viggle_v01": {
        "label": "Viggle Turbo v0.1 (4 steps)",
        "steps": 4,
        "guidance": 1.0,
        "filename": "Qwen-Image-2.1-viggle-turbo-4step-lora-r64.safetensors",
    },
    "viggle_v02": {
        "label": "Viggle Turbo v0.2 (5 steps)",
        "steps": 5,
        "guidance": 1.0,
        "filename": "Qwen-Image-2.1-viggle-turbo-v0.2-5step-lora-r256.safetensors",
    },
    "viggle_v021": {
        "label": "Viggle Turbo v0.2.1 (6 steps)",
        "steps": 6,
        "guidance": 1.0,
        "filename": "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors",
    },
}
_VIGGLE_REPO_URL = "https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo/resolve/main/"
for _profile in _VIGGLE_PROFILES.values():
    _profile["lora_url"] = _VIGGLE_REPO_URL + _profile["filename"]

_QWEN21_2K_VALUES = {
    "auto": "auto_2k",
    "21:9": "3136x1344",
    "16:9": "2752x1536",
    "9:16": "1536x2752",
    "3:2": "2528x1696",
    "2:3": "1696x2528",
    "4:3": "2400x1792",
    "3:4": "1792x2400",
    "1:1": "2048x2048",
}
_QWEN21_RESOLUTION_PRESETS = {
    "2k": {"label": "2K (native)", "values": _QWEN21_2K_VALUES},
}
_QWEN21_2K_PIXELS = 2048 * 2048


def _lora_filename(value):
    raw = str(value or "").replace("\\", "/")
    path = urlsplit(raw).path if raw.startswith(("http://", "https://")) else raw
    return path.rsplit("/", 1)[-1].casefold()


def _is_managed_viggle_lora(value):
    filename = _lora_filename(value)
    return any(filename == spec["filename"].casefold() for spec in _VIGGLE_PROFILES.values())


def _split_lora_multipliers(value):
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if not isinstance(value, str) or not value.strip():
        return []
    from shared.utils.loras_mutipliers import preparse_loras_multipliers
    return [str(item) for item in preparse_loras_multipliers(value)]


def _image_slots(value):
    if value is None or (isinstance(value, str) and not value.strip()):
        return []
    if isinstance(value, (list, tuple)):
        return [slot for slot in value if slot is not None and not (isinstance(slot, str) and not slot.strip())]
    return [value]


def apply_acceleration_profile(settings, *, selected_profile=None):
    """Normalize one managed Turbo adapter while preserving user LoRAs.

    The returned settings are safe to pass through validation repeatedly:
    only the three named Viggle files are managed, and all other adapter
    entries retain both their order and multiplier.
    """
    normalized = dict(settings)
    solver = selected_profile or normalized.get("sample_solver") or "default"
    solver = str(solver).strip().lower()
    if solver == "viggle":
        try:
            solver = {4: "viggle_v01", 5: "viggle_v02", 6: "viggle_v021"}[
                int(normalized.get("num_inference_steps", 5))
            ]
        except (KeyError, TypeError, ValueError):
            raise ValueError("Choose a Viggle Turbo profile with 4, 5, or 6 steps.")
    if solver in ("", "euler"):
        solver = "default"
    if solver not in ("default", *_VIGGLE_PROFILES):
        raise ValueError(f"Unsupported Qwen Image 2.1 acceleration profile: {solver!r}.")

    raw_loras = normalized.get("activated_loras") or []
    if isinstance(raw_loras, str):
        raw_loras = [raw_loras] if raw_loras else []
    else:
        raw_loras = list(raw_loras)
    raw_multipliers = _split_lora_multipliers(normalized.get("loras_multipliers"))
    raw_multipliers.extend(["1"] * max(0, len(raw_loras) - len(raw_multipliers)))

    kept_loras, kept_multipliers = [], []
    for index, lora in enumerate(raw_loras):
        if _is_managed_viggle_lora(lora):
            continue
        kept_loras.append(lora)
        kept_multipliers.append(raw_multipliers[index])

    normalized["sample_solver"] = solver
    if solver == "default":
        if len(kept_loras) != len(raw_loras):
            normalized["activated_loras"] = kept_loras
            normalized["loras_multipliers"] = " ".join(kept_multipliers)
        return normalized

    profile = _VIGGLE_PROFILES[solver]
    normalized["num_inference_steps"] = profile["steps"]
    normalized["guidance_scale"] = profile["guidance"]
    kept_loras.append(profile["lora_url"])
    kept_multipliers.append("1")
    normalized["activated_loras"] = kept_loras
    normalized["loras_multipliers"] = " ".join(kept_multipliers)
    return normalized


class family_handler:
    @staticmethod
    def query_model_def(base_model_type, model_def):
        return {
            "image_outputs": True,
            "dtype": "bf16",
            "guidance_max_phases": 1,
            "sample_solvers": [("FlowMatch Euler", "default")] + [
                (profile["label"], solver) for solver, profile in _VIGGLE_PROFILES.items()
            ],
            "qwen21_acceleration_profiles": {
                solver: {
                    "label": profile["label"],
                    "steps": profile["steps"],
                    "guidance": profile["guidance"],
                    "lora_url": profile["lora_url"],
                    "lora_weight": 1.0,
                }
                for solver, profile in _VIGGLE_PROFILES.items()
            },
            "compile": False,
            "fit_into_canvas_image_refs": 0,
            "vae_block_size": 32,
            "resolutions_categories": ["<=2k"],
            "resolution_presets": _QWEN21_RESOLUTION_PRESETS,
            "resolution_preset_order": ["auto", "480p", "540p", "720p", "1080p", "2k"],
            "supports_auto_aspect": True,
            "auto_resolution_budgets": {"auto_2k": _QWEN21_2K_PIXELS},
            "auto_resolution_fallbacks": {"auto_2k": "2048x2048"},
            "runtime_custom_settings": ["qwen21_kv_cache"],
            "custom_settings": [{
                "id": "qwen21_kv_cache",
                "name": "KV Cache",
                "label": "KV Cache",
                "type": "dropdown",
                "default": "Disabled",
                "choices": [("Disabled (lower VRAM)", "Disabled"), ("Enabled (faster denoising)", "Enabled")],
                "info": "Enabled retains reference K/V during denoising when the safe memory estimate permits it. It remains disabled for scheduled LoRA changes.",
            }],
            "inpaint_support": True,
            "inpaint_video_prompt_type": "VAG",
            "inpaint_with_image_ref": True,
            "image_ref_inpaint": True,
            "mask_strength_always_enabled": True,
            "inpaint_color": "FF0000",
            "guide_inpaint_color": "FF0000",
            "video_guide_outpainting": [1, 2],
            "outpainting_quantize_margins": 32,
            "guide_preprocessing": {
                "selection": ["", "PV", "DV", "EV", "SV", "CV", "V"],
                "labels": {"V": "Control Image"},
                "default": "",
            },
            "mask_preprocessing": {"selection": ["", "A"], "visible": True},
            "model_modes": {
                "choices": [("Masked Denoising", 0), ("LanPaint (2 steps)", 2),
                            ("LanPaint (5 steps)", 3), ("LanPaint (10 steps)", 4),
                            ("LanPaint (15 steps)", 5)],
                "default": 0,
                "label": "Inpainting Method",
                "image_modes": [2],
            },
            "profiles_dir": ["qwen21"],
            "no_background_removal": True,
            "max_image_refs": 10,
            "preserve_image_ref_alpha": True,
            "image_ref_choices": {
                "choices": [("None", ""), ("Main image / landscape first", "KI"),
                            ("Reference images", "I")],
                "letters_filter": "KI",
                "default": "I",
            },
            "at_least_one_image_ref_needed": False,
            "text_encoder_folder": ENCODER_FOLDER,
            "text_encoder_URLs": [
                build_hf_url("DeepBeepMeep/Ideogram4", ENCODER_FOLDER, "Qwen3-VL-8B-Instruct_bf16.safetensors"),
                build_hf_url("DeepBeepMeep/Ideogram4", ENCODER_FOLDER, "Qwen3-VL-8B-Instruct_int8_convrot.safetensors"),
            ],
        }

    @staticmethod
    def query_supported_types():
        return [MODEL_TYPE]

    @staticmethod
    def query_model_family():
        return "qwen"

    @staticmethod
    def query_family_infos():
        return {"qwen": (110, "Qwen")}

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def get_lora_dir(base_model_type, args, lora_root):
        # The old 20B adapters have different shapes and must not be offered.
        return getattr(args, "lora_dir_qwen21", None) or os.path.join(lora_root, "qwen21")

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        # Pin only processor metadata; existing local files and model-weight
        # downloads keep their paths and normal caching behavior.
        return [{
            "repoId": REPO,
            "sourceFolderList": ["qwen_image_21"],
            "fileList": [["qwen_image_21_vae.safetensors"]],
        }, {
            "repoId": REPO,
            "revision": PROCESSOR_REVISION,
            "sourceFolderList": [ENCODER_FOLDER],
            "fileList": [
                ["added_tokens.json", "chat_template.jinja", "config.json", "merges.txt",
                 "preprocessor_config.json", "special_tokens_map.json", "tokenizer.json",
                 "tokenizer_config.json", "video_preprocessor_config.json", "vocab.json"],
            ],
        }]

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, *,
                   text_encoder_filename=None, VAE_dtype=None, save_quantized=False, **kwargs):
        from .runtime import model_factory
        runtime = model_factory(model_filename, text_encoder_filename, model_def=model_def,
                                VAE_dtype=VAE_dtype, save_quantized=save_quantized, model_type=model_type)
        return runtime, {"transformer": runtime.transformer, "text_encoder": runtime.text_encoder,
                         "vae": runtime.vae}

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update(image_mode=1, video_prompt_type="I", batch_size=1,
                           num_inference_steps=40, guidance_scale=4.0, sample_solver="default",
                           resolution="1024x1024", remove_background_images_ref=0)

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        ui_defaults.setdefault("image_mode", 1)

    @staticmethod
    def apply_acceleration_profile(settings, *, selected_profile=None):
        return apply_acceleration_profile(settings, selected_profile=selected_profile)

    @staticmethod
    def validate_generative_settings(base_model_type, model_def, inputs):
        selector = str(inputs.get("video_prompt_type") or "")
        image_mode = inputs.get("image_mode", 1)
        has_mask = inputs.get("image_mask") is not None
        if isinstance(image_mode, str):
            try:
                image_mode = int(image_mode)
            except ValueError:
                image_mode = 1
        try:
            dims = inputs.get("outpainting_dims") or ()
            if isinstance(dims, str):
                dims = dims.split()
            outpainting = any(float(value) > 0 for value in dims)
        except (TypeError, ValueError):
            outpainting = False

        active_references = _image_slots(inputs.get("image_refs")) if "I" in selector else []
        active_control = "V" in selector or ((has_mask or outpainting) and image_mode == 2)
        control = None
        if active_control:
            for key in ("image_guide", "input_frames", "video_guide"):
                if inputs.get(key) is not None:
                    control = inputs[key]
                    break
        active_starts = _image_slots(inputs.get("image_start"))
        active_controls = _image_slots(control)
        active_nonreferences = active_starts + active_controls
        if len(active_nonreferences) + len(active_references) > 10:
            available_references = max(0, 10 - len(active_nonreferences))
            return (
                "Qwen Image 2.1 supports up to 10 active conditioning images; "
                f"with {len(active_nonreferences)} control/start image(s), use at most "
                f"{available_references} additional references."
            )
