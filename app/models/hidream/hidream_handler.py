import os

import torch
from PIL import Image
from .prompt_enhancer import HIDREAM_PROMPT_ENHANCER_INSTRUCTIONS


_PROJECT_REPO = "DeepBeepMeep/HiDream"
_ASSET_FOLDER = "hidream_o1"
_ASSET_FILES = [
    "chat_template.json",
    "config.json",
    "configuration.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
]


class family_handler:
    @staticmethod
    def query_model_def(base_model_type, model_def):
        is_dev = base_model_type == "hidream_o1_dev"
        return {
            "image_outputs": True,
            # HiDream's quanto-int8 layer wrappers in mmgp don't trigger
            # dequantization correctly unless torch.compile is applied to
            # the transformer at load time. Without compile, the model
            # produces noise output. Declaring "compile" here makes our
            # wgp.py treat it as required regardless of the user's global
            # server_config["compile"] setting (which may be "" by default).
            "compile": "transformer",
            "sample_solvers": [("Flash", "flash")] if is_dev else [("Default", "default")],
            "guidance_max_phases": 0 if is_dev else 1,
            "fit_into_canvas_image_refs": 0,
            "profiles_dir": [base_model_type],
            "flow_shift": True,
            "no_negative_prompt": True,
            "no_background_removal": True,
            "processor_folder": _ASSET_FOLDER,
            "vae_block_size": 32,
            "text_prompt_enhancer_instructions": HIDREAM_PROMPT_ENHANCER_INSTRUCTIONS,
            "image_prompt_enhancer_instructions": HIDREAM_PROMPT_ENHANCER_INSTRUCTIONS,
            "text_prompt_enhancer_max_tokens": 512,
            "image_prompt_enhancer_max_tokens": 512,
            "guide_preprocessing": {
                "selection": ["", "V", "PV", "DV", "EV"],
                "labels": {"V": "Use Control Image Unchanged"},
            },
            "image_ref_choices": {
                "choices": [
                    ("None", ""),
                    ("Conditional Image is first Main Subject / Landscape and may be followed by People / Objects", "KI"),
                    ("Conditional Images are References", "I"),
                ],
                "letters_filter": "KI",
                "default": "",
            },
        }

    @staticmethod
    def query_supported_types():
        return ["hidream_o1", "hidream_o1_dev"]

    @staticmethod
    def query_family_maps():
        return {}, {"hidream_o1": ["hidream_o1", "hidream_o1_dev"]}

    @staticmethod
    def query_model_family():
        return "hidream"

    @staticmethod
    def query_family_infos():
        return {"hidream": (130, "HiDream")}

    @staticmethod
    def register_lora_cli_args(parser, lora_root):
        parser.add_argument(
            "--lora-dir-hidream-o1",
            type=str,
            default=None,
            help=f"Path to a directory that contains HiDream O1 LoRAs (default: {os.path.join(lora_root, 'hidream_o1')})",
        )

    @staticmethod
    def get_lora_dir(base_model_type, args, lora_root):
        return getattr(args, "lora_dir_hidream_o1", None) or os.path.join(lora_root, "hidream_o1")

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return [
            {
                "repoId": _PROJECT_REPO,
                "sourceFolderList": [_ASSET_FOLDER],
                "fileList": [_ASSET_FILES],
            }
        ]

    @staticmethod
    def load_model(
        model_filename,
        model_type=None,
        base_model_type=None,
        model_def=None,
        quantizeTransformer=False,
        text_encoder_quantization=None,
        dtype=torch.bfloat16,
        VAE_dtype=torch.float32,
        mixed_precision_transformer=False,
        save_quantized=False,
        submodel_no_list=None,
        text_encoder_filename=None,
        **kwargs,
    ):
        # Maestro's Triton int8 kernel injection (shared/kernels/quanto_int8_inject.py)
        # is faster than optimum.quanto's default int8 forward, but it produces
        # incorrect outputs for HiDream's Qwen3VL layer pattern — confirmed by
        # log comparison: upstream Wan2GP (no Triton injection) generates clean
        # images, our Maestro install with the injection generates pure noise.
        # Disable globally before loading HiDream so the model uses the slower
        # but correct optimum.quanto path. Other models that loaded earlier in
        # the session keep their existing patched state; future models will
        # follow this disabled state until the user re-enables in Settings or
        # restarts Maestro.
        try:
            from shared.kernels.quanto_int8_inject import disable_quanto_int8_kernel
            disable_quanto_int8_kernel(notify_disabled=True)
        except Exception as e:
            print(f"[HiDream] Could not disable Triton int8 kernel injection: {e}")

        # disable_quanto_int8_kernel only undoes the Triton-kernel patch but
        # LEAVES IN PLACE the Maestro "default" Quanto patch
        # (_default_quanto_qbytes_linear_forward in quanto_int8_inject.py:208)
        # which replaces optimum.quanto's WeightQBytesLinearFunction.forward
        # with a custom implementation. That custom implementation works for
        # Flux / LTX-2 / Wan but produces wrong outputs for HiDream's Qwen3VL
        # layer pattern (confirmed by log diff vs working wan.git2 upstream
        # install which uses optimum.quanto's TRUE original forward).
        #
        # Restore the actual original by reading _BASE_PATCH_STATE which the
        # injection module saved BEFORE applying any patches.
        try:
            from shared.kernels import quanto_int8_inject as _inj
            if _inj._BASE_PATCH_STATE.enabled and _inj._BASE_PATCH_STATE.orig_forward is not None:
                from optimum.quanto.tensor.weights import qbytes as _qbytes
                _qbytes.WeightQBytesLinearFunction.forward = staticmethod(_inj._BASE_PATCH_STATE.orig_forward)
                _inj._BASE_PATCH_STATE.enabled = False
                _inj._BASE_PATCH_STATE.orig_forward = None
                print("[HiDream] Restored optimum.quanto's true original linear forward (was wrapped by Maestro default-kernel patch)")
        except Exception as e:
            print(f"[HiDream] Could not restore optimum.quanto's original forward: {e}")

        # IMPORTANT: torch.compile / torch._inductor caches compiled graphs to
        # disk. If a previous session compiled HiDream WITH the Triton kernels
        # active, the cached graph contains direct calls to
        # `torch.ops.wan2gp_int8.fused_quant_scaled_mm`. When we later disable
        # the Triton kernel injection (above), the kernel ops are not
        # registered — but the cached compiled graph still tries to call them,
        # crashing with "Triton backend not initialized".
        #
        # Two layers of cleanup:
        #   1. torch._dynamo.reset() — clear in-memory compile state so a
        #      fresh compile happens this session.
        #   2. Delete the on-disk torchinductor cache directory — remove
        #      persisted cached graphs with the broken kernel references.
        #
        # Other models lose their compile cache too. Cost is one extra compile
        # the next time those models run — acceptable trade for HiDream
        # working correctly out of the box on installs that previously
        # generated noise.
        try:
            import torch._dynamo
            torch._dynamo.reset()
            import shutil
            import tempfile
            cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
            if not cache_dir:
                user = os.environ.get("USERNAME") or os.environ.get("USER") or "user"
                cache_dir = os.path.join(tempfile.gettempdir(), f"torchinductor_{user.lower()}")
            if os.path.isdir(cache_dir):
                print(f"[HiDream] Clearing torch.compile inductor cache at {cache_dir} (avoid stale Triton-kernel graphs)")
                shutil.rmtree(cache_dir, ignore_errors=True)
        except Exception as e:
            print(f"[HiDream] Could not clear torch.compile inductor cache: {e}")

        from .hidream_main import model_factory

        pipe_processor = model_factory(
            checkpoint_dir="ckpts",
            model_filename=model_filename,
            model_type=model_type,
            model_def=model_def,
            base_model_type=base_model_type,
            quantizeTransformer=quantizeTransformer,
            dtype=dtype,
            save_quantized=save_quantized,
        )
        return pipe_processor, {"transformer": pipe_processor.transformer}

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        if base_model_type == "hidream_o1_dev":
            # NOTE on guidance_scale=1 for Dev: the model is CFG-distilled so
            # the actual generation forces guide_scale=0.0 inside hidream_main.py
            # regardless of this UI default. Showing 1 in the UI is for
            # consistency — users don't see a "0" CFG value that suggests
            # disabled guidance when in fact the model just doesn't use CFG
            # at all in its distilled form.
            ui_defaults.update({
                "guidance_scale": 1,
                "num_inference_steps": 28,
                "sample_solver": "flash",
                "flow_shift": 1.0,
            })
        else:
            ui_defaults.update({
                "guidance_scale": 5,
                "num_inference_steps": 50,
                "sample_solver": "default",
                "flow_shift": 3.0,
            })

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        if base_model_type == "hidream_o1_dev" and ui_defaults.get("sample_solver", "") in ("", "default"):
            ui_defaults["sample_solver"] = "flash"
        elif ui_defaults.get("sample_solver", "") == "":
            ui_defaults["sample_solver"] = "default"

    @staticmethod
    def preview_latents(base_model_type, latents, meta):
        if not torch.is_tensor(latents) or latents.dim() != 4 or latents.shape[0] != 3:
            return None
        image = latents.detach().float().cpu().clamp(-1, 1)
        channels, frames, height, width = image.shape
        image = image.permute(0, 2, 1, 3).reshape(channels, height, frames * width)
        image = image.add(1).mul(127.5).clamp(0, 255).to(torch.uint8)
        preview = Image.fromarray(image.permute(1, 2, 0).numpy())
        if preview.height > 0:
            scale = 200 / preview.height
            resampling_module = getattr(Image, "Resampling", Image)
            preview = preview.resize((max(1, int(round(preview.width * scale))), 200), resample=getattr(resampling_module, "BILINEAR", Image.BILINEAR))
        return preview
