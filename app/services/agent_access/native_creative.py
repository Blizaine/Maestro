"""Bind creative authoring to the live registry without loading an LLM."""
from __future__ import annotations

import copy

from .creative_context import CreativeError, CreativeService


def make_native_creative_service(
    backend, *, models, model_def, defaults, options, preferences, hardware,
    lora_hint, injected_keyframes, mature_mode, probe_audio=None,
):
    from services.authoring_guides import authoring_guide_bundle

    def resolve(context, request):
        from .authoring_plans import resolve_native_context

        params = context["params"]
        md = context["model_def"]
        refs = context.get("_input_references", context["references"])
        context["_input_references"] = copy.deepcopy(refs)
        director = context["workflow"].startswith("director.")
        if not refs and isinstance(params.get("minimax_h3_references"), list):
            refs = [{**row, "id": row.get("id") or f"reference-{index + 1}"} for index, row in enumerate(params["minimax_h3_references"])]
            context["references"] = refs
        if refs and md.get("omni_reference"):
            from models.minimax_h3.reference_manifest import validate_reference_manifest
            from services.studio_enhancement import canonical_h3_enhancement_inputs

            manifest = []
            for reference in refs:
                row = copy.deepcopy(reference)
                row["path"] = row.get("path") or row.get("asset") or ""
                row["type"] = row.get("type") or row.get("media_type", "image").split("/")[0]
                # UI's character label is identity; native scene/composition
                # are distinct image intents and must not be conflated.
                role = str(row.get("role") or "")
                if row["type"] == "image":
                    row.setdefault("image_intent", "identity" if role in {"character", "identity"} else role if role in {"scene", "composition", "style", "object"} else "identity")
                elif row["type"] == "audio":
                    row.setdefault("audio_intent", "drive" if role in {"drive", "soundtrack", "music"} else role if role in {"voice", "style", "sound"} else "voice")
                elif row["type"] == "video":
                    row.setdefault("video_intent", role if role in {"character", "motion", "scene"} else "character")
                manifest.append(row)
            normalized = validate_reference_manifest(manifest, require_visual=not director)
            canonical = canonical_h3_enhancement_inputs(context["brief"], normalized)
            params["minimax_h3_references"] = canonical["minimax_h3_references"]
            context["reference_context"] = canonical["reference_context"]
            context["brief"] = canonical["prompt"]
            # Canonical voice ordering owns <Audio N>; retain stable author IDs.
            context["references"] = [
                {key: row[key] for key in ("id", "type", "role", "image_intent", "video_intent", "audio_intent", "label") if key in row}
                for row in canonical["minimax_h3_references"]
            ]
        elif refs:
            images = []
            for reference in refs:
                path = reference.get("path") or reference.get("asset")
                if path and (reference.get("type") or reference.get("media_type", "image")).startswith("image"):
                    images.append(str(path))
            if images and context["workflow"].startswith("image."):
                params["image_refs"] = images
                params.setdefault("image_prompt_type", "I")
            elif images and context["workflow"] == "video.frames":
                params["image_start"] = images[0]
                params["image_prompt_type"] = "SE" if len(images) > 1 else "S"
                if len(images) > 1:
                    params["image_end"] = images[-1]
            if context["workflow"] == "audio.speech":
                voices = [row for row in refs if row.get("role") == "voice" and str(row.get("type") or row.get("media_type", "")).startswith("audio")]
                if len(voices) > int(md.get("max_voice_count") or 6):
                    raise CreativeError("The voice-reference count exceeds this speech model's supported count.")
                for index, voice in enumerate(voices, 1):
                    params["audio_guide" if index == 1 else f"audio_guide{index}"] = voice.get("path") or voice.get("asset")
                if voices:
                    choices = (options(context["model_type"]).get("audio_prompt_type_sources") or {}).get("selection", ["A"])
                    matches = [value for value in choices if "A" in value and ("B" in value) == (len(voices) > 1)]
                    if len(voices) > 1:
                        matches.sort(key=lambda value: "2" not in value)
                    if not matches:
                        raise CreativeError("This speech model does not support the requested voice-reference arrangement.")
                    params["audio_prompt_type"] = matches[0]
        # Keep paths private; author-facing IDs carry roles, not filesystem data.
        context["references"] = [{key: row[key] for key in ("id", "type", "role", "image_intent", "video_intent", "audio_intent", "label") if key in row} for row in context["references"]]
        if director:
            params["scene_description"] = context["brief"]
            if not md.get("omni_reference"):
                groups = {"character": [], "location": []}
                for reference in refs:
                    kind = str(reference.get("type") or reference.get("media_type") or "image").split("/")[0]
                    role = str(reference.get("role") or "identity")
                    path = reference.get("path") or reference.get("asset")
                    if kind == "image" and role in {"identity", "character", "scene"}:
                        groups["location" if role == "scene" else "character"].append((path, str(reference.get("label") or reference["id"])))
                    elif kind == "audio" and role == "voice":
                        params.update(voice_reference=path, voice_reference_role=str(reference.get("label") or ""))
                    elif kind == "image" or kind == "video":
                        raise CreativeError("This Director route supports character and scene images; use an Omni Reference model for composition, style, object or video references.")
                for group, values in groups.items():
                    params[group + "_ref_paths"] = [value[0] for value in values]
                    params[group + "_ref_labels"] = [value[1] for value in values]
                if groups["character"]:
                    params["reference_image_path"] = groups["character"][0][0]
            params["video_model"] = context["model_type"]
            params["pipeline_type"] = "music_video" if context["workflow"] == "director.music_video" else "short_film_story"
            if not params.get("video_params"):
                params["video_params"] = {key: copy.deepcopy(value) for key, value in params.items() if key not in {"video_params", "image_params"}}
            if not params.get("image_model"):
                params["shot_image_guidance"] = "prompt_only"
            if params["pipeline_type"] == "music_video":
                soundtrack = params.get("audio_path") or next((row.get("path") or row.get("asset") for row in refs if str(row.get("type") or row.get("media_type", "")).startswith("audio") and row.get("role") in {"drive", "soundtrack", "music"}), None)
                if not soundtrack:
                    raise CreativeError("Director music video requires an uploaded soundtrack in params.audio_path or a drive-audio reference.")
                params["audio_path"] = soundtrack
                if probe_audio is None:
                    raise CreativeError("Soundtrack duration inspection is unavailable.", 503)
                measured = probe_audio(soundtrack)
                if not measured or float(measured) <= 0:
                    raise CreativeError("The supplied soundtrack has no measurable audio duration.")
                params["source_duration_seconds"] = float(measured)
                params["duration_seconds"] = float(measured)
                params["video_params"]["duration_seconds"] = float(measured)
                context["source_duration_seconds"] = float(measured)
            elif not params.get("duration_seconds"):
                params["duration_seconds"] = 30
        hint = lora_hint(context["model_type"], params.get("activated_loras"))
        resolve_native_context(
            context, request, hardware(), lora_system_hint=hint,
            injected_keyframes=injected_keyframes(params),
        )
        if director:
            from services.director_model_compat import assess_director_model
            capability = assess_director_model(context["model_type"], md)["video"][params["pipeline_type"]]
            if not capability["compatible"]:
                raise CreativeError("This model cannot run the requested Director workflow: " + capability["reason"])
            if context.get("shot_image_policy") == "direct_references" and not any(str(row.get("type") or row.get("media_type") or "image").split("/")[0] in {"image", "video"} for row in refs):
                raise CreativeError("This Director model needs an image or video reference before authoring.")
            image_model = params.get("image_model")
            if image_model:
                image_definition = model_def(image_model)
                image_capability = assess_director_model(image_model, image_definition)["image"]
                if not image_capability["compatible"]:
                    raise CreativeError("Director image model is incompatible: " + image_capability["reason"])
                registry = models()
                rows = registry.get("models", []) if isinstance(registry, dict) else registry
                if not any(row.get("model_type") == image_model and row.get("is_downloaded") is True for row in rows):
                    raise CreativeError("The Director image model is not downloaded; authorize its download before authoring.", 409)
        if context["workflow"] == "audio.sfx":
            context["params"].update(sfx_mode=True, _mmaudio_variant="v2")
        context["reference_roles"] = {
            "identity": "Who the subject is; does not copy its source room, pose or lighting.",
            "scene": "The environment: room, location, objects, surfaces and spatial layout.",
            "composition": "How the view is framed: camera viewpoint, perspective and arrangement within the frame.",
            "style": "The visual medium, rendering treatment, palette and texture.",
            "object": "The specified object and its appearance, without inventing a person.",
            "voice": "Speaker identity, not the new script or target soundtrack.",
            "drive": "The exact source audio performance/soundtrack with native synchronization.",
        }

    def guides(mt, workflow, request, md):
        request = copy.deepcopy(request)
        voice_refs = [row for row in request.get("references", []) if row.get("role") == "voice"]
        if voice_refs:
            request.setdefault("voice_count", len(voice_refs))
        bundle = authoring_guide_bundle(
            mt, workflow, request, md,
            lora_guidance=lora_hint(mt, (request.get("params") or {}).get("activated_loras")),
            nsfw=bool(mature_mode()),
        )
        if str(request.get("workflow", "")).startswith("director."):
            from services.guide_loader import load_guide
            h3 = str(md.get("architecture") or mt).startswith("minimax_h3")
            bundle["director_guidance"] = {
                "shot_writing": load_guide("director", "minimax_h3_shot_breakdown" if h3 else "video_prompt_rules"),
                "camera_style": load_guide("director", "camera_style_guidance"),
                "contract": "Write actual image_prompt/video_prompt values for the server's editorial schedule. The connected agent performs all writing; native compilation and audio/reference contracts still apply.",
            }
            image_model = (request.get("params") or {}).get("image_model")
            if image_model:
                image_definition = model_def(image_model)
                if not image_definition:
                    raise CreativeError("Director image_model is not registered on this workstation.")
                bundle["shot_image_guides"] = authoring_guide_bundle(image_model, "image.generate", request, image_definition)
        return bundle

    return CreativeService(
        backend.runtime.directory, models=models, model_def=model_def,
        defaults=defaults, options=options, preferences=preferences,
        guides=guides, session=lambda: backend.session_id, resolve=resolve,
    )
