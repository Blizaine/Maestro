"""Exercise preview bridges without importing the GPU server at module load."""
from __future__ import annotations

import ast
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from PIL import Image

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

from services.generation_preview import configured_preview_mode, preview_mode


def isolated_function(path, name, namespace):
    tree = ast.parse((APP / path).read_text(encoding="utf-8"))
    node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
    node.decorator_list = []
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize(
    ("saved_mode", "expected"),
    [(None, "tiny_vae_video"), ("off", "off"), ("rgb", "rgb"), ("invalid", "off")],
)
def test_system_config_defaults_only_when_saved_preference_is_missing(saved_mode, expected):
    config = {} if saved_mode is None else {"generation_preview": saved_mode}
    get_system_config = isolated_function("launch.py", "get_system_config", {
        "wgp": SimpleNamespace(
            server_config=config,
            attention_modes_supported=[],
            args=SimpleNamespace(vram_safety_coefficient=0.8),
        ),
        "_APP_VERSION": "test",
        "configured_preview_mode": configured_preview_mode,
        "_get_linked_model_folders": lambda: [],
    })

    assert get_system_config()["generation_preview"] == expected


def test_load_models_preserves_explicit_off_over_the_default():
    tree = ast.parse((APP / "wgp.py").read_text(encoding="utf-8"))
    load_models = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "load_models"
    )
    preview_assignment = next(
        node for node in load_models.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Tuple)
            and any(isinstance(item, ast.Name) and item.id == "preview_decoder" for item in target.elts)
            for target in node.targets
        )
    )
    mode_expression = preview_assignment.value.args[0]
    selector = ast.FunctionDef(
        name="select_preview_mode",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="preview_mode"), ast.arg(arg="server_config")],
            kwonlyargs=[], kw_defaults=[], defaults=[],
        ),
        body=[ast.Return(value=mode_expression)],
        decorator_list=[],
    )
    namespace = {
        "configured_preview_mode": configured_preview_mode,
        "normalize_preview_mode": preview_mode,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[selector], type_ignores=[])),
                 "load-model-preview-mode", "exec"), namespace)

    assert namespace["select_preview_mode"]("off", {}) == "off"
    assert namespace["select_preview_mode"](None, {}) == "tiny_vae_video"
    assert namespace["select_preview_mode"](None, {"generation_preview": "off"}) == "off"


def make_callback(mode="off", decoder=None, sender=None, handler=None, config=None):
    gen = {"process_status": "process:main", "progress_status": "", "window_no": 2,
           "total_windows": 3, "prompt_no": 1, "prompts_max": 2}
    events = []
    namespace = {
        "torch": torch, "time": time, "gen_lock": threading.Lock(),
        "get_gen_info": lambda state: gen, "server_config": config or {},
        "get_model_handler": mock.Mock(return_value=handler or SimpleNamespace(get_rgb_factors=lambda: None)),
        "transformer_type": "test-model", "offloadobj": SimpleNamespace(tiny_vae=decoder, preview_notice=None),
        "wan_model": None, "merge_status_context": lambda status, phase: phase,
        "format_time": lambda seconds: "0s",
    }
    build = isolated_function("wgp.py", "build_callback", namespace)
    callback = build({}, SimpleNamespace(), sender or (lambda cmd, data=None: events.append((cmd, data))),
                     "", 24, preview_mode=mode, preview_duration=1.0)
    return callback, gen, events, namespace


def test_off_skips_model_handler_and_latent_capture_but_keeps_progress():
    callback, _, events, ns = make_callback()
    assert not callback.wants_preview(0)
    callback(0, torch.zeros(24, 2, 2, 2))
    callback.close_preview()
    ns["get_model_handler"].assert_not_called()
    assert [cmd for cmd, _ in events] == ["progress"]


def test_explicit_internal_off_does_not_inherit_live_video_default():
    callback, _, events, ns = make_callback("off", config={"generation_preview": "tiny_vae_video"})
    assert not callback.wants_preview(0)
    callback(0, torch.zeros(24, 2, 2, 2))
    callback.close_preview()
    ns["get_model_handler"].assert_not_called()
    assert [cmd for cmd, _ in events] == ["progress"]


def test_rgb_copies_detached_cpu_payload_only_when_due_and_labels_window():
    callback, gen, events, _ = make_callback("rgb")
    latent = torch.randn(16, 3, 2, 2, requires_grad=True)
    callback(0, latent)
    assert not callback.wants_preview(1)
    assert callback.wants_preview(4)
    callback(1, latent)
    payloads = [data for cmd, data in events if cmd == "preview"]
    assert len(payloads) == 1
    assert payloads[0]["context"]["window"] == 2
    assert payloads[0]["context"]["total_windows"] == 3
    assert payloads[0]["latents"].device.type == "cpu"
    assert not payloads[0]["latents"].requires_grad
    gen["abort"] = True
    assert not callback.wants_preview(23)
    callback.close_preview(cancel=True)


def test_tiny_decode_failure_continues_progress_and_falls_back_to_rgb():
    decoder = mock.Mock(side_effect=RuntimeError("optional decoder failed"))
    callback, _, events, _ = make_callback("tiny_vae_frames", decoder)
    latent = torch.zeros(24, 3, 2, 2)
    callback(0, latent)
    callback(1, latent)
    callback.close_preview()
    assert len([cmd for cmd, _ in events if cmd == "progress"]) == 2
    assert len([cmd for cmd, _ in events if cmd == "preview_notice"]) == 1
    assert [data["context"]["mode"] for cmd, data in events if cmd == "preview"] == ["rgb"]


def test_tiny_constructor_notice_follows_context_and_is_not_cleared():
    with mock.patch("shared.tinyvae.session.PreviewSession", side_effect=RuntimeError("no encoder")):
        callback, _, events, _ = make_callback("tiny_vae_video", object())
    assert [cmd for cmd, _ in events] == ["preview_context", "preview_notice"]
    assert events[0][1]["mode"] == "rgb"
    callback.close_preview()


def test_optional_transport_failure_does_not_break_generation_callback():
    def sender(cmd, data=None):
        if cmd.startswith("preview"):
            raise RuntimeError("closed preview transport")
    callback, _, _, _ = make_callback("rgb", sender=sender)
    callback(0, torch.zeros(16, 2, 2, 2))
    callback.preview_error(RuntimeError("optional preview error"))
    callback.close_preview()


def production_h3_preview_helpers(unpack):
    tree = ast.parse((APP / "models/minimax_h3/minimax_h3_main.py").read_text(encoding="utf-8"))
    names = {"_report_h3_preview_error", "_h3_video_scheduler_step", "_prepare_h3_clean_preview"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    ns = {"torch": torch, "unpatchify_video_tokens": unpack}
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])),
                 "h3-clean-preview-helpers", "exec"), ns)
    return ns


def h3_bridge(callback, *, audio_only=False, frozen=None, due=True, tail_rows=0,
              terminal=False, terminal_update=None, unpack_error=None):
    pack = isolated_function("models/minimax_h3/packing.py", "patchify_video_latents", {"torch": torch})
    unpack = isolated_function("models/minimax_h3/packing.py", "unpatchify_video_tokens", {"torch": torch})
    latent = torch.arange(24 * 2 * 4 * 4, dtype=torch.float32).reshape(1, 24, 2, 4, 4)
    packed = pack(latent, (1, 2, 2))
    generated_rows = packed.shape[0] - tail_rows
    rows = torch.cat((torch.full((2, 96), -999.0), packed))
    predicted = pack(latent + 2.0, (1, 2, 2))[:generated_rows]
    if due is not None:
        callback.wants_preview.return_value = due
    preview_unpack = mock.Mock(side_effect=unpack_error) if unpack_error else mock.Mock(wraps=unpack)
    ns = production_h3_preview_helpers(preview_unpack)

    def step(model_output, timestep, sample, *, return_dict, return_denoised=False):
        assert return_dict is False
        next_sample = sample + 7.0
        return (next_sample, predicted) if return_denoised else (next_sample,)

    scheduler = SimpleNamespace(step=mock.Mock(side_effect=step))
    next_rows, denoised_rows = ns["_h3_video_scheduler_step"](
        scheduler,
        torch.ones((generated_rows, 96)),
        0.25,
        rows[2 : 2 + generated_rows],
        callback=callback,
        index=0,
        audio_only=audio_only,
        frozen_target_video=frozen,
        generated_video_row_count=generated_rows,
    )
    rows[2 : 2 + generated_rows] = next_rows
    if terminal_update is not None:
        rows[2:] = terminal_update
    preview = ns["_prepare_h3_clean_preview"](
        callback,
        denoised_rows,
        rows,
        condition_row_count=2,
        generated_row_count=generated_rows,
        index=1 if terminal else 0,
        final_index=1 if terminal else 10,
        denoising_start_step=0,
        mask_end_step=0,
        source_video_rows=None,
        editable_mask_rows=None,
        num_latent_frames=2,
        latent_height=4,
        latent_width=4,
        patch_size=(1, 2, 2),
    )
    callback(0, preview)
    return preview, preview_unpack, rows, scheduler


def test_window_preparation_clears_old_media_before_next_callback():
    from services.generation_preview import GenerationPreviewCache
    from shared.preview_runtime import begin_preview_window
    cache = GenerationPreviewCache()
    cache.begin("windows")
    gen = {"window_no": 1, "total_windows": 2, "prompt_no": 1}
    events = []
    def send(cmd, context):
        events.append((cmd, context))
        cache.set_context("windows", context)
    begin_preview_window(send, gen, "tiny_vae_video")
    prior = events[-1][1]
    cache.publish("windows", Image.new("RGB", (4, 4)), prior)
    gen["window_no"] = 2
    begin_preview_window(send, gen, "tiny_vae_video")
    assert cache.fields({"id": "windows", "status": "running"})["preview"] is None
    assert cache.publish("windows", Image.new("RGB", (4, 4)), prior) is None
    begin_preview_window(mock.Mock(side_effect=RuntimeError("closed optional transport")), gen, "rgb")
    begin_preview_window(mock.Mock(side_effect=AssertionError("Off sent a preview event")), gen, "off")


def isolated_profile_setup(decoder, error):
    tree = ast.parse((APP / "wgp.py").read_text(encoding="utf-8"))
    load = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "load_models")
    index = next(i for i, node in enumerate(load.body) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "previous_last_offload" for target in node.targets))
    return_nodes = ast.parse("return wan_model, offloadobj").body
    body = [ast.Global(names=["preview_decoder", "wan_model", "offloadobj"])] + load.body[index:index + 4] + return_nodes
    function = ast.FunctionDef(name="recover_profile", args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[],
                                                                        kw_defaults=[], defaults=[]),
                               body=body, decorator_list=[])
    cancelled = type("LoadingCancelled", (Exception,), {})
    partial = SimpleNamespace(release=mock.Mock())
    fresh_model, fresh_offload = object(), SimpleNamespace()
    old_pipeline = {"transformer": object(), "tiny_vae": decoder}
    ns = {"preview_decoder": decoder, "preview_key": "taeh3", "pipe": old_pipeline,
          "mmgp_profile": 1, "compile_modules": "", "loras_transformer": [], "perc_reserved_mem_max": 0.5,
          "vram_safety_coefficient": 0.9, "transformer_dtype": torch.float16, "kwargs": {},
          "gc": SimpleNamespace(collect=mock.Mock()), "args": SimpleNamespace(gpu=""), "torch": torch,
          "model_type": "h3", "override_profile": 1, "output_type": "video", "preview_gen": {}, "model_kwargs": {},
          "wan_model": object(), "offloadobj": None,
          "offload": SimpleNamespace(LoadingCancelled=cancelled, last_offload_obj=None, flush_torch_caches=mock.Mock()),
          "load_models": mock.Mock(return_value=(fresh_model, fresh_offload))}
    def failing_profile(*args, **kwargs):
        ns["offload"].last_offload_obj = partial
        raise error
    ns["offload"].profile = mock.Mock(side_effect=failing_profile)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "preview-profile-recovery", "exec"), ns)
    return ns, old_pipeline, fresh_model, fresh_offload, partial


def test_preview_profile_failure_rebuilds_fresh_pipeline_once_without_tiny():
    ns, old_pipeline, fresh_model, fresh_offload, partial = isolated_profile_setup(object(), RuntimeError("tiny setup"))
    result = ns["recover_profile"]()
    assert result == (fresh_model, fresh_offload)
    assert old_pipeline == {}
    partial.release.assert_called_once()
    ns["offload"].flush_torch_caches.assert_called_once()
    ns["load_models"].assert_called_once_with("h3", 1, output_type="video", preview_mode="rgb", preview_gen={})
    assert fresh_offload.preview_decoder_key == "taeh3"
    assert "fast frames" in fresh_offload.preview_notice


def test_main_model_profile_failure_is_not_hidden_by_optional_recovery():
    ns, _, _, _, partial = isolated_profile_setup(None, RuntimeError("main model failed"))
    with pytest.raises(RuntimeError, match="main model failed"):
        ns["recover_profile"]()
    ns["load_models"].assert_not_called()
    partial.release.assert_not_called()


def test_profile_recovery_preserves_an_existing_auxiliary_offloader():
    ns, _, _, _, partial = isolated_profile_setup(object(), RuntimeError("before new offloader"))
    existing = SimpleNamespace(release=mock.Mock())
    ns["offload"].last_offload_obj = existing
    ns["offload"].profile = mock.Mock(side_effect=RuntimeError("before new offloader"))
    ns["recover_profile"]()
    existing.release.assert_not_called()
    partial.release.assert_not_called()
    assert ns["offload"].last_offload_obj is existing


def test_h3_preview_strips_condition_rows_and_preserves_normalized_24_channel_math():
    callback = mock.Mock()
    callback.wants_preview.return_value = True
    preview, unpack, rows, scheduler = h3_bridge(callback)
    unpack.assert_called_once()
    expected_rows = isolated_function(
        "models/minimax_h3/packing.py", "patchify_video_latents", {"torch": torch}
    )(torch.arange(24 * 2 * 4 * 4, dtype=torch.float32).reshape(1, 24, 2, 4, 4) + 2.0, (1, 2, 2))
    expected = isolated_function(
        "models/minimax_h3/packing.py", "unpatchify_video_tokens", {"torch": torch}
    )(expected_rows, 2, 4, 4, 24, (1, 2, 2))[0]
    assert torch.equal(preview, expected)
    assert torch.equal(callback.call_args.args[1], expected)
    assert torch.equal(rows[:2], torch.full((2, 96), -999.0))
    scheduler.step.assert_called_once()
    assert scheduler.step.call_args.kwargs["return_denoised"] is True


@pytest.mark.parametrize("audio,frozen,due", [(False, None, False), (True, None, True), (False, object(), True)])
def test_h3_skips_unpacking_for_off_audio_and_frozen_video(audio, frozen, due):
    callback = mock.Mock()
    callback.wants_preview.return_value = due
    _, unpack, _, scheduler = h3_bridge(callback, audio_only=audio, frozen=frozen, due=None)
    unpack.assert_not_called()
    if audio or frozen is not None:
        callback.wants_preview.assert_not_called()
    else:
        callback.wants_preview.assert_called_once_with(0)
    assert scheduler.step.call_args.kwargs.get("return_denoised", False) is False
    callback.assert_called_once_with(0, None)


def test_h3_clean_preview_uses_pre_step_estimate_and_preserves_target_condition_tail():
    callback = mock.Mock()
    callback.wants_preview.return_value = True
    preview, unpack, rows, scheduler = h3_bridge(callback, tail_rows=4)
    pack = isolated_function("models/minimax_h3/packing.py", "patchify_video_latents", {"torch": torch})
    unpatch = isolated_function("models/minimax_h3/packing.py", "unpatchify_video_tokens", {"torch": torch})
    source = torch.arange(24 * 2 * 4 * 4, dtype=torch.float32).reshape(1, 24, 2, 4, 4)
    clean_head = pack(source + 2.0, (1, 2, 2))[:4]
    conditioned_tail = pack(source, (1, 2, 2))[4:]
    expected = unpatch(torch.cat((clean_head, conditioned_tail)), 2, 4, 4, 24, (1, 2, 2))[0]
    assert torch.equal(preview, expected)
    unpack.assert_called_once()
    preview_tokens = unpack.call_args.args[0]
    assert preview_tokens.shape[0] == 8
    assert torch.equal(preview_tokens[:4], clean_head)
    assert torch.equal(preview_tokens[4:], conditioned_tail)
    assert not torch.equal(preview_tokens[:4], rows[2:6])
    assert scheduler.step.call_args.kwargs["return_denoised"] is True


def test_h3_terminal_preview_uses_post_update_target_rows():
    callback = mock.Mock()
    callback.wants_preview.return_value = True
    terminal_rows = torch.full((8, 96), 42.0)
    preview, unpack, _, _ = h3_bridge(callback, terminal=True, terminal_update=terminal_rows)
    expected = isolated_function(
        "models/minimax_h3/packing.py", "unpatchify_video_tokens", {"torch": torch}
    )(terminal_rows, 2, 4, 4, 24, (1, 2, 2))[0]
    assert torch.equal(preview, expected)
    assert torch.equal(unpack.call_args.args[0], terminal_rows)


def test_h3_source_preview_before_denoising_uses_clean_source_rows():
    unpack = isolated_function("models/minimax_h3/packing.py", "unpatchify_video_tokens", {"torch": torch})
    preview_unpack = mock.Mock(wraps=unpack)
    helpers = production_h3_preview_helpers(preview_unpack)
    callback = mock.Mock()
    prefix = torch.full((2, 96), -999.0)
    updated_target = torch.full((8, 96), 18.0)
    source = torch.arange(8 * 96, dtype=torch.float32).reshape(8, 96)
    video_rows = torch.cat((prefix, updated_target))

    preview = helpers["_prepare_h3_clean_preview"](
        callback,
        torch.full((6, 96), 4.0),
        video_rows,
        condition_row_count=2,
        generated_row_count=6,
        index=1,
        final_index=5,
        denoising_start_step=3,
        mask_end_step=0,
        source_video_rows=source,
        editable_mask_rows=None,
        num_latent_frames=2,
        latent_height=4,
        latent_width=4,
        patch_size=(1, 2, 2),
    )

    expected_rows = torch.cat((source[:6], updated_target[6:]))
    assert torch.equal(preview_unpack.call_args.args[0], expected_rows)
    assert torch.equal(source, torch.arange(8 * 96, dtype=torch.float32).reshape(8, 96))
    assert torch.equal(video_rows, torch.cat((prefix, updated_target)))
    assert preview.shape == (24, 2, 4, 4)


def test_h3_masked_clean_preview_blends_clean_source_without_mutating_render_state():
    unpack = isolated_function("models/minimax_h3/packing.py", "unpatchify_video_tokens", {"torch": torch})
    preview_unpack = mock.Mock(wraps=unpack)
    helpers = production_h3_preview_helpers(preview_unpack)
    callback = mock.Mock()
    prefix = torch.full((2, 96), -999.0)
    target_rows = torch.full((8, 96), 21.0)
    video_rows = torch.cat((prefix, target_rows))
    original_video_rows = video_rows.clone()
    denoised_rows = torch.full((6, 96), 9.0)
    scheduler_history = denoised_rows.detach()
    source_rows = torch.full((8, 96), 3.0)
    source_noise_rows = torch.full((8, 96), 100.0)
    source_buffer_rows = torch.full((8, 96), -12.0)
    editable_mask_rows = torch.ones((8, 96))
    editable_mask_rows[:3] = 0.0
    original_state = [
        tensor.clone()
        for tensor in (video_rows, denoised_rows, source_rows, source_noise_rows, source_buffer_rows, editable_mask_rows)
    ]

    helpers["_prepare_h3_clean_preview"](
        callback,
        denoised_rows,
        video_rows,
        condition_row_count=2,
        generated_row_count=6,
        index=1,
        final_index=5,
        denoising_start_step=0,
        mask_end_step=3,
        source_video_rows=source_rows,
        editable_mask_rows=editable_mask_rows,
        num_latent_frames=2,
        latent_height=4,
        latent_width=4,
        patch_size=(1, 2, 2),
    )

    expected_clean_rows = torch.cat((torch.full((3, 96), 3.0), torch.full((3, 96), 9.0), target_rows[6:]))
    assert torch.equal(preview_unpack.call_args.args[0], expected_clean_rows)
    assert torch.equal(scheduler_history, original_state[1])
    assert torch.equal(source_rows, original_state[2])
    assert torch.equal(source_noise_rows, original_state[3])
    assert torch.equal(source_buffer_rows, original_state[4])
    assert torch.equal(editable_mask_rows, original_state[5])
    assert torch.equal(video_rows, original_video_rows)


def test_h3_optional_unpack_failure_reports_notice_and_keeps_progress():
    callback = mock.Mock()
    callback.wants_preview.return_value = True
    _, unpack, _, _ = h3_bridge(callback, unpack_error=RuntimeError("preview unavailable"))
    unpack.assert_called_once()
    callback.preview_error.assert_called_once()
    callback.assert_called_once_with(0, None)


def test_h3_preview_cadence_failure_keeps_default_step_and_progress():
    callback = mock.Mock()
    callback.wants_preview.side_effect = RuntimeError("preview cadence unavailable")
    _, unpack, _, scheduler = h3_bridge(callback, due=None)
    unpack.assert_not_called()
    callback.preview_error.assert_called_once()
    assert scheduler.step.call_args.kwargs.get("return_denoised", False) is False
    callback.assert_called_once_with(0, None)


def test_h3_real_scheduler_failure_is_not_hidden_as_preview_failure():
    helpers = production_h3_preview_helpers(mock.Mock())
    callback = mock.Mock()
    callback.wants_preview.return_value = True
    scheduler = SimpleNamespace(step=mock.Mock(side_effect=RuntimeError("scheduler failed")))
    with pytest.raises(RuntimeError, match="scheduler failed"):
        helpers["_h3_video_scheduler_step"](
            scheduler,
            torch.ones(1, 96),
            0.5,
            torch.zeros(1, 96),
            callback=callback,
            index=0,
            audio_only=False,
            frozen_target_video=None,
            generated_video_row_count=1,
        )
    callback.preview_error.assert_not_called()


def test_ltx_callback_skips_off_and_optional_unpack_failure_keeps_progress():
    invoke = isolated_function("models/ltx2/ltx_pipelines/utils/helpers.py", "_invoke_callback", {
        "Callable": object, "LatentState": object, "VideoLatentTools": object,
    })
    tools = mock.Mock()
    callback = mock.Mock()
    callback.wants_preview.return_value = False
    invoke(callback, 2, 1, object(), tools)
    tools.clear_conditioning.assert_not_called()
    callback.assert_called_once_with(2, None, False, pass_no=1)
    callback.reset_mock()
    callback.wants_preview.return_value = True
    tools.clear_conditioning.side_effect = RuntimeError("optional unpack failure")
    invoke(callback, 3, 1, object(), tools)
    callback.preview_error.assert_called_once()
    callback.assert_called_once_with(3, None, False, pass_no=1)


def test_media_publishing_failure_does_not_change_job_status():
    from services.generation_preview import GenerationPreviewCache
    cache = GenerationPreviewCache()
    cache.begin("test")
    context = {"serial": 1, "mode": "rgb"}
    cache.set_context("test", context)
    consumer = isolated_function("launch.py", "_consume_generation_preview", {
        "_generation_previews": cache, "wgp": SimpleNamespace(generate_preview=mock.Mock(side_effect=RuntimeError("bad RGB"))),
    })
    job = {"id": "test", "status": "running"}
    consumer("test", {"latents": object(), "context": context}, "test-model")
    assert job["status"] == "running"
    assert "Generation will continue" in cache.fields(job)["preview_notice"]


def test_preview_endpoint_cannot_serve_completed_or_cross_job_media():
    from services.generation_preview import GenerationPreviewCache, preview_response
    from fastapi import HTTPException, Request
    cache = GenerationPreviewCache()
    context = {"serial": 1, "mode": "rgb"}
    cache.begin("active")
    cache.set_context("active", context)
    preview = cache.publish("active", Image.new("RGB", (4, 4)), context)
    jobs = {"active": {"status": "running"}, "finished": {"status": "completed"}, "other": {"status": "running"}}
    endpoint = isolated_function("launch.py", "get_generation_preview", {
        "_generation_previews": cache, "_jobs": jobs, "is_cancel_requested": lambda job: False,
        "HTTPException": HTTPException, "preview_response": preview_response, "Request": Request,
    })
    request = SimpleNamespace(headers={})
    assert endpoint("active", preview["revision"], request).status_code == 200
    for job_id in ("finished", "other", "unknown"):
        with pytest.raises(HTTPException) as error:
            endpoint(job_id, preview["revision"], request)
        assert error.value.status_code == 404
