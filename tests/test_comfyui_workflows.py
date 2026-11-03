"""CPU-only tests for ComfyUI API workflow normalization and bindings."""

from __future__ import annotations

import copy
import unittest

from app.services.comfyui_workflows import (
    API_EXPORT_GUIDANCE,
    MAX_WORKFLOW_BYTES,
    WorkflowError,
    apply_bindings,
    builtin_templates,
    normalize_workflow,
    preflight_workflow,
    validate_bindings,
)


def _graph():
    return {
        "1": {"class_type": "Loader", "inputs": {"model_name": "good.safetensors"}},
        "2": {"class_type": "Encode", "inputs": {"text": "original", "model": ["1", 0]}},
        "3": {"class_type": "Save", "inputs": {"image": ["2", 0]}},
    }


def _flux3_core_object_info(diffusion_model, text_encoder, vae, reference_image=None):
    def combo(values):
        return [values, {}]

    def schema(required, optional=None):
        return {"input": {"required": required, "optional": optional or {}}}

    info = {
        "UNETLoader": schema({"unet_name": combo([diffusion_model]), "weight_dtype": combo(["default"])}),
        "CLIPLoader": schema(
            {"clip_name": combo([text_encoder]), "type": combo(["flux3"])},
            {"device": combo(["default", "cpu"])},
        ),
        "VAELoader": schema({"vae_name": combo([vae])}),
        "CLIPTextEncode": schema({"text": ["STRING", {}], "clip": ["CLIP", {}]}),
        "EmptyFlux2LatentImage": schema(
            {"width": ["INT", {}], "height": ["INT", {}], "batch_size": ["INT", {}]}
        ),
        "KSampler": schema(
            {
                "seed": ["INT", {}],
                "steps": ["INT", {}],
                "cfg": ["FLOAT", {}],
                "sampler_name": combo(["euler"]),
                "scheduler": combo(["simple"]),
                "denoise": ["FLOAT", {}],
                "model": ["MODEL", {}],
                "positive": ["CONDITIONING", {}],
                "negative": ["CONDITIONING", {}],
                "latent_image": ["LATENT", {}],
            }
        ),
        "VAEDecode": schema({"samples": ["LATENT", {}], "vae": ["VAE", {}]}),
        "SaveImage": schema({"filename_prefix": ["STRING", {}], "images": ["IMAGE", {}]}),
    }
    if reference_image is not None:
        info.update(
            {
                "LoadImage": schema({"image": combo([reference_image])}),
                "VAEEncode": schema({"pixels": ["IMAGE", {}], "vae": ["VAE", {}]}),
                "ReferenceLatent": schema(
                    {"conditioning": ["CONDITIONING", {}]},
                    {"latent": ["LATENT", {}]},
                ),
            }
        )
    return info


class TestComfyWorkflowNormalization(unittest.TestCase):
    def test_unwraps_prompt_and_returns_detached_graph(self):
        source = _graph()
        wrapped = {"prompt": source, "client_id": "x"}
        normalized = normalize_workflow(wrapped)
        self.assertEqual(normalized, source)
        normalized["2"]["inputs"]["text"] = "changed"
        self.assertEqual(source["2"]["inputs"]["text"], "original")

    def test_rejects_ui_format_with_actionable_export_guidance(self):
        with self.assertRaises(WorkflowError) as raised:
            normalize_workflow({"nodes": [{"id": 1}], "links": []})
        self.assertIn(API_EXPORT_GUIDANCE, str(raised.exception))
        with self.assertRaises(WorkflowError) as raised:
            normalize_workflow({"1": {"id": 1, "type": "KSampler", "pos": [0, 0]}})
        self.assertIn("Save (API)", str(raised.exception))

    def test_checks_link_references_and_input_shape(self):
        graph = _graph()
        graph["2"]["inputs"]["model"] = ["missing", 0]
        with self.assertRaisesRegex(WorkflowError, "references missing node"):
            normalize_workflow(graph)
        with self.assertRaises(WorkflowError):
            normalize_workflow({"1": {"class_type": "NoInputs"}})

    def test_applies_node_and_text_size_bounds(self):
        many = {str(index): {"class_type": "T", "inputs": {}} for index in range(501)}
        with self.assertRaisesRegex(WorkflowError, "too many nodes"):
            normalize_workflow(many)
        large = {"1": {"class_type": "Text", "inputs": {"text": "x" * MAX_WORKFLOW_BYTES}}}
        with self.assertRaisesRegex(WorkflowError, "size limit"):
            normalize_workflow(large)


class TestComfyWorkflowBindings(unittest.TestCase):
    def setUp(self):
        self.graph = _graph()
        self.bindings = [
            {"name": "prompt", "label": "Prompt", "node_id": "2", "input": "text", "type": "string", "required": True},
            {"name": "steps", "node_id": "2", "input": "steps", "type": "integer", "default": 20, "min": 1, "max": 60},
        ]
        # The second binding can target an existing input while preserving the
        # graph API's exact node/input addressing contract.
        self.graph["2"]["inputs"]["steps"] = 12

    def test_validates_bindings_and_applies_values_without_mutation(self):
        original = copy.deepcopy(self.graph)
        self.assertEqual(validate_bindings(self.graph, self.bindings), [])
        updated = apply_bindings(self.graph, self.bindings, {"prompt": "a quiet mountain", "steps": 30})
        self.assertEqual(updated["2"]["inputs"]["text"], "a quiet mountain")
        self.assertEqual(updated["2"]["inputs"]["steps"], 30)
        self.assertEqual(updated["2"]["inputs"]["model"], ["1", 0])
        self.assertEqual(self.graph, original)

    def test_rejects_unknown_values_missing_required_and_bad_types_or_ranges(self):
        with self.assertRaisesRegex(WorkflowError, "Unknown binding"):
            apply_bindings(self.graph, self.bindings, {"prompt": "ok", "other": 1})
        with self.assertRaisesRegex(WorkflowError, "Required binding prompt"):
            apply_bindings(self.graph, self.bindings, {})
        with self.assertRaisesRegex(WorkflowError, "does not match type integer"):
            apply_bindings(self.graph, self.bindings, {"prompt": "ok", "steps": True})
        with self.assertRaisesRegex(WorkflowError, "above its maximum"):
            apply_bindings(self.graph, self.bindings, {"prompt": "ok", "steps": 61})

    def test_rejects_duplicate_and_linked_input_bindings(self):
        duplicate = [*self.bindings, dict(self.bindings[0])]
        self.assertTrue(any("duplicated" in error for error in validate_bindings(self.graph, duplicate)))
        shared_target = [
            {**self.bindings[0], "name": "positive_prompt"},
            {**self.bindings[0], "name": "prompt_variant"},
        ]
        errors = validate_bindings(self.graph, shared_target)
        self.assertTrue(any("shares graph input 2.text" in error for error in errors))
        with self.assertRaisesRegex(WorkflowError, "shares graph input 2.text"):
            apply_bindings(self.graph, shared_target, {"positive_prompt": "first", "prompt_variant": "second"})
        linked = [{"name": "model", "node_id": "2", "input": "model", "type": "string"}]
        self.assertTrue(any("cannot replace linked input" in error for error in validate_bindings(self.graph, linked)))

    def test_asset_bindings_require_nonempty_filename_strings(self):
        bindings = [{"name": "source", "node_id": "2", "input": "text", "type": "image", "required": True}]
        with self.assertRaisesRegex(WorkflowError, "does not match type image"):
            apply_bindings(self.graph, bindings, {"source": ""})
        updated = apply_bindings(self.graph, bindings, {"source": "upload_001.png"})
        self.assertEqual(updated["2"]["inputs"]["text"], "upload_001.png")


class TestComfyWorkflowPreflight(unittest.TestCase):
    def setUp(self):
        self.graph = {"1": {"class_type": "Loader", "inputs": {"ckpt_name": "good.safetensors", "mode": "safe"}}}
        self.object_info = {
            "Loader": {
                "input": {
                    "required": {
                        "ckpt_name": [["good.safetensors"], {}],
                        "mode": [["safe", "fast"], {}],
                    }
                }
            }
        }

    def test_preflight_accepts_advertised_checkpoint_and_rejects_missing_model_or_node(self):
        self.assertTrue(preflight_workflow(self.graph, self.object_info)["valid"])
        missing_model_graph = {"1": {"class_type": "Loader", "inputs": {"ckpt_name": "gone.safetensors", "mode": "safe"}}}
        result = preflight_workflow(missing_model_graph, self.object_info)
        self.assertFalse(result["valid"])
        self.assertEqual(result["missing_models"], [{"node_id": "1", "input": "ckpt_name", "value": "gone.safetensors"}])
        self.assertEqual(result["errors"], [])

        missing_node = preflight_workflow({"7": {"class_type": "CustomNode", "inputs": {}}}, self.object_info)
        self.assertEqual(missing_node["missing_nodes"], [{"node_id": "7", "class_type": "CustomNode"}])

    def test_preflight_uses_explicit_catalog_without_guessing_filename_type(self):
        graph = {"1": {"class_type": "Loader", "inputs": {"ckpt_name": "private-model.bin", "mode": "safe"}}}
        result = preflight_workflow(graph, self.object_info, {"checkpoints": ["private-model.bin"]})
        self.assertTrue(result["valid"])
        self.assertEqual(result["missing_models"], [])

        no_combo_schema = {
            "Loader": {"input": {"required": {"ckpt_name": ["STRING", {}], "mode": [["safe", "fast"], {}]}}}
        }
        missing = preflight_workflow(graph, no_combo_schema, {"checkpoints": ["another.bin"]})
        self.assertEqual(missing["missing_models"], [{"node_id": "1", "input": "ckpt_name", "value": "private-model.bin"}])
        present = preflight_workflow(graph, no_combo_schema, {"checkpoints": ["private-model.bin"]})
        self.assertTrue(present["valid"])

    def test_preflight_checks_combo_values_and_required_inputs(self):
        bad = {"1": {"class_type": "Loader", "inputs": {"ckpt_name": "good.safetensors", "mode": "turbo"}}}
        result = preflight_workflow(bad, self.object_info)
        self.assertFalse(result["valid"])
        self.assertTrue(any("not an advertised choice" in error for error in result["errors"]))
        omitted = {"1": {"class_type": "Loader", "inputs": {}}}
        result = preflight_workflow(omitted, self.object_info)
        self.assertTrue(any("missing required input ckpt_name" in error for error in result["errors"]))


class TestComfyBuiltInTemplates(unittest.TestCase):
    def test_templates_include_standard_checkpoint_and_cpu_connectivity_graph(self):
        templates = builtin_templates()
        self.assertEqual(
            [template["id"] for template in templates],
            [
                "sdxl_checkpoint_txt2img",
                "cpu_empty_image_smoke",
                "flux3_klein_distilled_txt2img",
                "flux3_klein_distilled_reference_edit",
            ],
        )
        self.assertTrue(all(set(template) == {"id", "name", "kind", "description", "workflow", "bindings"} for template in templates))
        sdxl = templates[0]
        self.assertIn("compatibility is not inferred", sdxl["description"])
        self.assertEqual(
            {node["class_type"] for node in sdxl["workflow"].values()},
            {"CheckpointLoaderSimple", "CLIPTextEncode", "EmptyLatentImage", "KSampler", "VAEDecode", "SaveImage"},
        )
        self.assertEqual(validate_bindings(sdxl["workflow"], sdxl["bindings"]), [])
        cpu = templates[1]
        self.assertEqual(
            {node["class_type"] for node in cpu["workflow"].values()},
            {"EmptyImage", "SaveImage"},
        )
        self.assertEqual(validate_bindings(cpu["workflow"], cpu["bindings"]), [])
        self.assertFalse(any(binding["name"] == "checkpoint" for binding in cpu["bindings"]))

    def test_flux3_klein_bindings_reference_graph_and_detached_templates(self):
        txt2img, edit = builtin_templates()[2:]
        values = {
            "diffusion_model": "selected_flux3.safetensors",
            "text_encoder": "selected_qwen3vl.safetensors",
            "vae": "selected_flux3_vae.safetensors",
            "positive_prompt": "a small red kite over a field",
            "width": 640,
            "height": 768,
            "steps": 16,
            "cfg": 1.0,
            "seed": 123,
        }
        bound_txt2img = apply_bindings(txt2img["workflow"], txt2img["bindings"], values)
        edit_values = {**values, "reference_image": "uploaded_reference.png"}
        bound_edit = apply_bindings(edit["workflow"], edit["bindings"], edit_values)

        self.assertEqual(bound_txt2img["1"]["inputs"]["unet_name"], values["diffusion_model"])
        self.assertEqual(bound_txt2img["2"]["inputs"]["clip_name"], values["text_encoder"])
        self.assertEqual(bound_txt2img["3"]["inputs"]["vae_name"], values["vae"])
        self.assertEqual(bound_txt2img["4"]["inputs"]["text"], values["positive_prompt"])
        self.assertEqual(bound_txt2img["7"]["inputs"]["steps"], values["steps"])
        self.assertEqual(bound_txt2img["7"]["inputs"]["cfg"], values["cfg"])
        self.assertEqual(bound_txt2img["7"]["inputs"]["seed"], values["seed"])

        self.assertEqual(bound_edit["10"]["inputs"]["image"], edit_values["reference_image"])
        self.assertEqual(bound_edit["11"]["inputs"]["pixels"], ["10", 0])
        self.assertEqual(bound_edit["12"]["inputs"]["conditioning"], ["4", 0])
        self.assertEqual(bound_edit["13"]["inputs"]["conditioning"], ["5", 0])
        self.assertEqual(bound_edit["12"]["inputs"]["latent"], ["11", 0])
        self.assertEqual(bound_edit["13"]["inputs"]["latent"], ["11", 0])
        self.assertEqual(bound_edit["7"]["inputs"]["positive"], ["12", 0])
        self.assertEqual(bound_edit["7"]["inputs"]["negative"], ["13", 0])
        self.assertEqual(bound_edit["6"]["inputs"]["width"], values["width"])
        self.assertEqual(bound_edit["6"]["inputs"]["height"], values["height"])
        self.assertEqual(bound_edit["7"]["inputs"]["latent_image"], ["6", 0])

        templates = builtin_templates()
        templates[2]["workflow"]["1"]["inputs"]["unet_name"] = "changed.safetensors"
        templates[3]["bindings"][0]["name"] = "changed"
        fresh = builtin_templates()
        self.assertEqual(fresh[2]["workflow"]["1"]["inputs"]["unet_name"], "")
        self.assertEqual(fresh[3]["bindings"][0]["name"], "diffusion_model")

    def test_flux3_klein_core_preflight_missing_component_and_numeric_bounds(self):
        txt2img, edit = builtin_templates()[2:]
        values = {
            "diffusion_model": "selected_flux3.safetensors",
            "text_encoder": "selected_qwen3vl.safetensors",
            "vae": "selected_flux3_vae.safetensors",
            "positive_prompt": "a small red kite over a field",
            "width": 512,
            "height": 512,
            "steps": 16,
            "cfg": 1.0,
            "seed": 123,
        }
        bound_txt2img = apply_bindings(txt2img["workflow"], txt2img["bindings"], values)
        object_info = _flux3_core_object_info(
            values["diffusion_model"], values["text_encoder"], values["vae"]
        )
        self.assertTrue(preflight_workflow(bound_txt2img, object_info)["valid"])

        missing_encoder_values = {**values, "text_encoder": "missing_qwen3vl.safetensors"}
        missing_encoder_graph = apply_bindings(
            txt2img["workflow"], txt2img["bindings"], missing_encoder_values
        )
        missing = preflight_workflow(missing_encoder_graph, object_info)
        self.assertEqual(
            missing["missing_models"],
            [{"node_id": "2", "input": "clip_name", "value": "missing_qwen3vl.safetensors"}],
        )

        edit_values = {**values, "reference_image": "uploaded_reference.png"}
        bound_edit = apply_bindings(edit["workflow"], edit["bindings"], edit_values)
        edit_object_info = _flux3_core_object_info(
            edit_values["diffusion_model"],
            edit_values["text_encoder"],
            edit_values["vae"],
            reference_image=edit_values["reference_image"],
        )
        self.assertTrue(preflight_workflow(bound_edit, edit_object_info)["valid"])
        with self.assertRaisesRegex(WorkflowError, "above its maximum"):
            apply_bindings(txt2img["workflow"], txt2img["bindings"], {**values, "width": 4096})
        with self.assertRaisesRegex(WorkflowError, "above its maximum"):
            apply_bindings(txt2img["workflow"], txt2img["bindings"], {**values, "cfg": 6.0})


if __name__ == "__main__":
    unittest.main()
