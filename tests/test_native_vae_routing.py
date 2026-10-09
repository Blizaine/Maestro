"""Native H3 decoder routing must preserve model settings and output geometry."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))
from services.native_vae import H3_VAE_X2, native_vae_selection
from services.media_info import processing_record


class NativeVaeRoutingTests(unittest.TestCase):
    def test_only_supported_h3_modes_select_optional_decoder(self):
        model = {"vae_upsamplers": {"h3_vae": [0, 1]}}
        for mode in (0, 1):
            self.assertEqual(native_vae_selection(model, H3_VAE_X2, mode), H3_VAE_X2)
        for method in ("", "lanczos2", "vae2"):
            self.assertIsNone(native_vae_selection(model, method, 0))
        for model_def, value, mode in (({}, H3_VAE_X2, 0), (model, "h3_vae*1.5", 0),
                                       (model, H3_VAE_X2, 2),
                                       ({**model, "audio_only": True}, H3_VAE_X2, 0)):
            with self.assertRaises(ValueError):
                native_vae_selection(model_def, value, mode)

    def test_native_pixel_postpass_returns_the_same_compact_output(self):
        tree = ast.parse((APP / "wgp.py").read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "perform_spatial_upsampling")
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "wgp.py", "exec"), namespace)
        # Identity check detects an accidental second resize or RGB conversion.
        compact_output = object()
        self.assertIs(namespace["perform_spatial_upsampling"](compact_output, H3_VAE_X2), compact_output)

    def test_native_toggle_reloads_once_without_losing_selected_text_encoder(self):
        tree = ast.parse((APP / "wgp.py").read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "generate_video")
        branch = next(node for node in function.body if isinstance(node, ast.If)
                      and "native_h3_vae_capability" in ast.unparse(node.test))
        code = compile(ast.Module(body=[branch], type_ignores=[]), "wgp.py", "exec")
        for loaded, requested, expected_reload in ((None, H3_VAE_X2, True),
                (H3_VAE_X2, H3_VAE_X2, False), (H3_VAE_X2, "", True), (None, "", False)):
            namespace = {"vae_upsampling": None, "native_h3_vae_capability": True,
                         "model_def": {"vae_upsamplers": {"h3_vae": [0, 1]}},
                         "spatial_upsampling": requested, "image_mode": 0,
                         "wan_model": SimpleNamespace(vae=SimpleNamespace(upsampling_set=loaded)),
                         "reload_needed": False, "_last_vae_upsampling": loaded,
                         "model_kwargs": {"minimax_h3_text_encoder": "gguf_q2_k"}}
            exec(code, namespace)
            self.assertEqual(namespace["reload_needed"], expected_reload)
            self.assertEqual(namespace["model_kwargs"]["minimax_h3_text_encoder"], "gguf_q2_k")
            self.assertEqual(namespace["model_kwargs"].get("VAE_upsampling"), requested or None)

    def test_gallery_provenance_distinguishes_decoder_from_pixel_upscaler(self):
        record = processing_record(spatial=H3_VAE_X2,
                    before={"width": 960, "height": 544, "fps": 24},
                    after={"width": 1920, "height": 1088, "fps": 24})
        self.assertEqual(record["stage"], "vae_decode")
        self.assertEqual(record["multiplier"], 2)
        self.assertEqual(record["output"]["height"], 1088)
        self.assertNotIn("elapsed_seconds", record)


if __name__ == "__main__":
    unittest.main()
