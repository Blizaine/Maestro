"""Chatterbox / IndexTTS2 compatibility with the pinned transformers, safetensors and mmgp."""
from __future__ import annotations

import ast
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
sys.path.insert(0, str(APP))

CHATTERBOX_TTS = APP / "models" / "TTS" / "chatterbox" / "mtl_tts.py"
INDEX_TTS2_INFER = APP / "models" / "TTS" / "index_tts2" / "infer_v2.py"


def calls_to(path: Path, dotted: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == dotted]


class ChatterboxCompatTests(unittest.TestCase):
    def test_checkpoints_keep_fp32_to_match_declared_model_dtype(self):
        # mmgp 3.8 converts to BF16 by default; ChatterboxPipeline declares FP32
        # and offload.profile asserts every parameter matches.
        loads = calls_to(CHATTERBOX_TTS, "offload.load_model_data")
        self.assertEqual(len(loads), 3)
        for call in loads:
            keywords = {kw.arg: kw.value for kw in call.keywords}
            self.assertIn("default_dtype", keywords, ast.unparse(call))
            self.assertIsNone(ast.literal_eval(keywords["default_dtype"]))

    def test_alignment_spy_keeps_kv_cache_passed_as_past_key_values(self):
        from transformers import LlamaConfig
        from transformers.cache_utils import DynamicCache
        from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding
        from models.TTS.chatterbox.models.t3.inference.alignment_stream_analyzer import forward_eager

        config = LlamaConfig(hidden_size=32, num_attention_heads=4, num_key_value_heads=4,
                             intermediate_size=64, num_hidden_layers=1, vocab_size=16)
        attention = LlamaAttention(config, layer_idx=0).eval()
        rotary = LlamaRotaryEmbedding(config)
        cache = DynamicCache()

        def step(start, length):
            hidden = torch.randn(1, length, config.hidden_size)
            positions = torch.arange(start, start + length).unsqueeze(0)
            with torch.no_grad():
                # transformers 4.57's LlamaDecoderLayer passes past_key_values.
                return forward_eager(attention, hidden_states=hidden,
                                     position_embeddings=rotary(hidden, positions),
                                     attention_mask=None, past_key_values=cache)

        step(0, 3)
        _, weights = step(3, 1)
        self.assertEqual(cache.get_seq_length(), 4)
        self.assertEqual(weights.shape[-1], 4)


class IndexTTS2CompatTests(unittest.TestCase):
    def test_vendored_generation_utils_import_on_pinned_transformers(self):
        from models.TTS.index_tts2.gpt import transformers_generation_utils as utils
        self.assertTrue(hasattr(utils, "GenerationMixin"))

    def test_semantic_codec_load_avoids_safetensors_load_model(self):
        # safetensors 0.8's load_model passes backend= to load_file, which
        # mmgp's load_file override does not accept.
        self.assertEqual(calls_to(INDEX_TTS2_INFER, "safetensors.torch.load_model"), [])
        import mmgp.offload  # noqa: F401  (installs the load_file override)
        import safetensors.torch

        source = torch.nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder, "codec.safetensors"))
            safetensors.torch.save_file(source.state_dict(), path)
            target = torch.nn.Linear(3, 2)
            target.load_state_dict(safetensors.torch.load_file(path))
        self.assertTrue(torch.equal(source.weight, target.weight))


if __name__ == "__main__":
    unittest.main()
