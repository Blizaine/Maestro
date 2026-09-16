"""GPU parity and actual gradient checks against the production YuE2 paths."""
from pathlib import Path
import sys
import unittest
import torch
from transformers import Qwen3Config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from models.TTS.yue2.modules import YuE2Config
from models.TTS.yue2.transformer import YuE2AR, YuE2Acoustic
from models.TTS.yue2.training_math import ar_condition, acoustic_forward
from models.TTS.yue2.artist_training import TrainableAdapter, sequence_loss


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA parity check')
class TrainingParityTests(unittest.TestCase):
    def test_lyric_cursor_and_ar_adapter_receive_finite_gradients(self):
        config = Qwen3Config(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16, intermediate_size=128, vocab_size=184704, max_position_embeddings=512)
        model = YuE2AR(config).cuda().bfloat16().eval().requires_grad_(False)
        model.configure_engine('legacy', lambda: False)
        with torch.no_grad():
            model.model.embed_tokens.weight.normal_(std=.05)
            model.lm_head.weight.normal_(std=.05)
        attention = model.model.layers[0].self_attn
        attention.q_proj = TrainableAdapter(attention.q_proj, 4)
        head = torch.nn.Linear(64, 64, bias=False, device='cuda', dtype=torch.float32)
        item = {'prefix': list(range(8)), 'codec': list(range(20)),
                'cursor': {'start': 1, 'end': 5, 'rows': list(range(4, 16)), 'columns': [0,1,2,3]*3, 'weights': [1.0]*12}}
        metrics = {}
        loss = sequence_loss(model, item, 100, gradients=True, cursor_head=head, metrics=metrics)
        loss.backward()
        self.assertGreater(metrics['cursor'], 0)
        for parameter in (head.weight, attention.q_proj.B):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_condition_and_gradient_checkpointing_match_playback(self):
        torch.manual_seed(31)
        config = dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, intermediate_size=128,
                      vocab_size=256, max_position_embeddings=512)
        ar = YuE2AR(Qwen3Config(**config)).cuda().bfloat16().eval().requires_grad_(False)
        with torch.no_grad():
            ar.model.embed_tokens.weight.normal_(std=.1)
        ar.configure_engine('legacy', lambda: False)
        ids = list(range(24))
        actual = ar.condition(ids)
        trained = ar_condition(ar, ids)
        for pair, expected in zip(trained, actual):
            for value, ref in zip(pair, expected):
                torch.testing.assert_close(value, ref, rtol=.01, atol=.006)
        model = YuE2Acoustic(YuE2Config(**config, latent_dim=8, max_latent_frames=512)).cuda().bfloat16().eval()
        state = torch.randn(32, 8, device='cuda', dtype=torch.bfloat16)
        t = torch.tensor(.3, device='cuda')
        cos, sin = model.rotary(torch.arange(24, 58, device='cuda')[None])
        pos = model.latent_pos_embed(torch.arange(34, device='cuda'))[None]
        with torch.inference_mode():
            expected = model([state.clone()], t, trained, cos, sin, pos).clone()
        result = acoustic_forward(model, state, t, trained, 24, gradients=True)
        torch.testing.assert_close(result, expected, rtol=.02, atol=.015)
        result.float().square().mean().backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(float(model.vae2llm.weight.grad.abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
