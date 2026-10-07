"""CPU-only device-policy and scheduler parity checks for the WanGP 17 port."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from shared.utils.default_device import call_with_default_device, generation_default_device, keep_default_device
from shared.utils.euler_scheduler import EulerScheduler
from shared.utils.lcm_scheduler import LCMScheduler
from models.wan.modules.posemb_layers import get_nd_rotary_pos_embed, get_rotary_pos_embed


class DevicePolicyTests(unittest.TestCase):
    def setUp(self):
        context = getattr(torch._GLOBAL_DEVICE_CONTEXT, "device_context", None)
        self.original = None if context is None else context.device
        self.addCleanup(torch.set_default_device, self.original)
        torch.set_default_device("meta")

    def test_audited_generation_drops_default_and_restores_even_after_exception(self):
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            with generation_default_device({"device_explicit": True}):
                self.assertEqual(torch.empty(1).device.type, "cpu")
                raise RuntimeError("generation failed")
        self.assertEqual(torch.empty(1).device.type, "meta")

    def test_unaudited_model_keeps_original_default(self):
        result = call_with_default_device({}, "legacy", lambda: torch.empty(1))
        self.assertEqual(result.device.type, "meta")

    def test_internal_profile_preserves_outer_generation_policy(self):
        with generation_default_device({"device_explicit": True}):
            with keep_default_device():
                torch.set_default_device("meta")
            self.assertEqual(torch.empty(1).device.type, "cpu")
        self.assertEqual(torch.empty(1).device.type, "meta")

    def test_euler_schedule_keeps_requested_device_and_numeric_values(self):
        schedule = EulerScheduler()
        values = schedule.set_timesteps(4, device="cpu")
        self.assertEqual(values.device.type, "cpu")
        self.assertEqual(values.dtype, torch.float32)
        self.assertEqual(values.shape, (4,))
        with generation_default_device({"device_explicit": True}):
            again = EulerScheduler().set_timesteps(4, device="cpu")
        torch.testing.assert_close(values, again)
        on_meta = EulerScheduler().set_timesteps(4, device="meta")
        self.assertEqual(on_meta.device.type, "meta")

    def test_lcm_schedule_builds_on_requested_device(self):
        scheduler = LCMScheduler()
        scheduler.set_timesteps(4, device="cpu")
        self.assertEqual(scheduler.timesteps.device.type, "cpu")
        self.assertEqual(scheduler.sigmas.device.type, "cpu")
        self.assertTrue(torch.all(scheduler.sigmas[:-1] >= scheduler.sigmas[1:]))

    def test_wan_rope_explicit_device_overrides_ambient_default(self):
        # The generation callers pass device= even on unflagged variants.
        # Exercise both their public helper signatures and the grid placement.
        actual = get_rotary_pos_embed((3, 4, 6), device="cpu")
        reference = get_nd_rotary_pos_embed((3, 2, 3), device="cpu")
        for value, expected in zip(actual, reference):
            self.assertEqual(value.device.type, "cpu")
            torch.testing.assert_close(value, expected)
        on_meta = get_rotary_pos_embed((3, 4, 6), enable_RIFLEx=True, device="meta")
        self.assertTrue(all(value.device.type == "meta" for value in on_meta))

    def test_ltx_schedule_preserves_values_on_requested_device(self):
        from models.ltx2.ltx_core.components.schedulers import LTX2Scheduler

        actual = LTX2Scheduler().execute(steps=4, device=torch.device("cpu"))
        self.assertEqual(actual.device.type, "cpu")
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(actual.shape, (5,))
        torch.testing.assert_close(actual[[0, -2, -1]], torch.tensor([1.0, 0.1, 0.0], device="cpu"))
        self.assertTrue(torch.all(actual[:-1] >= actual[1:]))

    def test_ltx_audio_processor_keeps_mel_buffers_and_waveform_together(self):
        from models.ltx2.ltx_core.model.audio_vae.ops import AudioProcessor

        processor = AudioProcessor(16000, 16, 64, 256, device=torch.device("cpu"))
        self.assertTrue(all(buffer.device.type == "cpu" for buffer in processor.buffers()))
        waveform = torch.randn(1, 1, 512, device="cpu")
        actual = processor.waveform_to_mel(waveform, 16000)
        self.assertEqual(actual.device.type, "cpu")
        self.assertEqual(actual.shape, (1, 1, 9, 16))
        self.assertTrue(torch.isfinite(actual).all())


if __name__ == "__main__":
    unittest.main()
