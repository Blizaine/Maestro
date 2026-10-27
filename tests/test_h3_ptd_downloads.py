"""Pinned asset and CPU loading contracts for MiniMax H3 PTD RefAsymOPD."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import types
import unittest
from urllib.parse import urlsplit
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))


class _Slice:
    def __init__(self, tensor):
        self.tensor = tensor

    def get_shape(self):
        return list(self.tensor.shape)

    def get_dtype(self):
        return "F32" if self.tensor.dtype == torch.float32 else "F16"


class _Handle:
    def __init__(self, tensors, metadata=None):
        self.tensors = tensors
        self._metadata = dict(metadata or {})
        self.reads = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def metadata(self):
        return dict(self._metadata)

    def keys(self):
        return list(self.tensors)

    def get_slice(self, key):
        return _Slice(self.tensors[key])

    def get_tensor(self, key):
        self.reads.append(key)
        return self.tensors[key]


class TestH3PTDDownloads(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from models.minimax_h3 import pdd, ptd
        from models.minimax_h3.minimax_h3_handler import family_handler

        cls.pdd = pdd
        cls.ptd = ptd
        cls.family_handler = family_handler

    def _metadata(self):
        return {
            "pdd_num_steps": str(self.ptd.PTD_PDD_INTERVALS),
            "pdd_block_size": str(self.ptd.PTD_PDD_BLOCK_SIZE),
            "ptd_format": self.ptd.PTD_PDD_RELATIVE_FORMAT,
            "ptd_num_hops": str(self.ptd.PTD_NUM_STEPS),
            "ptd_num_curv": str(self.ptd.PTD_NUM_STEPS),
            "source_repository": self.ptd.PTD_SOURCE_REPOSITORY,
            "source_revision": self.ptd.PTD_SOURCE_REVISION,
            "source_sha256": self.ptd.PTD_SOURCE_SHA256,
        }

    def _tensors(self):
        video_base = torch.tensor([[1.0, 2.0, 3.0], [-2.0, 1.0, 0.5]])
        audio_base = torch.tensor([[0.5, 1.5, -1.0]])
        video_bias = torch.tensor([0.25, -0.75])
        audio_bias = torch.tensor([0.125])
        video_offsets = torch.arange(1, 32, dtype=torch.float32).view(31, 1, 1)
        audio_offsets = torch.arange(2, 33, dtype=torch.float32).view(31, 1, 1)
        bias_offsets = torch.arange(1, 32, dtype=torch.float32)
        return {
            "pdd_heads.video_out.linear.weight": video_base,
            "pdd_heads.video_out.heads": video_offsets.expand(31, 2, 3).clone(),
            "pdd_heads.video_out.linear.bias": video_bias,
            "pdd_heads.video_out.head_biases": bias_offsets.view(31, 1).expand(31, 2).clone(),
            "pdd_heads.audio_out.linear.weight": audio_base,
            "pdd_heads.audio_out.heads": audio_offsets.expand(31, 1, 3).clone(),
            "pdd_heads.audio_out.linear.bias": audio_bias,
            "pdd_heads.audio_out.head_biases": bias_offsets.view(31, 1).clone(),
        }

    def _transformer(self):
        return types.SimpleNamespace(
            final_layer=types.SimpleNamespace(
                video_out=torch.nn.Linear(3, 2),
                audio_out=torch.nn.Linear(3, 1),
            )
        )

    def test_download_descriptor_uses_pinned_head_directory_and_keeps_lora_urls(self):
        model_def = {
            "minimax_h3_ptd_refasymopd": True,
            "pdd": True,
        }
        downloads = self.family_handler.query_model_files(
            [], "minimax_h3_ref2va", model_def
        )
        head_downloads = [
            item
            for item in downloads
            if self.ptd.PTD_HEAD_FILE
            in [name for group in item.get("fileList", []) for name in group]
        ]
        self.assertEqual(len(head_downloads), 1)
        head_download = head_downloads[0]
        self.assertEqual(head_download["repoId"], self.ptd.PTD_REPO)
        self.assertEqual(head_download["revision"], self.ptd.PTD_REVISION)
        self.assertEqual(head_download["sourceFolderList"], [self.ptd.PTD_FOLDER])
        self.assertEqual(head_download["fileList"], [[self.ptd.PTD_HEAD_FILE]])

        for step, url in enumerate(self.ptd.PTD_LORA_URLS, start=1):
            self.assertIn(
                f"/{self.ptd.PTD_REVISION}/loras/ptd_refasymopd/",
                url,
            )
            self.assertIn(f"hop{step:02d}_bf16.safetensors", url)
            self.assertNotIn(self.ptd.PTD_HEAD_FILE, url)

        for name in (
            "minimax_h3_ref2va_pruned_ptd_refasymopd",
            "minimax_h3_ref2va_ptd_refasymopd",
        ):
            profile = json.loads(
                (APP / "defaults" / f"{name}.json").read_text(encoding="utf-8")
            )
            self.assertTrue(profile["model"]["name"].isascii())
            self.assertIn("PTD RefAsymOPD 8-Step", profile["model"]["name"])
            urls = profile["model"]["loras"]
            self.assertEqual(len(urls), self.ptd.PTD_NUM_STEPS)
            for step, url in enumerate(urls, start=1):
                parsed = urlsplit(url)
                self.assertEqual(parsed.scheme, "https")
                self.assertEqual(parsed.netloc, "huggingface.co")
                self.assertIn(f"/{self.ptd.PTD_REVISION}/", url)
                self.assertEqual(
                    url.split("/")[-1],
                    f"minimax_h3_ref2va_ptd_refasymopd_8step_hop{step:02d}_bf16.safetensors",
                )
                self.assertFalse(parsed.query)

    def test_relative_heads_reconstruct_all_intervals_and_select_eight_fused_steps(self):
        tensors = self._tensors()
        handle = _Handle(tensors, self._metadata())
        transformer = self._transformer()

        with mock.patch("safetensors.safe_open", return_value=handle):
            controller = self.ptd.install_ptd_relative_heads(
                transformer,
                "ptd-relative.safetensors",
            )

        self.assertEqual(controller.num_steps, 8)
        self.assertEqual(set(handle.reads), set(tensors))
        expected_video_weights = torch.cat(
            (
                tensors["pdd_heads.video_out.linear.weight"].unsqueeze(0),
                tensors["pdd_heads.video_out.linear.weight"].unsqueeze(0)
                + tensors["pdd_heads.video_out.heads"],
            )
        )
        expected_video_biases = torch.cat(
            (
                tensors["pdd_heads.video_out.linear.bias"].unsqueeze(0),
                tensors["pdd_heads.video_out.linear.bias"].unsqueeze(0)
                + tensors["pdd_heads.video_out.head_biases"],
            )
        )
        expected_audio_weights = torch.cat(
            (
                tensors["pdd_heads.audio_out.linear.weight"].unsqueeze(0),
                tensors["pdd_heads.audio_out.linear.weight"].unsqueeze(0)
                + tensors["pdd_heads.audio_out.heads"],
            )
        )
        expected_audio_biases = torch.cat(
            (
                tensors["pdd_heads.audio_out.linear.bias"].unsqueeze(0),
                tensors["pdd_heads.audio_out.linear.bias"].unsqueeze(0)
                + tensors["pdd_heads.audio_out.head_biases"],
            )
        )
        self.assertTrue(torch.equal(controller.video_interval_weights, expected_video_weights))
        self.assertTrue(torch.equal(controller.video_interval_biases, expected_video_biases))
        self.assertTrue(torch.equal(controller.audio_interval_weights, expected_audio_weights))
        self.assertTrue(torch.equal(controller.audio_interval_biases, expected_audio_biases))

        video_sigmas = self.pdd.shifted_sigma(
            12.0,
            torch.linspace(1.0, 0.0, 9, dtype=torch.float64),
        )
        audio_sigmas = self.pdd.shifted_sigma(
            3.0,
            torch.linspace(1.0, 0.0, 9, dtype=torch.float64),
        )
        controller.configure_sigmas(video_sigmas, audio_sigmas)
        video_plans = self.pdd.pdd_sampling_plans_for_sigmas(
            video_sigmas, 12.0
        ).float()
        audio_plans = self.pdd.pdd_sampling_plans_for_sigmas(
            audio_sigmas, 3.0
        ).float()
        sample = torch.tensor([[0.75, -1.25, 2.0]])
        for step in range(8):
            controller.set_step(step)
            video_weight = torch.sum(
                expected_video_weights * video_plans[step, :, None, None], dim=0
            )
            video_bias = torch.sum(
                expected_video_biases * video_plans[step, :, None], dim=0
            )
            audio_weight = torch.sum(
                expected_audio_weights * audio_plans[step, :, None, None], dim=0
            )
            audio_bias = torch.sum(
                expected_audio_biases * audio_plans[step, :, None], dim=0
            )
            self.assertTrue(
                torch.allclose(
                    transformer.final_layer.video_out(sample),
                    torch.nn.functional.linear(sample, video_weight, video_bias),
                )
            )
            self.assertTrue(
                torch.allclose(
                    transformer.final_layer.audio_out(sample),
                    torch.nn.functional.linear(sample, audio_weight, audio_bias),
                )
            )

    def test_wrong_pinned_metadata_or_header_rejects_before_payload_read(self):
        cases = []
        bad_metadata = self._metadata()
        bad_metadata["ptd_format"] = "unrecognized"
        cases.append((self._tensors(), bad_metadata, "metadata mismatch"))

        bad_shape = self._tensors()
        bad_shape["pdd_heads.video_out.heads"] = torch.zeros((30, 2, 3))
        cases.append((bad_shape, self._metadata(), "shape mismatch"))

        bad_dtype = self._tensors()
        bad_dtype["pdd_heads.audio_out.head_biases"] = torch.zeros(
            (31, 1), dtype=torch.float16
        )
        cases.append((bad_dtype, self._metadata(), "must be FP32"))

        bad_keys = self._tensors()
        bad_keys["pdd_heads.video_out.heads_copy"] = torch.zeros((31, 2, 3))
        cases.append((bad_keys, self._metadata(), "tensor keys"))

        for tensors, metadata, message in cases:
            with self.subTest(message=message):
                handle = _Handle(tensors, metadata)
                with mock.patch("safetensors.safe_open", return_value=handle):
                    with self.assertRaisesRegex(ValueError, message):
                        self.ptd.install_ptd_relative_heads(
                            self._transformer(), "malformed.safetensors"
                        )
                self.assertEqual(handle.reads, [])


if __name__ == "__main__":
    unittest.main()
