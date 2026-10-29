"""Attention selection cannot change Flex's trained schedule or enable caches."""
# ruff: noqa: E402 -- repository app path bootstrap.
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from models.minimax_h3.pika_flex import normalize_pika_flex_attention, validate_pika_flex_request
from models.minimax_h3.pika_flex_assets import pika_flex_defaults


@pytest.mark.parametrize("mode", ["sage2", "sdpa", "sol", "sla"])
def test_requested_backend_survives_fixed_recipe_defaults(mode):
    for reference in (False, True):
        settings = pika_flex_defaults(reference, attention=mode)
        assert settings["override_attention"] == mode
        assert settings["num_inference_steps"] == 6
        assert settings["guidance_scale"] == 1
        assert settings["flow_shift"] == 12
        assert settings["audio_flow_shift"] == 3
        assert not settings["skip_steps_cache_type"]
        validate_pika_flex_request(settings)


@pytest.mark.parametrize("mode", [None, "", "auto", "default", "  SAGE2  "])
def test_new_dense_default_is_sage2(mode):
    assert normalize_pika_flex_attention(mode) == "sage2"


def test_unsupported_attention_and_first_block_cache_stay_rejected():
    with pytest.raises(ValueError, match="supports Sage2"):
        validate_pika_flex_request({"override_attention": "veda"})
    with pytest.raises(ValueError, match="accelerators"):
        validate_pika_flex_request({"override_attention": "sla", "skip_steps_cache_type": "first_block"})
