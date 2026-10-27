"""Bundled PTD URLs must match Maestro's actual basename-only LoRA loader."""
import json
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]


def test_ptd_bundled_urls_have_no_unsupported_upstream_folder_suffix():
    for name in ('minimax_h3_ref2va_pruned_ptd_refasymopd', 'minimax_h3_ref2va_ptd_refasymopd'):
        data = json.loads((ROOT / 'app/defaults' / (name + '.json')).read_text())
        urls = data['model']['loras']
        assert len(urls) == 8
        for hop, url in enumerate(urls, start=1):
            parsed = urlsplit(url)
            assert parsed.scheme == 'https' and parsed.netloc == 'huggingface.co'
            # download_models and get_transformer_loras both use url.split('/')[-1].
            # WanGP's newer '|subfolder' notation is not understood by either.
            filename = url.split('/')[-1]
            assert '|' not in filename and not parsed.query
            assert filename == f'minimax_h3_ref2va_ptd_refasymopd_8step_hop{hop:02d}_bf16.safetensors'
