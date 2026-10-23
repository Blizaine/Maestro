"""Verified MiniMax H3 VDN assets and narrow repair for their retired pin."""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit


VDN_REPO_ID = "DeepBeepMeep/MiniMax-H3"
VDN_LEGACY_REVISION = "304d34f7751f8ba9ca0eb55d5d10044234cdbfe2"
VDN_REVISION = "630975da14fd3c84169d6cd5092da3ce1287d137"

# Hugging Face LFS metadata for the four VDN files at VDN_REVISION. Keep the
# complete paths in the keys so URL repair cannot broaden to other repo files.
VDN_ASSET_SPECS = {
    "modules/minimax_h3_vdn_module_bf16.safetensors": {
        "size": 4_279_428_112,
        "sha256": "dec6981c7874f5b3bc92d1a02e256b673a3b3499dc1a124714bb3b19da602855",
    },
    "modules/minimax_h3_vdn_module_int8_convrot.safetensors": {
        "size": 2_166_148_396,
        "sha256": "dc22ad23ebee2ed893e79b9289ae6608d6eaa55f7a69e4c9bb8c19816e7b0c36",
    },
    "loras/minimax_h3_vdn_default_lora_bf16.safetensors": {
        "size": 334_026_912,
        "sha256": "58558fef506f88bb41649242de9b9b3a365da806b51b2e96afbbe1625222058a",
    },
    "loras/minimax_h3_vdn_turbo_8step_lora_bf16.safetensors": {
        "size": 851_452_696,
        "sha256": "24fc93c82fe84dc45d0627f4e72c637bc387d282ba18f60ed3b7f8c81089392c",
    },
}


def canonicalize_vdn_asset_url(url: str) -> str:
    """Repair only the four known H3 VDN URLs from the retired revision.

    Query strings and fragments are retained. Valid pinned URLs and every
    unrecognized host, revision, repository path, or asset path pass through.
    """

    if not isinstance(url, str):
        return url
    try:
        parts = urlsplit(url)
    except ValueError:
        return url

    if parts.scheme.casefold() != "https" or parts.netloc.casefold() != "huggingface.co":
        return url

    prefix = f"/{VDN_REPO_ID}/resolve/"
    if not parts.path.startswith(prefix):
        return url
    revision_and_path = parts.path[len(prefix):]
    revision, separator, asset_path = revision_and_path.partition("/")
    if (
        revision != VDN_LEGACY_REVISION
        or not separator
        or asset_path not in VDN_ASSET_SPECS
    ):
        return url

    repaired_path = f"{prefix}{VDN_REVISION}/{asset_path}"
    return urlunsplit((parts.scheme, parts.netloc, repaired_path, parts.query, parts.fragment))


__all__ = [
    "VDN_ASSET_SPECS",
    "VDN_LEGACY_REVISION",
    "VDN_REPO_ID",
    "VDN_REVISION",
    "canonicalize_vdn_asset_url",
]
