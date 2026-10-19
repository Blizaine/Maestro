"""Shared YuE2 download identity derived from the checked-in asset manifest."""

import json
from pathlib import Path


# This is the source-download snapshot identity. Existing artist style bundles
# retain their separate compatibility identity in services.music_styles.BASE_REVISION.
_ASSET_MANIFEST = Path(__file__).with_name("assets.json")
DOWNLOAD_REVISION = json.loads(_ASSET_MANIFEST.read_text(encoding="utf-8"))["revision"]
