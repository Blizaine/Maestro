# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from shared.utils import files_locator as fl
from .text_images import load_font

REPO = "DeepBeepMeep/Fibo2"
REPO_REVISION = "72bf0a84dd97688a4e9427a8c872114b1162c4b8"


def query_font_files():
    return {"repoId": REPO, "revision": REPO_REVISION,
            "sourceFolderList": ["fibo2"],
            "fileList": [["Aileron-Regular.otf", "DejaVuSans.ttf", "LICENSE-Aileron.txt", "LICENSE-DejaVu.txt"]]}


def load_fonts():
    definition = query_font_files()
    paths = {name: fl.locate_file(f"fibo2/{name}", error_if_none=False)
             for name in definition["fileList"][0]}
    if any(path is None for path in paths.values()):
        # Use the current branch's revision-aware downloader. Resolve this
        # lazily so importing the model never triggers a download.
        from wgp import process_files_def

        process_files_def(**definition)
        paths = {name: fl.locate_file(f"fibo2/{name}")
                 for name in definition["fileList"][0]}
    fonts = [load_font(paths[name]) for name in ("Aileron-Regular.otf", "DejaVuSans.ttf")]
    return fonts
