"""Explicit, hash-pinned installation of the experimental Windows 10 DLSS backend."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

PACKAGES = {
    "bridge": ("https://github.com/lisitskyaa/ComfyUI-DLSS5-NR/releases/download/v0.3.1/ComfyUI-DLSS5-NR-v0.3.1-windows-x64.zip", "219c9da4896a1b636fb3d74607e50914eb09853a6951d6921d8e8ee0e01a575a"),
    "nr": ("https://github.com/RankFTW/rhi-repo/releases/download/dlssnr-310.8.SF-v2/nvngx_dlssnr_310.8.SF-v2.zip", "1da35941894994eb087e017577829e492454e9bae3a6a9397027069ceb74955c"),
    "sr": ("https://github.com/RankFTW/rhi-repo/releases/download/dlss-310.8.0/nvngx_dlss_310.8.0.zip", "fb481660f7e952b87f91760e3afd7f9dc14cd2c3361b470e948d6346e4323009"),
    "worker": ("https://github.com/DeepBeepMeep/dlss5-visual-enhancer/releases/download/wangp-v1.1.3/WanGP-DLSS5-workers-v1.1.3.zip", "ec470d8eb990cc04fe142c037b2f9e84c1d59a70b111df51f110767897f5b0c2"),
}
# Only these entries are extracted; archives can never choose destination paths.
FILES = (
    ("bridge", "native/bin/dlss5nr_bridge.dll", "native/bin/dlss5nr_bridge.dll", "a60b0a8783daab45c5d001bb954459acf3e116f331b9218030bfc6378e4515dd"),
    ("bridge", "runtime/caller/nvngx.dll_comfy.dll", "runtime/caller/nvngx.dll_comfy.dll", "4b2913f2c90ef7721eda508c5b5962f49934f4ff654e7f4c1c245695cefd7591"),
    ("nr", "nvngx_dlssnr.dll", "runtime/nvngx_dlssnr.dll", "6eb209e764f39872625debd6abaf45e2bb6322f6f270f781f70c059ae30b3927"),
    ("sr", "nvngx_dlss.dll", "sr/dlss/nvngx_dlss.dll", "c85f971ce023c9f3492fc7455f0b01a24ba18ea39636407a846902c4360b0b7e"),
    ("worker", "nr-depth-worker.exe", "sr/host/nr-depth-worker.exe", "f8e2967912e5d596e8e36049370487b83620b0cb5845937b681cf835bafc6d0b"),
)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def install(app_root, caches):
    target = app_root / "dlss5/direct"
    # Never put the ReShade/NR hook into the plain Super Resolution host.
    if any((target / "sr/host" / name).exists() for name in ("dxgi.dll", "renodx-dlss5.addon64")):
        raise RuntimeError("The isolated SR host contains an unexpected hook; installation stopped.")
    with tempfile.TemporaryDirectory(prefix="maestro-direct-dlss-") as temporary:
        staging = Path(temporary)
        archives = {}
        for key, (url, expected) in PACKAGES.items():
            filename = url.rsplit("/", 1)[-1]
            source = next((folder / filename for folder in caches if (folder / filename).is_file()), None)
            if source is None:
                source = staging / filename
                print(f"Downloading {filename}...", flush=True)
                request = urllib.request.Request(url, headers={"User-Agent": "Maestro-DLSS-Installer"})
                with urllib.request.urlopen(request, timeout=60) as response, source.open("wb") as output:
                    shutil.copyfileobj(response, output)
            if digest(source) != expected:
                raise RuntimeError(f"Package checksum mismatch: {filename}")
            archives[key] = source
        staged = []
        for package, suffix, relative, expected in FILES:
            with zipfile.ZipFile(archives[package]) as archive:
                matches = [entry for entry in archive.infolist() if entry.filename.replace("\\", "/").endswith("/" + suffix) or entry.filename == suffix]
                if len(matches) != 1:
                    raise RuntimeError(f"Expected one archive entry: {suffix}")
                source = staging / "files" / relative
                source.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(matches[0]) as stream, source.open("wb") as output:
                    shutil.copyfileobj(stream, output)
            if digest(source) != expected:
                raise RuntimeError(f"Native file checksum mismatch: {relative}")
            destination = target / relative
            if destination.exists() and digest(destination) != expected:
                raise RuntimeError(f"Existing file differs; refusing to overwrite: {destination}")
            staged.append((source, destination))
        for package, suffix, relative in (
            ("bridge", "LICENSE", "LICENSE-ComfyUI-DLSS5-NR.txt"),
            ("bridge", "THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES-ComfyUI-DLSS5-NR.md"),
            ("worker", "LICENSE-DLSS5-Feeder.txt", "LICENSE-DLSS5-Feeder.txt"),
        ):
            with zipfile.ZipFile(archives[package]) as archive:
                matches = [entry for entry in archive.infolist() if entry.filename == suffix or entry.filename.endswith("/" + suffix)]
                if len(matches) != 1:
                    raise RuntimeError(f"Missing or ambiguous license: {suffix}")
                source = staging / relative
                source.write_bytes(archive.read(matches[0]))
                staged.append((source, target / relative))
        # Check every conflict before copying anything. Write enablement last.
        for source, destination in staged:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        manifest = {"enabled": True, "experimental": True,
                    "backend": "direct-nr-with-separate-sr", "bridge_version": "0.3.1",
                    "files": {relative: expected for _, _, relative, expected in FILES}}
        temporary_manifest = target / "enabled.json.tmp"
        temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf8")
        temporary_manifest.replace(target / "enabled.json")
    print("Experimental Windows 10 backend installed. Restart Maestro, then select DLSS in Tools > Upscale.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cache-dir", action="append", type=Path, default=[])
    parser.add_argument("--accept-third-party-risk", action="store_true")
    parser.add_argument("--disable", action="store_true", help="Disable this backend without deleting installed files")
    args = parser.parse_args()
    app_root = args.app_root.resolve()
    if os.name != "nt" or not (app_root / "wgp.py").is_file():
        parser.error("Run on Windows with a valid Maestro app directory")
    if args.disable:
        marker = app_root / "dlss5/direct/enabled.json"
        if marker.exists():
            manifest = json.loads(marker.read_text(encoding="utf8"))
            manifest["enabled"] = False
            marker.write_text(json.dumps(manifest, indent=2), encoding="utf8")
        print("Experimental backend disabled. Restart Maestro.")
        return
    print("Experimental local DLSS backend: community native bridge and modified, unsigned NVIDIA-derived NR runtime.\n"
          "These are not official NVIDIA releases or official Windows 10 support. The native worker runs separately;\n"
          "a stuck shutdown is forcibly contained. See docs/DLSS5.md and THIRD_PARTY_NOTICES.md for provenance and terms.")
    if not args.accept_third_party_risk and input("Type I ACCEPT to install: ") != "I ACCEPT":
        parser.error("Installation cancelled")
    install(app_root, args.cache_dir)


if __name__ == "__main__":
    main()
