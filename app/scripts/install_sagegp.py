"""Stage the optional SageGP wheel independently of SageAttention."""

import hashlib
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from urllib.parse import quote
from urllib.request import urlopen


VERSION = "1.0.2"
BASE_URL = f"https://github.com/deepbeepmeep/SageGP/releases/download/v{VERSION}/"
WHEEL_SUFFIX = f"sagegp-{VERSION}+cu130torch2.10.0andhigher-cp310-abi3"
WHEELS = {
    "Windows": ("win_amd64", "d85e0630b4dcb6ca5cb0a1dbd72c44287a760373570c15cb76fae64071cf09e9"),
    "Linux": ("linux_x86_64", "021c4e5e67d34c09110ed3402918750c0dac3b97ac090b1a53e4717b14daadc7"),
}


def compatible_wheel(system, machine, python_version, torch_version, cuda, capability):
    if (
        system not in WHEELS
        or machine.lower() not in {"amd64", "x86_64"}
        or python_version < (3, 10)
        or torch_version < (2, 10)
        or not (cuda or "").startswith("13.")
        or capability != (12, 0)
    ):
        return None
    platform_tag, digest = WHEELS[system]
    return f"{WHEEL_SUFFIX}-{platform_tag}.whl", digest


def install():
    from packaging.version import Version
    import torch

    if torch.version.hip or not torch.cuda.is_available():
        return
    asset = compatible_wheel(
        platform.system(), platform.machine(), sys.version_info[:2],
        Version(torch.__version__).release[:2], torch.version.cuda,
        tuple(torch.cuda.get_device_capability()),
    )
    if asset is None:
        return
    filename, digest = asset
    prefix = Path(sys.prefix).resolve()
    target = prefix / "maestro-sagegp"
    if (target / "sagegp" / "sm120" / "dense.py").is_file():
        print("[SageGP] Optional runtime is already installed.")
        return
    if target.exists():
        raise RuntimeError(f"Existing optional runtime is incomplete: {target}")

    with TemporaryDirectory(prefix=".maestro-sagegp-", dir=prefix) as tmp:
        workdir = Path(tmp)
        archive = workdir / filename
        with urlopen(BASE_URL + quote(filename), timeout=90) as response, archive.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        checksum = hashlib.sha256()
        with archive.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                checksum.update(block)
        if checksum.hexdigest() != digest:
            raise ValueError("Wheel checksum mismatch")
        staged = workdir / "site"
        subprocess.run(
            [shutil.which("uv") or "uv", "pip", "install", "--offline", "--no-deps", "--target",
             str(staged), str(archive)],
            check=True,
            timeout=180,
        )
        if not (staged / "sagegp" / "sm120" / "dense.py").is_file():
            raise RuntimeError("Wheel did not stage the expected package")
        staged.rename(target)
    print(f"[SageGP] Optional runtime {VERSION} installed.")


if __name__ == "__main__":
    try:
        install()
    except Exception as error:
        print(f"[SageGP] Optional installation unavailable: {error}")
