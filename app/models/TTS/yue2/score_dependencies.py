"""Install only missing, optional notation packages in the current app runtime."""
import importlib
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import time


def ensure_score_dependencies(cancelled=lambda: False):
    requirements = Path(__file__).with_name("requirements-score.txt").read_text().splitlines()
    missing = [line for line in requirements if line and not line.startswith("#")
               and importlib.util.find_spec(line.split("==")[0]) is None]
    if not missing:
        return
    if cancelled():
        raise InterruptedError("YuE2 score preparation cancelled")
    uv = shutil.which("uv")
    command = ([uv, "pip", "install", "--python", sys.executable] if uv else
               [sys.executable, "-m", "pip", "install", "--disable-pip-version-check"])
    # Numpy, scipy and six already belong to Maestro's core runtime. Never let
    # an optional score package resolve or replace Torch or the core AI stack.
    command += ["--no-deps", *missing]
    print("[YuE2] Installing optional score notation packages: " + ", ".join(missing), flush=True)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    started = time.monotonic()
    try:
        while True:
            if cancelled():
                raise InterruptedError("YuE2 score preparation cancelled")
            if time.monotonic() - started > 600:
                raise RuntimeError("YuE2 score package installation timed out; check the download connection and retry")
            try:
                output, _ = process.communicate(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                continue
        if process.returncode:
            raise RuntimeError("Could not install YuE2 score notation packages: " + output[-2000:])
        print(output.strip(), flush=True)
        importlib.invalidate_caches()
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
