"""Build the local CUDA LLM runtime where upstream only publishes CPU archives.

This does not modify Maestro's PyTorch environment or install system packages.
The CUDA toolkit is used only for this standalone llama.cpp executable.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import tempfile


def find_nvcc() -> str | None:
    for variable in ("CUDACXX", "CUDA_HOME", "CUDA_PATH"):
        value = os.environ.get(variable, "")
        if value:
            candidate = value if variable == "CUDACXX" else os.path.join(value, "bin", "nvcc")
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return os.path.abspath(candidate)
    return shutil.which("nvcc") or (
        "/usr/local/cuda/bin/nvcc"
        if os.access("/usr/local/cuda/bin/nvcc", os.X_OK) else None
    )


def probe_cuda(executable: str, environment: dict | None) -> tuple[list[str], str]:
    """Ask this executable for devices; installed Torch is not evidence of CUDA."""
    try:
        result = subprocess.run(
            [executable, "--list-devices"], env=environment,
            capture_output=True, text=True, timeout=30,
            **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return [], str(error)
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    # Only actual device rows count, not "loaded CUDA backend"/driver messages.
    devices = re.findall(r"^\s*(CUDA\d+):\s+\S.*$", output, re.MULTILINE)
    return (devices if result.returncode == 0 else []), output[-3000:]


def require_cuda(executable: str, environment: dict | None) -> list[str]:
    devices, diagnostic = probe_cuda(executable, environment)
    if not devices:
        raise RuntimeError(
            "CUDA was selected for the LLM, but llama-server cannot use a CUDA device. "
            "Check the NVIDIA driver and this runtime's CUDA libraries, or select CPU "
            "explicitly in Settings. Maestro will not silently run this GPU job on CPU.\n"
            f"Runtime: {executable}\n{diagnostic}"
        )
    return devices


def build_cuda_runtime(bin_dir: str, tag: str, environment_for) -> None:
    """Build a release in isolation and verify CUDA before replacing the cache."""
    if not re.fullmatch(r"b[1-9]\d*", tag):
        raise ValueError(f"Invalid llama.cpp build tag: {tag!r}")
    nvcc = find_nvcc()
    git = shutil.which("git")
    cmake = shutil.which("cmake")
    ninja = shutil.which("ninja")
    missing = [name for name, tool in (("git", git), ("cmake", cmake), ("CUDA toolkit (nvcc)", nvcc)) if not tool]
    if not ninja and not shutil.which("make"):
        missing.append("ninja or make")
    if missing:
        raise RuntimeError(
            "Linux GPU LLM setup needs a CUDA-enabled llama-server; upstream's Ubuntu "
            "archive is CPU-only. Missing build tools: " + ", ".join(missing) + ". "
            "Install the NVIDIA CUDA toolkit and C++ build tools, then retry Load in "
            "Settings (RTX 50-series requires CUDA 12.8 or newer). Alternatively, set "
            "MAESTRO_LLAMA_BIN to an existing CUDA llama-server directory. "
            "See docs/LLM-runtime.md. CPU remains available when explicitly selected."
        )

    os.makedirs(bin_dir, exist_ok=True)
    log_path = os.path.join(bin_dir, "llama-cuda-build.log")
    print(f"[LLM] Building llama.cpp {tag} with CUDA ({nvcc}); one-time setup may take several minutes.")
    print(f"[LLM] Build progress: {log_path}")
    with tempfile.TemporaryDirectory(prefix=".cuda-build-", dir=bin_dir) as work:
        source = os.path.join(work, "source")
        build = os.path.join(work, "build")
        cuda_root = os.path.dirname(os.path.dirname(os.path.realpath(nvcc)))
        cuda_libs = [path for path in (
            os.path.join(cuda_root, "lib64"), os.path.join(cuda_root, "lib"),
            *glob.glob(os.path.join(cuda_root, "targets", "*", "lib")),
        ) if os.path.isdir(path)]
        # Preserve toolkit lookup after the temporary build directory is removed.
        # $ORIGIN is literal here; subprocess receives an argv list, not a shell.
        configure = [
            cmake, "-S", source, "-B", build, "-DCMAKE_BUILD_TYPE=Release",
            "-DGGML_CUDA=ON", f"-DCMAKE_CUDA_COMPILER={nvcc}",
            # A shallow release clone has one commit; its count is not the
            # upstream build number and would otherwise trigger endless upgrades.
            f"-DLLAMA_BUILD_NUMBER={int(tag[1:])}",
            "-DCMAKE_CUDA_ARCHITECTURES=native", "-DGGML_BACKEND_DL=OFF",
            "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF",
            "-DLLAMA_BUILD_APP=OFF", "-DLLAMA_BUILD_UI=OFF",
            "-DLLAMA_USE_PREBUILT_UI=OFF", "-DLLAMA_OPENSSL=OFF", "-DLLAMA_CURL=OFF",
            "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
            "-DCMAKE_INSTALL_RPATH=" + ";".join(["$ORIGIN", *cuda_libs]),
        ]
        if ninja:
            configure += ["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}"]
        commands = [
            ("source download", [git, "clone", "--depth", "1", "--branch", tag,
                                 "https://github.com/ggml-org/llama.cpp.git", source], 300),
            ("configuration", configure, 300),
            ("compilation", [cmake, "--build", build, "--config", "Release", "--target", "llama-server",
                             "--parallel", str(min(os.cpu_count() or 2, 4))], 2400),
        ]
        with open(log_path, "w", encoding="utf-8") as log:
            for label, command, timeout in commands:
                print(f"[LLM] CUDA runtime: {label}...")
                log.write(f"\n{label}: {command!r}\n")
                log.flush()
                try:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout)
                except (OSError, subprocess.SubprocessError) as error:
                    raise RuntimeError(
                        f"Linux CUDA llama-server {label} failed. See {log_path}. "
                        "Check the installed CUDA toolkit and compatible C++ compiler; "
                        "the existing runtime was kept."
                    ) from error
        runtime = os.path.join(build, "bin")
        executable = os.path.join(runtime, "llama-server")
        require_cuda(executable, environment_for(executable))
        # Copy only build artifacts, not the source tree. Replace files atomically,
        # including SONAME aliases, without following an old destination symlink.
        # Replace the executable last, after all of its sibling dependencies.
        artifacts = [*glob.glob(os.path.join(runtime, "*.so*")), executable]
        for path in artifacts:
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=bin_dir, delete=False) as target:
                    temporary = target.name
                shutil.copy2(path, temporary)
                os.replace(temporary, os.path.join(bin_dir, os.path.basename(path)))
            finally:
                if temporary and os.path.isfile(temporary):
                    os.remove(temporary)
    require_cuda(os.path.join(bin_dir, "llama-server"), environment_for(os.path.join(bin_dir, "llama-server")))
