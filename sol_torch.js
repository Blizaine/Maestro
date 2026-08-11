const { isSolCapable, solRuntimeProfile } = require("./launcher_profile")

module.exports = async (kernel) => {
  if (!isSolCapable(kernel)) {
    throw new Error(
      "Maestro's optimized H3 Sol Engine supports NVIDIA SM89, SM90, SM100, and SM120 GPUs."
    )
  }

  const runtime = solRuntimeProfile(kernel)
  const windows = kernel.platform === "win32"
  const linux = kernel.platform === "linux"
  if (!windows && !linux) {
    throw new Error("Maestro's Sol Engine runtime is supported on Windows and Linux.")
  }

  const cudaArch = ({
    sm_89: "8.9",
    sm_90: "9.0",
    sm_100: "10.0",
    sm_120: "12.0",
  })[String(kernel.gpu_target || "").toLowerCase()] || "8.9"
  const env = linux ? {
    TORCH_CUDA_ARCH_LIST: cudaArch,
    MAX_JOBS: "4",
  } : undefined
  const message = windows ? [
    "uv pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130 --force-reinstall --no-deps",
    "{{args && args.xformers ? 'uv pip install xformers==0.0.35 --index-url https://download.pytorch.org/whl/cu130 --force-reinstall --no-deps' : ''}}",
    "uv pip install triton-windows==3.6.0.post25 --force-reinstall",
    "uv pip install https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post4/sageattention-2.2.0+cu130torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl --force-reinstall --no-deps",
    "uv pip install https://github.com/deepbeepmeep/kernels/releases/download/Light2xv/lightx2v_kernel-0.0.2+torch2.10.0-cp311-abi3-win_amd64.whl --force-reinstall --no-deps",
    "uv pip install https://github.com/nunchaku-ai/nunchaku/releases/download/v1.2.1/nunchaku-1.2.1+cu13.0torch2.10-cp311-cp311-win_amd64.whl --force-reinstall --no-deps",
    "uv pip install https://github.com/deepbeepmeep/kernels/releases/download/Flash2/flash_attn-2.8.3-cp311-cp311-win_amd64.whl --force-reinstall --no-deps",
  ] : [
    "uv pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130 --force-reinstall --no-deps",
    "{{args && args.xformers ? 'uv pip install xformers==0.0.35 --index-url https://download.pytorch.org/whl/cu130 --force-reinstall --no-deps' : ''}}",
    "uv pip install 'triton>=3.6,<3.7' --force-reinstall",
    "uv pip install 'setuptools<=75.8.2' ninja wheel --force-reinstall",
    "uv pip install --no-build-isolation git+https://github.com/thu-ml/SageAttention.git",
    "uv pip install https://github.com/deepbeepmeep/kernels/releases/download/Light2xv/lightx2v_kernel-0.0.2+torch2.10.0-cp311-abi3-linux_x86_64.whl --force-reinstall --no-deps",
    "uv pip install https://github.com/nunchaku-ai/nunchaku/releases/download/v1.2.1/nunchaku-1.2.1+cu13.0torch2.10-cp311-cp311-linux_x86_64.whl --force-reinstall --no-deps",
    "uv pip install flash-attn --no-build-isolation",
  ]

  return {
    run: [{
      method: "log",
      params: {
        raw: `Installing Maestro's optional ${runtime.label} runtime...`,
      },
    }, {
      method: "shell.run",
      params: {
        venv: "{{args && args.venv ? args.venv : null}}",
        path: "{{args && args.path ? args.path : '.'}}",
        ...(env ? { env } : {}),
        message,
      },
    }, {
      method: "fs.write",
      params: {
        path: runtime.marker,
        text: "Maestro H3 Sol Engine runtime installed. Delete this file and run the Sol repair action to reinstall it.",
      },
    }, {
      method: "fs.write",
      params: {
        path: runtime.flashMarker,
        text: "Maestro H3 Sol Engine optional FlashAttention wheel installed.",
      },
    }],
  }
}
