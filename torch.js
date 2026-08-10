const { isRtx50, runtimeProfile } = require("./launcher_profile")

module.exports = async (kernel) => {
  const runtime = runtimeProfile(kernel)
  const rtx50 = isRtx50(kernel)
  const windows = kernel.platform === "win32"
  const linux = kernel.platform === "linux"

  if (!windows && !linux) {
    throw new Error("Maestro's NVIDIA runtime is supported on Windows and Linux.")
  }

  let message
  let env = undefined

  if (rtx50 && windows) {
    message = [
      // Blackwell's native NVFP4 path requires the CUDA 13 / Torch 2.10 ABI.
      "uv pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130 --force-reinstall --no-deps",
      "{{args && args.xformers ? 'uv pip install xformers==0.0.35 --index-url https://download.pytorch.org/whl/cu130 --force-reinstall --no-deps' : ''}}",
      "uv pip install -U triton-windows",
      "uv pip install https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post4/sageattention-2.2.0+cu130torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl --force-reinstall --no-deps",
      "uv pip install https://github.com/deepbeepmeep/kernels/releases/download/Flash2/flash_attn-2.8.3-cp311-cp311-win_amd64.whl --force-reinstall --no-deps",
      "uv pip install https://github.com/deepbeepmeep/kernels/releases/download/Light2xv/lightx2v_kernel-0.0.2+torch2.10.0-cp311-abi3-win_amd64.whl --force-reinstall --no-deps",
      "uv pip install https://github.com/nunchaku-ai/nunchaku/releases/download/v1.2.1/nunchaku-1.2.1+cu13.0torch2.10-cp311-cp311-win_amd64.whl --force-reinstall --no-deps",
    ]
  } else if (rtx50 && linux) {
    message = [
      "uv pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130 --force-reinstall --no-deps",
      "{{args && args.xformers ? 'uv pip install xformers==0.0.35 --index-url https://download.pytorch.org/whl/cu130 --force-reinstall --no-deps' : ''}}",
      "uv pip install -U triton",
      "uv pip install 'setuptools<=75.8.2' ninja wheel --force-reinstall",
      "uv pip install --no-build-isolation git+https://github.com/thu-ml/SageAttention.git",
      "uv pip install flash-attn --no-build-isolation",
      "uv pip install https://github.com/deepbeepmeep/kernels/releases/download/Light2xv/lightx2v_kernel-0.0.2+torch2.10.0-cp311-abi3-linux_x86_64.whl --force-reinstall --no-deps",
      "uv pip install https://github.com/nunchaku-ai/nunchaku/releases/download/v1.2.1/nunchaku-1.2.1+cu13.0torch2.10-cp311-cp311-linux_x86_64.whl --force-reinstall --no-deps",
    ]
    env = {
      TORCH_CUDA_ARCH_LIST: "12.0",
      MAX_JOBS: "4",
    }
  } else if (windows) {
    // Preserve the known-good public runtime for RTX 20/30/40 systems.
    message = [
      "uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 {{args && args.xformers ? 'xformers==0.0.30' : ''}} --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps",
      "uv pip install triton-windows==3.3.1.post19",
      "uv pip install https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows/sageattention-2.2.0+cu128torch2.7.1-cp310-cp310-win_amd64.whl",
      "uv pip install https://huggingface.co/cocktailpeanut/wheels/resolve/main/flash_attn-2.8.2%2Bcu128torch2.7-cp310-cp310-win_amd64.whl",
    ]
  } else {
    message = [
      "uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 {{args && args.xformers ? 'xformers==0.0.30' : ''}} --index-url https://download.pytorch.org/whl/cu128 --force-reinstall",
      "uv pip install https://huggingface.co/MonsterMMORPG/SECourses_Premium_Flash_Attention/resolve/main/sageattention-2.1.1-cp310-cp310-linux_x86_64.whl",
      "uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.7.4+cu128torch2.7-cp310-cp310-linux_x86_64.whl",
      "uv pip install numpy==2.1.2",
    ]
  }

  return {
    run: [
      {
        method: "log",
        params: {
          raw: `Installing Maestro's ${runtime.label} acceleration runtime...`,
        },
      },
      {
        method: "shell.run",
        params: {
          venv: "{{args && args.venv ? args.venv : null}}",
          path: "{{args && args.path ? args.path : '.'}}",
          ...(env ? { env } : {}),
          message,
        },
      },
      {
        // update.js uses this hardware-specific marker to avoid unnecessary
        // multi-gigabyte reinstalls while still making interrupted migrations
        // resumable.
        method: "fs.write",
        params: {
          path: runtime.marker,
          text: `Maestro ${runtime.label} runtime installed. Delete this file and run Update to reinstall it.`,
        },
      },
    ],
  }
}
