const { isRtx50, isSolCapable, solRuntimeProfile } = require("./launcher_profile")

module.exports = async (kernel) => {
  if (!isSolCapable(kernel)) {
    throw new Error(
      "The optimized H3 Sol Engine requires an NVIDIA SM89, SM90, SM100, or SM120 GPU."
    )
  }
  if (isRtx50(kernel)) {
    throw new Error(
      "RTX 50 systems already receive the compatible CUDA 13 / Triton runtime through Maestro Update."
    )
  }
  const runtime = solRuntimeProfile(kernel)
  return {
    requires: {
      bundle: "ai",
    },
    run: [{
      method: "shell.run",
      params: {
        venv: runtime.env,
        venv_python: runtime.python,
        path: "app",
        message: [
          "uv pip install -r requirements.txt --index-strategy unsafe-best-match",
          "uv pip install hf-xet pip",
        ],
      },
    }, {
      method: "script.start",
      params: {
        uri: "sol_torch.js",
        params: {
          venv: runtime.env,
          path: "app",
          xformers: true,
        },
      },
    }, {
      method: "shell.run",
      params: {
        venv: runtime.env,
        venv_python: runtime.python,
        path: "app",
        message: "python scripts/install_gguf_kernels.py",
      },
    }, {
      method: "input",
      params: {
        title: "H3 Sol Engine runtime installed",
        description: "Start Maestro with the Sol Engine runtime, then enable Sol Engine in H3 Advanced settings.",
      },
    }],
  }
}
