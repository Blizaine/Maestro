const { isSolCapable, solRuntimeProfile } = require("./launcher_profile")

module.exports = async (kernel) => {
  if (!isSolCapable(kernel)) {
    throw new Error(
      "The optimized H3 Sol Engine requires an NVIDIA SM89, SM90, SM100, or SM120 GPU."
    )
  }
  const port = await kernel.port()
  const runtime = solRuntimeProfile(kernel)
  return {
    requires: {
      bundle: "ai",
    },
    daemon: true,
    run: [{
      when: `{{!exists('${runtime.marker}')}}`,
      method: "input",
      params: {
        title: "H3 performance runtime required",
        description: "Run Maestro's normal Update action to install or repair the H3 performance runtime, then use the normal Start button.",
      },
      next: null,
    }, {
      method: "shell.run",
      params: {
        venv: runtime.env,
        venv_python: runtime.python,
        env: {
          SERVER_PORT: port,
          MAESTRO_SOL_RUNTIME: "1",
        },
        path: "app",
        message: [
          "python launch.py {{args.compile ? '--compile' : ''}}",
        ],
        on: [{
          "event": "/(http:\/\/[0-9.:]+)/",
          "done": true,
        }],
      },
    }, {
      method: "local.set",
      params: {
        url: "{{input.event[1]}}",
      },
    }],
  }
}
