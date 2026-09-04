const { isSolCapable, solRuntimeProfile } = require("./launcher_profile")

module.exports = async (kernel) => {
  if (!isSolCapable(kernel)) {
    throw new Error(
      "The optimized H3 Sol Engine requires an NVIDIA SM89, SM90, SM100, or SM120 GPU."
    )
  }
  const fallbackPort = await kernel.port()
  const port = `{{local.remote_access && local.remote_access.enabled && local.remote_access.pinokio_port_lock && local.remote_access.target_port ? local.remote_access.target_port : ${fallbackPort}}}`
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
      when: "{{exists('app/settings/remote_access.json')}}",
      method: "json.get",
      params: {
        remote_access: "app/settings/remote_access.json",
      },
    }, {
      when: "{{platform === 'win32' && local.remote_access && local.remote_access.enabled && local.remote_access.windows_restore_task}}",
      method: "shell.run",
      params: {
        path: ".",
        message: {
          _: [
            "schtasks.exe",
            "/Run",
            "/TN",
            "Maestro Tailscale Serve",
          ],
        },
        on: [{
          event: "/ERROR:/i",
          break: false,
        }],
      },
    }, {
      // Keep the legacy Sol entry point self-healing too. This runs only when
      // an interrupted install/update left no complete Vite output.
      when: "{{exists('ui/package.json') && (!exists('ui/dist/index.html') || !exists('ui/dist/assets'))}}",
      method: "shell.run",
      params: {
        path: "ui",
        message: [
          "npm install",
          "npm run build",
        ],
      },
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
        port: "{{input.event[1].split(':').pop()}}",
      },
    }],
  }
}
