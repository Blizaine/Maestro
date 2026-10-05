const { isSolCapable, solRuntimeProfile } = require("./launcher_profile")

const appControlFailureEvent = "/(?:smart app control|application control|app control|device guard|wdac).*\\b(?:block(?:ed|ing)?|deny|denied|denial|prevent(?:ed|ing)?|restrict(?:ed|ion)?|unsigned|untrusted|reject(?:ed|ion)?|not allowed|not permitted)\\b|\\b(?:block(?:ed|ing)?|deny|denied|denial|prevent(?:ed|ing)?|restrict(?:ed|ion)?|unsigned|untrusted|reject(?:ed|ion)?|not allowed|not permitted)\\b.*(?:smart app control|application control|app control|device guard|wdac)|this app has been blocked by (?:your )?system administrator|blocked by group policy/i"
const readyUrlPattern = /^http:\/\/(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d):(?:[1-9]\d{0,3}|[1-5]\d{4}|6[0-4]\d{3}|65[0-4]\d{2}|655[0-2]\d|6553[0-5])$/
const hasReadyUrl = `input && input.event && typeof input.event[1] === "string" && ${readyUrlPattern}.test(input.event[1])`

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
        // Assigning even a null URL invokes Pinokio's sharing hook.
        // Remove stale readiness keys without starting a share before launch.
        method: "local.rm",
        params: ["url", "port"],
    }, {
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
          // Match English App Control / Device Guard failures that prevent
          // the unsigned interpreter from starting.
          event: appControlFailureEvent,
          break: true,
        }, {
          "event": "/(http:\/\/[0-9.:]+)/",
          "done": true,
        }],
      },
    }, {
      when: `{{${hasReadyUrl}}}`,
      method: "local.set",
      params: {
        url: "{{input.event[1]}}",
        port: "{{input.event[1].split(':').pop()}}",
      },
    }, {
      when: "{{!local.url || !local.port}}",
      method: "input",
      params: {
        title: "Maestro failed to start",
        description: "The backend stopped before reporting a valid local Web UI address. Check the Terminal output for the startup error. On Windows, App Control or Device Guard may have blocked the unsigned Python executable.",
      },
      next: null,
    }],
  }
}
