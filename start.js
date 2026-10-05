const {
  isRtx50,
  legacyRuntimeProfile,
  runtimeProfile,
} = require("./launcher_profile")

const appControlFailureEvent = "/(?:smart app control|application control|app control|device guard|wdac).*\\b(?:block(?:ed|ing)?|deny|denied|denial|prevent(?:ed|ing)?|restrict(?:ed|ion)?|unsigned|untrusted|reject(?:ed|ion)?|not allowed|not permitted)\\b|\\b(?:block(?:ed|ing)?|deny|denied|denial|prevent(?:ed|ing)?|restrict(?:ed|ion)?|unsigned|untrusted|reject(?:ed|ion)?|not allowed|not permitted)\\b.*(?:smart app control|application control|app control|device guard|wdac)|this app has been blocked by (?:your )?system administrator|blocked by group policy/i"
const readyUrlPattern = /^http:\/\/(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d):(?:[1-9]\d{0,3}|[1-5]\d{4}|6[0-4]\d{3}|65[0-4]\d{2}|655[0-2]\d|6553[0-5])$/
const hasReadyUrl = `input && input.event && typeof input.event[1] === "string" && ${readyUrlPattern}.test(input.event[1])`

module.exports = async (kernel) => {
  const fallbackPort = await kernel.port()
  // A successful one-time Tailscale setup records the exact Maestro backend
  // port it proxies. Reuse that port on later launches so the persistent
  // `tailscale serve --bg` route does not become stale when Pinokio assigns a
  // new dynamic port. If the user has never opted in, keep Pinokio's normal
  // conflict-safe dynamic port behavior.
  const port = `{{local.remote_access && local.remote_access.enabled && local.remote_access.pinokio_port_lock && local.remote_access.target_port ? local.remote_access.target_port : ${fallbackPort}}}`
  const runtime = runtimeProfile(kernel)
  const legacyRuntime = legacyRuntimeProfile(kernel)
  const hasRecoveryRuntime = runtime.env !== legacyRuntime.env
  const selectedEnv = hasRecoveryRuntime
    ? `{{exists('${runtime.marker}') ? '${runtime.env}' : '${legacyRuntime.env}'}}`
    : runtime.env
  const selectedPython = hasRecoveryRuntime
    ? `{{exists('${runtime.marker}') ? '${runtime.python}' : '${legacyRuntime.python}'}}`
    : runtime.python
  const runtimeGuard = isRtx50(kernel) ? [{
    when: `{{!exists('${runtime.marker}')}}`,
    method: "input",
    params: {
      title: "RTX 50 runtime upgrade required",
      description: "Run Update once to install Maestro's Python 3.11 / CUDA 13 acceleration environment, then start Maestro again. Your existing environment is preserved."
    },
    next: null
  }] : []
  // SERVER_NAME is intentionally NOT set here. The host-binding
  // decision lives in launch.py, which reads PINOKIO_SHARE_LOCAL
  // from the merged shell env (per-app ENVIRONMENT overrides global
  // there). kernel.envs in this start.js context only exposes the
  // global ENVIRONMENT, so a per-app override of PINOKIO_SHARE_LOCAL
  // wouldn't be visible if we made the decision here. See launch.py
  // bottom for the full priority chain.
  return {
    requires: {
      bundle: "ai",
    },
    daemon: true,
    run: [
      {
        // Assigning even a null URL invokes Pinokio's sharing hook.
        // Remove stale readiness keys without starting a share before launch.
        method: "local.rm",
        params: ["url", "port"],
      },
      ...runtimeGuard,
      {
        when: "{{exists('app/settings/remote_access.json')}}",
        method: "json.get",
        params: {
          remote_access: "app/settings/remote_access.json",
        },
      },
      {
        when: "{{platform === 'win32' && local.remote_access && local.remote_access.enabled && local.remote_access.windows_restore_task}}",
        method: "shell.run",
        params: {
          path: ".",
          // Tailscale Serve configuration requires elevation on Windows. The
          // user's one-time setup created this fixed on-demand task with their
          // approval, so later starts can restore the private route without a
          // new UAC prompt. A missing/deleted helper never blocks local start.
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
      },
      ...(hasRecoveryRuntime ? [{
        when: `{{!exists('${runtime.marker}')}}`,
        method: "log",
        params: {
          raw: "The preferred H3 acceleration runtime is not ready; starting the preserved compatibility runtime. Run Update to finish the automatic migration.",
        },
      }] : []),
      {
        // A pulled update can be interrupted after Git advances but before
        // Vite finishes. Build only when the served React bundle is missing,
        // so the next normal Start repairs that state without a Reset or
        // manual terminal commands.
        when: "{{exists('ui/package.json') && (!exists('ui/dist/index.html') || !exists('ui/dist/assets'))}}",
        method: "shell.run",
        params: {
          path: "ui",
          message: [
            "npm install",
            "npm run build",
          ],
        },
      },
      // SAM service starts on demand (launched by the backend when inpaint is used)
      // — not started here to avoid holding a CUDA context that wastes VRAM
      {
        method: "shell.run",
        params: {
          venv: selectedEnv,
          venv_python: selectedPython,
          env: {
            SERVER_PORT: port
          },
          path: "app",
          message: [
            "python launch.py {{args.compile ? '--compile' : ''}}"
          ],
          on: [{
            "event": "/Incorrect version of mmgp/i",
            "break": true
          }, {
            // Windows App Control / Device Guard can block the unsigned
            // interpreter before the backend starts. Stop on those messages
            // so the user sees the security failure directly.
            "event": appControlFailureEvent,
            "break": true
          }, {
            "event": "/(http:\/\/[0-9.:]+)/",
            "done": true
          }]
        }
      },
      {
        when: `{{${hasReadyUrl}}}`,
        method: "local.set",
        params: {
          url: "{{input.event[1]}}",
          port: "{{input.event[1].split(':').pop()}}"
        },
      },
      {
        when: "{{!local.url || !local.port}}",
        method: "input",
        params: {
          title: "Maestro failed to start",
          description: "The backend stopped before reporting a valid local Web UI address. Check the Terminal output for the startup error. On Windows, App Control or Device Guard may have blocked the unsigned Python executable."
        },
        next: null
      }
    ]
  }
}
