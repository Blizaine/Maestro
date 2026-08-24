const {
  isRtx50,
  legacyRuntimeProfile,
  runtimeProfile,
} = require("./launcher_profile")

module.exports = async (kernel) => {
  let port = await kernel.port()
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
      ...runtimeGuard,
      ...(hasRecoveryRuntime ? [{
        when: `{{!exists('${runtime.marker}')}}`,
        method: "log",
        params: {
          raw: "The preferred H3 acceleration runtime is not ready; starting the preserved compatibility runtime. Run Update to finish the automatic migration.",
        },
      }] : []),
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
            "event": "/(http:\/\/[0-9.:]+)/",
            "done": true
          }]
        }
      },
      {
        method: "local.set",
        params: {
          url: "{{input.event[1]}}",
          port: "{{input.event[1].split(':').pop()}}"
        }
      }
    ]
  }
}
