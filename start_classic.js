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
  // SERVER_NAME intentionally not set — wgp.py defaults to "localhost"
  // when SERVER_NAME isn't in the env. The classic UI is the legacy
  // secondary surface (Gradio); we don't surface a PINOKIO_SHARE_LOCAL
  // path here because (a) it would require patching upstream wgp.py
  // (intrusive / conflict risk on every Wan2GP sync) and (b) the
  // canonical UI is launch.py which DOES honor PINOKIO_SHARE_LOCAL
  // properly. Users who need LAN-shared classic can use the listen
  // flag manually via the Pinokio CLI.
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
            "python wgp.py --multiple-images {{args.compile ? '--compile' : ''}}"
          ],
          on: [{
            "event": "/(http:\/\/[0-9.:]+)/",
            "done": true
          }]
        }
      },
      {
        method: "local.set",
        params: {
          url: "{{input.event[1]}}"
        }
      }
    ]
  }
}
