const { legacyRuntimeProfile, runtimeProfile } = require("./launcher_profile")

// Explicit local opt-in only; it never restarts the running server.
module.exports = async (kernel) => {
  const preferred = runtimeProfile(kernel)
  const legacy = legacyRuntimeProfile(kernel)
  const selectedEnv = preferred.env !== legacy.env
    ? `{{exists('${preferred.marker}') ? '${preferred.env}' : '${legacy.env}'}}`
    : preferred.env
  return { run: [{ method: "shell.run", params: {
    path: "app", venv: selectedEnv,
    message: { _: ["python", "-m", "promptbench", "enable"] },
  } }] }
}
