const { legacyRuntimeProfile, runtimeProfile } = require("./launcher_profile")

// Run the client as the Pinokio desktop user so Windows key ownership/ACLs
// stay with that user rather than an external coding sandbox account.
module.exports = async (kernel) => {
  const preferred = runtimeProfile(kernel)
  const legacy = legacyRuntimeProfile(kernel)
  const selectedEnv = preferred.env !== legacy.env
    ? `{{exists('${preferred.marker}') ? '${preferred.env}' : '${legacy.env}'}}`
    : preferred.env
  return { run: [{ method: "shell.run", params: {
    path: "app", venv: selectedEnv,
    message: { _: ["python", "scripts/test_ssh_client.py", "--action", "{{args.action || 'prepare'}}"] },
  } }] }
}
