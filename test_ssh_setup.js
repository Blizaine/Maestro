// Optional diagnostic access. Normal tests do not need SSH.
// The fixed PowerShell helper performs the administrative setup, following
// tailscale_setup.js's structured shell.run / sudo pattern.
module.exports = async (kernel) => {
  const runId = require("crypto").randomUUID()
  const command = [
    kernel.which("powershell") || "powershell", "-NoProfile", "-NonInteractive",
    "-ExecutionPolicy", "Bypass", "-File",
    "{{path.resolve(cwd, 'app', 'scripts', 'setup_test_ssh.ps1')}}",
    "-PublicKeyFile", "{{args.public_key_file}}",
    "-ClientAddress", "{{args.client_address}}",
    "-AccountName", "{{args.account_name}}",
    "-RunId", runId,
  ]
  return { run: [{
    method: "shell.run",
    params: {
      path: "app",
      sudo: true,
      message: { _: command },
    },
  }, {
    // The elevated shell can return without stdout when Windows blocks or
    // cancels its launch. Verify a fresh receipt in an ordinary shell.
    method: "shell.run",
    params: { path: "app", message: { _: [...command, "-VerifyOnly"] } },
  }] }
}
