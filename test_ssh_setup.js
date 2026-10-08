// Optional diagnostic access. Normal tests do not need SSH.
// The fixed PowerShell helper performs the administrative setup, following
// tailscale_setup.js's structured shell.run / sudo pattern.
module.exports = async (kernel) => ({
  run: [{
    method: "shell.run",
    params: {
      path: "app",
      sudo: true,
      message: { _: [
        kernel.which("powershell") || "powershell", "-NoProfile", "-NonInteractive",
        "-ExecutionPolicy", "Bypass", "-File", "scripts/setup_test_ssh.ps1",
        "-PublicKeyFile", "{{args.public_key_file}}",
        "-ClientAddress", "{{args.client_address}}",
        "-AccountName", "{{args.account_name}}",
      ] },
    },
  }],
})
