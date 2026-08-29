// Optional user-owned private HTTPS access. This configures Tailscale Serve;
// it never enables Funnel and therefore never exposes Maestro publicly.
module.exports = {
  run: [{
    when: "{{!which('tailscale')}}",
    method: "input",
    params: {
      title: "Install Tailscale first",
      description: "Install the free Tailscale app from https://tailscale.com/download on this computer, sign in to your own account, then run this action again.",
    },
    next: null,
  }, {
    // The launcher keeps this action visible while Maestro is stopped so
    // users can discover it. Route creation must still wait for start.js to
    // capture the actual dynamic server port.
    when: "{{which('tailscale') && !args.port}}",
    method: "input",
    params: {
      title: "Start Maestro first",
      description: "Tailscale is installed. Start Maestro, then return to its Pinokio page and choose Secure Remote Access again. The action will use Maestro's actual running port.",
    },
    next: null,
  }, {
    when: "{{which('tailscale') && args.port}}",
    method: "shell.run",
    params: {
      path: ".",
      sudo: true,
      // Pinokio elevates each array entry independently on Windows. Keep this
      // to the one operation that actually requires administrator access so
      // setup produces one UAC prompt instead of three. The Maestro settings
      // panel performs the connection/status verification afterward.
      message: "\"{{which('tailscale')}}\" serve --bg --yes --https=443 http://127.0.0.1:{{args.port}}",
    },
  }, {
    when: "{{which('tailscale') && args.port}}",
    method: "json.set",
    params: {
      "app/settings/remote_access.json": {
        version: 2,
        enabled: true,
        target_port: "{{args.port}}",
        pinokio_port_lock: true,
      },
    },
  }, {
    when: "{{which('tailscale') && args.port}}",
    method: "notify",
    params: {
      html: "Private Maestro access is ready and will be restored automatically on future Maestro starts. Open Maestro Settings → Notifications to copy or scan the secure URL and finish phone notification setup.",
    },
  }],
}
