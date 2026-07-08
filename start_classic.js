module.exports = async (kernel) => {
  let port = await kernel.port()
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
      {
        method: "shell.run",
        params: {
          venv: "env",
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
