const readyEvent = "/^MAESTRO_AGENT_ACCESS_READY (https:\\/\\/127\\.0\\.0\\.1:\\d+)\\r?$/"

module.exports = async () => ({
  daemon: true,
  run: [{
    method: "local.rm",
    params: ["url", "port", "agent_access_url"],
  }, {
    when: "{{!exists('app/env-agent/pyvenv.cfg')}}",
    method: "shell.run",
    params: {
      venv: "env-agent",
      venv_python: "3.11",
      path: "app",
      message: "uv pip install -r requirements-agent-access.txt",
    },
  }, {
    method: "shell.run",
    params: {
      venv: "env-agent",
      venv_python: "3.11",
      path: "app",
      message: [
        "python -u -m services.agent_access.controller --app-root . {{args.setup ? '--setup' : ''}}",
      ],
      on: [{
        event: readyEvent,
        done: true,
      }],
    },
  }, {
    when: `input && input.event && typeof input.event[1] === "string" && /^https:\/\/127\.0\.0\.1:\d+$/.test(input.event[1])`,
    method: "local.set",
    params: {
      // local.url invokes Pinokio's automatic sharing hook. This TLS service
      // exposes LAN access only after the owner's explicit opt-in.
      agent_access_port: "{{input.event[1].split(':').pop()}}",
      agent_access_url: "{{input.event[1]}}",
    },
  }, {
    when: "{{!local.agent_access_url}}",
    method: "input",
    params: {
      title: "Agent Access did not become ready",
      description: "Check the Agent Access terminal for the startup error. Maestro can still be used normally.",
    },
    next: null,
  }],
})
