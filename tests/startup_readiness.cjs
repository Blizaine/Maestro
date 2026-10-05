"use strict";

const assert = require("node:assert/strict");
const start = require("../start.js");
const startSol = require("../start_sol.js");
const launcher = require("../pinokio.js");

const kernel = {
  gpu: "nvidia",
  gpu_target: "sm_89",
  gpu_driver: "580",
  platform: "win32",
  port: async () => 7860,
};

function compilePinokioEvent(value) {
  const match = /^\/(.*)\/([a-z]*)$/i.exec(value);
  assert.ok(match, `Expected a Pinokio event pattern, received ${value}`);
  return new RegExp(match[1], match[2]);
}

function evaluateTemplate(template, input, local) {
  const match = /^\{\{([\s\S]*)\}\}$/.exec(template);
  assert.ok(match, `Expected a Pinokio template expression, received ${template}`);
  return new Function("input", "local", `return (${match[1]})`)(input, local);
}

function startupSteps(config) {
  const clear = config.run[0];
  assert.equal(clear.method, "local.rm", "Startup must remove stale readiness without triggering URL sharing");
  assert.deepEqual(clear.params, ["url", "port"]);

  const server = config.run.find(
    (step) => step.method === "shell.run" && step.params.path === "app",
  );
  assert.ok(server, "Expected a backend shell.run step");

  const failed = config.run.find(
    (step) => step.method === "input" && step.params.title === "Maestro failed to start",
  );
  assert.ok(failed, "Expected a clear no-ready-URL failure step");

  const localSet = config.run.find(
    (step) => step.method === "local.set" && step.params.url === "{{input.event[1]}}",
  );
  assert.ok(localSet, "Expected url to remain set from input.event[1]");
  const serverIndex = config.run.indexOf(server);
  const localSetIndex = config.run.indexOf(localSet);
  const failedIndex = config.run.indexOf(failed);
  assert.equal(
    localSetIndex,
    serverIndex + 1,
    "The URL assignment must immediately consume the shell.run event before any conditional step can replace it",
  );
  assert.ok(failedIndex > localSetIndex, "The failure prompt must follow URL assignment");

  return { server, failed, localSet, serverIndex };
}

function simulateStartup(config, event, previousLocal = {}) {
  const clear = config.run[0];
  const { serverIndex } = startupSteps(config);
  let input = {};
  const local = { ...previousLocal };
  for (const key of clear.params) delete local[key];
  let stopped = false;

  for (let index = serverIndex; index < config.run.length; index += 1) {
    const step = config.run[index];
    if (step.method === "shell.run") {
      input = event === undefined ? {} : { event };
      continue;
    }

    const shouldRun = step.when === undefined
      || evaluateTemplate(step.when, input, local);
    if (!shouldRun) {
      // Pinokio advances past a skipped conditional step without preserving
      // the previous RPC return as the next step's input.
      input = {};
      continue;
    }

    if (step.method === "local.set") {
      for (const [key, value] of Object.entries(step.params)) {
        local[key] = typeof value === "string" && value.startsWith("{{")
          ? evaluateTemplate(value, input, local)
          : value;
      }
      input = {};
    } else if (step.method === "input") {
      stopped = true;
      break;
    }
  }

  assert.equal(
    stopped,
    !local.url || !local.port,
    "Startup must either stop or set one valid endpoint",
  );
  return { stopped, local };
}

function menuInfo(runningScript, local) {
  return {
    exists: () => true,
    running: (script) => script === runningScript,
    local: (script) => script === runningScript ? local : null,
  };
}

async function menuFor(runningScript, local) {
  return launcher.menu(kernel, menuInfo(runningScript, local));
}

async function checkLauncher(script, runningScript, openLabel) {
  const config = await script(kernel);
  assert.equal(config.daemon, true, `${runningScript} must keep the backend alive`);

  const { server } = startupSteps(config);
  const success = server.params.on.find((event) => event.done === true);
  assert.ok(success, "Expected the existing successful URL capture event");
  assert.equal(success.event, "/(http://[0-9.:]+)/");

  const match = compilePinokioEvent(success.event).exec("Serving Maestro at http://127.0.0.1:7860");
  assert.ok(match, "The success event must capture a normal local IPv4 endpoint");
  assert.equal(match[1], "http://127.0.0.1:7860");

  const failurePattern = server.params.on.find(
    (event) => event.break === true && /smart app control/i.test(event.event),
  );
  assert.ok(failurePattern, "Expected App Control / Device Guard break handling");
  const failureMatcher = compilePinokioEvent(failurePattern.event);
  for (const message of [
    "Windows Defender Application Control has blocked this app.",
    "Device Guard prevented this unsigned Python executable from starting.",
    "Smart App Control blocked an untrusted app.",
    "This app has been blocked by your system administrator.",
    "'C:\\pinokio\\api\\Maestro.git\\app\\env-rtx50\\Scripts\\python.exe' was blocked by your organization's Device Guard policy.",
  ]) {
    assert.match(message, failureMatcher, `Expected to recognize: ${message}`);
  }

  const ready = simulateStartup(config, match);
  assert.equal(ready.stopped, false);
  assert.deepEqual(ready.local, { url: match[1], port: "7860" });
  const stale = simulateStartup(config, undefined, ready.local);
  assert.equal(stale.stopped, true);
  assert.deepEqual(stale.local, {}, "A failed retry must remove the previous endpoint");
  const remoteAccess = { enabled: true, target_port: "7860" };
  const preserved = simulateStartup(config, undefined, { ...ready.local, remote_access: remoteAccess });
  assert.deepEqual(preserved.local, { remote_access: remoteAccess }, "Readiness reset must preserve remote-access preferences");

  for (const event of [
    undefined,
    ["Device Guard blocked this unsigned executable."],
    [],
    ["", undefined],
    ["{{input.event[1]}}", "{{input.event[1]}}"],
    ["javascript:alert(1)", "javascript:alert(1)"],
    ["https://127.0.0.1:7860", "https://127.0.0.1:7860"],
    ["http://127.0.0.1", "http://127.0.0.1"],
    ["http://999.999.999.999:7860", "http://999.999.999.999:7860"],
    ["http://127.0.0.1:65536", "http://127.0.0.1:65536"],
  ]) {
    const result = simulateStartup(config, event);
    assert.equal(result.stopped, true, `Expected startup to stop for ${String(event)}`);
    assert.deepEqual(result.local, {}, "Invalid events must not retain or set URL or port");
  }

  const validLocal = { url: "http://127.0.0.1:7860", port: "7860" };
  const validMenu = await menuFor(runningScript, validLocal);
  const openItem = validMenu.find((item) => item.text === openLabel);
  assert.ok(openItem, "A valid HTTP endpoint should remain available in the menu");
  assert.equal(openItem.href, validLocal.url);
  const tailscaleItem = validMenu.find((item) => item.href === "tailscale_setup.js");
  assert.equal(tailscaleItem.params.port, "7860");

  for (const local of [
    { url: "https://127.0.0.1:443", port: "443" },
    { url: "https://127.0.0.1", port: "443" },
    { url: "http://127.0.0.1", port: "80" },
  ]) {
    const items = await menuFor(runningScript, local);
    const openItem = items.find((item) => item.text === openLabel);
    assert.ok(openItem, `A valid HTTP(S) endpoint should be available: ${JSON.stringify(local)}`);
    assert.equal(openItem.href, local.url);
    assert.equal(
      items.find((item) => item.href === "tailscale_setup.js").params.port,
      local.port,
    );
  }

  for (const local of [
    null,
    { url: "{{input.event[1]}}", port: "7860" },
    { url: "http://127.0.0.1:7860/{{input.event[1]}}", port: "7860" },
    { url: "javascript:alert(1)", port: "7860" },
    { url: "ftp://127.0.0.1:7860", port: "7860" },
    { url: "https://127.0.0.1", port: "80" },
    { url: "http://127.0.0.1:99999", port: "99999" },
    { url: "http://127.0.0.1:7860", port: "{{port}}" },
    { url: "http://127.0.0.1:7860", port: "9000" },
  ]) {
    const items = await menuFor(runningScript, local);
    assert.equal(
      items.some((item) => item.text === openLabel),
      false,
      `The menu must hide an invalid endpoint: ${JSON.stringify(local)}`,
    );
    assert.equal(
      items.some((item) => item.href === "tailscale_setup.js"),
      false,
      "The menu must not pass an invalid port to Tailscale setup",
    );
    assert.equal(
      items.some((item) => typeof item.href === "string" && item.href.includes("{{")),
      false,
      "The menu must never expose an unresolved template",
    );
  }
}

(async () => {
  await checkLauncher(start, "start.js", "Open Web UI");
  await checkLauncher(startSol, "start_sol.js", "Open Web UI (Sol Runtime)");
  process.stdout.write("Startup readiness checks passed for normal and Sol launchers.\n");
})().catch((error) => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exitCode = 1;
});
