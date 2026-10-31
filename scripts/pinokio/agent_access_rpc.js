#!/usr/bin/env node
"use strict";

// Thin adapter over Pinokio's bundled pterm WebSocket RPC. Do not replace this
// with `pterm start`: that CLI registers process-exit handlers which stop the
// script when its short-lived helper exits.
const fs = require("fs");
const http = require("http");
const os = require("os");
const path = require("path");

const ENGINE_SCRIPTS = new Set(["start.js", "start_sol.js", "start_classic.js"]);
const STARTABLE_SCRIPTS = new Set([...ENGINE_SCRIPTS, "update.js", "agent_access.js"]);
const KNOWN_AUXILIARY_SCRIPTS = new Set([
  "install.js", "update.js", "reset.js", "sol_install.js", "sam_install.js",
  "torch.js", "huggingface_login.js", "agent_access.js",
]);
const APP_NAMES = new Set(["maestro", "maestro-dev", "maestro-main", "maestro.git", "maestro.git-dev", "maestro.git-main"]);
const RPC_TIMEOUT_MS = 4500;

function canonical(value) {
  return path.resolve(value).replace(/[\\/]+$/, "").toLocaleLowerCase("en-US");
}

function readPinokioHome(inputHome) {
  let candidate = typeof inputHome === "string" ? inputHome.trim() : "";
  if (!candidate) {
    try {
      const configPath = path.join(os.homedir(), ".pinokio", "config.json");
      const config = JSON.parse(fs.readFileSync(configPath, "utf8"));
      if (config && typeof config.home === "string") candidate = config.home.trim();
    } catch (_) {
      // Fall through to the explicit environment value.
    }
  }
  if (!candidate) candidate = String(process.env.PINOKIO_HOME || "").trim();
  if (!candidate || !path.isAbsolute(candidate)) throw new Error("PINOKIO_HOME is unavailable");
  const home = fs.realpathSync(candidate);
  if (!fs.statSync(path.join(home, "api")).isDirectory()) throw new Error("PINOKIO_HOME has no api directory");
  return home;
}

function resolveInstalledRoot(appRoot, pinokioHome) {
  const supplied = fs.realpathSync(appRoot);
  const projectRoot = path.basename(supplied).toLowerCase() === "app" ? path.dirname(supplied) : supplied;
  const root = fs.realpathSync(projectRoot);
  const name = path.basename(root).toLowerCase();
  const maestroSource = fs.existsSync(path.join(root, "app", "services", "agent_access", "__init__.py"))
    && fs.existsSync(path.join(root, "app", "launch.py")) && fs.existsSync(path.join(root, "VERSION"));
  if (!APP_NAMES.has(name) && !maestroSource) throw new Error("Only an installed Maestro app can be controlled");
  if (canonical(path.dirname(root)) !== canonical(path.join(pinokioHome, "api"))) {
    throw new Error("Lifecycle control is disabled outside PINOKIO_HOME/api");
  }
  const expectedApp = path.join(root, "app");
  if (!fs.existsSync(expectedApp) || canonical(path.resolve(supplied)) !== canonical(path.resolve(expectedApp))) {
    throw new Error("The supplied path is not the installed Maestro app directory");
  }
  for (const script of ["pinokio.js", "start.js", "update.js"]) {
    if (!fs.statSync(path.join(root, script), { throwIfNoEntry: false })?.isFile()) {
      throw new Error("The installed Maestro launcher is incomplete");
    }
  }
  return root;
}

function getStatus(baseUrl, appId) {
  return new Promise((resolve, reject) => {
    const url = new URL(`/apps/status/${encodeURIComponent(appId)}`, baseUrl);
    const request = http.get(url, { timeout: 3000, headers: { "x-pinokio-client": "pterm" } }, (response) => {
      const chunks = [];
      let size = 0;
      response.on("data", (chunk) => {
        size += chunk.length;
        if (size > 1024 * 1024) {
          request.destroy(new Error("Pinokio status response is too large"));
          return;
        }
        chunks.push(chunk);
      });
      response.on("end", () => {
        if (response.statusCode !== 200) return reject(new Error("Pinokio status was unavailable"));
        try {
          resolve(JSON.parse(Buffer.concat(chunks).toString("utf8")));
        } catch (_) {
          reject(new Error("Pinokio status response was invalid"));
        }
      });
    });
    request.on("timeout", () => request.destroy(new Error("Pinokio status timed out")));
    request.on("error", () => reject(new Error("Pinokio status is unavailable")));
  });
}

async function pinokioEndpoints(pinokioPackage) {
  const endpoint = require(path.join(pinokioPackage, "endpoint.js"));
  // Lifecycle must stay on the local Pinokio instance even if config.json
  // also contains a remote `access` host.
  const resolved = await endpoint.resolveHttpBaseUrl();
  const targetUrl = new URL(resolved);
  const localAddresses = new Set(["127.0.0.1", "localhost", "::1", ...Object.values(os.networkInterfaces()).flat().filter(Boolean).map(item => item.address)]);
  if (!localAddresses.has(targetUrl.hostname.replace(/^\[|\]$/g, ""))) {
    throw new Error("A local Pinokio control plane is required; remote hosts cannot control this installation");
  }
  const target = { protocol: targetUrl.protocol === "https:" ? "https" : "http", host: targetUrl.hostname, port: Number(targetUrl.port || (targetUrl.protocol === "https:" ? 443 : 80)) };
  return {
    http: await endpoint.resolveHttpBaseUrl(target),
    ws: await endpoint.resolveWsBaseUrl(target),
  };
}

async function getPinokioStatus(pinokioHome, projectRoot) {
  const ptermPackage = path.join(pinokioHome, "bin", "npm", "node_modules", "pterm");
  const endpoints = await pinokioEndpoints(ptermPackage);
  const status = await getStatus(endpoints.http, path.basename(projectRoot));
  if (!status || typeof status.path !== "string" || canonical(status.path) !== canonical(projectRoot)) {
    throw new Error("Pinokio could not verify this exact installed app");
  }
  if (!Array.isArray(status.running_scripts)) throw new Error("Pinokio status has no running script list");
  return { status, endpoints, ptermPackage };
}

function normalizedRunningScripts(status, projectRoot) {
  return status.running_scripts.map((entry) => {
    if (typeof entry !== "string" || !entry.trim()) throw new Error("Pinokio returned an unknown running script");
    const text = entry.trim().split("?", 1)[0];
    const target = path.resolve(path.isAbsolute(text) ? text : path.join(projectRoot, text));
    const relative = path.relative(projectRoot, target);
    if (relative === ".." || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) return "__foreign__";
    return relative.split(path.sep).join("/");
  });
}

async function runRpc(pinokioPackage, wsUrl, packet) {
  const RPC = require(path.join(pinokioPackage, "rpc.js"));
  const rpc = new RPC(wsUrl);
  let sent = false;
  let settled = false;
  let timer;
  return await new Promise((resolve) => {
    const finish = (value) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      try { rpc.close(); } catch (_) { /* Socket closure never sends stop. */ }
      resolve(value);
    };
    try {
      const pending = rpc.run(packet, (reply) => finish({ ok: true, dispatched: true, packet: reply }));
      if (rpc.ws && typeof rpc.ws.addEventListener === "function") {
        rpc.ws.addEventListener("open", () => { sent = true; }, { once: true });
      }
      Promise.resolve(pending).then(() => {
        if (!settled) finish(sent
          ? { ok: true, dispatched: true, packet: null }
          : { ok: false, dispatched: false, error: "Pinokio connection closed before dispatch" });
      }).catch(() => finish({ ok: false, dispatched: sent, error: "Pinokio RPC failed" }));
      timer = setTimeout(() => finish(sent
        ? { ok: true, dispatched: true, packet: null }
        : { ok: false, dispatched: false, error: "Pinokio connection timed out" }), RPC_TIMEOUT_MS);
    } catch (_) {
      finish({ ok: false, dispatched: sent, error: "Pinokio RPC could not be opened" });
    }
  });
}

async function dispatch(pinokioHome, projectRoot, operation, scriptName) {
  if (!STARTABLE_SCRIPTS.has(scriptName) || !["start", "stop"].includes(operation)) {
    throw new Error("Unsupported lifecycle operation");
  }
  if (operation === "stop" && !ENGINE_SCRIPTS.has(scriptName)) {
    throw new Error("Only a Maestro engine launcher can be stopped");
  }
  const { status, endpoints, ptermPackage } = await getPinokioStatus(pinokioHome, projectRoot);
  const running = normalizedRunningScripts(status, projectRoot);
  if (running.includes("__foreign__")) throw new Error("Pinokio reported a script outside the installed app root");
  const unknown = running.filter((item) => !ENGINE_SCRIPTS.has(item) && !KNOWN_AUXILIARY_SCRIPTS.has(item));
  if (unknown.length && !(operation === "start" && scriptName === "agent_access.js")) {
    throw new Error("Another Maestro script is running");
  }
  const runningEngines = running.filter((item) => ENGINE_SCRIPTS.has(item));
  const runningMaintenance = running.filter((item) => ["install.js", "update.js", "reset.js", "sol_install.js", "sam_install.js"].includes(item));
  if (runningEngines.length > 1 && scriptName !== "agent_access.js") {
    throw new Error("More than one Maestro launcher is already running");
  }
  if (operation === "stop" && runningMaintenance.length) throw new Error("A Maestro maintenance script is running");
  if (operation === "stop" && runningEngines.length && !running.includes(scriptName)) {
    throw new Error("The active Maestro launcher does not match the requested stop target");
  }
  if (operation === "stop" && !running.includes(scriptName)) {
    return { ok: true, dispatched: false, already_stopped: true };
  }
  if (operation === "start" && scriptName === "update.js" && runningEngines.length) {
    throw new Error("A Maestro engine is still running");
  }
  if (operation === "start" && runningMaintenance.some((item) => item !== scriptName)) {
    throw new Error("Another Maestro maintenance script is running");
  }
  if (operation === "start" && running.includes(scriptName)) {
    if (scriptName === "update.js" && runningEngines.length) throw new Error("A Maestro engine is still running");
    return { ok: true, dispatched: false, already_running: true };
  }
  if (operation === "start" && ENGINE_SCRIPTS.has(scriptName) && runningEngines.length) {
    throw new Error("Another Maestro launcher is already running");
  }
  const uri = path.join(projectRoot, scriptName);
  const packet = operation === "start"
    ? { uri, source: "pterm", client: { cols: 100, rows: 30, source: "pterm" } }
    : { method: "kernel.api.stop", params: { uri } };
  return runRpc(ptermPackage, endpoints.ws, packet);
}

async function autostart() {
  const pinokioHome = readPinokioHome("");
  const current = fs.realpathSync(process.cwd());
  const appRoot = path.basename(current).toLowerCase() === "app" ? current : path.join(current, "app");
  const projectRoot = resolveInstalledRoot(appRoot, pinokioHome);
  const envReady = fs.existsSync(path.join(projectRoot, "app", "env-agent", "pyvenv.cfg"));
  if (!envReady) return { ok: true, skipped: "env-agent is not installed" };
  const { status } = await getPinokioStatus(pinokioHome, projectRoot);
  const running = normalizedRunningScripts(status, projectRoot);
  if (running.includes("agent_access.js")) return { ok: true, already_running: true };
  const result = await dispatch(pinokioHome, projectRoot, "start", "agent_access.js");
  return result.ok
    ? { ok: true, dispatched: true, pending: true }
    : { ok: false, dispatched: result.dispatched === true, message: "Agent Access could not be started; continue using the Maestro app normally." };
}

async function main() {
  try {
    if (process.argv[2] === "autostart") {
      const result = await autostart();
      process.stdout.write(`${JSON.stringify(result)}\n`);
      // Starting Agent Access is optional for the model process. Log a
      // bounded result, but keep a missing/stopped Pinokio control plane from
      // interrupting normal Maestro startup.
      process.exitCode = 0;
      return;
    }
    const raw = await new Promise((resolve, reject) => {
      const chunks = [];
      let size = 0;
      process.stdin.on("data", (chunk) => {
        size += chunk.length;
        if (size > 16 * 1024) reject(new Error("Lifecycle request is too large"));
        else chunks.push(chunk);
      });
      process.stdin.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
      process.stdin.on("error", () => reject(new Error("Could not read lifecycle request")));
    });
    const input = JSON.parse(raw);
    const home = readPinokioHome(input.pinokio_home);
    const root = resolveInstalledRoot(input.app_root, home);
    let result;
    if (input.action === "status") {
      const { status } = await getPinokioStatus(home, root);
      result = { ok: true, status };
    } else if (input.action === "dispatch") {
      result = await dispatch(home, root, input.operation, input.script_name);
    } else {
      throw new Error("Unknown helper action");
    }
    process.stdout.write(`${JSON.stringify(result)}\n`);
  } catch (error) {
    process.stdout.write(`${JSON.stringify({ ok: false, dispatched: false, error: String(error?.message || "Pinokio lifecycle failed").slice(0, 240) })}\n`);
    process.exitCode = process.argv[2] === "autostart" ? 0 : 1;
  }
}

main();
