// Same-origin owner-session interceptor and API request regressions.
// Run: node tests/ui/agent_access.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');

function resolveEsbuild() {
  const candidates = [
    path.join(root, 'ui/node_modules/esbuild'),
    path.resolve(root, '../../..', 'ui/node_modules/esbuild'),
  ];
  const candidate = candidates.find(fs.existsSync);
  if (!candidate) throw new Error('esbuild is required to run the Agent Access UI test');
  return require(candidate);
}

async function loadAgentAccessApi(esbuild) {
  const source = fs.readFileSync(path.join(root, 'ui/src/api/agentAccess.ts'), 'utf8');
  const bundle = await esbuild.build({
    stdin: { contents: source, resolveDir: root, sourcefile: 'agentAccess.ts', loader: 'ts' },
    bundle: true,
    platform: 'node',
    format: 'cjs',
    write: false,
    target: 'node18',
  });
  const compiled = { exports: {} };
  new Function('module', 'exports', bundle.outputFiles[0].text)(compiled, compiled.exports);
  return compiled.exports;
}

function response(body, status = 200, contentType = 'application/json') {
  const data = typeof body === 'string' ? body : JSON.stringify(body);
  return new Response(data, { status, headers: { 'content-type': contentType } });
}

(async () => {
  const esbuild = resolveEsbuild();
  const api = await loadAgentAccessApi(esbuild);
  const priorWindow = global.window;
  const priorFetch = global.fetch;
  const priorCustomEvent = global.CustomEvent;
  const calls = [];
  const events = [];
  const nativeFetch = async input => {
    const url = new URL(input instanceof Request ? input.url : String(input), 'https://maestro.test/');
    calls.push(url.href);
    if (url.pathname === '/api/v1/agent-access/status') return response({ available: true, enabled: false });
    if (url.pathname === '/api/v1/agent-access/owner-login') {
      return response({ code: 'maestro_auth_required', message: 'Use a fresh code.' }, 401);
    }
    if (url.pathname === '/api/v1/protected') {
      return response({ detail: {
        code: 'maestro_auth_required',
        message: 'Owner session expired.',
        login_url: 'https://maestro.test/agent-access/login',
      } }, 401);
    }
    if (url.pathname === '/api/v1/top-level-protected') {
      return response({ code: 'maestro_auth_required', message: 'Owner session expired.' }, 401);
    }
    if (url.pathname === '/api/v1/ordinary-failure') {
      return response({ code: 'invalid_token', message: 'Different authorization failure.' }, 401);
    }
    if (url.pathname === '/api/v1/html-failure') {
      return response('{"code":"maestro_auth_required"}', 401, 'text/html');
    }
    return response({ ok: true });
  };

  if (typeof global.CustomEvent === 'undefined') {
    global.CustomEvent = class CustomEvent extends Event {
      constructor(type, init = {}) {
        super(type);
        this.detail = init.detail;
      }
    };
  }
  global.window = {
    location: { href: 'https://maestro.test/studio', origin: 'https://maestro.test' },
    fetch: nativeFetch,
    dispatchEvent: event => { events.push(event); return true; },
  };

  try {
    api.installAgentAccessAuthInterceptor();
    global.fetch = (...args) => global.window.fetch(...args);
    assert.equal(calls.length, 0, 'installing the interceptor does not make a startup request');

    await global.fetch('/api/v1/protected');
    assert.equal(events.length, 1, 'a same-origin JSON owner-session 401 emits one event');
    assert.equal(events[0].type, api.AGENT_ACCESS_AUTH_REQUIRED_EVENT);
    assert.equal(events[0].detail.code, 'maestro_auth_required');

    await global.fetch('/api/v1/ordinary-failure');
    await global.fetch('/api/v1/html-failure');
    await global.fetch('https://other.test/api/v1/protected');
    assert.equal(events.length, 1, 'non-owner 401s, non-JSON errors, and cross-origin requests are ignored');

    await global.fetch('/api/v1/top-level-protected');
    assert.equal(events.length, 2, 'the top-level error shape remains supported');

    await assert.rejects(
      api.loginAgentAccessOwner('one-time-code'),
      error => error.status === 401 && error.message === 'Use a fresh code.',
    );
    assert.equal(events.length, 2, 'owner login calls the saved native fetch and cannot recurse through the interceptor');
    assert.equal(calls.filter(url => url.endsWith('/api/v1/agent-access/owner-login')).length, 1);

    const status = await api.fetchAgentAccessStatus();
    assert.equal(status.available, true, 'same-origin Agent Access API calls remain readable');
    console.log('Agent Access API: startup is quiet, coded 401 signaling is same-origin, and owner login avoids recursion');
  } finally {
    global.window = priorWindow;
    global.fetch = priorFetch;
    if (priorCustomEvent === undefined) delete global.CustomEvent;
    else global.CustomEvent = priorCustomEvent;
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
