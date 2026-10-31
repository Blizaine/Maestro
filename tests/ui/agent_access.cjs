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

async function exerciseSessionGate(esbuild) {
  const filename = path.join(root, 'ui/src/components/AgentAccessSessionGate.tsx');
  const source = fs.readFileSync(filename, 'utf8');
  const built = await esbuild.build({
    stdin: { contents: source, resolveDir: path.dirname(filename), sourcefile: filename, loader: 'tsx' },
    bundle: false,
    platform: 'node',
    format: 'cjs',
    write: false,
    target: 'node18',
    jsx: 'automatic',
  });

  const hooks = [];
  let hookCursor = 0;
  const react = {
    useState(initial) {
      const index = hookCursor++;
      if (!hooks[index]) hooks[index] = { kind: 'state', value: typeof initial === 'function' ? initial() : initial };
      const slot = hooks[index];
      return [slot.value, value => { slot.value = typeof value === 'function' ? value(slot.value) : value; }];
    },
    useRef(initial) {
      const index = hookCursor++;
      if (!hooks[index]) hooks[index] = { kind: 'ref', value: { current: initial } };
      return hooks[index].value;
    },
    useCallback(callback) {
      hookCursor++;
      return callback;
    },
    useEffect(effect) {
      const index = hookCursor++;
      if (!hooks[index]) hooks[index] = { kind: 'effect', effect, ran: false };
    },
  };
  const jsx = (type, props) => ({ type, props: props || {} });
  class AgentAccessApiError extends Error {
    constructor(status, message) { super(message); this.status = status; }
  }
  let loginMode = 'reject';
  let freshCodeRequests = 0;
  const api = {
    AGENT_ACCESS_AUTH_REQUIRED_CODE: 'maestro_auth_required',
    AGENT_ACCESS_AUTH_REQUIRED_EVENT: 'maestro-agent-access-auth-required',
    AGENT_ACCESS_SESSION_RESTORED_EVENT: 'maestro-agent-access-session-restored',
    AgentAccessApiError,
    fetchAgentAccessStatus: async () => ({
      available: true, enabled: true, owner_configured: true,
      authenticated: false, owner: false,
    }),
    loginAgentAccessOwner: async () => {
      if (loginMode === 'reject') throw new AgentAccessApiError(401, 'Use a fresh code.');
      return { ok: true };
    },
    requestAgentAccessConsoleCode: async () => {
      freshCodeRequests += 1;
      return { message: 'A fresh code was sent to Pinokio.', expires_in: 60 };
    },
  };
  const mockedRequire = name => {
    if (name === 'react') return react;
    if (name === 'react/jsx-runtime') return { jsx, jsxs: jsx, Fragment: Symbol('Fragment') };
    if (name === 'lucide-react') return {
      ExternalLink: () => null, LockKeyhole: () => null, X: () => null,
    };
    if (name.includes('agentAccess')) return api;
    throw new Error(`Unexpected SessionGate import: ${name}`);
  };
  const compiled = { exports: {} };
  new Function('module', 'exports', 'require', built.outputFiles[0].text)(
    compiled, compiled.exports, mockedRequire,
  );

  const oldWindow = global.window;
  const oldCustomEvent = global.CustomEvent;
  const listeners = new Map();
  class TestCustomEvent extends Event {
    constructor(type, init = {}) { super(type); this.detail = init.detail; }
  }
  global.CustomEvent = TestCustomEvent;
  global.window = {
    location: { href: 'https://maestro.test/studio', origin: 'https://maestro.test', protocol: 'https:', hostname: 'maestro.test' },
    addEventListener(type, listener) {
      const set = listeners.get(type) || new Set();
      set.add(listener);
      listeners.set(type, set);
    },
    removeEventListener(type, listener) { listeners.get(type)?.delete(listener); },
    dispatchEvent(event) {
      for (const listener of listeners.get(event.type) || []) listener(event);
      return true;
    },
  };

  const render = () => {
    hookCursor = 0;
    const tree = compiled.exports.AgentAccessSessionGate();
    for (const slot of hooks) {
      if (slot?.kind === 'effect' && !slot.ran) {
        slot.ran = true;
        slot.cleanup = slot.effect();
      }
    }
    return tree;
  };
  const allNodes = (node, result = []) => {
    if (Array.isArray(node)) { for (const child of node) allNodes(child, result); return result; }
    if (!node || typeof node !== 'object') return result;
    result.push(node);
    allNodes(node.props?.children, result);
    return result;
  };
  const visibleText = node => {
    if (Array.isArray(node)) return node.map(visibleText).join(' ');
    if (typeof node === 'string' || typeof node === 'number') return String(node);
    if (!node || typeof node !== 'object') return '';
    return visibleText(node.props?.children);
  };
  const flush = () => new Promise(resolve => setImmediate(resolve));
  const authRequired = async () => {
    global.window.dispatchEvent(new TestCustomEvent(api.AGENT_ACCESS_AUTH_REQUIRED_EVENT, {
      detail: { code: api.AGENT_ACCESS_AUTH_REQUIRED_CODE },
    }));
    await flush();
    return render();
  };

  try {
    render();
    let tree = await authRequired();
    assert.ok(allNodes(tree).some(node => node.props?.role === 'dialog'), 'the first owner 401 opens the sign-in gate');

    let input = allNodes(tree).find(node => node.type === 'input');
    input.props.onChange({ target: { value: 'expired-code' } });
    tree = render();
    const form = allNodes(tree).find(node => node.type === 'form');
    await form.props.onSubmit({ preventDefault() {} });
    tree = render();
    assert.equal(allNodes(tree).find(node => node.props?.role === 'alert')?.props.children, 'Use a fresh code.');

    tree = await authRequired();
    assert.equal(allNodes(tree).find(node => node.props?.role === 'alert')?.props.children, 'Use a fresh code.', 'repeated 401s preserve the current error');
    assert.equal(allNodes(tree).find(node => node.type === 'input')?.props.value, 'expired-code', 'repeated 401s preserve the entered code');

    const dismiss = allNodes(tree).find(node => node.type === 'button' && node.props?.['aria-label'] === 'Dismiss sign-in');
    assert.ok(dismiss, 'the gate offers a dismiss control');
    dismiss.props.onClick();
    tree = render();
    assert.equal(allNodes(tree).some(node => node.props?.role === 'dialog'), false, 'dismissing hides the gate');
    const manualSignIn = allNodes(tree).find(node => node.type === 'button' && /sign in/i.test(visibleText(node)));
    assert.ok(manualSignIn, 'dismissal leaves a manual Sign in button');

    tree = await authRequired();
    assert.equal(allNodes(tree).some(node => node.props?.role === 'dialog'), false, 'dismissal suppresses automatic reopening');
    manualSignIn.props.onClick();
    await flush();
    tree = render();
    assert.ok(allNodes(tree).some(node => node.props?.role === 'dialog'), 'manual Sign in reopens the gate');
    const requestFreshCode = allNodes(tree).find(node =>
      ['button', 'a'].includes(node.type)
      && /fresh.*code|request.*code/i.test(visibleText(node)),
    );
    assert.ok(requestFreshCode, 'the open sign-in flow keeps a fresh-code request available');
    if (requestFreshCode.props.onClick) requestFreshCode.props.onClick();
    else assert.fail('fresh-code request control needs an onClick handler');
    await flush();
    tree = render();
    assert.equal(freshCodeRequests, 1, 'the fresh-code control uses the Agent Access code request');
    assert.equal(allNodes(tree).find(node => node.type === 'input')?.props.value, '', 'requesting a new code clears a stale input');
    assert.match(allNodes(tree).find(node => node.props?.role === 'status')?.props.children || '', /Pinokio/i);
    assert.match(allNodes(tree).find(node => node.props?.role === 'status')?.props.children || '', /1 minute\b/, 'code instructions use the server expiry instead of assuming ten minutes');
    console.log('Agent Access SessionGate: repeated 401 state, dismissal, manual sign-in, and fresh-code request passed');
  } finally {
    for (const slot of hooks) if (slot?.kind === 'effect' && slot.cleanup) slot.cleanup();
    global.window = oldWindow;
    if (oldCustomEvent === undefined) delete global.CustomEvent;
    else global.CustomEvent = oldCustomEvent;
  }
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
  const requests = [];
  const events = [];
  let ownerLoginMode = 'reject';
  let localChallengeAllowed = true;
  let localProof = '';
  let statusHttpStatus = 200;
  let statusPayload = { available: true, enabled: false };
  const nativeFetch = async (input, init = {}) => {
    const url = new URL(input instanceof Request ? input.url : String(input), 'https://maestro.test/');
    calls.push(url.href);
    const method = String(init.method || (input instanceof Request ? input.method : 'GET')).toUpperCase();
    requests.push({ url: url.href, method });
    if (url.pathname === '/api/v1/agent-access/local-challenge') {
      localProof = JSON.parse(init.body).proof;
      assert.match(localProof, /^[0-9a-f]{64}$/, 'approval supplies a cryptographically generated request proof');
      return localChallengeAllowed ? response({ nonce: 'local-nonce-fixture', expires_in: 120 }) : response({ detail: 'Local browser required' }, 403);
    }
    if (url.pathname === '/api/v1/agent-access/local-approve') {
      assert.deepEqual(JSON.parse(init.body), { nonce: 'local-nonce-fixture', proof: localProof, approve: true });
      return response({ ok: true });
    }
    if (url.pathname === '/api/v1/agent-access/status') {
      if (statusHttpStatus === 401) {
        return response({ code: 'maestro_auth_required', message: 'Owner session expired.' }, 401);
      }
      return response(statusPayload, statusHttpStatus);
    }
    if (url.pathname === '/api/v1/agent-access/owner-login') {
      if (ownerLoginMode === 'reject') {
        return response({ code: 'maestro_auth_required', message: 'Use a fresh code.' }, 401);
      }
      return response({ ok: true });
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

    ownerLoginMode = 'success';
    statusPayload = { available: true, enabled: true, authenticated: true, owner: true };
    const beforeVerifiedLogin = requests.length;
    await api.loginAgentAccessOwner('valid-owner-code');
    assert.deepEqual(
      requests.slice(beforeVerifiedLogin).map(({ url, method }) => [new URL(url).pathname, method]),
      [['/api/v1/agent-access/owner-login', 'POST'], ['/api/v1/agent-access/status', 'GET']],
      'a successful code POST is followed by a fresh status read',
    );
    assert.equal(events.length, 2, 'post-login verification stays on the saved native fetch');

    statusPayload = { available: true, enabled: true, authenticated: true, owner: false };
    const beforeUnpersistedLogin = requests.length;
    await assert.rejects(
      api.loginAgentAccessOwner('code-did-not-persist'),
      error => /fresh code|browser session|persist/i.test(error.message),
      'a successful POST without a persisted owner session returns an actionable error',
    );
    assert.deepEqual(
      requests.slice(beforeUnpersistedLogin).map(({ url, method }) => [new URL(url).pathname, method]),
      [['/api/v1/agent-access/owner-login', 'POST'], ['/api/v1/agent-access/status', 'GET']],
    );
    assert.equal(events.length, 2, 'a failed post-login check does not recurse into the auth interceptor');

    statusPayload = { available: true, enabled: true, authenticated: true, owner: true };
    statusHttpStatus = 401;
    const beforeStatusError = requests.length;
    await assert.rejects(api.loginAgentAccessOwner('status-check-401'));
    assert.deepEqual(
      requests.slice(beforeStatusError).map(({ url, method }) => [new URL(url).pathname, method]),
      [['/api/v1/agent-access/owner-login', 'POST'], ['/api/v1/agent-access/status', 'GET']],
      'even a failed status refresh is preserved as the login error',
    );
    assert.equal(events.length, 2, 'status verification 401 does not generate recursive login events');

    statusHttpStatus = 200;
    const beforeLocal = requests.length;
    await api.approveAgentAccessLocal();
    assert.deepEqual(requests.slice(beforeLocal).map(({ url, method }) => [new URL(url).pathname, method]), [
      ['/api/v1/agent-access/local-challenge', 'POST'],
      ['/api/v1/agent-access/local-approve', 'POST'],
      ['/api/v1/agent-access/status', 'GET'],
    ], 'explicit local approval obtains a challenge, consumes its matching proof, and verifies local access');
    assert.equal(events.length, 2);
    localChallengeAllowed = false;
    const beforeRemoteDenied = requests.length;
    await assert.rejects(api.approveAgentAccessLocal(), error => error.status === 403);
    assert.equal(requests.length, beforeRemoteDenied + 1, 'a rejected challenge never attempts owner approval');
    localChallengeAllowed = true;
    statusPayload = { ...statusPayload, authenticated: false, owner: false };
    await assert.rejects(api.approveAgentAccessLocal(), error => error.code === 'maestro_owner_session_missing' && /could not be verified/.test(error.message));
    assert.equal(events.length, 2, 'a missing local session cannot recursively open sign-in dialogs');
    const status = await api.fetchAgentAccessStatus();
    assert.equal(status.available, true, 'same-origin Agent Access API calls remain readable');
    await exerciseSessionGate(esbuild);
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
