// Guided Agent Access onboarding UI regressions.
// Run: node tests/ui/agent_access_onboarding.cjs
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
  if (!candidate) throw new Error('esbuild is required to run the Agent Access onboarding UI test');
  return require(candidate);
}

function sameDependencies(left, right) {
  return Boolean(left && right && left.length === right.length && left.every((value, index) => Object.is(value, right[index])));
}

function visibleText(node) {
  if (Array.isArray(node)) return node.map(visibleText).join(' ');
  if (typeof node === 'string' || typeof node === 'number') return String(node);
  if (!node || typeof node !== 'object') return '';
  return visibleText(node.props?.children);
}

function allNodes(node, result = []) {
  if (Array.isArray(node)) {
    for (const child of node) allNodes(child, result);
    return result;
  }
  if (!node || typeof node !== 'object') return result;
  result.push(node);
  allNodes(node.props?.children, result);
  return result;
}

function resolveComponents(node) {
  if (Array.isArray(node)) return node.map(resolveComponents);
  if (!node || typeof node !== 'object') return node;
  if (typeof node.type === 'function') return resolveComponents(node.type(node.props || {}));
  return { ...node, props: { ...node.props, children: resolveComponents(node.props?.children) } };
}

async function main() {
  const filename = path.join(root, 'ui/src/components/SettingsDrawer/AgentAccessSettingsPanel.tsx');
  const esbuild = resolveEsbuild();
  const built = await esbuild.build({
    stdin: { contents: fs.readFileSync(filename, 'utf8'), resolveDir: path.dirname(filename), sourcefile: filename, loader: 'tsx' },
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
    useMemo(factory, dependencies) {
      const index = hookCursor++;
      const slot = hooks[index];
      if (!slot || !sameDependencies(slot.dependencies, dependencies)) {
        hooks[index] = { kind: 'memo', dependencies, value: factory() };
      }
      return hooks[index].value;
    },
    useCallback(callback, dependencies) {
      return react.useMemo(() => callback, dependencies);
    },
    useEffect(effect, dependencies) {
      const index = hookCursor++;
      const slot = hooks[index];
      if (!slot || !sameDependencies(slot.dependencies, dependencies)) {
        hooks[index] = { kind: 'effect', effect, dependencies, cleanup: slot?.cleanup, pending: true };
      }
    },
  };
  const jsx = (type, props, key) => ({ type, props: props || {}, key });
  const icon = () => null;
  const icons = new Proxy({}, { get: () => icon });
  const status = {
    available: true,
    enabled: false,
    owner_configured: false,
    authenticated: false,
    owner: false,
    mcp_url: 'http://untrusted.example/mcp',
    lan_mcp_urls: ['https://maestro.test/mcp'],
    controller_url: 'javascript:alert(1)',
    local_setup_url: 'javascript:alert(1)',
    certificate_sha256: 'ab12cd34',
    installation_id: 'install-private-id',
    controller_status: 'ready',
    discovery_status: 'online',
    backend_online: true,
  };
  const grants = [{
    grant_id: 'grant-1',
    client_id: 'client-private-id',
    client_name: 'Example client',
    scopes: ['read'],
    created_at: '2026-10-31T12:00:00Z',
  }];
  const calls = { consoleCode: 0, localApproval: 0, login: [], settings: [], copied: [], grants: 0, audit: 0 };
  const api = {
    AGENT_ACCESS_AUTH_REQUIRED_EVENT: 'maestro-agent-access-auth-required',
    AGENT_ACCESS_SESSION_RESTORED_EVENT: 'maestro-agent-access-session-restored',
    AgentAccessApiError: class AgentAccessApiError extends Error {
      constructor(statusCode, message) { super(message); this.status = statusCode; }
    },
    fetchAgentAccessStatus: async () => ({ ...status }),
    fetchAgentAccessGrants: async () => { calls.grants += 1; return grants; },
    fetchAgentAccessAudit: async () => { calls.audit += 1; return []; },
    requestAgentAccessConsoleCode: async () => { calls.consoleCode += 1; return { expires_in: 120 }; },
    approveAgentAccessLocal: async () => {
      calls.localApproval += 1;
      Object.assign(status, { owner_configured: true, authenticated: true, owner: true });
      return { ok: true };
    },
    loginAgentAccessOwner: async code => {
      calls.login.push(code);
      Object.assign(status, { owner_configured: true, authenticated: true, owner: true });
      return { ok: true };
    },
    logoutAgentAccessOwner: async () => {
      Object.assign(status, { authenticated: false, owner: false });
      return { ok: true };
    },
    updateAgentAccessSettings: async settings => {
      calls.settings.push(settings);
      Object.assign(status, settings);
      return { ok: true };
    },
    revokeAgentAccessGrant: async () => ({ ok: true }),
    createAgentAccessBrowserCode: async () => ({ code: 'PAIR-1234', expires_in: 60 }),
  };
  const mockedRequire = name => {
    if (name === 'react') return react;
    if (name === 'react/jsx-runtime') return { jsx, jsxs: jsx, Fragment: Symbol('Fragment') };
    if (name === 'lucide-react') return icons;
    if (name.includes('agentAccess')) return api;
    throw new Error('Unexpected panel import: ' + name);
  };
  const compiled = { exports: {} };
  new Function('module', 'exports', 'require', built.outputFiles[0].text)(compiled, compiled.exports, mockedRequire);

  const oldWindow = global.window;
  const oldNavigator = Object.getOwnPropertyDescriptor(global, 'navigator');
  const listeners = new Map();
  global.window = {
    location: { href: 'https://maestro.test/settings', protocol: 'https:', hostname: 'maestro.test' },
    addEventListener(type, listener) {
      const set = listeners.get(type) || new Set();
      set.add(listener);
      listeners.set(type, set);
    },
    removeEventListener(type, listener) { listeners.get(type)?.delete(listener); },
    dispatchEvent(event) { for (const listener of listeners.get(event.type) || []) listener(event); return true; },
    setInterval: () => 1,
    clearInterval: () => {},
    setTimeout: () => 1,
  };
  Object.defineProperty(global, 'navigator', {
    configurable: true,
    value: { clipboard: { writeText: async value => { calls.copied.push(value); } } },
  });

  const render = () => {
    hookCursor = 0;
    const tree = resolveComponents(compiled.exports.AgentAccessSettingsPanel());
    for (const slot of hooks) {
      if (slot?.kind === 'effect' && slot.pending) {
        slot.pending = false;
        if (typeof slot.cleanup === 'function') slot.cleanup();
        slot.cleanup = slot.effect();
      }
    }
    return tree;
  };
  const flush = () => new Promise(resolve => setImmediate(resolve));
  const buttonByText = (tree, text) => allNodes(tree).find(node => node.type === 'button' && visibleText(node).includes(text));

  try {
    let tree = render();
    await flush();
    tree = render();
    const steps = allNodes(tree).filter(node => node.props?.['aria-current'] === 'step');
    assert.equal(steps.length, 1, 'one step is clearly marked as current');
    assert.match(visibleText(steps[0]), /Sign in to this browser once/);
    assert.match(visibleText(tree), /Allow agent connections/);
    assert.match(visibleText(tree), /Add an MCP client/);
    assert.match(visibleText(tree), /Before first setup/);
    assert.match(visibleText(tree), /other browsers/);
    assert.match(visibleText(tree), /turning agents off later keeps browser sign-in protection in place/);
    assert.equal(calls.settings.length, 0, 'fresh setup does not enable agent connections');
    assert.equal(buttonByText(tree, 'Copy Codex setup'), undefined, 'client setup waits for sign-in and explicit enablement');

    const requestCode = buttonByText(tree, 'Request Browser sign-in code');
    assert.ok(requestCode, 'fresh setup offers the Browser sign-in code action');
    let codeInput = allNodes(tree).find(node => node.type === 'input' && node.props.id === 'agent-access-owner-code');
    assert.ok(codeInput, 'a user who already has a Browser sign-in code can paste it without requesting another');
    assert.match(visibleText(tree), /Already have a Browser sign-in code/);
    codeInput.props.onChange({ target: { value: 'already-issued-code' } });
    tree = render();
    allNodes(tree).find(node => node.type === 'form').props.onSubmit({ preventDefault() {} });
    await flush();
    tree = render();
    assert.deepEqual(calls.login, ['already-issued-code'], 'an existing code can sign in without a new request');
    assert.equal(calls.consoleCode, 0);
    assert.match(visibleText(tree), /Agent connections are paused/);
    buttonByText(tree, 'Sign out').props.onClick();
    await flush();
    tree = render();

    const freshCodeRequest = buttonByText(tree, 'Request Browser sign-in code');
    assert.ok(freshCodeRequest, 'signed-out owner can request a fresh Browser sign-in code');
    freshCodeRequest.props.onClick();
    await flush();
    tree = render();
    assert.equal(calls.consoleCode, 1);
    assert.match(visibleText(tree), /Pinokio > this Maestro app > Agent Access Terminal/);
    assert.match(visibleText(tree), /Browser code:/);
    assert.match(visibleText(tree), /newest code/);
    assert.match(visibleText(tree), /single-use/);
    codeInput = allNodes(tree).find(node => node.type === 'input' && node.props.id === 'agent-access-owner-code');
    assert.ok(codeInput, 'requesting a code reveals the paste field');
    codeInput.props.onChange({ target: { value: 'stale-code' } });
    tree = render();
    buttonByText(tree, 'Request a new Browser sign-in code').props.onClick();
    await flush();
    tree = render();
    codeInput = allNodes(tree).find(node => node.type === 'input' && node.props.id === 'agent-access-owner-code');
    assert.equal(codeInput.props.value, '', 'requesting a newer code clears stale input');
    codeInput.props.onChange({ target: { value: 'current-code' } });
    tree = render();
    const ownerForm = allNodes(tree).find(node => node.type === 'form');
    ownerForm.props.onSubmit({ preventDefault() {} });
    await flush();
    tree = render();
    assert.deepEqual(calls.login, ['already-issued-code', 'current-code'], 'the pasted Browser sign-in code is submitted');
    assert.equal(calls.settings.length, 0, 'sign-in alone does not turn agent connections on');
    assert.ok(calls.grants > 0, 'agent grants load immediately after sign-in');
    assert.match(visibleText(tree), /Example client/);
    assert.match(visibleText(tree), /Revoke/);
    assert.match(visibleText(tree), /Agent connections are paused/);
    assert.match(visibleText(tree), /Sign in to this browser once/);
    assert.match(visibleText(tree), /Complete/);
    assert.match(visibleText(tree), /Current step/);
    assert.equal(buttonByText(tree, 'Copy Codex setup'), undefined, 'paused access keeps client setup as the next step');

    const connectionSwitch = allNodes(tree).find(node => node.props?.role === 'switch');
    assert.ok(connectionSwitch);
    assert.equal(connectionSwitch.props['aria-checked'], false);
    connectionSwitch.props.onClick();
    await flush();
    tree = render();
    assert.deepEqual(calls.settings, [{ enabled: true }], 'only the explicit switch enables agent connections');
    assert.ok(buttonByText(tree, 'Copy Codex setup'), 'enabled access reveals client setup choices');
    assert.ok(buttonByText(tree, 'Copy Claude setup'));
    assert.ok(buttonByText(tree, 'Copy server URL for another MCP client'));
    assert.ok(buttonByText(tree, 'Copy setup prompt for my agent'));
    assert.match(visibleText(tree), /trust Maestro’s TLS certificate/);
    assert.match(visibleText(tree), /SHA-256 fingerprint in Connection details/);

    const connectionDetails = allNodes(tree).find(node =>
      node.type === 'details'
      && visibleText(node.props?.children?.[0] ?? node).includes('Connection details'),
    );
    assert.ok(connectionDetails, 'technical details have a disclosure');
    assert.equal(connectionDetails.props.open, undefined, 'technical details are collapsed by default');
    assert.ok(allNodes(tree).every(node => !(node.type === 'a' && /^javascript:/i.test(node.props.href || ''))), 'unsafe controller and setup URLs are never linked');

    buttonByText(tree, 'Copy setup prompt for my agent').props.onClick();
    await flush();
    tree = render();
    assert.equal(calls.copied.length, 1);
    assert.ok(calls.copied[0].includes('MCP server URL: https://maestro.test/mcp'));
    assert.ok(calls.copied[0].includes('Expected public TLS certificate SHA-256 fingerprint: ab12cd34'));
    assert.ok(calls.copied[0].includes('Preserve the client’s existing configuration'));
    assert.ok(calls.copied[0].includes('Never disable TLS verification'));
    assert.ok(calls.copied[0].includes('normal OAuth approval flow'));
    assert.ok(calls.copied[0].includes('wait for my approval there'));
    assert.ok(calls.copied[0].includes('read-only server status or help request'));
    assert.ok(calls.copied[0].includes('Do not render or create images, video, audio'));
    assert.ok(calls.copied[0].includes('Do not try an unsupported install command'));
    assert.ok(!calls.copied[0].includes('already-issued-code'));
    assert.ok(!calls.copied[0].includes('current-code'));
    assert.match(visibleText(tree), /Setup prompt copied/);
    assert.match(visibleText(tree), /Paste it into chat with your agent/);
    assert.match(visibleText(tree), /does not connect or approve the client/);

    buttonByText(tree, 'Copy Codex setup').props.onClick();
    await flush();
    tree = render();
    assert.equal(calls.copied.length, 2);
    assert.ok(calls.copied[1].includes('https://maestro.test/mcp'));
    assert.ok(calls.copied[1].includes('TLS certificate SHA-256: ab12cd34'));
    assert.ok(!calls.copied[1].includes('untrusted.example'));

    buttonByText(tree, 'Copy server URL for another MCP client').props.onClick();
    await flush();
    assert.equal(calls.copied[2], 'https://maestro.test/mcp', 'other-client copy stays a plain validated URL');

    buttonByText(tree, 'Sign out').props.onClick();
    await flush();
    tree = render();
    assert.match(visibleText(tree), /Agent connections are allowed. Sign in to manage this setting/);
    assert.doesNotMatch(visibleText(tree), /Agent connections remain off until an owner explicitly allows them/);
    assert.equal(buttonByText(tree, 'Approve on this PC'), undefined, 'remote status never offers local approval');
    status.local_approval_available = true;
    global.window.dispatchEvent(new Event(api.AGENT_ACCESS_SESSION_RESTORED_EVENT));
    await flush();
    tree = render();
    assert.ok(buttonByText(tree, 'Approve on this PC'), 'server-confirmed local browser has a direct approval action');
    assert.equal(calls.localApproval, 0, 'rendering does not approve the browser');
    const beforeLocalCodeRequests = calls.consoleCode;
    const beforeLocalSettings = calls.settings.length;
    buttonByText(tree, 'Approve on this PC').props.onClick();
    await flush();
    tree = render();
    assert.equal(calls.localApproval, 1);
    assert.equal(calls.consoleCode, beforeLocalCodeRequests, 'local approval never requests a terminal code');
    assert.equal(calls.settings.length, beforeLocalSettings, 'local approval does not alter agent settings');
    assert.match(visibleText(tree), /This browser is signed in/);
    console.log('Agent Access onboarding: guided states, one-time code recovery, explicit enablement, disclosures, URL validation, and client copy passed');
  } finally {
    for (const slot of hooks) if (slot?.kind === 'effect' && typeof slot.cleanup === 'function') slot.cleanup();
    global.window = oldWindow;
    if (oldNavigator) Object.defineProperty(global, 'navigator', oldNavigator);
    else delete global.navigator;
  }
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
