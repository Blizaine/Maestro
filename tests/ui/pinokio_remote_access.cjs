// Pinokio Remote UI, endpoint, and trusted MCP URL regressions.
// Run: node tests/ui/pinokio_remote_access.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));

async function compile(filename, mockedRequire = () => ({})) {
  const built = await esbuild.build({
    stdin: {
      contents: fs.readFileSync(filename, 'utf8'),
      resolveDir: path.dirname(filename),
      sourcefile: filename,
      loader: filename.endsWith('.tsx') ? 'tsx' : 'ts',
    },
    bundle: false,
    platform: 'node',
    format: 'cjs',
    write: false,
    target: 'node18',
    jsx: 'automatic',
  });
  const module = { exports: {} };
  new Function('module', 'exports', 'require', built.outputFiles[0].text)(module, module.exports, mockedRequire);
  return module.exports;
}

function response(payload, status = 200) {
  return { ok: status >= 200 && status < 300, status, clone() { return { json: async () => payload }; } };
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

function createRenderer(jsx) {
  const hooks = [];
  let hookCursor = 0;
  let pendingEffects = [];
  const react = {
    useState(initial) {
      const index = hookCursor++;
      if (!hooks[index]) hooks[index] = { value: typeof initial === 'function' ? initial() : initial };
      const slot = hooks[index];
      return [slot.value, value => { slot.value = typeof value === 'function' ? value(slot.value) : value; }];
    },
    useMemo(factory, dependencies) {
      const index = hookCursor++;
      const slot = hooks[index];
      if (!slot || !sameDependencies(slot.dependencies, dependencies)) hooks[index] = { dependencies, value: factory() };
      return hooks[index].value;
    },
    useCallback(callback, dependencies) { return react.useMemo(() => callback, dependencies); },
    useEffect(effect, dependencies) {
      const index = hookCursor++;
      const slot = hooks[index];
      if (!slot || !sameDependencies(slot.dependencies, dependencies)) {
        hooks[index] = { dependencies };
        pendingEffects.push(effect);
      }
    },
  };
  const resolve = node => {
    if (Array.isArray(node)) return node.map(resolve);
    if (!node || typeof node !== 'object') return node;
    if (typeof node.type === 'function') return resolve(node.type(node.props || {}));
    return { ...node, props: { ...node.props, children: resolve(node.props?.children) } };
  };
  return {
    react,
    render(component, props) {
      hookCursor = 0;
      pendingEffects = [];
      const tree = resolve(component(props));
      const effects = pendingEffects;
      for (const effect of effects) effect();
      return tree;
    },
  };
}

function findNode(tree, predicate) {
  return allNodes(tree).find(predicate);
}

async function settle() {
  await new Promise(resolve => setImmediate(resolve));
  await new Promise(resolve => setImmediate(resolve));
}

async function testApi() {
  const api = await compile(path.join(root, 'ui/src/api/agentAccess.ts'));
  const originalFetch = global.fetch;
  const requests = [];
  const detected = {
    available: true,
    phase: 'ready',
    browser_url: 'https://pinokio.tail78df8a.ts.net:43000/',
    node_hostname: 'maestro-host',
    dns_name: 'pinokio.tail78df8a.ts.net',
    app_id: 'maestro-app',
    reason: null,
  };
  const managed = {
    ...detected,
    enabled: true,
    ready: true,
    mcp_url: 'https://pinokio.tail78df8a.ts.net:43000/mcp',
  };
  try {
    global.fetch = async (url, init = {}) => {
      requests.push({ url: String(url), method: init.method || 'GET', body: init.body ? JSON.parse(init.body) : null, headers: init.headers, credentials: init.credentials });
      if (String(url).endsWith('/remote-access/pinokio/status')) return response({ ...detected, owner_secret: 'must-not-reach-the-UI' });
      if (String(url).endsWith('/agent-access/pinokio')) return response({ ...managed, access_token: 'must-not-reach-the-UI' });
      if (String(url).endsWith('/agent-access/status')) return response({
        available: true,
        enabled: true,
        owner_configured: true,
        authenticated: true,
        owner: true,
        pinokio_remote: managed,
      });
      return response({ detail: 'not found' }, 404);
    };

    const detectedResult = await api.fetchPinokioRemoteAccessStatus();
    assert.deepEqual(detectedResult, detected, 'public detection reads only the typed browser status fields');
    assert.doesNotMatch(JSON.stringify(detectedResult), /must-not-reach-the-UI/);
    assert.deepEqual(requests[0], {
      url: '/api/v1/remote-access/pinokio/status', method: 'GET', body: null, headers: undefined, credentials: 'omit',
    });
    const managedResult = await api.updateAgentAccessPinokioRemoteSettings({ enabled: true });
    assert.equal(managedResult.mcp_url, managed.mcp_url);
    assert.doesNotMatch(JSON.stringify(managedResult), /must-not-reach-the-UI/);
    assert.deepEqual(requests[1], {
      url: '/api/v1/agent-access/pinokio',
      method: 'POST',
      body: { enabled: true },
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
    }, 'MCP access changes are explicit JSON POSTs');
    const agentStatus = await api.fetchAgentAccessStatus();
    assert.equal(agentStatus.pinokio_remote.ready, true, 'Agent Access status validates and exposes the managed Pinokio Remote status');

    global.fetch = async () => response({ detail: 'missing on older backend' }, 404);
    assert.equal(await api.fetchPinokioRemoteAccessStatus(), null, 'old backends gracefully report no Pinokio Remote detection endpoint');
    await assert.rejects(api.updateAgentAccessPinokioRemoteSettings({ enabled: true }), error => error.status === 404);

    global.fetch = async () => response({ available: true, phase: 'ready' });
    await assert.rejects(api.fetchPinokioRemoteAccessStatus(), error => error.status === 502, 'malformed JSON payloads are rejected');
  } finally {
    global.fetch = originalFetch;
  }
}

async function testComponent() {
  const jsx = (type, props, key) => ({ type, props: props || {}, key });
  const renderer = createRenderer(jsx);
  const icons = new Proxy({}, { get: () => () => null });
  const detected = {
    available: true,
    phase: 'ready',
    browser_url: 'https://pinokio.tail78df8a.ts.net:43000/',
    node_hostname: null,
    dns_name: 'pinokio.tail78df8a.ts.net',
    app_id: 'maestro-app',
    reason: null,
  };
  const notReady = { ...detected, phase: 'starting' };
  let publicStatus = detected;
  let failPublicStatus = false;
  const updates = [];
  const managedOff = { ...detected, enabled: false, ready: false, mcp_url: null };
  const managedReady = { ...detected, enabled: true, ready: true, mcp_url: 'https://pinokio.tail78df8a.ts.net:43000/mcp' };
  const api = {
    AgentAccessApiError: class AgentAccessApiError extends Error { constructor(status, message) { super(message); this.status = status; } },
    fetchPinokioRemoteAccessStatus: async () => {
      if (failPublicStatus) throw new Error('status refresh failed');
      return publicStatus;
    },
    updateAgentAccessPinokioRemoteSettings: async settings => {
      updates.push(settings);
      return { ...detected, enabled: settings.enabled, ready: false, mcp_url: null };
    },
  };
  const component = await compile(path.join(root, 'ui/src/components/SettingsDrawer/PinokioRemoteAccess.tsx'), name => {
    if (name === 'react') return renderer.react;
    if (name === 'react/jsx-runtime') return { jsx, jsxs: jsx, Fragment: Symbol('fragment') };
    if (name === 'lucide-react') return icons;
    if (name === 'qrcode.react') return { QRCodeSVG: props => jsx('qr-code', props) };
    if (name.includes('agentAccess')) return api;
    throw new Error(`Unexpected import: ${name}`);
  });

  const previousWindow = Object.getOwnPropertyDescriptor(global, 'window');
  Object.defineProperty(global, 'window', { configurable: true, value: { setTimeout: callback => { callback(); return 1; } } });
  try {
    let currentManaged = managedOff;
    const props = {
      ownerSession: true,
      remoteMcpEnabled: true,
      managedStatus: currentManaged,
      onManagedStatusChange: status => { currentManaged = status; },
      refreshOwnerStatus: async () => {
        currentManaged = managedReady;
        return { available: true, enabled: true, owner_configured: true, authenticated: true, owner: true, pinokio_remote: managedReady };
      },
    };
    const pause = () => new Promise(resolve => setImmediate(resolve));
    let tree = renderer.render(component.PinokioRemoteAccess, props);
    await settle();
    tree = renderer.render(component.PinokioRemoteAccess, props);
    const link = findNode(tree, node => node.type === 'a' && node.props.target === '_blank');
    assert.equal(link?.props.href, detected.browser_url, 'browser access uses the exact backend-published URL');
    assert.equal(findNode(tree, node => node.type === 'qr-code')?.props.value, detected.browser_url, 'QR uses the exact published browser URL');
    const toggle = findNode(tree, node => node.type === 'button' && node.props.role === 'switch');
    assert.equal(toggle?.props['aria-label'], 'Use Pinokio Remote for MCP');
    assert.equal(toggle?.props['aria-checked'], false, 'Pinokio MCP remains opt-in by default');
    await toggle.props.onClick();
    await pause();
    await pause();
    assert.deepEqual(updates, [{ enabled: true }], 'the explicit MCP opt-in posts enabled=true');
    assert.equal(currentManaged.enabled, true);
    assert.equal(currentManaged.ready, true, 'the owner status refresh, not the POST enabled flag, confirms readiness');
    tree = renderer.render(component.PinokioRemoteAccess, { ...props, managedStatus: currentManaged });
    assert.match(visibleText(tree), /MCP URL:\s+https:\/\/pinokio\.tail78df8a\.ts\.net:43000\/mcp/);
    assert.match(visibleText(tree), /leave Maestro and Agent Access running/i);

    publicStatus = notReady;
    const noReadyRenderer = createRenderer(jsx);
    const noReadyComponent = await compile(path.join(root, 'ui/src/components/SettingsDrawer/PinokioRemoteAccess.tsx'), name => {
      if (name === 'react') return noReadyRenderer.react;
      if (name === 'react/jsx-runtime') return { jsx, jsxs: jsx, Fragment: Symbol('fragment') };
      if (name === 'lucide-react') return icons;
      if (name === 'qrcode.react') return { QRCodeSVG: props => jsx('qr-code', props) };
      if (name.includes('agentAccess')) return api;
      throw new Error(`Unexpected import: ${name}`);
    });
    tree = noReadyRenderer.render(noReadyComponent.PinokioRemoteAccess, { ...props, managedStatus: managedOff });
    await settle();
    tree = noReadyRenderer.render(noReadyComponent.PinokioRemoteAccess, { ...props, managedStatus: managedOff });
    assert.equal(findNode(tree, node => node.type === 'qr-code'), undefined, 'a starting controller does not expose an unready browser URL as ready');
    assert.equal(findNode(tree, node => node.type === 'a' && node.props.target === '_blank'), undefined);

    failPublicStatus = true;
    const failedRenderer = createRenderer(jsx);
    const failedComponent = await compile(path.join(root, 'ui/src/components/SettingsDrawer/PinokioRemoteAccess.tsx'), name => {
      if (name === 'react') return failedRenderer.react;
      if (name === 'react/jsx-runtime') return { jsx, jsxs: jsx, Fragment: Symbol('fragment') };
      if (name === 'lucide-react') return icons;
      if (name === 'qrcode.react') return { QRCodeSVG: props => jsx('qr-code', props) };
      if (name.includes('agentAccess')) return api;
      throw new Error(`Unexpected import: ${name}`);
    });
    tree = failedRenderer.render(failedComponent.PinokioRemoteAccess, { ...props, managedStatus: managedReady });
    await settle();
    tree = failedRenderer.render(failedComponent.PinokioRemoteAccess, { ...props, managedStatus: managedReady });
    assert.equal(findNode(tree, node => node.type === 'qr-code'), undefined, 'a failed status refresh clears the stale ready QR code');
    assert.equal(findNode(tree, node => node.type === 'a' && node.props.target === '_blank'), undefined, 'a failed status refresh hides the stale browser link');
    assert.doesNotMatch(visibleText(tree), /pinokio\.tail78df8a\.ts\.net/, 'a failed refresh does not render the old published URL from cached owner status');
    assert.match(visibleText(tree), /status refresh failed/);
  } finally {
    if (previousWindow) Object.defineProperty(global, 'window', previousWindow);
    else delete global.window;
  }
}

async function main() {
  await testApi();
  await testComponent();

  const agentPanel = fs.readFileSync(path.join(root, 'ui/src/components/SettingsDrawer/AgentAccessSettingsPanel.tsx'), 'utf8');
  const notificationPanel = fs.readFileSync(path.join(root, 'ui/src/components/SettingsDrawer/NotificationSettingsPanel.tsx'), 'utf8');
  assert.match(agentPanel, /\[pinokioRemoteMcpUrl, tailscaleMcpUrl, usablePreferredMcpUrl/);
  assert.match(agentPanel, /Trusted Pinokio Remote HTTPS/);
  assert.match(agentPanel, /hosted cloud connectors are not automatically authorized by that network/);
  assert.match(notificationPanel, /Standalone Tailscale setup \(fallback\)/);
  assert.match(notificationPanel, /This PC does not need a second Tailscale install/);
  assert.match(notificationPanel, /fetchPinokioRemoteAccessStatus/);
  console.log('Pinokio Remote status, trusted MCP opt-in, published browser URL, and notification fallback passed');
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
