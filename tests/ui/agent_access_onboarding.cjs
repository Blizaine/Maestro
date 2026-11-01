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
    local_enabled: false,
    remote_enabled: false,
    owner_configured: false,
    authenticated: false,
    owner: false,
    mcp_url: 'http://untrusted.example/mcp',
    lan_mcp_urls: ['https://maestro.test/mcp'],
    tailscale: {
      enabled: false,
      installed: false,
      connected: false,
      hostname: null,
      certificate_ready: false,
      ready: false,
      mcp_url: null,
      error: null,
      expires_at: null,
    },
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
  let tailscalePollStatus = null;
  let tailscalePostError = null;
  const calls = { consoleCode: 0, localApproval: 0, login: [], settings: [], copied: [], grants: 0, audit: 0, statusFetches: 0, tailscaleReads: 0, tailscaleUpdates: [] };
  const api = {
    AGENT_ACCESS_AUTH_REQUIRED_EVENT: 'maestro-agent-access-auth-required',
    AGENT_ACCESS_SESSION_RESTORED_EVENT: 'maestro-agent-access-session-restored',
    AgentAccessApiError: class AgentAccessApiError extends Error {
      constructor(statusCode, message) { super(message); this.status = statusCode; }
    },
    fetchAgentAccessStatus: async () => {
      calls.statusFetches += 1;
      if (tailscalePollStatus) {
        status.tailscale = { ...tailscalePollStatus };
        status.preferred_mcp_url = tailscalePollStatus.mcp_url;
        tailscalePollStatus = null;
      }
      return { ...status, tailscale: { ...status.tailscale } };
    },
    fetchAgentAccessTailscaleStatus: async () => {
      calls.tailscaleReads += 1;
      return { ...status.tailscale };
    },
    fetchAgentAccessGrants: async () => { calls.grants += 1; return grants; },
    fetchAgentAccessAudit: async () => { calls.audit += 1; return []; },
    requestAgentAccessConsoleCode: async () => { calls.consoleCode += 1; return { expires_in: 120 }; },
    approveAgentAccessLocal: async () => {
      calls.localApproval += 1;
      Object.assign(status, { owner_configured: true, authenticated: true, owner: true, auth_method: 'local_device' });
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
      if (Object.hasOwn(settings, 'remote_enabled')) status.enabled = settings.remote_enabled;
      return { ok: true };
    },
    updateAgentAccessTailscaleSettings: async settings => {
      calls.tailscaleUpdates.push(settings);
      if (tailscalePostError) {
        const error = tailscalePostError;
        tailscalePostError = null;
        throw error;
      }
      if (settings.enabled) {
        status.tailscale = {
          enabled: true,
          installed: true,
          connected: true,
          hostname: 'desktop-abc.tailnet.ts.net',
          certificate_ready: false,
          ready: false,
          mcp_url: null,
          error: null,
          expires_at: null,
        };
        tailscalePollStatus = {
          enabled: true,
          installed: true,
          connected: true,
          hostname: 'desktop-abc.tailnet.ts.net',
          certificate_ready: true,
          ready: true,
          mcp_url: 'https://desktop-abc.tailnet.ts.net:8765/mcp',
          error: null,
          expires_at: '2027-10-31T12:00:00Z',
        };
      } else {
        status.tailscale = { ...status.tailscale, enabled: false, ready: false };
      }
      return { ...status.tailscale };
    },
    revokeAgentAccessGrant: async () => ({ ok: true }),
    createAgentAccessBrowserCode: async () => ({ code: 'PAIR-1234', expires_in: 60 }),
  };
  const mockedRequire = name => {
    if (name === 'react') return react;
    if (name === 'react/jsx-runtime') return { jsx, jsxs: jsx, Fragment: Symbol('Fragment') };
    if (name === 'lucide-react') return icons;
    if (name.endsWith('AgentAccessAdditionalSettings')) return { AgentGenerationDefaults: () => null, LocalMcpSettings: () => null };
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
    assert.match(visibleText(steps[0]), /Sign in to manage access/);
    assert.match(visibleText(tree), /Choose MCP access/);
    assert.match(visibleText(tree), /Local MCP · recommended on this computer/);
    assert.match(visibleText(tree), /Remote MCP · other computers/);
    assert.match(visibleText(tree), /Set up MCP clients/);
    assert.match(visibleText(tree), /Agent connections stay off until you allow them/);
    assert.match(visibleText(tree), /remote browsers and agents still need their own authorization/);
    assert.equal(buttonByText(tree, 'Copy ChatGPT desktop setup'), undefined, 'client setup waits for sign-in and explicit enablement');
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
    assert.equal(calls.tailscaleReads, 1, 'the first owner session performs one fresh Tailscale detection');
    assert.equal(calls.consoleCode, 0);
    assert.match(visibleText(tree), /Remote MCP is off/);
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
    assert.equal(calls.tailscaleReads, 2, 'a new owner session performs one fresh detection after sign-in');
    assert.match(visibleText(tree), /Example client/);
    assert.match(visibleText(tree), /Revoke/);
    assert.match(visibleText(tree), /Remote MCP is off/);
    assert.match(visibleText(tree), /Sign in to manage access/);
    assert.match(visibleText(tree), /Complete/);
    assert.match(visibleText(tree), /Current step/);
    assert.equal(buttonByText(tree, 'Copy Codex setup'), undefined, 'paused access keeps client setup as the next step');

    const localSwitch = allNodes(tree).find(node => node.props?.role === 'switch' && node.props['aria-label'] === 'Enable Local MCP');
    assert.ok(localSwitch, 'Local MCP has an independent owner-controlled switch');
    assert.equal(localSwitch.props['aria-checked'], false);
    localSwitch.props.onClick();
    await flush();
    tree = render();
    assert.deepEqual(calls.settings, [{ local_enabled: true }], 'enabling Local MCP sends only its own setting');
    assert.equal(status.local_enabled, true);
    assert.equal(status.remote_enabled, false);
    assert.match(visibleText(tree), /Local MCP only/);
    assert.equal(buttonByText(tree, 'Copy Codex setup'), undefined, 'Local MCP does not reveal remote URL setup');
    assert.equal(calls.tailscaleUpdates.length, 0, 'enabling Local MCP does not configure Tailscale');

    const remoteSwitch = allNodes(tree).find(node => node.props?.role === 'switch' && node.props['aria-label'] === 'Enable Remote MCP');
    assert.ok(remoteSwitch, 'Remote MCP has an independent switch');
    remoteSwitch.props.onClick();
    await flush();
    tree = render();
    assert.deepEqual(calls.settings, [{ local_enabled: true }, { remote_enabled: true }], 'enabling Remote MCP preserves Local MCP and sends only its own setting');
    assert.equal(status.local_enabled, true);
    assert.equal(status.remote_enabled, true);
    assert.match(visibleText(tree), /Local and Remote MCP on/);
    assert.equal(calls.tailscaleReads, 2, 'cached Agent Access refresh does not repeat Tailscale detection');
    assert.ok(buttonByText(tree, 'Copy Codex setup'), 'enabled access reveals client setup choices');
    assert.ok(buttonByText(tree, 'Copy Claude Code setup'));
    assert.ok(buttonByText(tree, 'Copy ChatGPT desktop setup'));
    assert.ok(buttonByText(tree, 'Copy server URL for another MCP client'));
    assert.ok(buttonByText(tree, 'Copy setup prompt for my agent'));
    assert.match(visibleText(tree), /Automatic setup depends on your agent’s available tools/);
    assert.match(visibleText(tree), /Certificate trust is a separate first-time step/);
    assert.match(visibleText(tree), /SHA-256 fingerprint in Connection details/);

    const localEnabledSwitch = allNodes(tree).find(node => node.props?.role === 'switch' && node.props['aria-label'] === 'Local MCP is enabled');
    localEnabledSwitch.props.onClick();
    await flush();
    tree = render();
    assert.deepEqual(calls.settings, [{ local_enabled: true }, { remote_enabled: true }, { local_enabled: false }], 'turning Local MCP off leaves Remote MCP on');
    assert.equal(status.local_enabled, false);
    assert.equal(status.remote_enabled, true);
    assert.match(visibleText(tree), /Remote MCP only/);

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
    assert.ok(calls.copied[0].includes('First inspect which MCP setup tools and client capabilities are actually available'));
    assert.ok(calls.copied[0].includes('If you do not have a server-registration tool, do not stop at a refusal'));
    assert.ok(calls.copied[0].includes('Settings > MCP servers > Add server > enter a server name > select Streamable HTTP'));
    assert.ok(calls.copied[0].includes('Do not assume a cloud agent can reach a private LAN URL'));
    assert.ok(calls.copied[0].includes('obtain the exact public TLS certificate'));
    assert.ok(calls.copied[0].includes('tell me the specific blocker'));
    assert.ok(calls.copied[0].includes('Never disable TLS verification'));
    assert.ok(calls.copied[0].includes('normal OAuth approval flow'));
    assert.ok(calls.copied[0].includes('wait for my approval there'));
    assert.ok(calls.copied[0].includes('read-only server status or help request'));
    assert.ok(calls.copied[0].includes('Do not render or create images, video, audio'));
    assert.ok(calls.copied[0].includes('Do not invent deep links, commands, certificate screens, or capabilities'));
    assert.ok(!calls.copied[0].includes('already-issued-code'));
    assert.ok(!calls.copied[0].includes('current-code'));
    assert.match(visibleText(tree), /Setup prompt copied/);
    assert.match(visibleText(tree), /Paste it into chat with your agent/);
    assert.match(visibleText(tree), /does not (register, )?connect or approve the client/);

    const settingsBeforeChatGPTCopy = calls.settings.length;
    buttonByText(tree, 'Copy ChatGPT desktop setup').props.onClick();
    await flush();
    tree = render();
    assert.equal(calls.copied.length, 2);
    assert.ok(calls.copied[1].includes('ChatGPT desktop MCP setup for Maestro (manual steps)'));
    assert.ok(calls.copied[1].includes('Server URL: https://maestro.test/mcp'));
    assert.ok(calls.copied[1].includes('Expected public TLS certificate SHA-256 fingerprint: ab12cd34'));
    assert.ok(calls.copied[1].includes('Settings > MCP servers > Add server'));
    assert.ok(calls.copied[1].includes('Enter a distinct server name of your choice'));
    assert.ok(calls.copied[1].includes('select Streamable HTTP'));
    assert.ok(calls.copied[1].includes('select Save'));
    assert.ok(calls.copied[1].includes('Restart ChatGPT, then select Authenticate'));
    assert.ok(calls.copied[1].includes('verify that its SHA-256 exactly matches the fingerprint above'));
    assert.ok(calls.copied[1].includes('Connection details > Download public TLS certificate'));
    assert.ok(calls.copied[1].includes('browser trust alone does not prove the MCP client will accept it'));
    assert.ok(calls.copied[1].includes('Being on the same Wi-Fi does not mean a cloud-hosted agent can reach a private LAN URL'));
    assert.ok(calls.copied[1].includes('review the requested scopes and approve only the access you intend'));
    assert.ok(calls.copied[1].includes('read-only server status or help request only'));
    assert.ok(calls.copied[1].includes('Do not render or create images, video, audio'));
    assert.ok(!calls.copied[1].includes('Authorization:'));
    assert.ok(!calls.copied[1].includes('already-issued-code'));
    assert.match(visibleText(tree), /does not configure, connect, or approve the server/);
    assert.equal(calls.settings.length, settingsBeforeChatGPTCopy, 'copying ChatGPT instructions never changes server settings or enables connections');

    buttonByText(tree, 'Copy Codex setup').props.onClick();
    await flush();
    tree = render();
    assert.equal(calls.copied.length, 3);
    assert.ok(calls.copied[2].includes('https://maestro.test/mcp'));
    assert.ok(calls.copied[2].includes('TLS certificate SHA-256: ab12cd34'));
    assert.ok(!calls.copied[2].includes('untrusted.example'));

    buttonByText(tree, 'Copy server URL for another MCP client').props.onClick();
    await flush();
    assert.equal(calls.copied[3], 'https://maestro.test/mcp', 'other-client copy stays a plain validated URL');

    buttonByText(tree, 'Sign out').props.onClick();
    await flush();
    tree = render();
    assert.match(visibleText(tree), /Remote MCP · other computers:  On/);
    assert.match(visibleText(tree), /Sign in to manage either transport/);
    assert.doesNotMatch(visibleText(tree), /Agent connections remain off until an owner explicitly allows them/);
    assert.equal(buttonByText(tree, 'Copy ChatGPT desktop setup'), undefined, 'signing out gates client setup instructions');
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
    assert.match(visibleText(tree), /This PC is approved for local management/);
    assert.equal(buttonByText(tree, 'Sign out'), undefined, 'local device authority is not presented as a cookie session');

    const originalFingerprint = status.certificate_sha256;
    status.certificate_sha256 = '';
    global.window.dispatchEvent(new Event(api.AGENT_ACCESS_SESSION_RESTORED_EVENT));
    await flush();
    tree = render();
    assert.equal(buttonByText(tree, 'Copy ChatGPT desktop setup').props.disabled, true, 'ChatGPT setup requires a reported certificate fingerprint');
    assert.equal(buttonByText(tree, 'Copy setup prompt for my agent').props.disabled, true, 'generic setup prompt also requires a reported certificate fingerprint');
    status.certificate_sha256 = originalFingerprint;
    global.window.dispatchEvent(new Event(api.AGENT_ACCESS_SESSION_RESTORED_EVENT));
    await flush();
    tree = render();

    assert.match(visibleText(tree), /Tailscale is not installed on this computer/);
    assert.match(visibleText(tree), /Settings → Notifications → Private phone access/);
    assert.ok(buttonByText(tree, 'Refresh / detect'), 'owner can refresh or detect Tailscale state without an install action');
    assert.equal(buttonByText(tree, 'Use Tailscale'), undefined, 'Tailscale cannot be enabled before installation and sign-in are detected');

    const readsBeforeManualDetect = calls.tailscaleReads;
    status.tailscale = { ...status.tailscale, installed: true, connected: false };
    buttonByText(tree, 'Refresh / detect').props.onClick();
    await flush();
    tree = render();
    assert.equal(calls.tailscaleReads, readsBeforeManualDetect + 1, 'refresh uses the owner-only fresh Tailscale detection endpoint');
    assert.match(visibleText(tree), /Open Tailscale, sign this computer into your network/);
    assert.ok(buttonByText(tree, 'Refresh / detect'));
    assert.equal(buttonByText(tree, 'Use Tailscale'), undefined, 'an installed but disconnected Tailscale client cannot provision HTTPS');

    status.tailscale = { ...status.tailscale, connected: true };
    buttonByText(tree, 'Refresh / detect').props.onClick();
    await flush();
    tree = render();
    assert.equal(calls.tailscaleReads, readsBeforeManualDetect + 2);
    assert.ok(buttonByText(tree, 'Use Tailscale'));
    assert.match(visibleText(tree), /Select Use Tailscale to provision the trusted HTTPS certificate/);

    buttonByText(tree, 'Use Tailscale').props.onClick();
    await flush();
    tree = render();
    assert.deepEqual(calls.tailscaleUpdates, [{ enabled: true }], 'enabling Tailscale is one explicit provisioning request');
    assert.match(visibleText(tree), /Tailscale HTTPS is provisioning/);
    assert.doesNotMatch(visibleText(tree), /Trusted Tailscale HTTPS transport is ready/);
    assert.ok(visibleText(tree).includes('https://maestro.test/mcp'));
    assert.ok(!visibleText(tree).includes('https://desktop-abc.tailnet.ts.net:8765/mcp'), 'hostname or enabled state alone does not make a Tailscale URL preferred');

    await new Promise(resolve => setTimeout(resolve, 1100));
    await flush();
    tree = render();
    assert.equal(calls.statusFetches > 0, true, 'provisioning polls the cache-only Agent Access status endpoint');
    assert.match(visibleText(tree), /Trusted Tailscale HTTPS transport is ready/);
    const addressSelectReady = allNodes(tree).find(node => node.type === 'select' && node.props['aria-label'] === 'MCP connection address');
    assert.equal(addressSelectReady?.props.value, 'https://desktop-abc.tailnet.ts.net:8765/mcp', 'ready Tailscale MCP URL is the default choice');
    assert.match(visibleText(tree), /The agent computer must be signed into the same Tailscale network/);

    const tailscaleCopyStart = calls.copied.length;
    buttonByText(tree, 'Copy setup prompt for my agent').props.onClick();
    await flush();
    tree = render();
    const tailscaleAgentCopy = calls.copied[tailscaleCopyStart];
    assert.ok(tailscaleAgentCopy.includes('MCP server URL: https://desktop-abc.tailnet.ts.net:8765/mcp'));
    assert.ok(tailscaleAgentCopy.includes('same Tailscale network'));
    assert.ok(tailscaleAgentCopy.includes('trusted Tailscale HTTPS'));
    assert.ok(!tailscaleAgentCopy.includes('Expected public TLS certificate SHA-256 fingerprint'));
    assert.ok(!tailscaleAgentCopy.includes('Download public TLS certificate'));
    assert.ok(!tailscaleAgentCopy.includes('verify that its SHA-256 exactly matches'));
    assert.ok(!tailscaleAgentCopy.includes('Authorization:'));

    buttonByText(tree, 'Copy ChatGPT desktop setup').props.onClick();
    await flush();
    tree = render();
    const tailscaleChatGptCopy = calls.copied[tailscaleCopyStart + 1];
    assert.ok(tailscaleChatGptCopy.includes('Server URL: https://desktop-abc.tailnet.ts.net:8765/mcp'));
    assert.ok(tailscaleChatGptCopy.includes('computer running ChatGPT must be signed into the same Tailscale network'));
    assert.ok(!tailscaleChatGptCopy.includes('Expected public TLS certificate SHA-256 fingerprint'));
    assert.ok(!tailscaleChatGptCopy.includes('Download public TLS certificate'));
    assert.ok(!tailscaleChatGptCopy.includes('fingerprint above'));
    assert.ok(tailscaleChatGptCopy.includes('Do not claim the server has been added, connected, authenticated, or approved'));

    buttonByText(tree, 'Copy Codex setup').props.onClick();
    await flush();
    tree = render();
    const tailscaleCodexCopy = calls.copied[tailscaleCopyStart + 2];
    assert.ok(tailscaleCodexCopy.includes('url = "https://desktop-abc.tailnet.ts.net:8765/mcp"'));
    assert.ok(tailscaleCodexCopy.includes('trusted Tailscale HTTPS'));
    assert.ok(!tailscaleCodexCopy.includes('TLS certificate SHA-256'));
    assert.ok(!tailscaleCodexCopy.includes('Download public TLS certificate'));

    buttonByText(tree, 'Copy Claude Code setup').props.onClick();
    await flush();
    tree = render();
    const tailscaleClaudeCopy = calls.copied[tailscaleCopyStart + 3];
    assert.ok(tailscaleClaudeCopy.includes('Claude Code MCP setup for Maestro'));
    assert.ok(tailscaleClaudeCopy.includes('computer running Claude Code, signed into the same Tailscale network as Maestro'));
    assert.ok(tailscaleClaudeCopy.includes('claude mcp add --transport http maestro "https://desktop-abc.tailnet.ts.net:8765/mcp"'));
    assert.ok(tailscaleClaudeCopy.includes('In Claude Code, run /mcp to complete OAuth authentication'));
    assert.ok(tailscaleClaudeCopy.includes('read-only status or help request'));
    assert.ok(tailscaleClaudeCopy.includes('Anthropic says those requests come from its cloud'));
    assert.ok(tailscaleClaudeCopy.includes('private LAN or Tailscale URL'));
    assert.ok(!tailscaleClaudeCopy.includes('then paste this server URL'));
    assert.ok(!tailscaleClaudeCopy.includes('TLS certificate SHA-256'));
    assert.ok(!tailscaleClaudeCopy.includes('Download public TLS certificate'));
    assert.ok(!tailscaleClaudeCopy.includes('Authorization:'));
    assert.ok(tailscaleClaudeCopy.includes('No bearer token is included'));

    const addressSelect = allNodes(tree).find(node => node.type === 'select' && node.props['aria-label'] === 'MCP connection address');
    addressSelect.props.onChange({ target: { value: 'https://maestro.test/mcp' } });
    tree = render();
    assert.equal(allNodes(tree).find(node => node.type === 'select' && node.props['aria-label'] === 'MCP connection address').props.value, 'https://maestro.test/mcp', 'an explicit LAN selection overrides the Tailscale default');
    const lanCopyIndex = calls.copied.length;
    buttonByText(tree, 'Copy setup prompt for my agent').props.onClick();
    await flush();
    tree = render();
    const lanAgentCopy = calls.copied[lanCopyIndex];
    assert.ok(lanAgentCopy.includes('MCP server URL: https://maestro.test/mcp'));
    assert.ok(lanAgentCopy.includes('Expected public TLS certificate SHA-256 fingerprint: ab12cd34'));
    assert.ok(lanAgentCopy.includes('Download public TLS certificate'));
    assert.ok(lanAgentCopy.includes('verify that its SHA-256 exactly matches the fingerprint above'));
    buttonByText(tree, 'Copy Claude Code setup').props.onClick();
    await flush();
    tree = render();
    const lanClaudeCopy = calls.copied[lanCopyIndex + 1];
    assert.ok(lanClaudeCopy.includes('claude mcp add --transport http maestro "https://maestro.test/mcp"'));
    assert.ok(lanClaudeCopy.includes('Expected public TLS certificate SHA-256 fingerprint: ab12cd34'));
    assert.ok(lanClaudeCopy.includes('verify that its SHA-256 exactly matches the fingerprint above'));
    assert.ok(!lanClaudeCopy.includes('Authorization:'));
    assert.equal(calls.tailscaleUpdates.length, 1, 'copying setup never toggles Tailscale');

    buttonByText(tree, 'Disable Tailscale').props.onClick();
    await flush();
    tree = render();
    assert.deepEqual(calls.tailscaleUpdates, [{ enabled: true }, { enabled: false }]);
    assert.match(visibleText(tree), /Tailscale HTTPS transport was disabled/);
    assert.ok(visibleText(tree).includes('https://maestro.test/mcp'), 'the explicit LAN choice remains selected when Tailscale is disabled');
    assert.ok(!visibleText(tree).includes('https://desktop-abc.tailnet.ts.net:8765/mcp'));

    const writesBeforeAmbiguousFailure = calls.tailscaleUpdates.length;
    const statusReadsBeforeAmbiguousFailure = calls.statusFetches;
    tailscalePostError = new Error('Tailscale request timed out');
    buttonByText(tree, 'Use Tailscale').props.onClick();
    await flush();
    tree = render();
    assert.equal(calls.tailscaleUpdates.length, writesBeforeAmbiguousFailure + 1, 'an ambiguous POST is not retried automatically');
    assert.equal(calls.statusFetches, statusReadsBeforeAmbiguousFailure + 1, 'the UI reconciles an ambiguous POST with the cache-only status endpoint');
    assert.match(visibleText(tree), /Tailscale request timed out/);
    assert.match(visibleText(tree), /Current Tailscale status was refreshed after the request failed/);

    const enabledRemoteSwitch = allNodes(tree).find(node => node.props?.role === 'switch' && node.props['aria-label'] === 'Remote MCP is enabled');
    enabledRemoteSwitch.props.onClick();
    await flush();
    tree = render();
    assert.deepEqual(calls.settings, [{ local_enabled: true }, { remote_enabled: true }, { local_enabled: false }, { remote_enabled: false }], 'turning Remote MCP off preserves the current Local MCP value');
    assert.equal(status.local_enabled, false);
    assert.equal(status.remote_enabled, false);
    assert.match(visibleText(tree), /Off/);
    assert.equal(calls.tailscaleUpdates.length, 3, 'disabling Remote MCP does not silently change the separate Tailscale setting');

    console.log('Agent Access onboarding: owner recovery, independent local/remote toggles, Tailscale readiness, URL selection, certificate trust, and client copy passed');
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
