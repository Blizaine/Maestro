// Local Agent Access API payload, scope, and config safety regressions.
// Run: node tests/ui/agent_access_local.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');

function resolveEsbuild() {
  const candidate = path.join(root, 'ui/node_modules/esbuild');
  if (!fs.existsSync(candidate)) throw new Error('esbuild is required to run the local Agent Access UI test');
  return require(candidate);
}

async function compile(esbuild, filename, mockedRequire = () => ({})) {
  const built = await esbuild.build({
    stdin: { contents: fs.readFileSync(filename, 'utf8'), resolveDir: path.dirname(filename), sourcefile: filename, loader: filename.endsWith('.tsx') ? 'tsx' : 'ts' },
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
  return {
    ok: status >= 200 && status < 300,
    status,
    clone() { return { json: async () => payload }; },
  };
}

function sameDependencies(left, right) {
  return Boolean(left && right && left.length === right.length && left.every((value, index) => Object.is(value, right[index])));
}

function visibleText(node) {
  if (Array.isArray(node)) return node.map(visibleText).join(' ');
  if (typeof node === 'string' || typeof node === 'number') return String(node);
  if (!node || typeof node !== 'object') return '';
  const children = node.type === 'details' && !node.props?.open
    ? (Array.isArray(node.props?.children) ? node.props.children.find(child => child?.type === 'summary') : null)
    : node.props?.children;
  return visibleText(children);
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
      if (!hooks[index]) hooks[index] = { kind: 'state', value: typeof initial === 'function' ? initial() : initial };
      const slot = hooks[index];
      return [slot.value, value => { slot.value = typeof value === 'function' ? value(slot.value) : value; }];
    },
    useRef(initial) {
      const index = hookCursor++;
      if (!hooks[index]) hooks[index] = { kind: 'ref', value: { current: initial } };
      return hooks[index].value;
    },
    useMemo(factory, dependencies) {
      const index = hookCursor++;
      const slot = hooks[index];
      if (!slot || !sameDependencies(slot.dependencies, dependencies)) hooks[index] = { kind: 'memo', dependencies, value: factory() };
      return hooks[index].value;
    },
    useCallback(callback, dependencies) { return react.useMemo(() => callback, dependencies); },
    useEffect(effect, dependencies) {
      const index = hookCursor++;
      const slot = hooks[index];
      if (!slot || !sameDependencies(slot.dependencies, dependencies)) {
        hooks[index] = { kind: 'effect', dependencies, cleanup: slot?.cleanup };
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
    render(ComponentToRender, nextProps) {
      hookCursor = 0;
      pendingEffects = [];
      const tree = resolve(ComponentToRender(nextProps));
      const effects = pendingEffects;
      for (const effect of effects) effect();
      return tree;
    },
  };
}

function findNode(tree, predicate) {
  return allNodes(tree).find(predicate);
}

function buttonByText(tree, label) {
  return findNode(tree, node => node.type === 'button' && visibleText(node).trim() === label);
}

function buttonsContaining(tree, text) {
  return allNodes(tree).filter(node => node.type === 'button' && visibleText(node).includes(text));
}

function openDetailsWithText(tree, text) {
  const details = findNode(tree, node => node.type === 'details' && visibleText(node).includes(text));
  if (details) details.props.open = true;
  return details;
}

async function main() {
  const esbuild = resolveEsbuild();
  const apiPath = path.join(root, 'ui/src/api/agentAccess.ts');
  const api = await compile(esbuild, apiPath);
  const requests = [];
  const originalFetch = global.fetch;
  const creativeFixture = {
    defaults: { 'audio.speech': { model_type: 'voice-v1', duration_seconds: 0, resolution: 'high' } },
    workflows: [{ id: 'audio.speech', label: 'Speech' }, { id: 'image.generate', label: 'Image generation' }],
    models: [{ model_type: 'voice-v1', name: 'Voice v1', workflows: ['audio.speech'], is_downloaded: true }],
    version: 1,
  };
  global.fetch = async (url, init = {}) => {
    requests.push({ url: String(url), method: init.method || 'GET', body: init.body ? JSON.parse(init.body) : null });
    if (String(url).endsWith('/extension')) {
      return {
        ok: true,
        status: 200,
        headers: { get: () => 'application/octet-stream' },
        blob: async () => new Blob([Buffer.from('mcpb-zip')]),
      };
    }
    if (String(url).endsWith('/local-clients') && init.method === 'POST') {
      const input = JSON.parse(init.body);
      return response({
        id: 'local-client-1234',
        name: input.name,
        scopes: input.scopes,
        command: 'C:\\Program Files\\Maestro\\python.exe',
        args: ['C:\\Users\\A Name\\Maestro\\app\\maestro_mcp.py', '--client', 'local-client-1234'],
        access_token: 'must-not-leak-bearer-token',
        credential_path: 'must-not-leak-private-file-path',
      });
    }
    if (String(url).endsWith('/local-clients')) {
      return response({ clients: [{
        id: 'local-client-saved',
        name: 'Saved local client',
        scopes: ['observe'],
        command: 'C:\\Program Files\\Maestro\\python.exe',
        args: ['C:\\Maestro\\app\\maestro_mcp.py', '--client', 'local-client-saved'],
        access_token: 'must-not-leak-list-token',
        credential_path: 'must-not-leak-list-path',
      }] });
    }
    if (String(url).endsWith('/creative-defaults')) return response(creativeFixture);
    return response({ ok: true });
  };

  try {
    await api.updateAgentAccessSettings({ local_enabled: true });
    await api.updateAgentAccessSettings({ remote_enabled: false });
    assert.deepEqual(requests.slice(0, 2).map(({ method, body }) => ({ method, body })), [
      { method: 'PUT', body: { local_enabled: true } },
      { method: 'PUT', body: { remote_enabled: false } },
    ], 'each transport toggle sends only its own setting with PUT');

    const created = await api.createAgentAccessLocalClient({ name: ' Local Codex ', scopes: ['observe', 'admin', 'observe', 'invalid'] });
    assert.deepEqual(requests[2], {
      url: '/api/v1/agent-access/local-clients',
      method: 'POST',
      body: { name: 'Local Codex', scopes: ['observe', 'admin'] },
    }, 'local client creation sends a trimmed name and the deliberately selected scopes');
    assert.deepEqual(Object.keys(created).sort(), ['args', 'command', 'created_at', 'grant_id', 'id', 'name', 'revoked', 'scopes'].sort(), 'the API exposes only the safe client fields');
    assert.equal(JSON.stringify(created).includes('must-not-leak'), false, 'local client response tokens and credential paths are stripped');
    assert.deepEqual(created.scopes, ['observe', 'admin']);

    const listed = await api.fetchAgentAccessLocalClients();
    assert.equal(listed.length, 1);
    assert.equal(JSON.stringify(listed).includes('must-not-leak'), false, 'list responses strip token and credential path fields');
    assert.deepEqual(listed[0].args, ['C:\\Maestro\\app\\maestro_mcp.py', '--client', 'local-client-saved']);
    const extension = await api.fetchAgentAccessLocalClientExtension('local-client-saved');
    assert.equal(extension.size, 8);
    assert.equal(requests[4].url, '/api/v1/agent-access/local-clients/local-client-saved/extension');
    await assert.rejects(api.fetchAgentAccessLocalClientExtension('../path'), /local client ID is invalid/i);

    const creative = await api.fetchAgentAccessCreativeDefaults();
    assert.deepEqual(creative.workflows.map(item => item.id), ['audio.speech', 'image.generate']);
    assert.equal(creative.defaults['audio.speech'].duration_seconds, 0);
    const savedDefaults = { 'audio.speech': { model_type: 'voice-v1', duration_seconds: 0 } };
    await api.updateAgentAccessCreativeDefaults(savedDefaults);
    assert.deepEqual(requests[6], {
      url: '/api/v1/agent-access/creative-defaults',
      method: 'PUT',
      body: { defaults: savedDefaults },
    }, 'creative defaults change only through an explicit PUT save');

    const componentPath = path.join(root, 'ui/src/components/SettingsDrawer/AgentAccessAdditionalSettings.tsx');
    const fakeIcon = () => null;
    const icons = new Proxy({}, { get: () => fakeIcon });
    const jsx = (type, props, key) => ({ type, props: props || {}, key });
    const renderer = createRenderer(jsx);
    const component = await compile(esbuild, componentPath, name => {
      if (name === 'react') return renderer.react;
      if (name === 'react/jsx-runtime') return { jsx, jsxs: jsx, Fragment: Symbol('fragment') };
      if (name === 'lucide-react') return icons;
      if (name.includes('agentAccess')) return api;
      throw new Error(`Unexpected import: ${name}`);
    });
    const windowsClient = {
      id: 'client-space-1',
      name: 'Windows client',
      scopes: ['observe'],
      command: "C:\\Program Files\\Maestro O'Neil\\python.exe",
      args: ["C:\\Users\\A Name\\Maestro O'Neil\\app\\maestro_mcp.py", '--client', 'client-space-1'],
    };
    const config = JSON.parse(api.formatLocalAgentClientJson({ ...windowsClient, access_token: 'extra-secret-field' }));
    const entry = Object.values(config.mcpServers)[0];
    assert.equal(entry.command, windowsClient.command);
    assert.deepEqual(entry.args, windowsClient.args);
    assert.equal(JSON.stringify(config).includes('extra-secret-field'), false, 'copied JSON contains only command and args, never response secrets');
    const codex = api.formatLocalAgentClientCodexConfig(windowsClient);
    assert.match(codex, /^\[mcp_servers\.maestro-clientspace1\]/, 'Codex receives a unique config.toml section for this client');
    assert.ok(codex.includes(`command = ${JSON.stringify(windowsClient.command)}`));
    assert.ok(codex.includes(`args = [${windowsClient.args.map(value => JSON.stringify(value)).join(', ')}]`));
    assert.equal(codex.includes('extra-secret-field'), false);
    const windowsClaude = api.formatLocalAgentClientClaudeCommand(windowsClient);
    assert.match(windowsClaude, /^& 'claude' 'mcp' 'add'/, 'Windows Claude Code commands use PowerShell call operator before the quoted executable');
    assert.ok(windowsClaude.includes("'C:\\Program Files\\Maestro O''Neil\\python.exe'"), 'PowerShell command quoting preserves spaces and escapes apostrophes');
    assert.ok(windowsClaude.includes("'C:\\Users\\A Name\\Maestro O''Neil\\app\\maestro_mcp.py'"));
    const posixClaude = api.formatLocalAgentClientClaudeCommand({ ...windowsClient, command: '/usr/local/Maestro/bin/python', args: ["/home/Lee's Folder/maestro_mcp.py", '--client', 'posix-client'] });
    assert.match(posixClaude, /^'claude' 'mcp' 'add'/, 'POSIX commands do not include the PowerShell call operator');
    assert.ok(posixClaude.includes("'/home/Lee'\\''s Folder/maestro_mcp.py'"), 'POSIX command quoting preserves spaces and apostrophes');

    const previousWindow = Object.getOwnPropertyDescriptor(global, 'window');
    const previousDocument = Object.getOwnPropertyDescriptor(global, 'document');
    const previousNavigator = Object.getOwnPropertyDescriptor(global, 'navigator');
    const previousCreateObjectUrl = Object.getOwnPropertyDescriptor(URL, 'createObjectURL');
    const previousRevokeObjectUrl = Object.getOwnPropertyDescriptor(URL, 'revokeObjectURL');
    const clipboard = [];
    const downloaded = [];
    Object.defineProperty(global, 'window', { configurable: true, value: { setTimeout: () => 1 } });
    Object.defineProperty(global, 'navigator', { configurable: true, value: { clipboard: { writeText: async value => clipboard.push(value) } } });
    Object.defineProperty(global, 'document', { configurable: true, value: {
      createElement: tag => ({ tag, style: {}, href: '', download: '', click() { downloaded.push(this.download); }, remove() {} }),
      body: { appendChild() {} },
    } });
    Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: () => 'blob:local-extension' });
    Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: () => {} });
    const componentProps = { ownerSession: true, enabled: true, available: true, refreshKey: 1 };
    const settle = async () => { await new Promise(resolve => setImmediate(resolve)); };
    let tree;
    try {
      tree = renderer.render(component.LocalMcpSettings, componentProps);
      await settle();
      tree = renderer.render(component.LocalMcpSettings, componentProps);
      assert.match(visibleText(tree), /Saved local client/, 'an existing local client is available before creating a new one');
      assert.match(visibleText(tree), /existing grant/, 'existing grants are labeled as reusable');
      assert.match(visibleText(tree), /Step 1 · Choose the app/, 'new-client flow starts with target selection');

      openDetailsWithText(tree, 'Saved local client');
      let clientTarget = findNode(tree, node => node.type === 'select' && node.props['aria-label'] === 'Saved local client app setup target');
      assert.ok(clientTarget, 'existing clients can choose a target app without creating a new grant');
      clientTarget.props.onChange({ target: { value: 'chatgpt' } });
      tree = renderer.render(component.LocalMcpSettings, componentProps);
      openDetailsWithText(tree, 'Saved local client');
      assert.match(visibleText(tree), /ChatGPT desktop → Settings → MCP servers → Add server/);
      assert.match(visibleText(tree), /name the server\s+Maestro/i);
      assert.ok(buttonByText(tree, 'Copy launch command'));
      assert.equal(buttonsContaining(tree, 'Copy').filter(button => visibleText(button).includes('launch command')).length, 1, 'ChatGPT has a single relevant primary action');
      assert.equal(buttonByText(tree, 'Copy Codex config section'), undefined, 'the Codex action is not shown for ChatGPT');
      assert.doesNotMatch(visibleText(tree), /mcpServers/, 'raw configuration is hidden while Manual setup details is collapsed');
      await buttonByText(tree, 'Copy launch command').props.onClick();
      await settle();
      tree = renderer.render(component.LocalMcpSettings, componentProps);
      assert.match(clipboard[0], /^"C:\\Program Files\\Maestro\\python\.exe"/);
      assert.doesNotMatch(clipboard[0], /^& /, 'ChatGPT desktop receives a GUI command line without a PowerShell invocation operator');
      assert.match(visibleText(findNode(tree, node => node.props?.role === 'status')), /does not connect/i, 'copying setup is not reported as a successful connection');

      clientTarget = findNode(tree, node => node.type === 'select' && node.props['aria-label'] === 'Saved local client app setup target');
      clientTarget.props.onChange({ target: { value: 'codex' } });
      tree = renderer.render(component.LocalMcpSettings, componentProps);
      openDetailsWithText(tree, 'Saved local client');
      assert.ok(buttonByText(tree, 'Copy Codex config section'));
      assert.equal(buttonByText(tree, 'Copy launch command'), undefined);
      assert.match(visibleText(tree), /%USERPROFILE%\\.codex\\config\.toml/);

      clientTarget = findNode(tree, node => node.type === 'select' && node.props['aria-label'] === 'Saved local client app setup target');
      clientTarget.props.onChange({ target: { value: 'claude-desktop' } });
      tree = renderer.render(component.LocalMcpSettings, componentProps);
      openDetailsWithText(tree, 'Saved local client');
      assert.ok(buttonByText(tree, 'Download Claude Desktop extension'));
      assert.match(visibleText(tree), /Claude Desktop → Settings → Extensions → Install Extension/);
      assert.match(visibleText(tree), /same computer where Maestro runs/);
      await buttonByText(tree, 'Download Claude Desktop extension').props.onClick();
      await settle();
      tree = renderer.render(component.LocalMcpSettings, componentProps);
      openDetailsWithText(tree, 'Saved local client');
      assert.ok(downloaded.some(filename => filename.endsWith('.mcpb')), 'Claude Desktop action downloads the existing client extension');
      assert.ok(requests.some(request => request.url.endsWith('/local-clients/local-client-saved/extension')));
      assert.match(visibleText(findNode(tree, node => node.props?.role === 'status')), /does not install or connect/i);

      clientTarget = findNode(tree, node => node.type === 'select' && node.props['aria-label'] === 'Saved local client app setup target');
      clientTarget.props.onChange({ target: { value: 'claude-code' } });
      tree = renderer.render(component.LocalMcpSettings, componentProps);
      openDetailsWithText(tree, 'Saved local client');
      assert.ok(buttonByText(tree, 'Copy Claude Code command'));
      assert.match(visibleText(tree), /PowerShell on Windows/);
      const manualDetails = findNode(tree, node => node.type === 'details' && visibleText(node).includes('Manual setup details'));
      assert.ok(manualDetails, 'raw config stays inside optional manual details');
      manualDetails.props.open = true;
      assert.match(visibleText(manualDetails), /Claude Code command/);

      clientTarget = findNode(tree, node => node.type === 'select' && node.props['aria-label'] === 'Saved local client app setup target');
      clientTarget.props.onChange({ target: { value: 'other' } });
      tree = renderer.render(component.LocalMcpSettings, componentProps);
      openDetailsWithText(tree, 'Saved local client');
      assert.ok(buttonByText(tree, 'Copy MCP JSON'));
      assert.match(visibleText(tree), /local STDIO MCP configuration under/);

      const createTarget = findNode(tree, node => node.type === 'select' && node.props.id === 'agent-access-local-target');
      createTarget.props.onChange({ target: { value: 'codex' } });
      tree = renderer.render(component.LocalMcpSettings, componentProps);
      const nameInput = findNode(tree, node => node.type === 'input' && node.props.id === 'agent-access-local-client-name');
      assert.equal(nameInput.props.value, 'Codex on this PC', 'the app picker suggests a matching local client name');
      const requestCount = requests.length;
      const createForm = findNode(tree, node => node.type === 'form');
      await createForm.props.onSubmit({ preventDefault() {} });
      await settle();
      tree = renderer.render(component.LocalMcpSettings, componentProps);
      const createRequest = requests.slice(requestCount).find(request => request.method === 'POST' && request.url.endsWith('/local-clients'));
      assert.deepEqual(createRequest.body, { name: 'Codex on this PC', scopes: ['observe'] }, 'client creation grants only the visibly selected default scope');
      assert.match(visibleText(tree), /creation does not connect it/i);
      const createdSetup = findNode(tree, node => node.type === 'details' && node.props.open === true && visibleText(node).includes('Codex on this PC'));
      assert.ok(createdSetup, 'the new client opens directly to its next setup step');
      assert.equal(findNode(createdSetup, node => node.type === 'select' && node.props.autoFocus === true)?.props.value, 'codex', 'the new client app picker receives focus');
      assert.ok(buttonByText(tree, 'Copy Codex config section'), 'the newly created client immediately exposes its app-specific action');
    } finally {
      for (const [key, descriptor] of [['window', previousWindow], ['document', previousDocument], ['navigator', previousNavigator]]) {
        if (descriptor) Object.defineProperty(global, key, descriptor);
        else delete global[key];
      }
      for (const [key, descriptor] of [['createObjectURL', previousCreateObjectUrl], ['revokeObjectURL', previousRevokeObjectUrl]]) {
        if (descriptor) Object.defineProperty(URL, key, descriptor);
        else delete URL[key];
      }
    }

    const source = fs.readFileSync(componentPath, 'utf8');
    assert.match(source, /Choose this client’s access scopes/);
    assert.match(source, /Step 1 · Choose the app/);
    assert.match(source, /Existing local clients/);
    assert.match(source, /Manual setup details/);
    assert.match(source, /ChatGPT desktop → Settings → MCP servers → Add server/);
    assert.match(source, /Download Claude Desktop extension/);
    assert.match(source, /Claude Desktop → Settings → Extensions → Install Extension/);
    assert.match(source, /same computer where Maestro runs/);
    assert.doesNotMatch(source, /Copy ChatGPT STDIO JSON/);
    assert.match(source, /setOpenClientId\(client\.id\)/, 'new clients open directly to their app setup');
    assert.doesNotMatch(source, /Add custom connector/);
    assert.match(source, /flex flex-wrap/);
    assert.match(source, /min-w-0/);
    assert.match(source, /grid-cols-1/);
    assert.match(source, /sm:grid-cols-2/);
    assert.match(source, /const imageWorkflow = id\.startsWith\('image\.'\)/, 'image workflows do not render duration controls');
    assert.match(source, /0 = Auto/);
    assert.match(source, /onClick=\{\(\) => void save\(\)\}/, 'creative defaults have an explicit Save button');
    assert.match(source, /const invalidateSetup = useCallback\(\(\) => \{[\s\S]*?setClients\(\[\]\)[\s\S]*?setClientTargets\(\{\}\)[\s\S]*?setCopied\(null\)[\s\S]*?setDownloadingClientId\(null\)/, 'invalidating local setup clears stale clients, target selection, copy state, and download state');
    assert.match(source, /\[invalidateSetup, refresh, refreshKey\]/, 'a local grant refresh invalidates and reloads the child setup');
    assert.match(source, /if \(generation !== setupGeneration\.current\) return/, 'a download completing after revocation does not publish stale setup');
    assert.match(source, /className="min-w-0 break-all[^\"]*">Form value:/, 'long current default IDs wrap in narrow layouts');
    assert.match(source, /className="min-w-0 break-all[^\"]*">Last saved by Maestro:/, 'long saved default IDs wrap in narrow layouts');
    const remotePanel = fs.readFileSync(path.join(root, 'ui/src/components/SettingsDrawer/AgentAccessSettingsPanel.tsx'), 'utf8');
    assert.match(remotePanel, /grant\.transport === 'local'\) setLocalClientRefreshKey/);
    assert.match(remotePanel, /<LocalMcpSettings[^>]*refreshKey=\{localClientRefreshKey\}/);
    assert.doesNotMatch(remotePanel, /Add custom connector/);
    assert.match(remotePanel, /Do not use this private HTTP URL for Claude Desktop/);
    console.log('Agent Access local setup: independent payloads, scope selection, secret stripping, safe quoting, and compact responsive layout passed');
  } finally {
    global.fetch = originalFetch;
  }
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
