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
      return response({
        id: 'local-client-1234',
        name: 'Local Codex',
        scopes: ['observe', 'admin'],
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
    const component = await compile(esbuild, componentPath, name => {
      if (name === 'react') return {};
      if (name === 'react/jsx-runtime') return { jsx: fakeIcon, jsxs: fakeIcon, Fragment: Symbol('fragment') };
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
    assert.ok(windowsClaude.includes("'C:\\Program Files\\Maestro O''Neil\\python.exe'"), 'PowerShell command quoting preserves spaces and escapes apostrophes');
    assert.ok(windowsClaude.includes("'C:\\Users\\A Name\\Maestro O''Neil\\app\\maestro_mcp.py'"));
    const posixClaude = api.formatLocalAgentClientClaudeCommand({ ...windowsClient, command: '/usr/local/Maestro/bin/python', args: ["/home/Lee's Folder/maestro_mcp.py", '--client', 'posix-client'] });
    assert.ok(posixClaude.includes("'/home/Lee'\\''s Folder/maestro_mcp.py'"), 'POSIX command quoting preserves spaces and apostrophes');

    const source = fs.readFileSync(componentPath, 'utf8');
    assert.match(source, /Choose this client’s access scopes/);
    assert.match(source, /Copy Codex config\.toml/);
    assert.match(source, /Copy ChatGPT STDIO JSON/);
    assert.match(source, /Download Claude Desktop extension/);
    assert.match(source, /Settings → Extensions → Install Extension/);
    assert.match(source, /same computer where this Maestro instance runs/);
    assert.doesNotMatch(source, /Add custom connector/);
    assert.match(source, /flex flex-wrap/);
    assert.match(source, /min-w-0/);
    assert.match(source, /grid grid-cols-1 gap-2 sm:grid-cols-2/);
    assert.match(source, /const imageWorkflow = id\.startsWith\('image\.'\)/, 'image workflows do not render duration controls');
    assert.match(source, /0 = Auto/);
    assert.match(source, /onClick=\{\(\) => void save\(\)\}/, 'creative defaults have an explicit Save button');
    assert.match(source, /const invalidateSetup = useCallback\(\(\) => \{[\s\S]*?setClients\(\[\]\)[\s\S]*?setCopied\(null\)[\s\S]*?setDownloadingClientId\(null\)/, 'invalidating local setup clears stale clients, copy state, and download state');
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
