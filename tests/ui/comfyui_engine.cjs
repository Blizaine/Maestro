// Run: node tests/ui/comfyui_engine.cjs
// Mocks the API and browser. No ComfyUI server, model files, GPU, or generation is used.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
let chromium = null;
let playwrightUnavailable = null;
try { ({chromium} = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright')); }
catch (error) { playwrightUnavailable = error; }

async function compileApi() {
  const filename = path.join(root, 'ui/src/api/comfyui.ts');
  const built = await esbuild.build({
    stdin: {contents: fs.readFileSync(filename, 'utf8'), resolveDir: path.dirname(filename), sourcefile: filename, loader: 'ts'},
    bundle: false,
    platform: 'node',
    format: 'cjs',
    write: false,
    target: 'node18',
    logLevel: 'silent',
  });
  const module = {exports: {}};
  new Function('module', 'exports', 'require', built.outputFiles[0].text)(module, module.exports, require);
  return module.exports;
}

function response(payload, status = 200) {
  return {ok: status >= 200 && status < 300, status, json: async () => payload};
}

async function checkRouteApiPayload() {
  const api = await compileApi();
  const requests = [];
  const originalFetch = global.fetch;
  global.fetch = async (url, init = {}) => {
    requests.push({url: String(url), method: init.method || 'GET', body: init.body ? JSON.parse(init.body) : null});
    return response({workflow_id: 'image-template', enabled: false, kind: 'image', name: 'Checkpoint Image'});
  };
  try {
    const route = {
      workflow_id: 'image-template',
      enabled: false,
      field_bindings: {prompt: 'positive_prompt', num_inference_steps: 'steps'},
      input_defaults: {checkpoint: 'model.safetensors', include_audio: false, count: 24},
      kind: 'image',
    };
    await api.saveComfyUIRoute('photo/model', route);
    assert.deepEqual(requests, [{
      url: '/api/v1/comfyui/routes/photo%2Fmodel',
      method: 'PUT',
      body: route,
    }], 'The typed route API sends mappings and scalar defaults to the model-specific route endpoint');
  } finally { global.fetch = originalFetch; }
}

const workflowFixtures = [
  {
    id: 'image-template', name: 'Checkpoint Image', kind: 'image', workflow: {},
    bindings: [
      {name: 'prompt', label: 'Positive prompt', node_id: '1', input: 'text', type: 'string', required: true},
      {name: 'steps', label: 'Steps', node_id: '2', input: 'steps', type: 'integer', required: true, min: 1, max: 80},
      {name: 'include_audio', label: 'Include audio', node_id: '3', input: 'enabled', type: 'boolean', required: true, default: false},
      {name: 'checkpoint', label: 'Checkpoint', node_id: '4', input: 'ckpt_name', type: 'choice', required: true, options: ['model-a.safetensors', 'model-b.safetensors']},
      {name: 'style', label: 'Style', node_id: '5', input: 'text', type: 'string', required: true},
      {name: 'start_image', label: 'Start image', node_id: '6', input: 'image', type: 'image'},
    ],
  },
  {id: 'video-template', name: 'Video only', kind: 'video', workflow: {}, bindings: []},
];

async function createBrowserHarness() {
  // Agent worktrees live several directories below the repository root. Vite's
  // esbuild optimizer probes ancestors while resolving CommonJS packages, and
  // that traversal can cross the sandbox boundary. Use the repository's
  // shallow UI dependency tree when available while serving the exact source
  // under test from this worktree.
  const shallowUiRoot = path.resolve(root, '..', '..', '..', 'ui');
  const uiRoot = fs.existsSync(path.join(shallowUiRoot, 'node_modules', 'react', 'package.json'))
    ? shallowUiRoot
    : path.join(root, 'ui');
  const componentPath = path.join(root, 'ui/src/components/Sidebar/ComfyUIEngineControl.tsx')
    .replaceAll('\\', '/');
  const componentUrl = componentPath.startsWith(uiRoot.replaceAll('\\', '/'))
    ? `/${path.relative(uiRoot, componentPath).replaceAll('\\', '/')}`
    : `/@fs/${componentPath}`;
  const directory = fs.mkdtempSync(path.join(uiRoot, 'comfyui-engine-browser-test-'));
  const name = path.basename(directory);
  const entry = [
    "import React from 'react';",
    "import {createRoot} from 'react-dom/client';",
    `import {ComfyUIEngineControl} from '${componentUrl}';`,
    'window.mountComfyUIEngine = (modelType, kind) => createRoot(document.getElementById("root")).render(',
    '  React.createElement(ComfyUIEngineControl, {modelType, kind}),',
    ');',
    'window.mountComfyUIEngine("photo-model", "image");',
  ].join('\n');
  fs.writeFileSync(path.join(directory, 'index.html'), '<!doctype html><html><body><div id="root"></div><script type="module" src="./engine.tsx"></script></body></html>');
  fs.writeFileSync(path.join(directory, 'engine.tsx'), entry);
  const {createServer} = require(path.join(uiRoot, 'node_modules/vite'));
  const server = await createServer({
    root: uiRoot,
    configFile: false,
    appType: 'spa',
    logLevel: 'error',
    // The harness has no package entry HTML for Vite's scanner to discover.
    // Optimize the app's CJS dependencies explicitly so React exports proper
    // browser ESM bindings during the test.
    optimizeDeps: {noDiscovery: true, include: ['react', 'react/jsx-runtime', 'react/jsx-dev-runtime', 'react-dom/client', 'lucide-react', 'zustand']},
    resolve: {
      alias: [
        {find: /^react$/, replacement: path.join(uiRoot, 'node_modules/react/index.js')},
        {find: /^react\/jsx-runtime$/, replacement: path.join(uiRoot, 'node_modules/react/jsx-runtime.js')},
        {find: /^react\/jsx-dev-runtime$/, replacement: path.join(uiRoot, 'node_modules/react/jsx-dev-runtime.js')},
        {find: /^react-dom\/client$/, replacement: path.join(uiRoot, 'node_modules/react-dom/client.js')},
        {find: /^lucide-react$/, replacement: path.join(uiRoot, 'node_modules/lucide-react/dist/esm/lucide-react.js')},
        {find: /^zustand$/, replacement: path.join(uiRoot, 'node_modules/zustand/esm/index.mjs')},
      ],
    },
    server: {host: '127.0.0.1', port: 0},
  });
  try {
    await server.listen();
    const address = server.httpServer.address();
    return {
      origin: `http://127.0.0.1:${address.port}`,
      pagePath: `/${name}/index.html`,
      async close() {
        await server.close();
        fs.unlinkSync(path.join(directory, 'engine.tsx'));
        fs.unlinkSync(path.join(directory, 'index.html'));
        fs.rmdirSync(directory);
      },
    };
  } catch (error) {
    await server.close();
    fs.unlinkSync(path.join(directory, 'engine.tsx'));
    fs.unlinkSync(path.join(directory, 'index.html'));
    fs.rmdirSync(directory);
    throw error;
  }
}

async function checkComponentSyntax() {
  const filename = path.join(root, 'ui/src/components/Sidebar/ComfyUIEngineControl.tsx');
  const transformed = await esbuild.transform(fs.readFileSync(filename, 'utf8'), {
    loader: 'tsx',
    jsx: 'automatic',
    target: 'es2020',
  });
  assert.ok(transformed.code.includes('Open workflow library'), 'The engine control exposes the workflow library');
  assert.ok(transformed.code.includes('unsupported options are rejected'), 'The engine control explains unsupported native options');
  assert.ok(transformed.code.includes('Duration (seconds)'), 'Audio routes can map Native duration_seconds');
  assert.ok(transformed.code.includes('Second image reference'), 'Image/video routes expose a second ordered reference slot');
}

function jsonResponse(route, value, status = 200) {
  return route.fulfill({status, contentType: 'application/json', body: JSON.stringify(value)});
}

async function checkConfigurationFlow(browser, harness) {
  const state = {requests: [], validations: [], routeSaves: [], route: null};
  const page = await browser.newPage({viewport: {width: 430, height: 900}});
  page.setDefaultTimeout(7000);
  const pageErrors = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') pageErrors.push(message.text()); });
  await page.route(`${harness.origin}/**/*`, async route => {
    const request = route.request();
    const url = new URL(request.url());
    const pathName = url.pathname;
    if (!pathName.startsWith('/api/v1/comfyui/')) return route.continue();
    const body = request.postDataJSON?.() ?? null;
    state.requests.push({path: pathName, method: request.method(), body});
    if (pathName === '/api/v1/comfyui/status' && request.method() === 'GET') {
      return jsonResponse(route, {enabled: true, endpoint: 'http://127.0.0.1:8188', connected: false,
        routes: state.route ? {'photo-model': state.route} : {}});
    }
    if (pathName === '/api/v1/comfyui/workflows' && request.method() === 'GET') {
      return jsonResponse(route, {workflows: workflowFixtures, templates: []});
    }
    if (pathName === '/api/v1/comfyui/workflows/image-template/validate' && request.method() === 'POST') {
      state.validations.push(body);
      return jsonResponse(route, {valid: true, errors: [], missing_nodes: [], missing_models: [], pending_media_upload: true});
    }
    if (pathName === '/api/v1/comfyui/routes/photo-model' && request.method() === 'PUT') {
      state.routeSaves.push(body);
      state.route = {name: 'Checkpoint Image', ...body};
      return jsonResponse(route, state.route);
    }
    return jsonResponse(route, {detail: `Unexpected request ${request.method()} ${pathName}`}, 404);
  });
  const response = await page.goto(`${harness.origin}${harness.pagePath}`);
  assert.equal(response?.status(), 200, 'Vite serves the component test harness.');
  try {
    await page.getByRole('button', {name: /Generation engine/}).waitFor({state: 'visible'});
  } catch (error) {
    throw new Error(`${error.message}\nAPI requests: ${JSON.stringify(state.requests)}\nBrowser errors: ${JSON.stringify(pageErrors)}`);
  }
  assert.equal(await page.getByRole('button', {name: /Generation engine/}).getByText('Native', {exact: true}).count(), 2,
    'The collapsed engine control shows Native as both the selected route and status');
  assert.ok(state.requests.some(item => item.path === '/api/v1/comfyui/status' && item.method === 'GET'));
  assert.ok(state.requests.some(item => item.path === '/api/v1/comfyui/workflows' && item.method === 'GET'));
  assert.equal(state.requests.some(item => item.path.includes('/connection/test')), false, 'Loading control state does not ping ComfyUI');

  await page.getByRole('button', {name: /Generation engine/}).click();
  const workflowSelect = page.getByLabel('Verified workflow');
  assert.deepEqual(await workflowSelect.locator('option').allTextContents(), ['Choose a image workflow', 'Checkpoint Image'], 'Only same-kind workflows are offered');
  assert.equal(await page.getByLabel('Native field mapping: Prompt').inputValue(), 'prompt', 'Prompt is mapped automatically');
  assert.equal(await page.getByLabel('Native field mapping: Steps').inputValue(), 'steps', 'Native steps map to a typed integer workflow input');
  assert.equal(await page.getByRole('checkbox', {name: 'Enabled'}).isChecked(), false, 'A boolean binding retains its false default');
  assert.equal(await page.getByLabel('Default for Start image').count(), 0, 'Media bindings cannot be assigned fixed filename defaults');
  assert.equal(await page.getByLabel('Default for Style').isEnabled(), true);
  assert.equal(await page.getByRole('button', {name: 'Save routing'}).isDisabled(), true, 'A required unmapped binding must receive a fixed value or mapping');

  await page.getByLabel('Native field mapping: Start image').selectOption('start_image');
  await page.getByLabel('Default for Checkpoint').selectOption('1');
  await page.getByLabel('Default for Style').fill('cinematic');
  await page.getByRole('button', {name: 'Validate workflow'}).click();
  await page.getByText('Workflow preflight passed. Saving routing does not start generation.').waitFor({state: 'visible'});
  assert.deepEqual(state.validations.at(-1), {
    inputs: {
      include_audio: false,
      checkpoint: 'model-b.safetensors',
      style: 'cinematic',
      prompt: 'Maestro route validation',
      steps: 1,
      start_image: '__maestro_pending_media__',
    },
  }, 'Validation includes typed integer, boolean, selected choice and pending media binding values');

  await page.getByRole('button', {name: 'Save routing'}).click();
  try {
    const engineButton = page.getByRole('button', {name: /Generation engine/});
    await page.waitForFunction(() => {
      const button = Array.from(document.querySelectorAll('button')).find(item => item.textContent?.includes('Generation engine'));
      return Boolean(button?.textContent?.includes('Checkpoint Image') && button.textContent?.includes('ComfyUI'));
    });
    const collapsedText = await engineButton.innerText();
    assert.ok(collapsedText.includes('Checkpoint Image') && collapsedText.includes('ComfyUI'),
      'The collapsed control shows the configured workflow name after refreshed status arrives');
  } catch (error) {
    throw new Error(`${error.message}\nPage text: ${await page.locator('body').innerText()}\nRequests: ${JSON.stringify(state.requests)}\nValidations: ${JSON.stringify(state.validations)}\nRoute saves: ${JSON.stringify(state.routeSaves)}\nBrowser errors: ${JSON.stringify(pageErrors)}`);
  }
  assert.deepEqual(state.routeSaves, [{
    workflow_id: 'image-template',
    enabled: true,
    field_bindings: {prompt: 'prompt', num_inference_steps: 'steps', image_start: 'start_image'},
    input_defaults: {include_audio: false, checkpoint: 'model-b.safetensors', style: 'cinematic'},
    kind: 'image',
  }], 'Saved route keeps integer/media mappings and fixed scalar defaults');
  assert.equal(state.validations.length, 2, 'Save revalidates the selected workflow before configuring routing');
  assert.equal(state.requests.some(item => item.path.includes('/run') || item.path.includes('/queue')), false, 'Configuration never runs or queues a workflow');
  assert.deepEqual(pageErrors, [], 'Engine control has no browser errors');
  await page.close();
}

async function checkNativeWorksWhenComfyIsOffline(browser, harness) {
  const state = {
    requests: [],
    routeSaves: [],
    route: {workflow_id: 'image-template', enabled: true, kind: 'image', name: 'Existing Portrait Route',
      field_bindings: {prompt: 'prompt'}, input_defaults: {checkpoint: 'model-a.safetensors'}},
  };
  const page = await browser.newPage();
  page.setDefaultTimeout(7000);
  await page.route(`${harness.origin}/**/*`, async route => {
    const request = route.request();
    const url = new URL(request.url());
    const pathName = url.pathname;
    if (!pathName.startsWith('/api/v1/comfyui/')) return route.continue();
    const body = request.postDataJSON?.() ?? null;
    state.requests.push({path: pathName, method: request.method(), body});
    if (pathName === '/api/v1/comfyui/status' && request.method() === 'GET') {
      return jsonResponse(route, {enabled: true, endpoint: 'http://127.0.0.1:8188', connected: false,
        routes: {'photo-model': state.route}});
    }
    if (pathName === '/api/v1/comfyui/workflows' && request.method() === 'GET') return jsonResponse(route, {workflows: workflowFixtures, templates: []});
    if (pathName === '/api/v1/comfyui/routes/photo-model' && request.method() === 'PUT') {
      state.routeSaves.push(body);
      state.route = {workflow_id: 'image-template', kind: 'image', name: 'Existing Portrait Route', ...body};
      return jsonResponse(route, state.route);
    }
    return jsonResponse(route, {detail: `Unexpected request ${request.method()} ${pathName}`}, 404);
  });
  await page.goto(`${harness.origin}${harness.pagePath}`);
  await page.getByRole('button', {name: /Generation engine/}).waitFor({state: 'visible'});
  assert.equal(await page.getByText('Existing Portrait Route', {exact: true}).count(), 1);
  await page.getByRole('button', {name: /Generation engine/}).click();
  await page.getByRole('button', {name: 'Use Native'}).click();
  await page.getByText('Native', {exact: true}).last().waitFor({state: 'visible'});
  assert.deepEqual(state.routeSaves, [{
    workflow_id: 'image-template',
    enabled: false,
    field_bindings: {prompt: 'prompt'},
    input_defaults: {checkpoint: 'model-a.safetensors'},
    kind: 'image',
  }], 'Choosing Native persists a model-specific disable, including from an offline ComfyUI status');
  assert.equal(state.requests.some(item => item.path.includes('/connection/test') || item.path.includes('/run') || item.path.includes('/queue')), false,
    'Switching to Native never tests ComfyUI or submits generation work');
  await page.close();
}

async function main() {
  await checkRouteApiPayload();
  await checkComponentSyntax();
  if (!chromium) {
    console.log(`ComfyUI route API payload and component syntax passed. Browser behavior checks skipped: ${playwrightUnavailable?.code || 'Playwright unavailable'}.`);
    return;
  }
  const harness = await createBrowserHarness();
  const executablePath = process.env.MAESTRO_CHROME || (process.platform === 'win32' ? 'C:/Program Files/Google/Chrome/Application/chrome.exe' : undefined);
  try {
    const browser = await chromium.launch({headless: true, ...(executablePath ? {executablePath} : {})});
    try {
      await checkConfigurationFlow(browser, harness);
      await checkNativeWorksWhenComfyIsOffline(browser, harness);
    } finally { await browser.close(); }
  } finally { await harness.close(); }
  console.log('ComfyUI route API, typed mappings/defaults, workflow verification, Native-offline routing, and no-implicit-run checks passed.');
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
