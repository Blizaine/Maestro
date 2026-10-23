// Exercise App's real startup effect and gallery store with held HTTP replies.
// Every request is intercepted; no running app, settings or media are changed.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const playwright = require(process.env.MAESTRO_PLAYWRIGHT || 'C:/Users/bliza/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const origin = 'http://gallery-startup.test';

function deferred() {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
}

function gate(held = true) {
  const started = deferred(), released = deferred();
  if (!held) released.resolve();
  return { started, wait: released.promise, release: released.resolve };
}

function within(promise, label) {
  let timer;
  return Promise.race([
    promise,
    new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(`Timed out waiting for ${label}`)), 5000); }),
  ]).finally(() => clearTimeout(timer));
}

const image = (workspace, name = `${workspace}.png`) => ({
  name, workspace, id: `${workspace}/${name}`, type: 'image', mode: 'image',
  path: `${workspace}/${name}`, created_at: 1, size: 50, metadata_ready: true,
  url: `/api/v1/file/${encodeURIComponent(name)}?workspace=${encodeURIComponent(workspace)}`,
});

async function buildBundle() {
  const bundle = await esbuild.build({
    stdin: { contents: [
      "import React from 'react'; import {createRoot} from 'react-dom/client';",
      "import App from './src/App'; import {useStore} from './src/stores/useStore';",
      'window.store = useStore; const idle = async () => {}; const loadModels = async () => useStore.setState({modelsLoaded: true, modelsLoading: false});',
      // Keep gallery hydration real; unrelated model/GPU startup is outside this test.
      "useStore.setState({modelsLoaded: true, generationMode: 'video', servicesConfig: {}, loadModels, loadSystemConfig: idle, loadServicesConfig: idle, loadLlmStatus: idle, loadLlmModels: idle, loadPipelineList: idle, reconnectJobs: idle, loadSystemStats: idle, loadSystemDetect: idle});",
      "const app = <App/>; createRoot(document.getElementById('root')).render(location.search.includes('strict') ? <React.StrictMode>{app}</React.StrictMode> : app);",
    ].join('\n'), resolveDir: path.join(root, 'ui'), loader: 'tsx' },
    bundle: true, write: false, jsx: 'automatic',
    define: { 'process.env.NODE_ENV': '"development"' }, logLevel: 'silent',
  });
  const assets = path.join(root, 'ui/dist/assets');
  const cssFile = fs.readdirSync(assets).find(name => name.endsWith('.css'));
  assert.ok(cssFile, 'the built UI stylesheet is present');
  return { js: bundle.outputFiles[0].text, css: fs.readFileSync(path.join(assets, cssFile), 'utf8') };
}

async function openFixture(browser, bundle, options = {}) {
  const page = await browser.newPage({ viewport: options.viewport || { width: 1360, height: 900 } });
  const workspaceGate = options.workspaceGate || gate(false);
  const outputGates = options.outputGates || [];
  const outputQueries = [], externalRequests = [], mutations = [], pageErrors = [];
  page.on('pageerror', error => pageErrors.push(error.stack || error.message));
  await page.addInitScript(() => localStorage.setItem('maestro_welcome_seen_v1', '1'));
  await page.route('**/*', async route => {
    const request = route.request(), url = new URL(request.url()), method = request.method();
    const json = (body, status = 200) => route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });
    if (url.origin !== origin) { externalRequests.push(request.url()); return route.abort(); }
    if (url.pathname === '/') return route.fulfill({ contentType: 'text/html', body:
      '<meta name="viewport" content="width=device-width,initial-scale=1"><style>html,body,#root{height:100%;margin:0}</style>' +
      '<link rel="stylesheet" href="/app.css"><div id="root"></div><script src="/bundle.js"></script>' });
    if (url.pathname === '/bundle.js') return route.fulfill({ contentType: 'application/javascript', body: bundle.js });
    if (url.pathname === '/app.css') return route.fulfill({ contentType: 'text/css', body: bundle.css });
    if (!url.pathname.startsWith('/api/')) return route.fulfill({ status: 200, body: '' });
    if (method !== 'GET' && method !== 'HEAD' && method !== 'OPTIONS') mutations.push(`${method} ${url.pathname}`);

    if (method === 'GET' && url.pathname === '/api/v1/workspaces') {
      workspaceGate.started.resolve();
      await workspaceGate.wait;
      if (options.workspaceStatus && options.workspaceStatus !== 200) return json({ detail: 'Fixture read failed' }, options.workspaceStatus);
      return json({ active: 'Saved-folder', workspaces: [{ name: 'default', path: 'default' }, { name: 'Saved-folder', path: 'Saved-folder' }] });
    }
    if (method === 'GET' && url.pathname === '/api/v1/outputs') {
      const index = outputQueries.length;
      outputQueries.push({ workspace: url.searchParams.get('workspace'), search: url.searchParams.get('search') });
      if (outputGates[index]) { outputGates[index].started.resolve(); await outputGates[index].wait; }
      if (options.failOutputIndex === index) return json({ detail: 'Fixture read failed' }, options.outputStatus || 503);
      const workspace = url.searchParams.get('workspace') || 'default';
      const body = options.outputResponse
        ? options.outputResponse(url, index)
        : { outputs: [image(workspace === '__all__' ? 'Folder-A' : workspace)], total: 1 };
      return json(body);
    }
    if (url.pathname === '/api/v1/media-flow/capabilities') return json({
      neural_rendering: { available: false, reason: 'Isolated test' },
      frame_generation: { available: false, factors: [], reason: 'Isolated test' },
    });
    if (url.pathname.includes('/metadata')) return json({ params: { prompt: 'Saved gallery image' } });
    if (url.pathname.includes('/file/') || url.pathname.includes('/thumbnail/')) return route.fulfill({
      contentType: 'image/svg+xml', body: '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360"><rect width="640" height="360" fill="#334155"/></svg>',
    });
    return json({ checks: [], downloads: [], jobs: [], items: [], entries: [], queue: [], pipelines: [], recipes: [],
      loras: [], models: [], families: [], presets: [], capabilities: {}, outputs: [], total: 0 });
  });
  await page.goto(`${origin}/${options.strict ? '?strict=1' : ''}`);
  return { page, workspaceGate, outputGates, outputQueries, externalRequests, mutations, pageErrors };
}

async function loading(page, phase) {
  await page.getByRole('status').filter({ hasText: 'Loading workspace...' }).waitFor({ state: 'visible', timeout: 5000 });
  assert.equal(await page.getByText('Your generated videos will appear here.', { exact: true }).count(), 0,
    `${phase}: first-run empty guidance stays hidden while startup is pending`);
}

async function auditLoading(page) {
  await page.evaluate(() => {
    const audit = window.galleryAudit = { active: true, problems: [] };
    const visible = element => element && element.getClientRects().length > 0 && getComputedStyle(element).visibility !== 'hidden';
    const sample = () => {
      if (!audit.active) return;
      if (![...document.querySelectorAll('[role="status"]')].some(el => visible(el) && el.textContent.includes('Loading workspace...')) && !audit.problems.includes('status disappeared')) audit.problems.push('status disappeared');
      if ([...document.querySelectorAll('p')].some(el => visible(el) && el.textContent.includes('Your generated videos will appear here.')) && !audit.problems.includes('empty guidance appeared')) audit.problems.push('empty guidance appeared');
      requestAnimationFrame(sample);
    };
    requestAnimationFrame(sample);
  });
}

const frame = page => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
const endAudit = page => page.evaluate(() => { window.galleryAudit.active = false; return window.galleryAudit.problems; });
const ready = page => page.waitForFunction(() => {
  const state = window.store?.getState();
  return state && !state.galleryInitializing && !state.outputsLoading;
}, null, { timeout: 5000 });

async function noSideEffects(fixture) {
  assert.deepEqual(fixture.externalRequests, [], 'all browser requests stay on the mocked origin');
  assert.deepEqual(fixture.mutations, [], 'startup does not send mutation requests');
  assert.deepEqual(fixture.pageErrors, [], 'startup has no browser runtime errors');
}

async function closeFixture(fixture) {
  fixture.workspaceGate.release();
  fixture.outputGates.forEach(responseGate => responseGate.release());
  await fixture.page.close();
}

async function heldCard(browser, bundle, { strict = false, caseName } = {}) {
  const workspaces = gate(), outputs = gate();
  const fixture = await openFixture(browser, bundle, { workspaceGate: workspaces, outputGates: [outputs], strict });
  const { page } = fixture;
  try {
    await within(workspaces.started.promise, 'workspace lookup');
    await loading(page, 'workspace lookup');
    assert.deepEqual(await page.evaluate(() => {
      const s = window.store.getState();
      return { controlsReady: s.modelsLoaded && !s.modelsLoading && !s.modelsLoadError,
        galleryInitializing: s.galleryInitializing, outputsLoading: s.outputsLoading };
    }), { controlsReady: true, galleryInitializing: true, outputsLoading: false },
    'model controls are ready while the saved workspace lookup is still pending');
    await auditLoading(page);

    workspaces.release();
    await within(outputs.started.promise, 'media listing');
    await loading(page, 'media listing');
    assert.deepEqual(await page.evaluate(() => {
      const s = window.store.getState(); return [s.activeWorkspace, s.galleryInitializing, s.outputsLoading];
    }), ['Saved-folder', true, true]);
    await frame(page);
    assert.deepEqual(await endAudit(page), [], 'loading stays visible across both held reads');
    assert.deepEqual(fixture.outputQueries[0], { workspace: 'Saved-folder', search: null }, 'the saved folder is restored before listing media');

    outputs.release();
    await ready(page);
    await page.locator('[data-feed-index]').first().waitFor({ state: 'visible', timeout: 5000 });
    assert.equal(await page.locator('[data-feed-index]').count(), 1, 'the restored media card is rendered');
    if (!strict) {
      await page.reload();
      await ready(page);
      await page.locator('[data-feed-index]').first().waitFor({ state: 'visible', timeout: 5000 });
      assert.ok(fixture.outputQueries.slice(1).some(query => query.workspace === 'Saved-folder'), 'refresh reloads the saved folder');
    }
    await noSideEffects(fixture);
    console.log(`PASS gallery startup: ${caseName}`);
  } finally { await closeFixture(fixture); }
}

async function outputsFirstMobile(browser, bundle) {
  const workspaces = gate();
  const fixture = await openFixture(browser, bundle, {
    workspaceGate: workspaces, viewport: { width: 390, height: 844 },
    outputResponse: () => ({ outputs: [], total: 0 }),
  });
  const { page } = fixture;
  try {
    await within(workspaces.started.promise, 'mobile workspace lookup');
    await loading(page, 'mobile workspace lookup');
    workspaces.release();
    await ready(page);
    await page.getByText('Your generated videos will appear here.', { exact: true }).waitFor({ state: 'visible', timeout: 5000 });
    assert.equal(await page.locator('[data-feed-index]').count(), 0, 'the successful empty result renders no cards');
    assert.equal(fixture.outputQueries[0]?.workspace, 'Saved-folder', 'the immediate output read uses the restored folder');
    await noSideEffects(fixture);
    console.log('PASS gallery startup: outputs-first mobile empty result');
  } finally { await closeFixture(fixture); }
}

async function failedReadsSettle(browser, bundle) {
  const fixture = await openFixture(browser, bundle, {
    workspaceGate: gate(false), workspaceStatus: 503, failOutputIndex: 0, outputStatus: 503,
  });
  try {
    await ready(fixture.page);
    assert.deepEqual(await fixture.page.evaluate(() => {
      const s = window.store.getState(); return [s.galleryInitializing, s.outputsLoading];
    }), [false, false], 'failed reads do not leave startup loading stuck');
    await noSideEffects(fixture);
    console.log('PASS gallery startup: failed reads settle');
  } finally { await closeFixture(fixture); }
}

async function allFoldersRace(browser, bundle) {
  const workspaces = gate(), stale = gate(), current = gate();
  const fixture = await openFixture(browser, bundle, {
    workspaceGate: workspaces, outputGates: [stale, current],
    outputResponse: (_url, index) => index === 0
      ? { outputs: [image('Folder-A', 'stale.png')], total: 1 }
      : { outputs: [image('Folder-B', 'current.png')], total: 1 },
  });
  const { page } = fixture;
  try {
    await within(workspaces.started.promise, 'all-folders workspace lookup');
    await loading(page, 'all-folders workspace lookup');
    await auditLoading(page);
    await page.evaluate(() => window.store.getState().switchWorkspace('__all__'));
    await within(stale.started.promise, 'first All folders request');
    assert.equal(fixture.outputQueries[0]?.workspace, '__all__');
    workspaces.release();
    await within(current.started.promise, 'restored All folders request');
    assert.equal(fixture.outputQueries[1]?.workspace, '__all__');
    await loading(page, 'All folders media listing');

    const staleResponse = page.waitForResponse(response => {
      const url = new URL(response.url());
      return url.origin === origin && url.pathname === '/api/v1/outputs' && response.status() === 200;
    }, { timeout: 5000 });
    stale.release();
    await staleResponse;
    await page.waitForFunction(() => {
      const s = window.store?.getState(); return s?.galleryInitializing && s.outputsLoading && s.outputs.length === 0;
    }, null, { timeout: 5000 });
    await frame(page);
    assert.deepEqual(await endAudit(page), [], 'the status remains visible while the newer request is pending');

    current.release();
    await page.waitForFunction(() => {
      const s = window.store?.getState(); return s && !s.galleryInitializing && !s.outputsLoading && s.outputs[0]?.name === 'current.png';
    }, null, { timeout: 5000 });
    await page.locator('[data-feed-index]').first().waitFor({ state: 'visible', timeout: 5000 });
    assert.equal(await page.evaluate(() => window.store.getState().browsingAllFolders), true);
    assert.equal(await page.evaluate(() => window.store.getState().activeWorkspace), 'Saved-folder');
    await noSideEffects(fixture);
    console.log('PASS gallery startup: All folders startup race');
  } finally { stale.release(); current.release(); await closeFixture(fixture); }
}

(async () => {
  const bundle = await buildBundle();
  const browser = await playwright.chromium.launch({ headless: true,
    ...(process.platform === 'win32' ? { executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe' } : {}) });
  try {
    await heldCard(browser, bundle, { caseName: 'workspace-first' });
    await outputsFirstMobile(browser, bundle);
    await heldCard(browser, bundle, { strict: true, caseName: 'workspace-first-strict' });
    await failedReadsSettle(browser, bundle);
    await allFoldersRace(browser, bundle);
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
