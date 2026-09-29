// Run with: node tests/ui/status_polling.cjs
// Mounts the real HardwareStatusBar and DownloadStatusBanner with every request
// intercepted and a fake clock. Page visibility and window focus are simulated,
// because a headless page is always visible and focused.
const assert = require('node:assert/strict');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const playwright = require(process.env.MAESTRO_PLAYWRIGHT || 'C:/Users/bliza/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

const pageStateScript = () => {
  window.pageState = {visible: true, focused: true};
  Object.defineProperty(Document.prototype, 'visibilityState', {configurable: true,
    get: () => (window.pageState.visible ? 'visible' : 'hidden')});
  Object.defineProperty(Document.prototype, 'hidden', {configurable: true, get: () => !window.pageState.visible});
  Document.prototype.hasFocus = () => window.pageState.visible && window.pageState.focused;
  window.setPageState = next => {
    const before = {...window.pageState};
    Object.assign(window.pageState, next);
    if (before.visible !== window.pageState.visible) document.dispatchEvent(new Event('visibilitychange'));
    if (before.focused !== window.pageState.focused) window.dispatchEvent(new Event(window.pageState.focused ? 'focus' : 'blur'));
  };
};

const stats = {cpu: {percent: 3}, ram: {percent: 40, used_gb: 12.5, total_gb: 32},
  gpu: {available: true, percent: 0, vram_used_gb: 1, vram_total_gb: 24, vram_percent: 4},
  model: {name: null, model_type: null, loaded: false}};

const download = overrides => ({file_id: 'a', filename: 'model.safetensors', started_at: 1, last_active_at: 1,
  downloaded_bytes: 1024 * 1024, total_bytes: 4 * 1024 * 1024, status: 'downloading', seconds_since_progress: 0,
  ...overrides});

async function openHarness(browser, bundle, component, {strict = false} = {}) {
  const page = await browser.newPage();
  const errors = [];
  const statsRequests = [];
  // Download requests stay open until the test answers them, like a held long-poll.
  const downloadRequests = [];
  const failed = new Set();
  page.on('pageerror', error => errors.push(error.stack));
  page.on('requestfailed', request => failed.add(request));
  await page.clock.install();
  await page.addInitScript(pageStateScript);
  await page.route('**/*', route => {
    const url = new URL(route.request().url());
    if (url.pathname === '/') return route.fulfill({contentType: 'text/html', body: '<div id="root"></div><script src="/bundle.js"></script>'});
    if (url.pathname === '/bundle.js') return route.fulfill({contentType: 'application/javascript', body: bundle});
    if (url.pathname === '/api/v1/system-stats') {
      statsRequests.push(url.search);
      return route.fulfill({contentType: 'application/json', body: JSON.stringify(stats)});
    }
    if (url.pathname === '/api/v1/downloads/active') {
      downloadRequests.push({since: url.searchParams.get('since'), route, answered: false});
      return undefined;
    }
    return route.fulfill({contentType: 'application/json', body: '{}'});
  });
  await page.goto(`http://maestro.test/${strict ? '?strict' : ''}`);
  await page.evaluate(which => window.mount(which), component);
  const open = () => downloadRequests.filter(r => !r.answered && !failed.has(r.route.request()));
  const answer = async (body, status = 200) => {
    const pending = open();
    assert.equal(pending.length, 1, 'Exactly one download request is open');
    pending[0].answered = true;
    await pending[0].route.fulfill({status, contentType: 'application/json', body: JSON.stringify(body)});
    await settle(page);
  };
  return {page, errors, statsRequests, downloadRequests, failed, open, answer};
}

// Let fetch callbacks and React effects settle without advancing the fake clock.
const settle = page => page.evaluate(() => new Promise(resolve => setTimeout(resolve, 0)))
  .then(() => page.waitForTimeout(50));

async function assertHardwareStatusPolling(browser, bundle) {
  const {page, errors, statsRequests} = await openHarness(browser, bundle, 'hardware');
  await settle(page);
  assert.equal(statsRequests.length, 1, 'Stats load immediately on mount');
  await page.clock.runFor(2000); await settle(page);
  assert.equal(statsRequests.length, 2, 'Stats refresh every 2 s while visible');

  await page.evaluate(() => window.setPageState({focused: false})); await settle(page);
  await page.clock.runFor(2000); await settle(page);
  assert.equal(statsRequests.length, 3, 'A visible but unfocused window keeps its 2 s stats cadence');

  await page.evaluate(() => window.setPageState({visible: false})); await settle(page);
  const hiddenAt = statsRequests.length;
  await page.clock.runFor(10_000); await settle(page);
  assert.equal(statsRequests.length, hiddenAt, 'A hidden tab sends no stats requests');

  await page.evaluate(() => window.setPageState({visible: true, focused: true})); await settle(page);
  assert.equal(statsRequests.length, hiddenAt + 1, 'Stats refresh at once when the tab becomes visible');
  assert.deepEqual(errors, []);
  await page.close();
}

async function assertDownloadLongPoll(browser, bundle) {
  const {page, errors, downloadRequests, open, answer} = await openHarness(browser, bundle, 'downloads');
  await settle(page);
  assert.deepEqual(downloadRequests.map(r => r.since), [null], 'The first request carries no version');
  await answer({downloads: [], version: 'v1'});
  assert.equal(open()[0]?.since, 'v1', 'The focused window holds the next request with the version it has');

  await answer({downloads: [download()], version: 'v2'});
  assert.ok(await page.getByText('Downloading model files').isVisible(), 'A change is shown at once');
  assert.equal(open()[0]?.since, 'v2');

  await page.clock.runFor(25_000);
  await answer({downloads: [download()], version: 'v2'});
  assert.equal(open()[0]?.since, 'v2', 'A hold that timed out is renewed at once');

  const beforeFast = downloadRequests.length;
  await answer({downloads: [download()], version: 'v2'});
  assert.equal(downloadRequests.length, beforeFast, 'An unchanged answer that did not wait is not followed at once');
  await page.clock.runFor(2000); await settle(page);
  assert.equal(open()[0]?.since, 'v2', 'The long-poll resumes after 2 s');

  const beforeBlur = downloadRequests.length;
  await page.evaluate(() => window.setPageState({focused: false})); await settle(page);
  await page.clock.runFor(10_000); await settle(page);
  assert.equal(downloadRequests.length, beforeBlur, 'Losing focus (another app, DevTools) keeps the same held request');
  assert.equal(open()[0]?.since, 'v2', 'A visible but unfocused window keeps long-polling');
  await page.evaluate(() => window.setPageState({focused: true})); await settle(page);
  assert.equal(downloadRequests.length, beforeBlur, 'Regaining focus changes nothing');

  await page.evaluate(() => window.setPageState({visible: false})); await settle(page);
  assert.equal(open().length, 0, 'Hiding the tab aborts the held request');
  const hiddenAt = downloadRequests.length;
  await page.clock.runFor(10_000); await settle(page);
  assert.equal(downloadRequests.length, hiddenAt, 'A hidden tab sends no download requests');
  await page.evaluate(() => window.setPageState({visible: true})); await settle(page);
  assert.equal(downloadRequests.length, hiddenAt + 1, 'Showing the tab asks again at once');
  assert.equal(open()[0]?.since, 'v2', 'It asks with the version it already has, so the server answers only if something changed');

  await answer({downloads: [download({seconds_since_progress: 31})], version: 'v3'});
  assert.ok(await page.getByText('No progress for 31s.', {exact: false}).isVisible());
  const stalledAt = downloadRequests.length;
  await page.clock.runFor(3000); await settle(page);
  assert.ok(await page.getByText('No progress for 34s.', {exact: false}).isVisible(), 'The stall counter keeps counting locally');
  assert.equal(downloadRequests.length, stalledAt, 'Counting needs no requests');

  await answer({detail: 'boom'}, 500);
  assert.equal(await page.getByText('model.safetensors').count(), 0, 'An error clears the banner, as before');
  assert.equal(open().length, 0);
  await page.clock.runFor(2000); await settle(page);
  assert.equal(open()[0]?.since, null, 'Requests resume 2 s after an error');
  assert.deepEqual(errors, []);
  await page.close();
}

async function assertOlderServerFallback(browser, bundle) {
  const {page, errors, open, answer} = await openHarness(browser, bundle, 'downloads');
  await settle(page);
  await answer({downloads: []});
  assert.equal(open().length, 0, 'A server without versions is not asked again at once');
  await page.clock.runFor(2000); await settle(page);
  assert.equal(open()[0]?.since, null, 'It is polled every 2 s instead');
  assert.deepEqual(errors, []);
  await page.close();
}

async function assertStrictModeRunsOneLoop(browser, bundle) {
  const {page, errors, open, answer} = await openHarness(browser, bundle, 'downloads', {strict: true});
  await settle(page);
  await answer({downloads: [], version: 'v1'});
  assert.equal(open().length, 1, 'StrictMode double mounting leaves one held request');
  assert.deepEqual(errors, []);
  await page.close();
}

(async () => {
  const built = await esbuild.build({stdin: {contents: [
    "import React from 'react'; import {createRoot} from 'react-dom/client';",
    "import {HardwareStatusBar} from './src/components/Sidebar/HardwareStatusBar';",
    "import {DownloadStatusBanner} from './src/components/DownloadStatusBanner';",
    "import {useStore} from './src/stores/useStore';",
    'window.store = useStore;',
    "window.mount = which => { const app = which === 'hardware' ? <HardwareStatusBar/> : <DownloadStatusBanner/>; createRoot(document.getElementById('root')).render(location.search.includes('strict') ? <React.StrictMode>{app}</React.StrictMode> : app); };",
  ].join('\n'), resolveDir: path.join(root, 'ui'), loader: 'tsx'}, bundle: true, write: false,
    jsx: 'automatic', define: {'process.env.NODE_ENV': '"development"'}, logLevel: 'silent'});
  const bundle = built.outputFiles[0].text;
  const browser = await playwright.chromium.launch({headless: true,
    ...(process.platform === 'win32' ? {executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'} : {})});
  try {
    await assertHardwareStatusPolling(browser, bundle);
    await assertDownloadLongPoll(browser, bundle);
    await assertOlderServerFallback(browser, bundle);
    await assertStrictModeRunsOneLoop(browser, bundle);
    console.log('Status polling: hardware stats cadence, download long-poll, fallback and StrictMode passed');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
