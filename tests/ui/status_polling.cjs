// Run with: node tests/ui/status_polling.cjs
// Mounts the real HardwareStatusBar with every request intercepted and a fake
// clock. Page visibility and window focus are simulated, because a headless
// page is always visible and focused.
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

async function openHarness(browser, bundle, {strict = false} = {}) {
  const page = await browser.newPage();
  const errors = [];
  const statsRequests = [];
  page.on('pageerror', error => errors.push(error.stack));
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
    return route.fulfill({contentType: 'application/json', body: '{}'});
  });
  await page.goto(`http://maestro.test/${strict ? '?strict' : ''}`);
  await page.evaluate(() => window.mount());
  return {page, errors, statsRequests};
}

// Let fetch callbacks and React effects settle without advancing the fake clock.
const settle = page => page.evaluate(() => new Promise(resolve => setTimeout(resolve, 0)))
  .then(() => page.waitForTimeout(50));

async function assertHardwareStatusPolling(browser, bundle) {
  const {page, errors, statsRequests} = await openHarness(browser, bundle);
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

(async () => {
  const built = await esbuild.build({stdin: {contents: [
    "import React from 'react'; import {createRoot} from 'react-dom/client';",
    "import {HardwareStatusBar} from './src/components/Sidebar/HardwareStatusBar';",
    "import {useStore} from './src/stores/useStore';",
    'window.store = useStore;',
    "window.mount = () => { const app = <HardwareStatusBar/>; createRoot(document.getElementById('root')).render(location.search.includes('strict') ? <React.StrictMode>{app}</React.StrictMode> : app); };",
  ].join('\n'), resolveDir: path.join(root, 'ui'), loader: 'tsx'}, bundle: true, write: false,
    jsx: 'automatic', define: {'process.env.NODE_ENV': '"development"'}, logLevel: 'silent'});
  const bundle = built.outputFiles[0].text;
  const browser = await playwright.chromium.launch({headless: true,
    ...(process.platform === 'win32' ? {executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'} : {})});
  try {
    await assertHardwareStatusPolling(browser, bundle);
    console.log('Status polling: hardware stats visible-only cadence passed');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
