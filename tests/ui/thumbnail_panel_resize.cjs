// Exercise the resizable thumbnail panel in the real app with all HTTP calls
// intercepted. Run after MainContent's controlled open/close integration lands.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const { chromium } = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');
async function capture(page, name) {
  if (!process.env.MAESTRO_THUMBNAIL_SHOTS) return;
  fs.mkdirSync(process.env.MAESTRO_THUMBNAIL_SHOTS, { recursive: true });
  await page.screenshot({ path: path.join(process.env.MAESTRO_THUMBNAIL_SHOTS, `${name}.png`) });
}

const makeOutput = index => {
  const type = index > 0 && index % 17 === 0 ? 'audio' : index % 11 === 0 ? 'video' : 'image';
  const name = `asset-${index}.${type === 'audio' ? 'mp3' : type === 'video' ? 'mp4' : 'png'}`;
  return {
    id: `Saved/${name}`, name, workspace: 'Saved', type, mode: type, path: `Saved/${name}`,
    url: `/api/v1/file/${encodeURIComponent(name)}?workspace=Saved`,
    size: 1024, created_at: 1, favorite: false, metadata_ready: true,
  };
};

const bundleContents = [
  "import React from 'react'; import {createRoot} from 'react-dom/client';",
  "import App from './src/App'; import {useStore} from './src/stores/useStore';",
  'window.store = useStore; const idle = async () => {};',
  'window.setup = (files) => {',
  '  useStore.setState({loadModels: idle, loadOutputs: idle, loadWorkspaces: idle,',
  '    loadSystemConfig: idle, loadServicesConfig: idle, loadLlmStatus: idle, loadLlmModels: idle,',
  '    loadPipelineList: idle, reconnectJobs: idle, loadSystemStats: idle, loadSystemDetect: idle,',
  "    generationMode: 'image', sidebarMode: 'studio', sidebarOpen: false, models: [], families: [],",
  "    outputs: files, outputsTotal: files.length, selectedOutput: 0, activeWorkspace: 'Saved',",
  '    browsingAllFolders: true, browsingUploads: false, mediaFilter: \'all\', jobs: [],',
  '    isGenerating: false, isEnhancing: false, settingsOpen: false, servicesConfig: {},',
  "    params: {...useStore.getState().params, model_type: 'qwen_image_21_7B', image_mode: 1, prompt: 'Thumbnail resize'}});",
  '  createRoot(document.getElementById("root")).render(<App/>);',
  '};',
].join('\n');

async function buildBundle() {
  return esbuild.build({
    stdin: { contents: bundleContents, resolveDir: path.join(root, 'ui'), loader: 'tsx' },
    bundle: true, write: false, jsx: 'automatic',
    define: { 'process.env.NODE_ENV': '"development"' }, logLevel: 'silent',
  });
}

async function setupPage(page, files, mutations, errors) {
  page.on('pageerror', error => errors.push(error.stack || error.message));
  await page.addInitScript(() => {
    localStorage.setItem('maestro_welcome_seen_v1', '1');
    if (!sessionStorage.getItem('__thumbnail_panel_test_seeded')) {
      localStorage.removeItem('maestro-thumbnail-panel-width');
      localStorage.removeItem('maestro-thumbnail-panel-open');
      sessionStorage.setItem('__thumbnail_panel_test_seeded', '1');
    }
  });
  await page.route('**/*', async route => {
    const request = route.request();
    const url = new URL(request.url());
    const json = body => route.fulfill({ contentType: 'application/json', body: JSON.stringify(body) });
    if (['POST', 'PUT', 'PATCH', 'DELETE'].includes(request.method())) mutations.push(`${request.method()} ${url.pathname}`);
    if (url.pathname === '/') return route.fulfill({ contentType: 'text/html', body:
      '<html data-theme="graphite-studio"><meta name="viewport" content="width=device-width,initial-scale=1"><style>html,body,#root{height:100%;margin:0}</style><link rel="stylesheet" href="/app.css"><div id="root"></div><script src="/bundle.js"></script>' });
    if (url.pathname === '/bundle.js') return route.fulfill({ contentType: 'application/javascript', body: page.bundleText });
    if (url.pathname === '/app.css') return route.fulfill({ contentType: 'text/css', body: page.cssText });
    if (url.pathname.includes('/file/')) return route.fulfill({ contentType: 'image/svg+xml', body:
      '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360"><rect width="640" height="360" fill="#334155"/></svg>' });
    return json({ checks: [], downloads: [], jobs: [], items: [], entries: [], queue: [], pipelines: [],
      recipes: [], loras: [], models: [], families: [], presets: [], capabilities: {}, outputs: [], total: 0 });
  });
  await page.goto('http://thumbnail-panel.test/');
  await page.evaluate(files => window.setup(files), files);
  await page.locator('[data-feed-index="0"]').waitFor();
}

async function openPanel(page) {
  if (await page.locator('[data-testid="thumbnail-panel"]').count()) return;
  await page.locator('button[title="Show thumbnails"]').click();
  await page.locator('[data-testid="thumbnail-panel"]').waitFor();
}

async function scrollThumbnailTo(page, index) {
  await page.evaluate(targetIndex => {
    const thumb = document.querySelector('[data-thumb-index]');
    let pane = thumb;
    while (pane && !/(auto|scroll)/.test(getComputedStyle(pane).overflowY)) pane = pane.parentElement;
    if (!pane || !thumb) throw new Error('Thumbnail scroll area was not found');
    const rowStride = thumb.getBoundingClientRect().height + 6;
    pane.scrollTop = Math.max(0, targetIndex * rowStride - pane.clientHeight / 2);
    pane.dispatchEvent(new Event('scroll', { bubbles: true }));
  }, index);
  await page.locator(`[data-thumb-index="${index}"]`).waitFor();
}

async function metrics(page) {
  return page.evaluate(() => {
    const panel = document.querySelector('[data-testid="thumbnail-panel"]');
    const gallery = document.querySelector('[data-testid="gallery-content"]');
    const feed = document.querySelector('[data-testid="gallery-feed"]');
    const thumb = document.querySelector('[data-thumb-index]');
    return {
      panelWidth: panel?.getBoundingClientRect().width,
      galleryWidth: gallery?.clientWidth,
      feedWidth: feed?.clientWidth,
      thumbHeight: thumb?.getBoundingClientRect().height,
      maxWidth: Number(panel?.querySelector('[aria-label="Resize thumbnails"]')?.getAttribute('aria-valuemax')),
    };
  });
}

async function dragHandle(page, deltaX) {
  const handle = page.locator('[aria-label="Resize thumbnails"]');
  const box = await handle.boundingBox();
  assert.ok(box, 'resize separator is visible');
  const beforeSelected = await page.evaluate(() => window.store.getState().selectedOutput);
  const bodyStyles = await page.evaluate(() => ({ cursor: document.body.style.cursor, userSelect: document.body.style.userSelect }));
  await page.mouse.move(box.x + box.width / 2, box.y + Math.min(20, box.height / 2));
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 - deltaX, box.y + Math.min(20, box.height / 2), { steps: 6 });
  await page.mouse.up();
  await page.waitForTimeout(50);
  assert.equal(await page.evaluate(() => window.store.getState().selectedOutput), beforeSelected,
    'resizer pointerup does not select a thumbnail');
  assert.deepEqual(await page.evaluate(() => ({ cursor: document.body.style.cursor, userSelect: document.body.style.userSelect })),
    bodyStyles, 'drag restores body cursor and user selection styles');
}

async function main() {
  const bundle = await buildBundle();
  const assets = path.join(root, 'ui/dist/assets');
  const cssFile = fs.readdirSync(assets).find(name => name.endsWith('.css'));
  const browser = await chromium.launch({ headless: true, ...(process.platform === 'win32'
    ? { executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe' } : {}) });
  const files = Array.from({ length: 160 }, (_, index) => makeOutput(index));
  const errors = [];
  const mutations = [];
  const desktop = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  const page = await desktop.newPage();
  page.setDefaultTimeout(8000);
  page.bundleText = bundle.outputFiles[0].text;
  page.cssText = fs.readFileSync(path.join(assets, cssFile), 'utf8');

  try {
    await setupPage(page, files, mutations, errors);
    await openPanel(page);
    const initial = await metrics(page);
    const initialWidth = initial.panelWidth;
    await capture(page, 'desktop-compact');
    await dragHandle(page, 120);
    await page.waitForFunction(oldWidth => document.querySelector('[data-testid="thumbnail-panel"]')
      ?.getBoundingClientRect().width > oldWidth + 90, initialWidth);
    const widened = await metrics(page);
    assert.ok(widened.feedWidth < initial.feedWidth, 'widening the panel shrinks the gallery feed');
    assert.ok(widened.thumbHeight > initial.thumbHeight + 30, 'thumbnail rows grow to a 16:9 height at the wider panel width');
    await capture(page, 'desktop-widened');
    assert.equal(await page.evaluate(() => localStorage.getItem('maestro-thumbnail-panel-width')),
      String(Math.round(widened.panelWidth)), 'dragged width is persisted');

    const separator = page.locator('[aria-label="Resize thumbnails"]');
    await separator.focus();
    await page.keyboard.press('End');
    assert.equal((await metrics(page)).panelWidth, 360, 'End moves to the maximum width');
    await page.keyboard.press('Home');
    assert.equal((await metrics(page)).panelWidth, 80, 'Home moves to the minimum width');
    await page.keyboard.press('ArrowLeft');
    assert.equal((await metrics(page)).panelWidth, 100, 'ArrowLeft widens the panel');
    await page.keyboard.press('ArrowRight');
    assert.equal((await metrics(page)).panelWidth, 80, 'ArrowRight shrinks the panel');
    await page.keyboard.press('ArrowRight');
    assert.equal((await metrics(page)).panelWidth, 80, 'keyboard resize stops at the minimum');

    await page.keyboard.press('End');
    const savedWidth = await page.evaluate(() => localStorage.getItem('maestro-thumbnail-panel-width'));
    await page.setViewportSize({ width: 768, height: 900 });
    await page.waitForTimeout(80);
    const clamped = await metrics(page);
    if (clamped.galleryWidth >= 320) {
      assert.ok(clamped.panelWidth <= clamped.galleryWidth - 240, 'responsive width leaves at least 240px for the feed when possible');
      assert.ok(clamped.feedWidth >= 240, 'the feed retains usable width');
      assert.equal(await page.evaluate(() => localStorage.getItem('maestro-thumbnail-panel-width')), savedWidth,
        'responsive clamping preserves the preferred saved width');
    }

    await page.setViewportSize({ width: 1280, height: 900 });
    await page.waitForFunction(width => document.querySelector('[data-testid="thumbnail-panel"]')
      ?.getBoundingClientRect().width === width, Number(savedWidth));
    await page.reload();
    await page.evaluate(files => window.setup(files), files);
    await openPanel(page);
    assert.equal((await metrics(page)).panelWidth, Number(savedWidth), 'saved width is restored after reload');
    assert.equal(await page.locator('#gallery-thumbnail-panel button[title="Show thumbnails"], #gallery-thumbnail-panel button[title="Hide thumbnails"]').count(), 0,
      'the panel contains no old protruding toggle');

    await scrollThumbnailTo(page, 140);
    await page.locator('[data-thumb-index="140"]').click();
    await page.waitForFunction(() => window.store.getState().selectedOutput === 140);
    await page.locator('[data-thumb-index="140"]').waitFor();

    await page.locator('button[title="Hide thumbnails"]').click();
    await page.locator('[data-testid="thumbnail-panel"]').waitFor({ state: 'detached' });
    await page.locator('button[title="Show thumbnails"]').click();
    await page.locator('[data-testid="thumbnail-panel"]').waitFor();
    assert.ok(await page.locator('[data-thumb-index="140"]').count(), 'reopened panel keeps the distant active item visible');

    // A temporarily empty folder must not prevent a later populated panel
    // from measuring its narrower gallery container.
    await page.evaluate(() => window.store.setState({ outputs: [], outputsTotal: 0, selectedOutput: 0 }));
    await page.locator('[data-testid="thumbnail-panel"]').waitFor({ state: 'detached' });
    await page.setViewportSize({ width: 768, height: 900 });
    await page.evaluate(files => window.store.setState({ outputs: files, outputsTotal: files.length }), files);
    await page.locator('[data-testid="thumbnail-panel"]').waitFor();
    const repopulated = await metrics(page);
    assert.ok(repopulated.panelWidth <= repopulated.galleryWidth - 240, 'empty-to-populated panel applies responsive bounds');
    assert.equal(await page.evaluate(() => localStorage.getItem('maestro-thumbnail-panel-width')), savedWidth);
    await page.setViewportSize({ width: 1280, height: 900 });
    await page.waitForTimeout(80);

    // Closing the controlled panel during an active drag must restore global
    // interaction styles even though the component itself stays mounted.
    const dragStyles = await page.evaluate(() => ({ cursor: document.body.style.cursor, userSelect: document.body.style.userSelect }));
    const dragBox = await separator.boundingBox();
    assert.equal(await separator.evaluate(element => {
      const bounds = element.getBoundingClientRect();
      return element.contains(document.elementFromPoint(bounds.left + bounds.width / 2, bounds.top + 20));
    }), true, 'resizer remains reachable after responsive changes');
    await page.mouse.move(dragBox.x + dragBox.width / 2, dragBox.y + 20);
    await page.mouse.down();
    assert.equal(await page.evaluate(() => document.body.style.cursor), 'col-resize');
    await page.locator('button[title="Hide thumbnails"]').dispatchEvent('click');
    await page.locator('[data-testid="thumbnail-panel"]').waitFor({ state: 'detached' });
    await page.mouse.up();
    assert.deepEqual(await page.evaluate(() => ({ cursor: document.body.style.cursor, userSelect: document.body.style.userSelect })), dragStyles);
    await page.reload();
    await page.evaluate(files => window.setup(files), files);
    assert.equal(await page.locator('[data-testid="thumbnail-panel"]').count(), 0, 'desktop collapsed choice survives reload');
    await openPanel(page);
    assert.equal((await metrics(page)).panelWidth, Number(savedWidth), 'reopening restores preferred width');
    console.log('PASS desktop resize, keyboard bounds, persistence, responsive clamp, navigation, and toolbar toggle');

    const mobileContext = await browser.newContext({
      viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, hasTouch: true, isMobile: true,
    });
    const mobile = await mobileContext.newPage();
    mobile.setDefaultTimeout(8000);
    mobile.bundleText = bundle.outputFiles[0].text;
    mobile.cssText = page.cssText;
    try {
      await setupPage(mobile, files, mutations, errors);
      assert.equal(await mobile.locator('[data-testid="thumbnail-panel"]').count(), 0, 'mobile starts with thumbnails closed');
      assert.equal(await mobile.locator('button.fixed.bottom-4.right-4').count(), 0, 'mobile has no floating opener');
      await openPanel(mobile);
      const mobileBefore = await metrics(mobile);
      assert.ok(mobileBefore.panelWidth >= 100, 'mobile keeps the existing usable minimum width');
      await capture(mobile, 'mobile-overlay');
      await dragHandle(mobile, 40);
      assert.ok((await metrics(mobile)).panelWidth > mobileBefore.panelWidth, 'mobile overlay can be resized');
      await mobile.getByRole('button', { name: 'Close thumbnails' }).click();
      await mobile.locator('[data-testid="thumbnail-panel"]').waitFor({ state: 'detached' });
      await openPanel(mobile);
      await scrollThumbnailTo(mobile, 120);
      await mobile.locator('[data-thumb-index="120"]').click();
      await mobile.waitForFunction(() => window.store.getState().selectedOutput === 120);
      await mobile.locator('[data-testid="thumbnail-panel"]').waitFor({ state: 'detached' });
      console.log('PASS mobile overlay open/close and thumbnail selection');
    } finally {
      await mobileContext.close();
    }

    assert.deepEqual(errors, [], 'the isolated page has no browser runtime errors');
    assert.deepEqual(mutations, [], 'the isolated test sends no mutating API request');
  } finally {
    await desktop.close();
    await browser.close();
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
