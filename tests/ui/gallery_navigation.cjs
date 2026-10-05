// Exercise real thumbnail clicks against the virtualized feed with every API
// intercepted. Run MAESTRO_NAV_BASELINE=1 to bundle the preserved pre-fix UI.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const { chromium } = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');
const artifacts = path.join(root, '.codex-tmp/gallery-scroll-20261001/tests');
const baseline = process.env.MAESTRO_NAV_BASELINE === '1';
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));

const makeOutput = (index, overrides = {}) => {
  const workspace = overrides.workspace || (index === 5 ? 'north' : index === 85 ? 'south' : 'Saved');
  const name = overrides.name || (index === 5 || index === 85 ? 'duplicate.png' : `asset-${index}.png`);
  const type = overrides.type || (index > 0 && index % 17 === 0 ? 'audio' : 'image');
  return {
    id: `${workspace}/${name}`, name, workspace, type, mode: type, path: `${workspace}/${name}`,
    url: `/api/v1/file/${encodeURIComponent(name)}?workspace=${encodeURIComponent(workspace)}`,
    size: 1024, created_at: 1, favorite: false, metadata_ready: true,
    ...overrides,
  };
};

const bundleContents = [
  "import React from 'react'; import {createRoot} from 'react-dom/client';",
  "import App from './src/App'; import {useStore} from './src/stores/useStore';",
  'window.store = useStore; const idle = async () => {};',
  'window.setup = (files, strict = false) => {',
  '  useStore.setState({loadModels: idle, loadOutputs: idle, loadWorkspaces: idle,',
  '    loadSystemConfig: idle, loadServicesConfig: idle, loadLlmStatus: idle, loadLlmModels: idle,',
  '    loadPipelineList: idle, reconnectJobs: idle, loadSystemStats: idle, loadSystemDetect: idle,',
  "    generationMode: 'image', sidebarMode: 'studio', sidebarOpen: false, models: [], families: [],",
  "    outputs: files, outputsTotal: files.length, selectedOutput: 0, activeWorkspace: 'Saved',",
  "    browsingAllFolders: true, browsingUploads: false, mediaFilter: 'all', jobs: [],",
  '    isGenerating: false, isEnhancing: false, settingsOpen: false, servicesConfig: {},',
  "    params: {...useStore.getState().params, model_type: 'qwen_image_21_7B', image_mode: 1, prompt: 'Gallery navigation regression'}});",
  '  const app = <App/>;',
  '  createRoot(document.getElementById("root")).render(strict ? <React.StrictMode>{app}</React.StrictMode> : app);',
  '};',
].join('\n');

async function buildBundle() {
  const baselineSources = new Map([
    ['ui/src/components/MainContent/MainContent.tsx', '.codex-tmp/gallery-scroll-20261001/MainContent.before.tsx'],
    ['ui/src/components/MainContent/ThumbnailGallery.tsx', '.codex-tmp/gallery-scroll-20261001/ThumbnailGallery.before.tsx'],
  ]);
  const plugins = baseline ? [{
    name: 'preserved-gallery-baseline',
    setup(build) {
      build.onLoad({ filter: /\.tsx$/ }, args => {
        const relative = path.relative(root, args.path).replace(/\\/g, '/');
        const snapshot = baselineSources.get(relative);
        if (!snapshot) return null;
        return { contents: fs.readFileSync(path.join(root, snapshot), 'utf8'), loader: 'tsx' };
      });
    },
  }] : [];
  return esbuild.build({
    stdin: { contents: bundleContents, resolveDir: path.join(root, 'ui'), loader: 'tsx' },
    bundle: true, write: false, jsx: 'automatic',
    define: { 'process.env.NODE_ENV': '"development"' }, logLevel: 'silent', plugins,
  });
}

async function openThumbnails(page, mobile) {
  if (mobile && await page.locator('[data-thumb-index]').count() === 0) {
    await page.locator('button[title="Show thumbnails"]').click();
    await page.locator('[data-thumb-index]').first().waitFor();
  }
}

async function scrollThumbnailTo(page, index, mobile) {
  await openThumbnails(page, mobile);
  await page.evaluate(targetIndex => {
    const first = document.querySelector('[data-thumb-index]');
    let pane = first;
    while (pane && !/(auto|scroll)/.test(getComputedStyle(pane).overflowY)) pane = pane.parentElement;
    if (!pane) throw new Error('Thumbnail scroll area was not found');
    pane.scrollTop = Math.max(0, targetIndex * 51 - pane.clientHeight / 2);
    pane.dispatchEvent(new Event('scroll', { bubbles: true }));
  }, index);
  await page.locator(`[data-thumb-index="${index}"]`).waitFor();
}

async function clickThumbnail(page, index, mobile) {
  await scrollThumbnailTo(page, index, mobile);
  const thumbnail = page.locator(`[data-thumb-index="${index}"]`);
  await thumbnail.click();
}

async function traceNavigation(page, identity) {
  if (process.env.MAESTRO_NAV_TRACE !== '1') return;
  await page.evaluate(expectedId => {
    window.__galleryNavigationTrace = [];
    const sample = frame => {
      const state = window.store.getState();
      const outputs = state.filteredOutputs();
      const index = outputs.findIndex(file => file.id === expectedId);
      const item = document.querySelector(`[data-feed-index="${index}"]`);
      const feed = item && window.galleryScrollParent(item);
      const visible = feed ? [...feed.querySelectorAll('[data-feed-index]')]
        .map(node => Number(node.dataset.feedIndex)) : [];
      const rect = item?.getBoundingClientRect();
      window.__galleryNavigationTrace.push({ frame,
        scrollTop: feed?.scrollTop, itemTop: rect?.top, itemHeight: rect?.height,
        styleTop: item?.style.top, feedWidth: feed?.clientWidth,
        listHeight: item?.parentElement?.style.height,
        mountedCount: visible.length, minIndex: Math.min(...visible), maxIndex: Math.max(...visible),
        selectedIndex: state.selectedOutput,
      });
      if (frame < 59) requestAnimationFrame(() => sample(frame + 1));
    };
    requestAnimationFrame(() => sample(0));
  }, identity);
  await page.waitForFunction(() => window.__galleryNavigationTrace?.length >= 50, null, { timeout: 3000 });
  const trace = await page.evaluate(() => window.__galleryNavigationTrace);
  fs.mkdirSync(artifacts, { recursive: true });
  const tracePath = path.join(artifacts,
    `navigation-trace-${baseline ? 'baseline' : 'current'}-${identity.replace(/[^a-z0-9.-]/gi, '_')}.json`);
  fs.writeFileSync(tracePath, `${JSON.stringify(trace, null, 2)}\n`);
  console.log(`Navigation trace ${identity}: ${tracePath}; first=${JSON.stringify(trace.slice(0, 8))}; last=${JSON.stringify(trace.slice(-4))}`);
}

async function readMetrics(page, identity) {
  return page.evaluate(expectedId => {
    const state = window.store.getState();
    const outputs = state.filteredOutputs();
    const targetIndex = outputs.findIndex(file => file.id === expectedId);
    const output = outputs[targetIndex];
    const item = document.querySelector(`[data-feed-index="${targetIndex}"]`);
    const feed = item && window.galleryScrollParent(item);
    const thumb = document.querySelector(`[data-thumb-index="${targetIndex}"]`);
    const thumbPane = thumb && window.galleryScrollParent(thumb);
    if (!item || !feed) return { targetIndex, selectedId: output?.id, mounted: false };
    const rect = item.getBoundingClientRect();
    const viewport = feed.getBoundingClientRect();
    const visibleHeight = Math.max(0, Math.min(rect.bottom, viewport.bottom) - Math.max(rect.top, viewport.top));
    const thumbRect = thumb?.getBoundingClientRect();
    const thumbViewport = thumbPane?.getBoundingClientRect();
    return {
      targetIndex, targetId: output?.id,
      selectedId: outputs[state.selectedOutput]?.id,
      mounted: true,
      itemTop: rect.top, itemHeight: rect.height,
      feedTop: viewport.top, feedHeight: viewport.height, feedScrollTop: feed.scrollTop,
      visibleHeight,
      visibleRatio: visibleHeight / Math.min(rect.height, feed.clientHeight),
      thumbnailVisible: !!thumbRect && !!thumbViewport
        && Math.min(thumbRect.bottom, thumbViewport.bottom) > Math.max(thumbRect.top, thumbViewport.top),
      thumbnailScrollTop: thumbPane?.scrollTop ?? null,
      feedCenterX: viewport.left + viewport.width / 2,
      feedCenterY: viewport.top + viewport.height / 2,
    };
  }, identity);
}

async function waitForLayout(page, identity, mobile) {
  await openThumbnails(page, mobile);
  try {
    await page.waitForFunction(expectedId => {
    const state = window.store.getState();
    const outputs = state.filteredOutputs();
    const index = outputs.findIndex(file => file.id === expectedId);
    const item = document.querySelector(`[data-feed-index="${index}"]`);
    const thumb = document.querySelector(`[data-thumb-index="${index}"]`);
    const feed = item && window.galleryScrollParent(item);
    const thumbPane = thumb && window.galleryScrollParent(thumb);
    if (!item || !feed || !thumb || !thumbPane) return false;
    const rect = item.getBoundingClientRect();
    const viewport = feed.getBoundingClientRect();
    const visibleHeight = Math.max(0, Math.min(rect.bottom, viewport.bottom) - Math.max(rect.top, viewport.top));
    if (visibleHeight < Math.min(rect.height, feed.clientHeight) * 0.8) return false;
    const thumbRect = thumb.getBoundingClientRect();
    const thumbViewport = thumbPane.getBoundingClientRect();
    // Check the actual feed and thumbnail geometry for several stable frames.
    const snapshot = [feed.scrollTop, rect.top, rect.height]
      .map(value => Math.round(value));
    window.__galleryNavigationTimeline ||= [];
    window.__galleryNavigationTimeline.push({ time: Math.round(performance.now()), snapshot });
    if (window.__galleryNavigationTimeline.length > 40) window.__galleryNavigationTimeline.shift();
    const previous = window.__galleryNavigationSettle;
    const same = previous?.identity === expectedId
      && snapshot.every((value, i) => Math.abs(value - previous.snapshot[i]) <= 1);
    window.__galleryNavigationSettle = {
      identity: expectedId, snapshot, frames: same ? previous.frames + 1 : 0,
    };
    return Math.min(thumbRect.bottom, thumbViewport.bottom) > Math.max(thumbRect.top, thumbViewport.top)
      && window.__galleryNavigationSettle.frames >= 3;
    }, identity, { polling: 'raf', timeout: 10000 });
  } catch (error) {
    console.log(`Layout settle diagnostic ${identity}: ${JSON.stringify(await page.evaluate(expectedId => {
      const state = window.store.getState(), outputs = state.filteredOutputs();
      const index = outputs.findIndex(file => file.id === expectedId);
      const item = document.querySelector(`[data-feed-index="${index}"]`);
      const thumb = document.querySelector(`[data-thumb-index="${index}"]`);
      const feed = item && window.galleryScrollParent(item);
      const thumbPane = thumb && window.galleryScrollParent(thumb);
      const rect = item?.getBoundingClientRect(), viewport = feed?.getBoundingClientRect();
      const thumbRect = thumb?.getBoundingClientRect(), thumbViewport = thumbPane?.getBoundingClientRect();
      return { index, selectedId: outputs[state.selectedOutput]?.id,
        rect: rect && { top: rect.top, height: rect.height },
        itemStyleTop: item?.style.top,
        parentTop: item?.parentElement?.getBoundingClientRect().top,
        feed: feed && { top: viewport.top, height: viewport.height, scrollTop: feed.scrollTop },
        thumb: thumbRect && { top: thumbRect.top, height: thumbRect.height },
        thumbPane: thumbPane && { top: thumbViewport.top, height: thumbViewport.height, scrollTop: thumbPane.scrollTop },
        settle: window.__galleryNavigationSettle, timeline: window.__galleryNavigationTimeline };
    }, identity))}`);
    throw error;
  }
}

async function assertNavigation(page, index, identity, mobile, label) {
  await waitForLayout(page, identity, mobile);
  const metrics = await readMetrics(page, identity);
  assert.equal(metrics.selectedId, identity, `${label}: the clicked asset remains selected`);
  assert.ok(metrics.mounted && metrics.visibleRatio >= 0.9,
    `${label}: the clicked asset is substantially visible in the main feed (${JSON.stringify(metrics)})`);
  assert.equal(metrics.thumbnailVisible, true,
    `${label}: the clicked thumbnail remains visible (${JSON.stringify(metrics)})`);
  assert.equal(metrics.targetIndex, index,
    `${label}: the asset keeps the expected filtered position`);
  console.log(`PASS navigation ${label}: ${JSON.stringify(metrics)}`);
  return metrics;
}

async function manualWheel(page, previousId) {
  const before = await readMetrics(page, previousId);
  await page.mouse.move(before.feedCenterX, before.feedCenterY);
  await page.mouse.wheel(0, Math.max(450, Math.round(before.feedHeight * 0.55)));
  await page.waitForFunction(({ identity, oldTop }) => {
    const outputs = window.store.getState().filteredOutputs();
    const state = window.store.getState();
    const selected = outputs[state.selectedOutput];
    const index = selected ? outputs.findIndex(file => file.id === selected.id) : -1;
    const item = document.querySelector(`[data-feed-index="${index}"]`);
    const feed = item && window.galleryScrollParent(item);
    return !!feed && feed.scrollTop > oldTop + 40 && selected?.id !== identity;
  }, { identity: previousId, oldTop: before.feedScrollTop }, { polling: 'raf', timeout: 5000 });
  const after = await readMetrics(page, await page.evaluate(() => {
    const outputs = window.store.getState().filteredOutputs();
    const state = window.store.getState();
    return outputs[state.selectedOutput]?.id;
  }));
  assert.ok(after.visibleRatio > 0, 'manual feed wheel selection points to a visible card');
  console.log(`PASS manual wheel selection: ${JSON.stringify(after)}`);
}

function metadataFor(name) {
  const longPrompt = name === 'asset-41.png'
    ? Array.from({ length: 32 }, (_, index) => `delayed detail marker line ${index + 1}: a quiet lake under the northern lights, with layered reflections and distant mountains`).join('\n')
    : `Prompt for ${name}`;
  return { source: 'sidecar', params: { model_type: 'qwen_image_21_7B', prompt: longPrompt, seed: 42 } };
}

async function runScenario(browser, bundle, css, files, scenario) {
  const context = await browser.newContext({
    viewport: { width: scenario.width, height: scenario.height },
    deviceScaleFactor: scenario.mobile ? 2 : 1,
    hasTouch: !!scenario.mobile,
    isMobile: !!scenario.mobile,
  });
  const page = await context.newPage();
  page.setDefaultTimeout(8000);
  const errors = [];
  const mutations = [];
  page.on('pageerror', error => errors.push(error.stack || error.message));
  await page.addInitScript(() => {
    localStorage.setItem('maestro_welcome_seen_v1', '1');
    window.galleryScrollParent = node => {
      for (let element = node; element; element = element.parentElement) {
        if (/(auto|scroll)/.test(getComputedStyle(element).overflowY) && element.clientHeight > 100) return element;
      }
      return null;
    };
  });
  await page.route('**/*', async route => {
    const request = route.request();
    const url = new URL(request.url());
    const json = body => route.fulfill({ contentType: 'application/json', body: JSON.stringify(body) });
    if (['POST', 'PUT', 'PATCH', 'DELETE'].includes(request.method())) mutations.push(`${request.method()} ${url.pathname}`);
    if (url.pathname === '/') return route.fulfill({ contentType: 'text/html', body:
      '<meta name="viewport" content="width=device-width,initial-scale=1"><style>html,body,#root{height:100%;margin:0}</style><link rel="stylesheet" href="/app.css"><div id="root"></div><script src="/bundle.js"></script>' });
    if (url.pathname === '/bundle.js') return route.fulfill({ contentType: 'application/javascript', body: bundle.outputFiles[0].text });
    if (url.pathname === '/app.css') return route.fulfill({ contentType: 'text/css', body: css });
    if (url.pathname.includes('/outputs/') && url.pathname.endsWith('/metadata')) {
      const name = decodeURIComponent(url.pathname.split('/').at(-2));
      if (name === 'asset-41.png') await pause(1200);
      return json(metadataFor(name));
    }
    if (url.pathname.includes('/file/')) {
      return route.fulfill({ contentType: 'image/svg+xml', body:
        '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360"><rect width="640" height="360" fill="#334155"/></svg>' });
    }
    return json({ checks: [], downloads: [], jobs: [], items: [], entries: [], queue: [], pipelines: [],
      recipes: [], loras: [], models: [], families: [], presets: [], capabilities: {}, outputs: [], total: 0 });
  });

  try {
    await page.goto('http://gallery-navigation.test/');
    await page.evaluate(({ outputFiles, strict }) => window.setup(outputFiles, strict),
      { outputFiles: files, strict: scenario.strict });
    await page.locator('[data-feed-index="0"]').waitFor();
    if (scenario.mobile) await openThumbnails(page, true);
    else await page.locator('[data-thumb-index]').first().waitFor();
    console.log(`Scenario ${scenario.name}: feed=${JSON.stringify(await page.evaluate(() => {
      const item = document.querySelector('[data-feed-index="0"]');
      const feed = window.galleryScrollParent(item);
      return { width: feed.clientWidth, height: feed.clientHeight };
    }))}`);

    if (scenario.baseline) {
      await clickThumbnail(page, 5, false);
      await assertNavigation(page, 5, files[5].id, false, 'baseline-interior-short-card');
    } else if (scenario.mobile) {
      for (const index of [5, 120, 0, 149]) {
        await clickThumbnail(page, index, true);
        await assertNavigation(page, index, files[index].id, true, `${scenario.name}-index-${index}`);
      }
    } else if (scenario.strict) {
      await clickThumbnail(page, 5, false);
      await assertNavigation(page, 5, files[5].id, false, 'strict-mode-short-card');
      await manualWheel(page, files[5].id);
    } else {
      for (const index of [5, 120, 7]) {
        await clickThumbnail(page, index, false);
        if (index === 120) await traceNavigation(page, files[index].id);
        await assertNavigation(page, index, files[index].id, false, `desktop-index-${index}`);
      }
      await manualWheel(page, files[7].id);
      for (const index of [0, 149]) {
        await clickThumbnail(page, index, false);
        await assertNavigation(page, index, files[index].id, false, `desktop-edge-${index}`);
      }

      // A real details toggle opens before its delayed metadata response. The
      // prompt then adds height while the selected card stays in the feed.
      const delayedMetadata = page.waitForRequest(request => request.url().includes('/outputs/asset-41.png/metadata'));
      await clickThumbnail(page, 41, false);
      await waitForLayout(page, files[41].id, false);
      const detailCard = page.locator('[data-feed-index="41"]');
      await delayedMetadata;
      await detailCard.getByRole('button', { name: 'Show media details', exact: true }).click();
      const beforeHeight = await detailCard.evaluate(element => element.getBoundingClientRect().height);
      await page.waitForFunction(() => document.querySelector('[data-feed-index="41"]')
        ?.textContent?.includes('delayed detail marker line 32'));
      await page.waitForTimeout(80);
      const afterHeight = await page.locator('[data-feed-index="41"]')
        .evaluate(element => element.getBoundingClientRect().height);
      assert.ok(afterHeight > beforeHeight + 80,
        `delayed metadata expands the actual card (${beforeHeight}px -> ${afterHeight}px)`);
      await assertNavigation(page, 41, files[41].id, false, 'delayed-details-height-change');

      await clickThumbnail(page, 120, false);
      await assertNavigation(page, 120, files[120].id, false, 'navigation-after-expanded-card');

      // Successive trusted clicks must leave the last clicked identity active.
      await clickThumbnail(page, 25, false);
      await clickThumbnail(page, 104, false);
      await assertNavigation(page, 104, files[104].id, false, 'rapid-repeated-clicks');

      // Appending preserves the selected file. Reordering moves the two
      // same-name files across the list while keeping the clicked workspace's
      // identity selected and visible.
      const northId = files[5].id;
      const southId = files[85].id;
      await clickThumbnail(page, 5, false);
      await waitForLayout(page, northId, false);
      // Insert a new row between pointer-down and pointer-up. The button's
      // displayed index changes, while the clicked output identity stays the
      // same; this catches stale index closures at the real input boundary.
      await scrollThumbnailTo(page, 5, false);
      const clickedThumbnail = page.locator('[data-thumb-index="5"]');
      const thumbnailBox = await clickedThumbnail.boundingBox();
      await page.mouse.move(thumbnailBox.x + thumbnailBox.width / 2, thumbnailBox.y + thumbnailBox.height / 2);
      await page.mouse.down();
      const prepended = makeOutput(151, { name: 'inserted.png', workspace: 'Incoming' });
      await page.evaluate(file => {
        const state = window.store.getState();
        const outputs = [file, ...state.outputs];
        window.store.setState({ outputs, outputsTotal: outputs.length });
      }, prepended);
      await page.mouse.up();
      await assertNavigation(page, 6, northId, false, 'identity-survives-pointerdown-list-insert');

      await page.evaluate(appended => {
        const state = window.store.getState();
        const outputs = [...state.outputs, appended];
        window.store.setState({ outputs, outputsTotal: outputs.length });
      }, makeOutput(150, { name: 'appended.png', workspace: 'Later' }));
      await waitForLayout(page, northId, false);
      assert.equal((await readMetrics(page, northId)).selectedId, northId, 'append preserves selected workspace identity');

      await page.evaluate(({ north, south }) => {
        const state = window.store.getState();
        const first = state.outputs.find(file => file.id === north);
        const second = state.outputs.find(file => file.id === south);
        const reordered = [second, ...state.outputs.filter(file => file.id !== north && file.id !== south), first];
        window.store.setState({ outputs: reordered, outputsTotal: reordered.length });
      }, { north: northId, south: southId });
      await waitForLayout(page, northId, false);
      let reorderedMetrics = await readMetrics(page, northId);
      assert.equal(reorderedMetrics.selectedId, northId, 'reordering preserves the clicked same-name workspace asset');
      assert.notEqual(reorderedMetrics.targetIndex, 0, 'the other workspace duplicate occupies a different position');
      assert.equal(reorderedMetrics.targetId, northId);
      await assertNavigation(page, reorderedMetrics.targetIndex, northId, false, 'reordered-workspace-identity');

      await page.evaluate(({ north, south }) => {
        const state = window.store.getState();
        const outputs = state.outputs.map(file => ({
          ...file, favorite: file.id === north || file.id === south,
        }));
        const filtered = outputs.filter(file => file.favorite);
        window.store.setState({ outputs, mediaFilter: 'favorites', selectedOutput: 0 });
      }, { north: northId, south: southId });
      await waitForLayout(page, southId, false);
      const resetMetrics = await readMetrics(page, southId);
      assert.equal(resetMetrics.selectedId, southId, 'a filter scope change resets to the first remaining result');
      await clickThumbnail(page, 1, false);
      await assertNavigation(page, 1, northId, false, 'filtered-same-name-workspace-identity');
      const filteredMetrics = await readMetrics(page, northId);
      assert.equal(filteredMetrics.selectedId, northId, 'filtering retains a clicked asset that remains in the result set');
      assert.equal(filteredMetrics.targetIndex, 1, 'filter remaps the selected same-name asset by workspace identity');
      assert.equal(filteredMetrics.thumbnailVisible, true, 'the selected duplicate remains visible in filtered thumbnails');

      await page.setViewportSize({ width: 1024, height: 900 });
      await waitForLayout(page, northId, false);
      assert.equal((await readMetrics(page, northId)).selectedId, northId, 'resizing preserves the selected identity');
    }

    assert.deepEqual(errors, [], 'the isolated page has no browser runtime errors');
    assert.deepEqual(mutations, [], 'the isolated test sends no mutating API request');
    console.log(`PASS gallery navigation scenario: ${scenario.name}`);
  } catch (error) {
    const filename = `navigation-${scenario.name.replace(/[^a-z0-9-]/gi, '-')}-failure.png`;
    await page.screenshot({ path: path.join(artifacts, filename), fullPage: false });
    error.message += `\nBrowser errors: ${JSON.stringify(errors)}\nMutation requests: ${JSON.stringify(mutations)}`;
    throw error;
  } finally {
    await context.close();
  }
}

async function main() {
  fs.mkdirSync(artifacts, { recursive: true });
  const bundle = await buildBundle();
  const assetDirectory = path.join(root, 'ui/dist/assets');
  const cssName = fs.readdirSync(assetDirectory).find(name => name.endsWith('.css'));
  const css = fs.readFileSync(path.join(assetDirectory, cssName), 'utf8');
  const files = Array.from({ length: 150 }, (_, index) => makeOutput(index));
  const browser = await chromium.launch({ headless: true, ...(process.platform === 'win32'
    ? { executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe' } : {}) });
  const scenarios = baseline
    ? [{ name: 'baseline-short-selection', width: 1280, height: 1450, strict: false, baseline: true }]
    : [
      { name: 'desktop', width: 1280, height: 1450, strict: false },
      { name: 'desktop-strict', width: 1280, height: 1450, strict: true },
      { name: 'mobile', width: 390, height: 844, strict: true, mobile: true },
    ];
  try {
    for (const scenario of scenarios) await runScenario(browser, bundle, css, files, scenario);
  } finally {
    await browser.close();
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
