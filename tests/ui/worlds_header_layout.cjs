// Run: node tests/ui/worlds_header_layout.cjs
// Exercises the real shared SidebarHeader and Settings drawer with production Tailwind CSS.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const uiRoot = path.join(root, 'ui');
const esbuild = require(path.join(uiRoot, 'node_modules/esbuild'));
const {chromium} = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');

async function captureHeader(page, name) {
  if (!process.env.MAESTRO_HEADER_SHOTS) return;
  const directory = path.resolve(process.env.MAESTRO_HEADER_SHOTS);
  fs.mkdirSync(directory, {recursive: true});
  await page.getByTestId('sidecar-header').screenshot({path: path.join(directory, `${name}.png`)});
}

const services = {
  llm_provider: 'local',
  llm_model_id: '',
  llm_device: 'cpu',
  show_experimental: true,
  auto_performance: true,
  nsfw_mode: false,
  worlds_enabled: false,
};

async function buildHarness() {
  const source = `
    import React from 'react';
    import {createRoot} from 'react-dom/client';
    import {useStore} from './src/stores/useStore';
    import {SidebarHeader} from './src/components/Sidebar/SidebarHeader';
    import {SettingsDrawer} from './src/components/SettingsDrawer/SettingsDrawer';
    import {AppModeToggle, MaestroBrand, WorldsButton} from './src/components/AppModeNavigation';
    import {Settings} from 'lucide-react';

    window.store = useStore;

    // A test-only copy of the old, single-row desktop header. It is kept
    // offscreen so its measured overflow proves the geometry assertion catches
    // the original failure, while all passing checks exercise SidebarHeader.
    function LegacyHeaderProbe() {
      return <aside data-testid="legacy-sidebar" aria-hidden="true" style={{position:'fixed', left:'-10000px', top:0, width:420, visibility:'hidden'}}>
        <div className="flex h-14 shrink-0 items-center justify-between border-b border-border px-4">
          <MaestroBrand />
          <div data-testid="legacy-actions" className="flex items-center gap-2">
            <AppModeToggle />
            <WorldsButton size="sm" onClick={() => {}} />
            <button type="button" className="p-1.5 rounded-lg"><Settings size={16} /></button>
          </div>
        </div>
      </aside>;
    }

    function Harness() {
      const mobile = window.matchMedia('(max-width: 767px)').matches;
      const sidebarOpen = useStore(s => s.sidebarOpen);
      const worldsOpen = useStore(s => s.worldsOpen);
      const setSidebarOpen = useStore(s => s.setSidebarOpen);
      const setWorldsOpen = useStore(s => s.setWorldsOpen);
      const toggleSettings = useStore(s => s.toggleSettings);
      return <>
        <div className="relative flex h-full min-h-0 w-full overflow-hidden">
          <aside
            data-testid="sidebar"
            aria-hidden={mobile ? !sidebarOpen : undefined}
            className={mobile
              ? 'maestro-sidebar fixed top-0 h-dvh w-[380px] max-w-[94vw] bg-bg-secondary border-r border-border z-50 flex flex-col transition-[left] duration-300 ease-in-out ' + (sidebarOpen ? 'left-0' : '-left-full')
              : 'maestro-sidebar relative w-[420px] h-full shrink-0 bg-bg-secondary border-r border-border flex flex-col'}>
            <SidebarHeader
              mobile={mobile}
              onOpenWorlds={() => setWorldsOpen(true)}
              worldsOpen={worldsOpen}
              onSettings={() => toggleSettings()}
              onClose={() => setSidebarOpen(false)}
            />
          </aside>
          <main data-testid="gallery" className="relative min-w-0 flex-1 h-full">
            <button
              type="button"
              data-testid="gallery-control"
              aria-label="Adjacent gallery control"
              className="absolute z-0 rounded border border-border px-1 py-1 text-[10px]"
              style={{left: mobile ? 'calc(min(380px, 94vw) + 2px)' : '8px', top: '24px'}}>
              Gallery
            </button>
          </main>
        </div>
        <SettingsDrawer />
        <LegacyHeaderProbe />
      </>;
    }

    window.mountHeaderHarness = worldsEnabled => {
      useStore.setState({
        servicesConfig: {
          llm_provider: 'local', llm_model_id: '', llm_device: 'cpu',
          show_experimental: true, auto_performance: true, nsfw_mode: false,
          worlds_enabled: worldsEnabled,
        },
        servicesConfigLoading: false,
        systemConfig: {app_version: '3.3.0'},
        settingsOpen: false,
        settingsTab: 'integrations',
        worldsOpen: false,
        sidebarOpen: true,
        sidebarMode: 'studio',
      });
      createRoot(document.getElementById('root')).render(<Harness />);
    };
  `;
  const bundle = await esbuild.build({
    stdin: {contents: source, resolveDir: uiRoot, loader: 'tsx'},
    bundle: true,
    nodePaths: [path.dirname(path.dirname(require.resolve(path.join(uiRoot, 'node_modules/react/package.json'))))],
    write: false,
    jsx: 'automatic',
    format: 'iife',
    define: {'process.env.NODE_ENV': '"test"'},
    logLevel: 'silent',
  });

  const assetsDir = path.join(uiRoot, 'dist/assets');
  const cssFile = fs.readdirSync(assetsDir)
    .filter(name => /^index-.*\.css$/.test(name))
    .map(name => path.join(assetsDir, name))
    .sort((a, b) => fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs)[0];
  assert.ok(cssFile, 'Load the CSS emitted by the public UI build');
  const css = fs.readFileSync(cssFile, 'utf8');
  assert.match(css, /\.flex-wrap/, 'The public build CSS includes wrapping utilities');

  return {js: bundle.outputFiles[0].text, css};
}

function htmlFor(js, css, worldsEnabled) {
  const script = js.replace(/<\/script/gi, '<\\/script');
  return `<!doctype html><html><head><meta charset="utf-8"><style>${css}</style>
    <style>html,body,#root{width:100%;height:100%;margin:0}body{overflow:hidden}*{transition-duration:0s!important}</style>
    </head><body><div id="root"></div><script>${script}</script>
    <script>window.mountHeaderHarness(${JSON.stringify(worldsEnabled)});</script></body></html>`;
}

function setupHttpMocks(page, js, css, saved, requests) {
  return page.route('**/*', async route => {
    const request = route.request();
    const url = new URL(request.url());
    const key = url.pathname;
    if (key === '/' && request.method() === 'GET') {
      return route.fulfill({contentType: 'text/html', body: htmlFor(js, css, saved.worlds_enabled === true)});
    }
    if (key === '/api/v1/services-config') {
      if (request.method() === 'PUT') {
        const body = request.postDataJSON();
        requests.push({method: request.method(), path: key, body});
        Object.assign(saved, body);
        return route.fulfill({json: {status: 'ok', updated: body}});
      }
      requests.push({method: request.method(), path: key});
      return route.fulfill({json: saved});
    }
    if (key === '/maestro-home-icon-orange.png') {
      return route.fulfill({contentType: 'image/png', body: fs.readFileSync(path.join(uiRoot, 'dist/maestro-home-icon-orange.png'))});
    }
    if (request.method() !== 'GET') requests.push({method: request.method(), path: key});
    return route.fulfill({json: {}});
  });
}

async function headerMetrics(page, name) {
  const button = page.getByRole('button', {name, exact: true});
  await button.waitFor({state: 'visible'});
  const metrics = await button.evaluate(element => {
    const sidecar = document.querySelector('[data-testid="sidebar"]');
    const gallery = document.querySelector('[data-testid="gallery-control"]');
    const rect = element.getBoundingClientRect();
    const sidebarRect = sidecar.getBoundingClientRect();
    const galleryRect = gallery.getBoundingClientRect();
    const hit = document.elementFromPoint(rect.left + rect.width / 2, rect.top + rect.height / 2);
    const intersects = rect.left < galleryRect.right && rect.right > galleryRect.left
      && rect.top < galleryRect.bottom && rect.bottom > galleryRect.top;
    return {
      name: element.getAttribute('aria-label') || element.innerText.trim(),
      left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom,
      sidebarLeft: sidebarRect.left, sidebarRight: sidebarRect.right,
      hitTarget: hit === element || element.contains(hit),
      intersectsGallery: intersects,
    };
  });
  assert.ok(metrics.left >= metrics.sidebarLeft - 1, `${name} starts inside the sidebar: ${JSON.stringify(metrics)}`);
  assert.ok(metrics.right <= metrics.sidebarRight + 1, `${name} ends inside the sidebar: ${JSON.stringify(metrics)}`);
  assert.equal(metrics.intersectsGallery, false, `${name} does not overlap adjacent gallery controls: ${JSON.stringify(metrics)}`);
  assert.equal(metrics.hitTarget, true, `${name} wins pointer hit-testing at its center: ${JSON.stringify(metrics)}`);
  return button;
}

async function checkOriginalOverflow(page) {
  const result = await page.getByTestId('legacy-sidebar').evaluate(sidebar => {
    const bounds = sidebar.getBoundingClientRect();
    const controls = sidebar.querySelector('[data-testid="legacy-actions"]');
    const right = controls.getBoundingClientRect().right;
    return {sidebarRight: bounds.right, controlsRight: right, overflow: right > bounds.right + 1};
  });
  assert.equal(result.overflow, true,
    `The legacy single-row header must overflow at 420px with v3.3.0 and Worlds enabled: ${JSON.stringify(result)}`);
}

async function main() {
  const {js, css} = await buildHarness();
  const browser = await chromium.launch({headless: true,
    ...(process.platform === 'win32' ? {executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'} : {}),
    args: ['--disable-gpu'],
  });
  const requests = [];
  const saved = {...services};
  const errors = [];
  try {
    const desktopContext = await browser.newContext({viewport: {width: 1280, height: 900}});
    const desktop = await desktopContext.newPage();
    desktop.on('pageerror', error => errors.push(error.message));
    await setupHttpMocks(desktop, js, css, saved, requests);
    await desktop.goto('http://worlds-header.test/');

    assert.equal(await desktop.getByTestId('sidebar').getByText('v3.3.0', {exact: true}).count(), 1, 'The real Maestro brand gets the overflow-triggering app version');
    assert.equal(await desktop.getByRole('button', {name: 'Open Worlds'}).count(), 0, 'Worlds stays hidden while disabled');
    for (const mode of ['Director', 'Studio', 'Editor']) await headerMetrics(desktop, mode);
    const desktopSettings = await headerMetrics(desktop, 'Settings');
    await desktopSettings.click();
    await desktop.getByRole('heading', {name: 'Settings', exact: true}).waitFor();
    const worldsSwitch = desktop.getByRole('switch', {name: 'Worlds mode (Experimental)'});
    assert.equal(await worldsSwitch.getAttribute('aria-checked'), 'false', 'The real Services settings panel starts disabled');
    await worldsSwitch.click();
    await worldsSwitch.waitFor({state: 'visible'});
    await desktop.waitForFunction(() => window.store.getState().servicesConfig.worlds_enabled === true);
    assert.equal(saved.worlds_enabled, true, 'The mocked settings endpoint persists opt-in');
    assert.equal(await desktop.getByRole('button', {name: 'Open Worlds'}).count(), 1, 'Enabling Worlds adds the real navigation button');
    await desktop.evaluate(() => window.store.getState().setSettingsOpen(false));
    await checkOriginalOverflow(desktop);

    const enabledHeader = ['Director', 'Studio', 'Editor', 'Open Worlds', 'Settings'];
    for (const name of enabledHeader) await headerMetrics(desktop, name);
    await captureHeader(desktop, 'desktop-worlds-enabled');
    const worlds = desktop.getByRole('button', {name: 'Open Worlds', exact: true});
    await worlds.click();
    assert.equal(await worlds.getAttribute('aria-expanded'), 'true', 'Worlds remains hit-test clickable in the fixed header');
    assert.equal(await desktop.evaluate(() => window.store.getState().worldsOpen), true);

    // Reload from the mocked persisted config. The feature remains enabled,
    // while the open panel state is transient.
    await desktop.reload();
    assert.equal(await desktop.getByRole('button', {name: 'Open Worlds'}).count(), 1, 'Opt-in survives a browser reload');
    assert.equal(await desktop.evaluate(() => window.store.getState().worldsOpen), false, 'Reload closes the transient Worlds panel');
    await desktop.getByRole('button', {name: 'Settings', exact: true}).click();
    await desktop.getByRole('heading', {name: 'Settings', exact: true}).waitFor();
    const enabledSwitch = desktop.getByRole('switch', {name: 'Worlds mode (Experimental)'});
    assert.equal(await enabledSwitch.getAttribute('aria-checked'), 'true');
    await enabledSwitch.click();
    await desktop.waitForFunction(() => window.store.getState().servicesConfig.worlds_enabled === false);
    assert.equal(await desktop.getByRole('button', {name: 'Open Worlds'}).count(), 0, 'Disabling from actual Settings removes the header entry');
    await desktop.evaluate(() => window.store.getState().setSettingsOpen(false));
    await desktop.reload();
    assert.equal(await desktop.getByRole('button', {name: 'Open Worlds'}).count(), 0, 'The disabled choice also survives reload');
    assert.equal(await desktop.evaluate(() => window.store.getState().servicesConfig.worlds_enabled), false);
    assert.equal(await desktop.getByRole('button', {name: 'Settings', exact: true}).count(), 1, 'Settings remains available when Worlds is disabled');
    await desktopContext.close();

    // Existing enabled installs are exercised at the narrowest common mobile
    // widths. The adjacent fake gallery control sits just outside the sidecar.
    Object.assign(saved, services, {worlds_enabled: true});
    for (const width of [320, 390, 430]) {
      saved.worlds_enabled = true;
      const context = await browser.newContext({viewport: {width, height: 844}});
      const page = await context.newPage();
      page.on('pageerror', error => errors.push(error.message));
      await setupHttpMocks(page, js, css, saved, requests);
      await page.goto('http://worlds-header.test/');
      assert.equal(await page.getByRole('button', {name: 'Open Worlds', exact: true}).count(), 1, `Worlds is available at ${width}px`);
      assert.equal(await page.getByRole('button', {name: 'Settings', exact: true}).count(), 0, 'The mobile sidecar exposes Close instead of desktop Settings');
      for (const name of ['Director', 'Studio', 'Editor', 'Open Worlds', 'Close sidecar']) {
        await headerMetrics(page, name);
      }
      await captureHeader(page, `mobile-${width}-worlds-enabled`);
      await page.getByRole('button', {name: 'Open Worlds', exact: true}).click();
      assert.equal(await page.getByRole('button', {name: 'Open Worlds', exact: true}).getAttribute('aria-expanded'), 'true');
      await page.getByRole('button', {name: 'Close sidecar', exact: true}).click();
      await page.waitForFunction(() => window.store.getState().sidebarOpen === false);
      assert.equal(await page.getByTestId('sidebar').getAttribute('aria-hidden'), 'true', 'The mobile close action dismisses the sidecar');
      saved.worlds_enabled = false;
      await page.reload();
      assert.equal(await page.getByRole('button', {name: 'Open Worlds', exact: true}).count(), 0);
      for (const name of ['Director', 'Studio', 'Editor', 'Close sidecar']) await headerMetrics(page, name);
      await context.close();
    }

    assert.deepEqual(errors, [], 'The mounted real components produce no browser runtime errors');
    assert.ok(requests.some(request => request.method === 'PUT' && request.path === '/api/v1/services-config'),
      'The real Settings toggle writes only to the mocked Services config endpoint');
    assert.ok(requests.every(request => request.method === 'GET' || (request.method === 'PUT' && request.path === '/api/v1/services-config')),
      `No real settings, generation, or download writes are made: ${JSON.stringify(requests.filter(request => request.method !== 'GET'))}`);
    assert.ok(!requests.some(request => /generate|download|jobs|worlds/i.test(request.path)),
      'The layout regression never calls generation or download endpoints');
    console.log('Worlds header layout checks passed: legacy overflow control, Settings opt-in/disable and reload, desktop 420px, mobile 320/390/430, bounds, overlap, and hit-testing.');
  } finally {
    await browser.close();
  }
}

main().catch(error => {console.error(error); process.exitCode = 1;});
