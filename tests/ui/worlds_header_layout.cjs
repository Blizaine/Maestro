// Run: node tests/ui/worlds_header_layout.cjs
// Real SidebarHeader + Video workflow picker, production Tailwind CSS, mocked HTTP.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const uiRoot = path.join(root, 'ui');
const esbuild = require(path.join(uiRoot, 'node_modules/esbuild'));
const {chromium} = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');

async function buildHarness() {
  const source = `
    import React from 'react';
    import {createRoot} from 'react-dom/client';
    import {useStore} from './src/stores/useStore';
    import {SidebarHeader} from './src/components/Sidebar/SidebarHeader';
    import {VideoWorkflowSelector} from './src/components/Sidebar/VideoWorkflowSelector';
    import {SidebarLayoutContext} from './src/components/Sidebar/SidebarPanels';
    import {SettingsDrawer} from './src/components/SettingsDrawer/SettingsDrawer';

    window.store = useStore;
    function Harness() {
      const mobile = window.matchMedia('(max-width: 767px)').matches;
      const sidebarOpen = useStore(s => s.sidebarOpen);
      const setSidebarOpen = useStore(s => s.setSidebarOpen);
      const setWorldsOpen = useStore(s => s.setWorldsOpen);
      const toggleSettings = useStore(s => s.toggleSettings);
      const [sidebar, setSidebar] = React.useState(null);
      const layout = React.useMemo(() => ({sidebar, settings:null}), [sidebar]);
      return <>
        <div className="relative flex h-full min-h-0 w-full overflow-hidden">
          <aside ref={setSidebar} data-testid="sidebar" aria-hidden={mobile ? !sidebarOpen : undefined}
            className={mobile
              ? 'maestro-sidebar fixed top-0 h-dvh w-[380px] max-w-[94vw] bg-bg-secondary border-r border-border z-50 flex flex-col transition-[left] duration-300 ease-in-out ' + (sidebarOpen ? 'left-0' : '-left-full')
              : 'maestro-sidebar relative w-[420px] h-full shrink-0 bg-bg-secondary border-r border-border flex flex-col'}>
            <SidebarHeader mobile={mobile} onSettings={() => toggleSettings()} onClose={() => setSidebarOpen(false)} />
            <div className="min-h-0 flex-1 p-3">
              <SidebarLayoutContext.Provider value={layout}>
              <VideoWorkflowSelector onOpenWorlds={() => setWorldsOpen(true)} />
              </SidebarLayoutContext.Provider>
            </div>
          </aside>
          <main className="relative min-w-0 flex-1 h-full">
            <button data-testid="gallery-control" type="button" aria-label="Adjacent gallery control"
              className="absolute z-0 rounded border border-border px-1 py-1 text-[10px]"
              style={{left: mobile ? 'calc(min(380px, 94vw) + 2px)' : '8px', top: '24px'}}>Gallery</button>
          </main>
        </div>
        <SettingsDrawer />
      </>;
    }
    window.mountHeaderHarness = worldsEnabled => {
      useStore.setState({
        servicesConfig: {llm_provider:'local', worlds_enabled:worldsEnabled, show_experimental:true},
        servicesConfigLoading: false,
        systemConfig: {app_version:'3.3.0'},
        settingsOpen: false, settingsTab: 'integrations', sidebarOpen: true,
        sidebarMode: 'studio', generationMode: 'video', studioVideoWorkflow: 'frames',
      });
      createRoot(document.getElementById('root')).render(<Harness />);
    };
  `;
  const bundle = await esbuild.build({
    stdin: {contents: source, resolveDir: uiRoot, loader: 'tsx'},
    bundle: true, write: false, jsx: 'automatic',
    define: {'process.env.NODE_ENV': '"test"'}, logLevel: 'silent',
  });
  const assetsDir = path.join(uiRoot, 'dist/assets');
  const cssFile = fs.readdirSync(assetsDir).filter(name => /^index-.*\.css$/.test(name))
    .map(name => path.join(assetsDir, name))
    .sort((a, b) => fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs)[0];
  assert.ok(cssFile, 'Load CSS from the public UI build');
  return {js: bundle.outputFiles[0].text, css: fs.readFileSync(cssFile, 'utf8')};
}

async function capture(page, name) {
  if (!process.env.MAESTRO_HEADER_SHOTS) return;
  const directory = path.resolve(process.env.MAESTRO_HEADER_SHOTS);
  fs.mkdirSync(directory, {recursive: true});
  await page.screenshot({path: path.join(directory, `${name}.png`), fullPage: true});
}

function htmlFor(js, css, worldsEnabled) {
  return `<!doctype html><html><head><meta charset="utf-8"><style>${css}</style>
    <style>html,body,#root{width:100%;height:100%;margin:0}body{overflow:hidden}*{transition-duration:0s!important}</style>
    </head><body><div id="root"></div><script>${js.replace(/<\/script/gi, '<\\/script')}</script>
    <script>window.mountHeaderHarness(${JSON.stringify(worldsEnabled)});</script></body></html>`;
}

async function headerMetrics(page, name) {
  const header = page.getByTestId('sidecar-header');
  const button = header.getByRole('button', {name, exact: true});
  await button.waitFor({state: 'visible'});
  const metrics = await button.evaluate(element => {
    const sidebar = document.querySelector('[data-testid="sidebar"]').getBoundingClientRect();
    const gallery = document.querySelector('[data-testid="gallery-control"]').getBoundingClientRect();
    const rect = element.getBoundingClientRect();
    const hit = document.elementFromPoint(rect.left + rect.width / 2, rect.top + rect.height / 2);
    return {
      left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom,
      sidebarLeft: sidebar.left, sidebarRight: sidebar.right,
      hit: hit === element || element.contains(hit),
      overlapsGallery: rect.left < gallery.right && rect.right > gallery.left
        && rect.top < gallery.bottom && rect.bottom > gallery.top,
    };
  });
  assert.ok(metrics.left >= metrics.sidebarLeft - 1, `${name} stays inside the sidebar: ${JSON.stringify(metrics)}`);
  assert.ok(metrics.right <= metrics.sidebarRight + 1, `${name} stays inside the sidebar: ${JSON.stringify(metrics)}`);
  assert.equal(metrics.overlapsGallery, false, `${name} does not overlap adjacent gallery controls`);
  assert.equal(metrics.hit, true, `${name} wins hit-testing at its center`);
  return button;
}

async function main() {
  const {js, css} = await buildHarness();
  const browser = await chromium.launch({headless:true,
    ...(process.platform === 'win32' ? {executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'} : {}),
    args:['--disable-gpu'],
  });
  const errors = [], writes = [];
  try {
    for (const width of [1280, 320, 390, 430]) {
      const context = await browser.newContext({viewport:{width, height:900}});
      const page = await context.newPage();
      page.on('pageerror', error => errors.push(error.message));
      await page.route('**/*', route => {
        const request = route.request();
        const pathname = new URL(request.url()).pathname;
        if (pathname === '/') return route.fulfill({contentType:'text/html', body:htmlFor(js, css, width !== 1280)});
        if (request.method() !== 'GET') writes.push({method:request.method(), path:pathname});
        if (pathname === '/maestro-home-icon-orange.png') {
          return route.fulfill({contentType:'image/png', body:fs.readFileSync(path.join(uiRoot, 'dist/maestro-home-icon-orange.png'))});
        }
        return route.fulfill({json:{}});
      });
      await page.goto('http://worlds-header.test/');
      const header = page.getByTestId('sidecar-header');
      assert.equal(await header.getByRole('button', {name:/^Worlds/}).count(), 0,
        `Worlds never occupies a header row at ${width}px, even when enabled`);
      const height = (await header.boundingBox()).height;
      assert.ok(height <= 72, `The header remains a single compact row at ${width}px (height ${height})`);
      assert.equal(await page.getByTestId('sidebar').getByText('v3.3.0', {exact:true}).count(), 1);
      for (const name of ['Director', 'Studio', 'Editor', width === 1280 ? 'Settings' : 'Close sidecar']) {
        await headerMetrics(page, name);
      }

      const workflow = page.locator('button[title^="Text, first / last"]');
      if (width === 1280) {
        await workflow.click();
        const disabledDialog = page.getByRole('dialog', {name:'Choose a workflow'});
        assert.equal(await disabledDialog.getByRole('button', {name:/^Worlds/}).count(), 0,
          'The Video menu hides Worlds before opt-in');
        await workflow.click();
        await page.evaluate(() => window.store.setState({servicesConfig: {
          ...window.store.getState().servicesConfig, worlds_enabled: true,
        }}));
      }
      await workflow.click();
      const dialog = page.getByRole('dialog', {name:'Choose a workflow'});
      const worlds = dialog.getByRole('button', {name:/^Worlds/});
      await worlds.waitFor();
      assert.equal(await header.getByRole('button', {name:/^Worlds/}).count(), 0,
        'The enabled Worlds action stays in the Video workflow menu, outside the header');
      assert.ok((await header.boundingBox()).height <= 72, 'Opt-in does not add a header row');
      for (const name of ['Director', 'Studio', 'Editor', width === 1280 ? 'Settings' : 'Close sidecar']) {
        await headerMetrics(page, name);
      }
      if (width === 1280) {
        await capture(page, 'desktop-video-workflow-worlds');
        await workflow.click();
        await header.getByRole('button', {name:'Settings', exact:true}).click();
        await page.getByRole('heading', {name:'Settings', exact:true}).waitFor();
        assert.equal(await page.getByRole('switch', {name:'Worlds mode (Experimental)'}).getAttribute('aria-checked'), 'true');
      } else {
        if (width === 320) await capture(page, 'mobile-320-header');
        await workflow.click();
        await header.getByRole('button', {name:'Close sidecar', exact:true}).click();
        await page.waitForFunction(() => window.store.getState().sidebarOpen === false);
        assert.equal(await page.getByTestId('sidebar').getAttribute('aria-hidden'), 'true');
      }
      assert.deepEqual(writes, [], 'Header and menu checks make no settings or generation writes');
      assert.deepEqual(errors, [], 'The real header and workflow picker have no browser runtime errors');
      await context.close();
    }
    console.log('Worlds stays in the Video workflow menu; the header remains compact and clickable at desktop 420px and mobile 320/390/430px.');
  } finally {
    await browser.close();
  }
}

main().catch(error => {console.error(error); process.exitCode = 1;});
