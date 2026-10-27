// Real Settings/navigation/store, mocked HTTP only: no generation or downloads.
const assert = require('node:assert/strict');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const {chromium} = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');

(async () => {
  const bundle = await esbuild.build({
    absWorkingDir: path.join(root, 'ui'),
    stdin: {contents: `
      import React from 'react'; import {createRoot} from 'react-dom/client';
      import {useStore} from './src/stores/useStore';
      import {WorldsButton} from './src/components/AppModeNavigation';
      import {ServicesSettingsPanel} from './src/components/SettingsDrawer/ServicesSettingsPanel';
      window.store = useStore;
      function Harness() {
        const open = useStore(s => s.worldsOpen);
        const setOpen = useStore(s => s.setWorldsOpen);
        return <><WorldsButton open={open} onClick={() => setOpen(true)}/><ServicesSettingsPanel/></>;
      }
      createRoot(document.getElementById('root')).render(<Harness/>);
    `, resolveDir: path.join(root, 'ui'), loader: 'tsx'},
    bundle: true, write: false, jsx: 'automatic', logLevel: 'silent',
    define: {'process.env.NODE_ENV': '"test"'},
  });
  const browser = await chromium.launch({headless: true,
    ...(process.platform === 'win32' ? {executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'} : {}),
    args: ['--disable-gpu'],
  });
  try {
    for (const width of [1200, 390]) {
      const context = await browser.newContext({viewport: {width, height: 900}});
      const page = await context.newPage();
      const errors = [], writes = [];
      let failSave = false;
      // Existing installations can already expose other beta features. Missing
      // Worlds preferences must still stay off, independently of that choice.
      let saved = {llm_provider: 'local', llm_model_id: '', llm_device: 'cpu',
        show_experimental: true, auto_performance: true, nsfw_mode: false};
      page.on('pageerror', error => errors.push(error.message));
      await page.route('**/*', async route => {
        const request = route.request();
        const key = new URL(request.url()).pathname;
        if (key === '/') return route.fulfill({contentType: 'text/html', body: '<div id="root"></div>'});
        if (key === '/api/v1/services-config') {
          if (request.method() === 'PUT') {
            const body = request.postDataJSON();
            writes.push({path: key, body});
            if (failSave) return route.fulfill({status: 500, json: {detail: 'Could not save setting.'}});
            saved = {...saved, ...body};
            return route.fulfill({json: {status: 'ok'}});
          }
          return route.fulfill({json: saved});
        }
        if (request.method() !== 'GET') writes.push({path: key, body: request.postDataJSON()});
        return route.fulfill({json: {items: [], models: [], jobs: [], outputs: []}});
      });
      async function boot() {
        await page.goto('http://worlds-setting.test');
        await page.addScriptTag({content: bundle.outputFiles[0].text});
        await page.evaluate(() => window.store.getState().loadServicesConfig());
      }
      await page.goto('http://worlds-setting.test');
      await page.addScriptTag({content: bundle.outputFiles[0].text});
      const worlds = page.getByRole('button', {name: 'Open Worlds'});
      assert.equal(await worlds.count(), 0, 'No navigation flash before settings load');
      await page.evaluate(() => window.store.getState().setWorldsOpen(true));
      assert.equal(await page.evaluate(() => window.store.getState().worldsOpen), false);
      await page.evaluate(() => window.store.getState().loadServicesConfig());
      const toggle = page.getByRole('switch', {name: 'Worlds mode (Experimental)'});
      await toggle.waitFor();
      assert.equal(await toggle.getAttribute('aria-checked'), 'false', 'Missing preference defaults off');
      assert.equal(await worlds.count(), 0, 'Other beta features do not enable Worlds');

      await toggle.click();
      await worlds.waitFor();
      assert.equal(await toggle.getAttribute('aria-checked'), 'true');
      assert.match(await worlds.getAttribute('title'), /Experimental/);
      assert.deepEqual(writes.at(-1), {path: '/api/v1/services-config', body: {worlds_enabled: true}});
      await worlds.click();
      assert.equal(await page.evaluate(() => window.store.getState().worldsOpen), true);

      await boot();
      await worlds.waitFor();
      assert.equal(await toggle.getAttribute('aria-checked'), 'true', 'Enabled choice survives reload');
      assert.equal(await page.evaluate(() => window.store.getState().worldsOpen), false, 'The panel itself is transient');
      await worlds.click();
      await toggle.click();
      await worlds.waitFor({state: 'detached'});
      assert.equal(await page.evaluate(() => window.store.getState().worldsOpen), false, 'Disabling closes the panel');
      assert.deepEqual(writes.at(-1).body, {worlds_enabled: false});
      await boot();
      await toggle.waitFor();
      assert.equal(await toggle.getAttribute('aria-checked'), 'false', 'Disabled choice survives reload');
      assert.equal(await worlds.count(), 0);

      failSave = true;
      await toggle.click();
      await page.getByRole('alert').waitFor();
      assert.equal(await toggle.getAttribute('aria-checked'), 'false', 'Failed saves do not enable the feature');
      assert.equal(await worlds.count(), 0);
      assert.equal(saved.auto_performance, true);
      assert.equal(saved.nsfw_mode, false);
      assert.equal(saved.show_experimental, true);
      assert.ok(writes.every(write => write.path === '/api/v1/services-config'), 'No generation, cancellation or other setting writes');
      assert.deepEqual(errors, []);
      await context.close();
    }
    console.log('Desktop/phone Worlds default-off, opt-in, reload, disable/close, failed-save and unrelated-setting checks passed.');
  } finally {
    await browser.close();
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
