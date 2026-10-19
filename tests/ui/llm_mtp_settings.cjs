// Mounted Integrations controls; synthetic origin, no live settings or inference.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const playwright = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');
const saluki = 'Kujira/Underdog-Saluki-27B-1.0-MTP-Abliterated-GGUF';
const gemma = 'Abhiray/gemma-4-E4B-it-heretic-GGUF';
const models = [
  {id: saluki, label: 'Underdog Saluki', size_hint: '12 GB VRAM', provider: 'local', supports_mtp: true},
  {id: gemma, label: 'Gemma', size_hint: '8 GB VRAM', provider: 'local', supports_mtp: false},
];

(async () => {
  const bundle = await esbuild.build({stdin: {contents: `
    import React from 'react'; import {createRoot} from 'react-dom/client';
    import {useStore} from './src/stores/useStore';
    import {ServicesSettingsPanel} from './src/components/SettingsDrawer/ServicesSettingsPanel';
    window.store = useStore;
    window.mount = () => createRoot(document.getElementById('root')).render(<ServicesSettingsPanel/>);
  `, resolveDir: path.join(root, 'ui'), loader: 'tsx'}, bundle: true, write: false, jsx: 'automatic',
    define: {'process.env.NODE_ENV': '"development"'}, logLevel: 'silent'});
  const browser = await playwright.chromium.launch({headless: true,
    ...(process.platform === 'win32' ? {executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'} : {})});
  const output = path.join(root, '.codex-tmp/mtp-20261014');
  fs.mkdirSync(output, {recursive: true});
  try {
    for (const width of [1100, 390]) {
      const page = await browser.newPage({viewport: {width, height: 900}});
      const errors = [], requests = [];
      let config = {llm_provider: 'local', llm_model_id: saluki, llm_device: 'cuda',
        llm_mtp_acceleration: false, enhance_llm_model_id: '', enhance_llm_device: 'cuda',
        show_experimental: false, nsfw_mode: false};
      page.on('pageerror', error => errors.push(error.message));
      await page.route('**/*', async route => {
        const pathname = new URL(route.request().url()).pathname;
        if (!pathname.startsWith('/api/')) return route.fulfill({contentType: 'text/html',
          body: '<body style="background:#111;color:#eee"><main id="root" style="max-width:430px;margin:auto;padding:16px"></main></body>'});
        requests.push({path: pathname, method: route.request().method()});
        let body = {};
        if (pathname === '/api/v1/services-config') {
          if (route.request().method() === 'PUT') {
            const partial = route.request().postDataJSON(); config = {...config, ...partial};
            body = {status: 'ok', updated: partial};
          } else body = config;
        } else if (pathname === '/api/v1/llm/models') body = {models};
        return route.fulfill({contentType: 'application/json', body: JSON.stringify(body)});
      });
      await page.goto('http://llm-mtp-settings.test');
      await page.addScriptTag({content: bundle.outputFiles[0].text});
      const assets = path.join(root, 'ui/dist/assets');
      await page.addStyleTag({content: fs.readFileSync(path.join(assets, fs.readdirSync(assets).find(file => file.endsWith('.css'))), 'utf8')});
      await page.evaluate(({config, models, saluki}) => {
        window.store.setState({servicesConfig: config, servicesConfigLoading: false, llmModels: models,
          llmStatus: {loaded: true, model_id: saluki, device: 'cuda', provider: 'local', mtp_acceleration: false}});
        window.mount();
      }, {config, models, saluki});
      const setting = page.getByTestId('llm-mtp-setting');
      const toggle = setting.getByRole('checkbox', {name: /^MTP acceleration/});
      await toggle.waitFor();
      assert.equal(await toggle.isChecked(), false, 'MTP defaults off');
      assert.equal(await toggle.isEnabled(), true, 'Eligible local writer exposes the opt-in');
      assert.match(await setting.innerText(), /Experimental/);
      await toggle.click();
      await page.waitForFunction(() => window.store.getState().servicesConfig.llm_mtp_acceleration === true);
      assert.equal(await toggle.isChecked(), true);
      assert.equal(config.llm_mtp_acceleration, true, 'Saving reaches the services API');
      assert.match(await setting.getByTestId('llm-mtp-active-state').innerText(), /MTP off/);
      assert.match(await setting.innerText(), /compatible writer next loads/);
      assert.ok(!requests.some(row => row.path === '/api/v1/llm/load' || row.path === '/api/v1/llm/unload'),
        'Saving the preference never interrupts the loaded writer');

      const modelSelect = page.locator('select').filter({has: page.locator(`option[value="${saluki}"]`)}).first();
      await modelSelect.selectOption(gemma);
      await page.waitForFunction(gemma => window.store.getState().servicesConfig.llm_model_id === gemma, gemma);
      assert.equal(await toggle.isEnabled(), false, 'Weights without MTP cannot enable drafting');
      assert.equal(await toggle.isChecked(), false);
      assert.equal(config.llm_mtp_acceleration, true, 'Changing models retains the compatible-writer preference');
      await modelSelect.selectOption(saluki);
      await page.waitForFunction(saluki => window.store.getState().servicesConfig.llm_model_id === saluki, saluki);
      assert.equal(await toggle.isChecked(), true);
      await page.evaluate(() => window.store.setState({llmStatus: {...window.store.getState().llmStatus, mtp_acceleration: true}}));
      assert.match(await setting.getByTestId('llm-mtp-active-state').innerText(), /MTP on/);
      await toggle.click();
      await page.waitForFunction(() => window.store.getState().servicesConfig.llm_mtp_acceleration === false);
      assert.equal(await toggle.isChecked(), false);
      assert.equal(config.llm_mtp_acceleration, false);
      await setting.scrollIntoViewIfNeeded();
      await page.screenshot({path: path.join(output, `mtp-settings-${width}.png`)});
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'The toggle fits mobile without horizontal overflow');

      await page.evaluate(({gemma, saluki}) => {
        const store = window.store;
        store.setState({servicesConfig: {...store.getState().servicesConfig, llm_model_id: gemma, enhance_llm_model_id: saluki}});
      }, {gemma, saluki});
      assert.equal(await toggle.isEnabled(), true, 'A separate supported enhancement writer can use the preference');
      await page.evaluate(() => window.store.setState({servicesConfig: {...window.store.getState().servicesConfig, llm_provider: 'remote'}}));
      assert.equal(await setting.count(), 0, 'Remote/API writers do not expose local MTP controls');
      assert.deepEqual(errors, []);
      await page.close();
    }
    console.log('Local MTP Integrations: desktop/mobile, opt-in persistence, eligibility, active state, enhancement writer, remote exclusion and no live reload passed');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
