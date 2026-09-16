// Exercise App's real startup effect and gallery store with delayed HTTP replies.
// Every request is intercepted; no running app, user settings or media are changed.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const playwright = require(process.env.MAESTRO_PLAYWRIGHT || 'C:/Users/bliza/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
const image = workspace => ({name: `${workspace}.png`, workspace, id: `${workspace}/image.png`,
  type: 'image', mode: 'image', path: `${workspace}/image.png`, created_at: 1, size: 50,
  metadata_ready: true, url: `/api/v1/file/${workspace}.png?workspace=${workspace}`});

(async () => {
  const bundle = await esbuild.build({stdin: {contents: [
    "import React from 'react'; import {createRoot} from 'react-dom/client';",
    "import App from './src/App'; import {useStore} from './src/stores/useStore';",
    "window.store = useStore; const idle = async () => {};",
    // Keep gallery hydration real; unrelated model/GPU startup is outside this test.
    "useStore.setState({loadModels: idle, loadSystemConfig: idle, loadServicesConfig: idle, loadLlmStatus: idle, loadLlmModels: idle, loadPipelineList: idle, reconnectJobs: idle, loadSystemStats: idle, loadSystemDetect: idle});",
    "const app = <App/>; createRoot(document.getElementById('root')).render(location.search.includes('strict') ? <React.StrictMode>{app}</React.StrictMode> : app);",
  ].join('\n'), resolveDir: path.join(root, 'ui'), loader: 'tsx'}, bundle: true, write: false,
    jsx: 'automatic', define: {'process.env.NODE_ENV': '"development"'}, logLevel: 'silent'});
  const assets = path.join(root, 'ui/dist/assets');
  const css = fs.readFileSync(path.join(assets, fs.readdirSync(assets).find(name => name.endsWith('.css'))), 'utf8');
  const browser = await playwright.chromium.launch({headless: true,
    ...(process.platform === 'win32' ? {executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'} : {})});
  const failures = [];
  try {
    for (const scenario of [
      {name: 'workspace-first', workspaceDelay: 20, outputDelay: 180},
      {name: 'outputs-first', workspaceDelay: 180, outputDelay: 0},
      {name: 'workspace-first-strict', workspaceDelay: 20, outputDelay: 180},
      {name: 'all-folders-during-startup', workspaceDelay: 300, outputDelay: 0, all: true},
    ]) {
      const page = await browser.newPage({viewport: {width: 1360, height: 900}});
      const errors = [], outputRequests = [];
      page.on('pageerror', error => errors.push(error.stack));
      await page.addInitScript(() => localStorage.setItem('maestro_welcome_seen_v1', '1'));
      await page.route('**/*', async route => {
        const url = new URL(route.request().url());
        const json = body => route.fulfill({contentType: 'application/json', body: JSON.stringify(body)});
        if (url.pathname === '/') return route.fulfill({contentType: 'text/html', body:
          '<meta name="viewport" content="width=device-width,initial-scale=1"><style>html,body,#root{height:100%;margin:0}</style><link rel="stylesheet" href="/app.css"><div id="root"></div><script src="/bundle.js"></script>'});
        if (url.pathname === '/bundle.js') return route.fulfill({contentType: 'application/javascript', body: bundle.outputFiles[0].text});
        if (url.pathname === '/app.css') return route.fulfill({contentType: 'text/css', body: css});
        if (url.pathname === '/api/v1/media-flow/capabilities') return json({
          neural_rendering: {available: false, reason: 'Isolated test'},
          frame_generation: {available: false, factors: [], reason: 'Isolated test'},
        });
        if (url.pathname === '/api/v1/workspaces') {
          await pause(scenario.workspaceDelay);
          return json({active: 'Saved-folder', workspaces: ['default', 'Saved-folder'].map(name => ({name, path: name}))});
        }
        if (url.pathname === '/api/v1/outputs') {
          const workspace = url.searchParams.get('workspace');
          outputRequests.push(workspace);
          await pause(scenario.outputDelay);
          return json({outputs: [image(workspace)], total: 1});
        }
        if (url.pathname.includes('/metadata')) return json({params: {prompt: 'Saved gallery image'}});
        if (url.pathname.includes('/file/')) return route.fulfill({contentType: 'image/svg+xml', body:
          '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360"><rect width="640" height="360" fill="#334155"/></svg>'});
        return json({checks: [], downloads: [], jobs: [], items: [], entries: [], queue: [], pipelines: [], recipes: [], loras: [], models: [], families: [], presets: [], capabilities: {}});
      });
      try {
        await page.goto(`http://gallery-startup.test/?${scenario.name}`);
        if (scenario.all) {
          await page.waitForFunction(() => !!window.store);
          await page.evaluate(() => window.store.getState().switchWorkspace('__all__'));
        }
        const expected = scenario.all ? '__all__' : 'Saved-folder';
        await page.waitForFunction(expected => {
          const state = window.store?.getState();
          return state?.activeWorkspace === 'Saved-folder' && !state.outputsLoading && state.outputs[0]?.workspace === expected;
        }, expected, {timeout: 3000});
        assert.ok(outputRequests.includes(expected), 'requests the restored browsing folder');
        await page.locator('[data-feed-index]').first().waitFor({timeout: 3000});
        assert.equal(await page.locator('[data-feed-index]').count(), 1, 'restored media is rendered');
        assert.deepEqual(errors, [], 'startup has no browser errors');
        if (!scenario.all) {
          outputRequests.length = 0;
          await page.reload();
          await page.waitForFunction(() => window.store?.getState().outputs[0]?.workspace === 'Saved-folder', null, {timeout: 3000});
          assert.ok(outputRequests.includes('Saved-folder'), 'browser refresh reloads the saved folder');
        }
        console.log(`PASS gallery startup: ${scenario.name}`);
      } catch (error) {
        failures.push(`${scenario.name}: ${error.message}`);
        console.error(`FAIL ${scenario.name}: requests=${JSON.stringify(outputRequests)} errors=${JSON.stringify(errors)}`);
      } finally {
        await page.close();
      }
    }
  } finally {
    await browser.close();
  }
  assert.deepEqual(failures, [], failures.join('\n'));
})().catch(error => {console.error(error); process.exitCode = 1;});
