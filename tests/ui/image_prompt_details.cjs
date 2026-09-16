// Isolated rendering test: no request can reach the running app.
const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const playwright = require(process.env.MAESTRO_PLAYWRIGHT || 'C:/Users/bliza/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

(async () => {
  const bundle = await esbuild.build({stdin: {contents: `
    import React from 'react'; import {createRoot} from 'react-dom/client';
    import {MediaFeedItem} from './src/components/MainContent/MediaFeedItem';
    const root = createRoot(document.getElementById('root'));
    window.showImage = id => root.render(<MediaFeedItem key={id} file={{name: id + '.jpg',
      url: '/picture.svg', type: 'image', mode: 'image', metadata_ready: true}}
      index={0} isActive={true} onActivate={() => {}} onPlaybackStart={() => {}} onMeasured={() => {}}/>);
  `, resolveDir: path.join(root, 'ui'), loader: 'tsx'}, bundle: true, write: false,
    jsx: 'automatic', define: {'process.env.NODE_ENV': '"development"'}, logLevel: 'silent'});
  const assets = path.join(root, 'ui/dist/assets');
  const css = fs.readFileSync(path.join(assets, fs.readdirSync(assets).find(name => name.endsWith('.css'))), 'utf8');
  const browser = await playwright.chromium.launch({headless: true,
    ...(process.platform === 'win32' ? {executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'} : {})});
  try {
    const page = await browser.newPage({viewport: {width: 850, height: 900}});
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.route('**/*', async route => {
      const endpoint = new URL(route.request().url()).pathname;
      if (endpoint === '/') return route.fulfill({contentType: 'text/html', body: `<style>${css}</style><div id="root"></div><script>${bundle.outputFiles[0].text}</script>`});
      if (endpoint === '/picture.svg' || endpoint.startsWith('/api/v1/file/')) return route.fulfill({contentType: 'image/svg+xml', body: '<svg xmlns="http://www.w3.org/2000/svg" width="500" height="200"><rect width="500" height="200" fill="#444"/></svg>'});
      if (endpoint.endsWith('/metadata')) {
        const plain = endpoint.includes('/plain.');
        return route.fulfill({contentType: 'application/json', body: JSON.stringify({source: 'sidecar', params: {
          model_type: 'flux2_klein_9b', prompt: 'A detailed blue vase on a wooden table.',
          ...(!plain ? {_prompt_enhancement: {version: 1, state: 'complete',
            original_prompt: 'My source idea: a blue vase.', enhanced_prompt: 'A detailed blue vase on a wooden table.'}} : {}),
        }})});
      }
      return route.fulfill({contentType: 'application/json', body: '{}'});
    });
    await page.goto('http://image-details.test');
    for (const id of ['queued', 'immediate', 'plain']) {
      await page.evaluate(id => window.showImage(id), id);
      const details = page.getByRole('button', {name: 'Show generation details', exact: true});
      await details.click();
      await page.getByText('A detailed blue vase on a wooden table.', {exact: true}).last().waitFor();
      if (id === 'plain') assert.equal(await page.getByText('Original prompt', {exact: true}).count(), 0);
      else {
        await page.getByText('Enhanced prompt', {exact: true}).waitFor();
        await page.getByText('Original prompt', {exact: true}).waitFor();
        await page.getByText('My source idea: a blue vase.', {exact: true}).waitFor();
        await page.getByRole('button', {name: 'Copy original prompt', exact: true}).waitFor();
      }
    }
    assert.deepEqual(errors, []);
    console.log('Image details: queued/immediate source and enhanced prompts, copy action, and plain-image behavior passed');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
