// Actual React components and API client, with isolated character data.
// Run after the UI build: node tests/ui/character_images.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const playwright = require(process.env.MAESTRO_PLAYWRIGHT ||
  'C:/Users/bliza/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

(async () => {
  const bundle = await esbuild.build({
    stdin: {contents: `
      import React from 'react';
      import {createRoot} from 'react-dom/client';
      import {CharacterBrowser} from './src/components/Characters/CharacterBrowser';
      createRoot(document.getElementById('root')).render(<React.StrictMode><CharacterBrowser/></React.StrictMode>);
    `, resolveDir: path.join(root, 'ui'), loader: 'tsx'},
    bundle: true, write: false, jsx: 'automatic', define: {'process.env.NODE_ENV': '"development"'}, logLevel: 'silent',
  });
  const assets = path.join(root, 'ui/dist/assets');
  const css = fs.readFileSync(path.join(assets, fs.readdirSync(assets).find(name => name.endsWith('.css'))), 'utf8');
  // Chromium attachment downloads may bypass Playwright route interception.
  const downloads = http.createServer((request, response) => {
    response.writeHead(200, {'Content-Type': 'application/zip', 'Content-Disposition': 'attachment; filename="reference.images.zip"'});
    response.end('test archive');
  });
  await new Promise(resolve => downloads.listen(0, '127.0.0.1', resolve));
  const browser = await playwright.chromium.launch({headless: true,
    ...(process.env.MAESTRO_CHROME || process.platform === 'win32'
      ? {executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'} : {})});
  try {
    const page = await browser.newPage({viewport: {width: 390, height: 844}});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    const card = {id: 'reference-person', name: 'Reference Person', created_at: 1, updated_at: 1,
      visual: {type: 'image', path: '/original.png', url: '/picture.svg'},
      refmod: {kind: 'image', path: '/reference.safetensors', tokens: 2048}};
    const views = {version: 1, source: 'refmod', cover_id: 'view-0001',
      selected_ids: ['view-0001', 'view-0002', 'view-0003', 'view-0004'], notes: [],
      items: Array.from({length: 19}, (_, index) => ({id: `view-${String(index + 1).padStart(4, '0')}`,
        width: 512, height: 512, url: `/picture.svg?v=${index}`}))};
    let recoveries = 0, failRecovery = false;
    const saves = [];
    await page.route('**/*', async route => {
      const request = route.request(), url = new URL(request.url());
      const json = data => route.fulfill({contentType: 'application/json', body: JSON.stringify(data)});
      if (url.pathname === '/api/v1/characters') return json({characters: [card]});
      if (url.pathname.endsWith('/images/recover')) { recoveries++; return json({id: 'recovery'}); }
      if (url.pathname.startsWith('/api/v1/character-transfers/')) {
        if (failRecovery) return json({status: 'failed', error: 'Unable to decode this view.'});
        card.image_views = structuredClone(views);
        card.visual.thumbnail_url = '/picture.svg?v=recovered';
        return json({status: 'completed', result: {character: card}});
      }
      if (url.pathname.endsWith('/images') && request.method() === 'PUT') {
        const selection = request.postDataJSON();
        saves.push(selection);
        Object.assign(card.image_views, selection);
        card.visual.thumbnail_url = `/picture.svg?v=${selection.cover_id}`;
        return json(card);
      }
      if (url.pathname.endsWith('/images.zip')) return route.continue();
      if (url.pathname === '/picture.svg') return route.fulfill({contentType: 'image/svg+xml',
        body: '<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512"><rect width="512" height="512" fill="#334155"/><circle cx="256" cy="195" r="105" fill="#d6aa87"/><path d="M65 512 Q256 170 447 512" fill="#94a3b8"/></svg>'});
      return route.fulfill({contentType: 'text/html', body: '<meta name="viewport" content="width=device-width,initial-scale=1"><div id="root" style="height:100dvh"></div>'});
    });
    await page.goto(`http://127.0.0.1:${downloads.address().port}`);
    await page.addStyleTag({content: css});
    await page.addScriptTag({content: bundle.outputFiles[0].text});
    const trigger = page.getByRole('button', {name: 'Images for Reference Person'});
    await trigger.click();
    const dialog = page.getByRole('dialog');
    await dialog.getByRole('button', {name: 'Preview image 19', exact: true}).waitFor();
    assert.equal(recoveries, 1, 'StrictMode must not start duplicate GPU work');
    assert.equal(await dialog.getByRole('checkbox').count(), 19, 'Every recovered view is available');
    assert.ok(await dialog.evaluate(el => el.scrollWidth <= el.clientWidth), 'No horizontal overflow at 390px');
    await dialog.getByRole('button', {name: 'Preview image 19', exact: true}).click();
    assert.equal(await dialog.getByRole('button', {name: 'Preview image 19', exact: true}).getAttribute('aria-pressed'), 'true', 'Last view is reachable by scrolling');
    await dialog.getByRole('button', {name: 'Preview image 2', exact: true}).click();
    await dialog.getByRole('button', {name: 'Use as cover', exact: true}).click();
    await dialog.getByRole('checkbox', {name: 'Select image 1', exact: true}).uncheck();
    await dialog.getByRole('button', {name: 'Save selection', exact: true}).click();
    await dialog.waitFor({state: 'hidden'});
    assert.deepEqual(saves[0], {selected_ids: ['view-0002', 'view-0003', 'view-0004'], cover_id: 'view-0002'});
    assert.ok((await page.getByRole('img', {name: 'Reference Person', exact: true}).getAttribute('src')).includes('view-0002'), 'Library cover refreshes');
    assert.equal(await trigger.evaluate(el => document.activeElement === el), true, 'Closing returns focus to the image button');
    await trigger.click();
    await dialog.getByRole('checkbox', {name: 'Select image 2', exact: true}).waitFor();
    assert.equal(recoveries, 1, 'Reopening uses saved PNGs');
    await dialog.getByRole('checkbox', {name: 'Select image 2', exact: true}).uncheck();
    await dialog.getByRole('checkbox', {name: 'Select image 3', exact: true}).uncheck();
    await dialog.getByRole('checkbox', {name: 'Select image 4', exact: true}).uncheck();
    assert.equal(await dialog.getByRole('button', {name: 'Save selection', exact: true}).isDisabled(), true, 'Empty selections cannot be saved');
    await dialog.getByRole('checkbox', {name: 'Select image 19', exact: true}).check();
    const downloading = page.waitForEvent('download');
    await dialog.getByRole('button', {name: 'Download selected', exact: true}).click();
    assert.equal((await downloading).suggestedFilename(), 'reference.images.zip');
    assert.deepEqual(saves[1], {selected_ids: ['view-0019'], cover_id: 'view-0019'}, 'Downloads use the current selection');
    failRecovery = true;
    await dialog.getByRole('button', {name: 'Recover again', exact: true}).click();
    await dialog.getByRole('alert').waitFor();
    assert.ok((await dialog.getByRole('alert').innerText()).includes('Unable to decode'));
    assert.equal(await dialog.getByRole('checkbox').count(), 19, 'A recovery failure retains usable images');
    await page.setViewportSize({width: 1440, height: 1000});
    assert.ok(await dialog.evaluate(el => el.scrollWidth <= el.clientWidth), 'Desktop dialog fits');
    await dialog.getByRole('button', {name: 'Save selection', exact: true}).focus();
    await page.keyboard.press('Tab');
    assert.equal(await dialog.getByRole('button', {name: 'Close character images'}).evaluate(el => document.activeElement === el), true, 'Tab stays inside dialog');
    await page.keyboard.press('Escape');
    await dialog.waitFor({state: 'hidden'});
    assert.deepEqual(errors, [], 'Character recovery renders without browser errors');
    console.log('Character gallery: recovery, all views, mobile scrolling, cover refresh, selection, ZIP download, failure recovery and keyboard checks passed');
  } finally {await browser.close(); await new Promise(resolve => downloads.close(resolve));}
})().catch(error => {console.error(error); process.exitCode = 1;});
