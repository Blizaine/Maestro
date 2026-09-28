// Focused Frames-mode native picker regression. The app bundle and all API
// calls are isolated; this test does not contact the running Maestro service.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const playwright = require(process.env.MAESTRO_PLAYWRIGHT ||
  'C:/Users/bliza/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

(async () => {
  const bundle = await esbuild.build({stdin: {contents: `
    import React from 'react'; import {createRoot} from 'react-dom/client';
    import {useStore} from './src/stores/useStore';
    import {InputsPanel} from './src/components/Sidebar/InputsPanel';
    window.store=useStore; const root=createRoot(document.getElementById('root'));
    window.mount=()=>{
      const state=useStore.getState();
      useStore.setState({generationMode:'video',studioVideoWorkflow:'frames',
        params:{...state.params,model_type:'minimax_h3_fused_turbo',image_mode:0,
          image_start:undefined,image_end:undefined,image_refs:undefined,frames_positions:undefined,
          video_prompt_type:'',audio_prompt_type:''},
        models:[],enabledModels:new Set(),modelOptions:{fps:24,supports_end_frame:false},
        startImage:null,endImage:null,imageRefs:[],durationSeconds:4,
        slidingWindowSeconds:4,slidingWindowOverlap:0});
      root.render(<InputsPanel/>);
    };
  `, resolveDir: path.join(root, 'ui'), loader: 'tsx'}, bundle: true, write: false,
    jsx: 'automatic', define: {'process.env.NODE_ENV': '"development"'}, logLevel: 'silent'});
  const assets = path.join(root, 'ui/dist/assets');
  const cssName = fs.readdirSync(assets).find(name => name.endsWith('.css'));
  assert.ok(cssName, 'UI build CSS exists for the hidden picker and Frames tile');
  const css = fs.readFileSync(path.join(assets, cssName), 'utf8');
  const browser = await playwright.chromium.launch({headless: true,
    ...(process.platform === 'win32' ? {executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'} : {})});
  try {
    const context = await browser.newContext({viewport: {width: 390, height: 844}, isMobile: true, hasTouch: true});
    const page = await context.newPage();
    page.setDefaultTimeout(5000);
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/*', async route => {
      const url = new URL(route.request().url());
      if (url.pathname === '/') return route.fulfill({contentType: 'text/html', body:
        '<meta name="viewport" content="width=device-width,initial-scale=1"><div id="root" style="height:100dvh"></div>'});
      if (url.pathname.startsWith('/api/')) return route.fulfill({json: {}});
      return route.fulfill({status: 404, body: ''});
    });
    await page.goto('http://frames-picker.test');
    await page.addStyleTag({content: css});
    await page.addScriptTag({content: bundle.outputFiles[0].text});
    await page.evaluate(() => window.mount());

    const input = page.getByLabel('Choose a frame image');
    assert.equal(await input.count(), 1, 'Frames keeps a single dedicated image picker mounted');
    assert.equal(await input.evaluate(node => node.isConnected), true, 'the picker remains attached to the document');

    // An empty selection models cancelling the native chooser; it must leave
    // the current frame empty and allow the user to open the picker again.
    const cancelledPickerPromise = page.waitForEvent('filechooser');
    await page.getByRole('button', {name: 'Start frame', exact: true}).tap();
    const cancelledPicker = await cancelledPickerPromise;
    assert.equal(await cancelledPicker.element().evaluate(element => element.isConnected), true,
      'the native chooser is backed by a connected input');
    await cancelledPicker.setFiles([]);
    await page.waitForTimeout(50);
    assert.equal(await page.evaluate(() => window.store.getState().startImage), null,
      'cancelling the picker does not set a start frame');

    const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=', 'base64');
    const photo = {name: 'start-frame.png', mimeType: 'image/png', buffer: png};
    const pickerPromise = page.waitForEvent('filechooser');
    await page.getByRole('button', {name: 'Start frame', exact: true}).tap();
    const picker = await pickerPromise;
    assert.equal(await picker.element().evaluate(element => element.isConnected), true,
      'Frames opens the native picker from its mounted input');
    // A rerender while the native chooser is pending must not detach the input
    // or lose the selected file when the picker returns.
    await page.evaluate(() => window.store.setState({durationSeconds: 4.5}));
    assert.equal(await picker.element().evaluate(element => element.isConnected), true,
      'the picker input survives a panel rerender while selection is pending');
    await picker.setFiles(photo);
    await page.waitForFunction(() => window.store.getState().startImage?.name === 'start-frame.png');
    const firstPreview = page.locator('img[alt="Frame Start"]');
    await firstPreview.waitFor();
    await page.waitForFunction(() => {
      const image = document.querySelector('img[alt="Frame Start"]');
      return image?.complete && image.naturalWidth > 0;
    });
    assert.ok((await firstPreview.getAttribute('src')).startsWith('blob:'), 'the chosen photo is shown as the start-frame preview');
    assert.equal(await input.evaluate(node => node.value), '', 'the input resets after a successful selection');

    await page.getByRole('button', {name: 'Remove', exact: true}).click();
    await page.getByRole('button', {name: 'Start frame', exact: true}).waitFor();
    const retryPickerPromise = page.waitForEvent('filechooser');
    await page.getByRole('button', {name: 'Start frame', exact: true}).tap();
    const retryPicker = await retryPickerPromise;
    await retryPicker.setFiles(photo);
    await page.waitForFunction(() => window.store.getState().startImage?.name === 'start-frame.png');
    await page.waitForFunction(() => {
      const image = document.querySelector('img[alt="Frame Start"]');
      return image?.complete && image.naturalWidth > 0;
    });
    assert.equal(await input.evaluate(node => node.value), '', 'the same image can be selected again after removal');

    // Enabling end-frame capability exercises the same smart handler's second
    // input role, while keeping the original start file in place.
    await page.evaluate(() => window.store.setState({modelOptions: {fps: 24, supports_end_frame: true}}));
    await page.getByRole('button', {name: 'Add frame', exact: true}).waitFor();
    const endPickerPromise = page.waitForEvent('filechooser');
    await page.getByRole('button', {name: 'Add frame', exact: true}).tap();
    const endPicker = await endPickerPromise;
    await endPicker.setFiles({...photo, name: 'end-frame.png'});
    await page.waitForFunction(() => window.store.getState().endImage?.name === 'end-frame.png');
    await page.waitForFunction(() => {
      const image = document.querySelector('img[alt="Frame End"]');
      return image?.complete && image.naturalWidth > 0;
    });
    assert.equal(await page.evaluate(() => window.store.getState().startImage?.name), 'start-frame.png',
      'adding an end frame preserves the start frame');
    assert.deepEqual(errors, [], 'the Frames picker runs without browser errors');
    console.log('Frames mobile image picker stays mounted through selection, cancellation, rerender, and same-file retry');
    await context.close();
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
