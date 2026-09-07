// Actual React, API client and Zustand paths in an isolated mobile/desktop browser.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const playwright = require(process.env.MAESTRO_PLAYWRIGHT ||
  'C:/Users/bliza/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

(async () => {
  const bundle = await esbuild.build({stdin: {contents: `
    import React from 'react';
    import {createRoot} from 'react-dom/client';
    import {useStore} from './src/stores/useStore';
    import {ViggleControls} from './src/components/Sidebar/ViggleControls';
    import {ImageRefSection} from './src/components/Sidebar/ImageRefSection';
    import {GenerateButton} from './src/components/Sidebar/GenerateButton';
    window.store = useStore;
    function App() {
      const mode = useStore(s => s.generationMode);
      return <div className="mx-auto max-w-sm space-y-4 p-4">{mode === 'video' ? <ViggleControls/> : <ImageRefSection/>}<GenerateButton/></div>;
    }
    window.mount = () => createRoot(document.getElementById('root')).render(<React.StrictMode><App/></React.StrictMode>);
  `, resolveDir: path.join(root, 'ui'), loader: 'tsx'}, bundle: true, write: false,
    jsx: 'automatic', define: {'process.env.NODE_ENV': '"development"'}, logLevel: 'silent'});
  const assets = path.join(root, 'ui/dist/assets');
  const css = fs.readFileSync(path.join(assets, fs.readdirSync(assets).find(name => name.endsWith('.css'))), 'utf8');
  const browser = await playwright.chromium.launch({headless: true,
    ...(process.env.MAESTRO_CHROME || process.platform === 'win32'
      ? {executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'} : {})});
  try {
    const page = await browser.newPage({viewport: {width: 390, height: 844}});
    const errors = [], requests = [], uploads = [];
    page.on('pageerror', error => errors.push(error.message));
    const views = {version: 1, source: 'refmod', cover_id: 'view-0002', selected_ids: ['view-0002'], notes: [],
      items: Array.from({length: 19}, (_, n) => ({id: `view-${String(n + 1).padStart(4, '0')}`,
        width: 512, height: 512, url: `/picture.svg?view=${n + 1}`}))};
    const person = {id: 'blaine', name: 'Blaine', created_at: 1, updated_at: 1,
      visual: {type: 'image', path: '/original.png', url: '/picture.svg'}, image_views: views};
    const recovered = {...person, id: 'refmod', name: 'Recovered Person', image_views: undefined,
      refmod: {kind: 'image', path: '/refmod.safetensors', tokens: 2048}};
    const result = {signature: 'preview-signature', image_path: '/uploads/viggle_prepared_test.png',
      image_url: '/picture.svg?prepared', width: 720, height: 1280, frame_seconds: 0,
      image_model: 'flux2_klein_9b', prompt: 'Replace'};
    let recoveries = 0, failPreview = false, runningPreview = false, cancelled = false;
    await page.route('**/*', async route => {
      const request = route.request(), url = new URL(request.url());
      const json = data => route.fulfill({contentType: 'application/json', body: JSON.stringify(data)});
      if (url.pathname === '/api/v1/characters') return json({characters: [person, recovered,
        ...Array.from({length: 14}, (_, n) => ({...person, id: `extra${n}`, name: `Person ${n + 1}`}))]});
      if (url.pathname.endsWith('/images/recover')) {recoveries++; return json({id: 'recovery'});}
      if (url.pathname.startsWith('/api/v1/character-transfers/')) {
        recovered.image_views = views;
        return json({status: 'completed', result: {character: recovered}});
      }
      if (url.pathname === '/api/v1/upload') {
        uploads.push(request.postData());
        return json({path: `/uploads/reference-${uploads.length}.png`, url: '/picture.svg'});
      }
      if (url.pathname === '/api/v1/generate') {
        const body = request.postDataJSON(); requests.push(body);
        cancelled = false;
        return json({job_id: String(requests.length), status: body._queue_mode === 'held' ? 'held' : 'queued'});
      }
      if (url.pathname.startsWith('/api/v1/status/')) return json({
        status: cancelled ? 'cancelled' : failPreview ? 'failed' : runningPreview ? 'running' : 'completed',
        error: failPreview ? 'Klein could not prepare this frame.' : null,
        message: 'Preparing character…', progress: 25, output_files: [], viggle_preparation: result});
      if (url.pathname.includes('/cancel/')) {cancelled = true; return json({status: 'cancelled'});}
      if (url.pathname === '/api/v1/outputs') return json({outputs: [], total: 0});
      if (url.pathname.startsWith('/api/')) return json({});
      if (url.pathname === '/picture.svg') return route.fulfill({contentType: 'image/svg+xml',
        body: '<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512"><rect width="512" height="512" fill="#334155"/><circle cx="256" cy="190" r="105" fill="#d6aa87"/><path d="M65 512 Q256 165 447 512" fill="#94a3b8"/></svg>'});
      return route.fulfill({contentType: 'text/html', body: '<meta name="viewport" content="width=device-width,initial-scale=1"><div id="root"></div>'});
    });
    await page.goto('http://maestro.test');
    await page.addStyleTag({content: css});
    await page.addScriptTag({content: bundle.outputFiles[0].text});
    await page.evaluate(() => {
      window.store.setState({generationMode: 'video', studioVideoWorkflow: 'animate',
        params: {...window.store.getState().params, model_type: 'viggle_animate', video_guide: '/uploads/source.mp4',
          image_mode: 0, seed: 42, _viggle_source_seconds: 5.2, resolution: 'auto_480p'},
        modelOptions: {model_type: 'viggle_animate', architecture: 'viggle_animate', fps: 24, frames_minimum: 124, frames_steps: 17},
        models: [{model_type: 'viggle_animate', architecture: 'viggle_animate', name: 'Viggle', family: 'minimax_h3'}],
        durationSeconds: 124 / 24});
      window.mount();
    });
    await page.getByRole('button', {name: 'Use a character', exact: true}).click();
    await page.getByRole('button', {name: 'Characters', exact: true}).click();
    let dialog = page.getByRole('dialog');
    await dialog.getByRole('button', {name: 'Choose Person 14', exact: true}).scrollIntoViewIfNeeded();
    assert.ok(await dialog.evaluate(el => el.scrollWidth <= el.clientWidth), 'Library fits mobile');
    await dialog.getByRole('button', {name: 'Choose Blaine', exact: true}).click();
    assert.equal(await dialog.getByRole('button', {name: 'Use character view 2', exact: true}).getAttribute('aria-pressed'), 'true', 'Default is saved cover');
    await dialog.getByRole('button', {name: 'Use character view 19', exact: true}).click();
    await dialog.getByRole('button', {name: 'Use this image', exact: true}).click();
    await dialog.waitFor({state: 'hidden'});
    assert.equal(recoveries, 0, 'Cached native PNGs need no GPU recovery');
    assert.equal(await page.evaluate(() => window.store.getState().params.viggle_character.view_id), 'view-0019');
    await page.getByLabel('Appearance (optional)', {exact: true}).fill('A dark leather jacket and blue jeans.');
    await page.getByRole('button', {name: 'Preview replacement frame', exact: true}).click();
    await page.getByRole('img', {name: 'Prepared character frame', exact: true}).waitFor();
    assert.equal(requests[0]._viggle_prepare_only, true);
    assert.equal(requests[0].viggle_character.appearance_prompt, 'A dark leather jacket and blue jeans.');
    assert.ok(requests[0].viggle_character.swap_prompt.startsWith('Replace the main character in source frame'));
    await page.evaluate(() => window.store.getState().startGeneration('queue'));
    assert.equal(requests[1]._viggle_prepare_only, undefined, 'Generate must animate, even after a preview');
    assert.equal(requests[1].model_type, 'viggle_animate');
    assert.equal(requests[1]._viggle_prepared.signature, result.signature, 'Unchanged preview is reusable');
    assert.equal(requests[1].viggle_character.view_id, 'view-0019');
    await page.getByRole('spinbutton', {name: 'Source frame seconds'}).fill('0.5');
    assert.equal(await page.getByRole('img', {name: 'Prepared character frame'}).count(), 0, 'New timestamp invalidates preview');
    await page.evaluate(() => window.store.getState().startGeneration('queue'));
    assert.equal(requests[2].image_refs.length, 0, 'Automatic generation accepts no manually edited frame');
    assert.equal(requests[2].viggle_character.frame_seconds, 0.5);
    failPreview = true;
    await page.getByRole('button', {name: 'Preview replacement frame'}).click();
    await page.getByRole('alert').filter({hasText: 'Klein could not'}).waitFor();
    assert.equal(await page.getByRole('img', {name: 'Prepared character frame'}).count(), 0);
    failPreview = false; runningPreview = true;
    await page.getByRole('button', {name: 'Preview replacement frame'}).click();
    await page.getByRole('button', {name: 'Cancel preparation'}).click();
    await page.getByRole('alert').filter({hasText: 'cancelled'}).waitFor();
    runningPreview = false;
    await page.getByRole('button', {name: 'Use an edited frame'}).click();
    await page.getByLabel('Upload edited frame', {exact: true}).setInputFiles({name: 'edit.png', mimeType: 'image/png', buffer: Buffer.from('image')});
    await page.getByRole('img', {name: 'Edited reference frame'}).waitFor();
    await page.evaluate(() => window.store.getState().startGeneration('queue'));
    assert.equal(requests.at(-1).viggle_character, undefined, 'Manual route stays independent');
    assert.equal(requests.at(-1).image_refs.length, 1);

    await page.evaluate(character => window.store.setState({jobs: [], params: {...window.store.getState().params,
      viggle_character: character, _viggle_prepared: undefined, _viggle_edited_frame: undefined}}), requests[2].viggle_character);
    await page.evaluate(() => window.store.getState().startGeneration());
    await page.getByRole('img', {name: 'Prepared character frame'}).waitFor();
    assert.equal(await page.evaluate(() => window.store.getState().params._viggle_prepared.signature), result.signature,
      'Normal Generate exposes its completed preparation frame');
    if (process.env.MAESTRO_UI_SCREENSHOT_DIR) {
      fs.mkdirSync(process.env.MAESTRO_UI_SCREENSHOT_DIR, {recursive: true});
      await page.screenshot({path: path.join(process.env.MAESTRO_UI_SCREENSHOT_DIR, 'viggle-mobile.png'), fullPage: true});
    }

    // Ordinary image refs retain the first source, select recovered views, and enforce the model budget.
    await page.evaluate(() => window.store.setState({generationMode: 'image', studioImageWorkflow: 'generate',
      imageRefs: [new File(['source'], 'source.png', {type: 'image/png'})], imageRefType: 'KI',
      params: {...window.store.getState().params, model_type: 'flux2_klein_9b', image_mode: 1, prompt: 'Replace the source person with image 2.'},
      modelOptions: {model_type: 'flux2_klein_9b', architecture: 'flux2_klein_9b', max_image_refs: 3,
        image_ref_choices: {choices: [['Main and people', 'KI']]}},
      models: [{model_type: 'flux2_klein_9b', architecture: 'flux2_klein_9b', supports_image_edit: true, supports_ref_images: true}]}));
    await page.getByRole('button', {name: 'Characters', exact: true}).click();
    dialog = page.getByRole('dialog');
    await dialog.getByRole('button', {name: 'Choose Recovered Person', exact: true}).click();
    await dialog.getByRole('button', {name: 'Use character view 1', exact: true}).click();
    if (process.env.MAESTRO_UI_SCREENSHOT_DIR) await page.screenshot({path: path.join(process.env.MAESTRO_UI_SCREENSHOT_DIR, 'character-picker-mobile.png')});
    assert.equal(await dialog.getByRole('button', {name: 'Use character view 3', exact: true}).isDisabled(), true, 'Selection respects remaining model slots');
    await dialog.getByRole('button', {name: 'Use 2 images', exact: true}).click();
    await dialog.waitFor({state: 'hidden'});
    assert.equal(recoveries, 1, 'RefMod without cached views uses native recovery');
    const refs = await page.evaluate(() => window.store.getState().imageRefs.map(file => file.name));
    assert.equal(refs[0], 'source.png');
    assert.ok(refs[1].includes('view-0002') && refs[2].includes('view-0001'), 'Selected reference order is preserved');
    assert.equal(await page.getByRole('button', {name: 'Characters', exact: true}).isDisabled(), true);
    await page.evaluate(() => window.store.getState().startGeneration('queue'));
    assert.equal(requests.at(-1).image_refs.length, 3);
    assert.equal(requests.at(-1).viggle_character, undefined);
    assert.equal(requests.at(-1)._viggle_prepared, undefined);
    await page.evaluate(() => window.store.setState({studioImageWorkflow: 'inpaint', imageWorkflowSourcePath: '/source.png', imageWorkflowMaskPath: '/mask.png',
      imageRefs: window.store.getState().imageRefs.slice(1), params: {...window.store.getState().params, image_mode: 2}}));
    assert.equal(await page.getByRole('button', {name: 'Characters', exact: true}).isDisabled(), true, 'Inpaint source consumes one reference slot');
    await page.evaluate(() => window.store.getState().startGeneration('queue'));
    assert.equal(requests.at(-1).image_guide, '/source.png');
    assert.equal(requests.at(-1).image_refs.length, 2, 'Reference-capable inpaint submits the selected character images');
    assert.ok(requests.at(-1).video_prompt_type.includes('I'));
    await page.setViewportSize({width: 1440, height: 1000});
    assert.ok(await page.locator('#root').evaluate(el => el.scrollWidth <= el.clientWidth), 'Desktop fits');
    assert.deepEqual(errors, [], 'No React errors or update loops');
    console.log('Viggle characters: native picker, mobile scrolling, reference order/limits, automatic and manual payloads, preview, invalidation, failure, cancellation, and image/inpaint refs passed');
  } finally {await browser.close();}
})().catch(error => {console.error(error); process.exitCode = 1;});
