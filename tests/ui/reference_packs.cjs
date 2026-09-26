// Real components/store/API client; isolated mocked library, no GPU or user assets.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const playwright = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');
const views = ['face_closeup', 'full_body_front', 'full_body_three_quarter', 'upper_body_three_quarter', 'full_body_back'];
const labels = ['Face close-up', 'Full body — front', 'Full body — three-quarter', 'Upper body — three-quarter', 'Full body — back'];
(async () => {
  const bundle = await esbuild.build({stdin: {contents: `
    import React from 'react'; import {createRoot} from 'react-dom/client';
    import {CharacterBrowser} from './src/components/Characters/CharacterBrowser';
    import {OmniReferenceSection} from './src/components/Sidebar/OmniReferenceSection';
    import {useStore} from './src/stores/useStore';
    window.store = useStore;
    const scope = new URLSearchParams(location.search).get('scope');
    useStore.setState({modelOptions: {omni_reference: true}, selectedModelPerMode: {video: ''}});
    createRoot(document.getElementById('root')).render(scope ? <OmniReferenceSection scope={scope}/> : <CharacterBrowser/>);
  `, resolveDir: path.join(root, 'ui'), loader: 'tsx'}, bundle: true, write: false, jsx: 'automatic',
    define: {'process.env.NODE_ENV': '"development"'}, logLevel: 'silent'});
  const assets = path.join(root, 'ui/dist/assets');
  const css = fs.readFileSync(path.join(assets, fs.readdirSync(assets).find(name => name.endsWith('.css'))), 'utf8');
  const browser = await playwright.chromium.launch({headless: true, ...(process.env.MAESTRO_CHROME ? {executablePath: process.env.MAESTRO_CHROME} : {})});
  try {
    const page = await browser.newPage({viewport: {width: 390, height: 844}});
    page.setDefaultTimeout(10000);
    const errors = []; page.on('pageerror', e => errors.push(e.message));
    let character; const saves = [];
    await page.route('**/*', async route => {
      const req = route.request(), url = new URL(req.url());
      const json = data => route.fulfill({contentType: 'application/json', body: JSON.stringify(data)});
      if (url.pathname === '/api/v1/characters') return json({characters: character ? [character] : []});
      if (url.pathname === '/api/v1/upload') return json({path: '/uploads/view.png', filename: 'view.png', url: '/picture.svg'});
      if (url.pathname.endsWith('/reference-packs') && req.method() === 'POST') {
        const body = req.postDataJSON(); saves.push(body);
        character ||= {id: '0123456789abcdef', name: body.name, created_at: 1, updated_at: 1,
          visual: {type: 'image', path: '/cover.png', filename: 'face.png', url: '/picture.svg'}, reference_pack: {version: 1, revisions: []}};
        const number = character.reference_pack.revisions.length + 1;
        character.reference_pack.revisions.push({id: `revision${number}`, number, label: body.label, parent_revision_id: body.base_revision_id,
          created_at: number, approved: false,
          images: Object.fromEntries(views.map((view, i) => [view, {path: `/pack/${number}/${view}.png`, url: `/picture.svg?view=${i}`, label: labels[i], inherited: !(view in body.images)}]))});
        return json(character);
      }
      if (url.pathname.endsWith('/review')) {
        character.reference_pack.revisions.find(r => url.pathname.includes(r.id)).approved = req.postDataJSON().approved;
        return json(character);
      }
      if (url.pathname === '/picture.svg') return route.fulfill({contentType: 'image/svg+xml', body: '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="140"><rect width="100" height="140" fill="#456"/></svg>'});
      if (url.pathname.startsWith('/api/')) return json({});
      return route.fulfill({contentType: 'text/html', body: `<html><head><style>${css}</style></head><body><div id="root"></div><script>${bundle.outputFiles[0].text}</script></body></html>`});
    });
    await page.goto('http://maestro.test');
    await page.getByRole('button', {name: 'Import five-view pack', exact: true}).click();
    let dialog = page.getByRole('dialog', {name: 'Import five-view Reference Pack', exact: true});
    await dialog.getByLabel('Pack character name').fill('Pack Person');
    assert.equal(await dialog.getByRole('button', {name: 'Save draft revision'}).isDisabled(), true);
    const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aX1sAAAAASUVORK5CYII=', 'base64');
    for (const label of labels) await dialog.getByLabel(label, {exact: true}).setInputFiles({name: 'view.png', mimeType: 'image/png', buffer: png});
    assert.equal(await dialog.getByAltText('Selected replacement').count(), 5);
    await dialog.getByLabel('Revision label').fill('Original outfit');
    await dialog.getByRole('button', {name: 'Save draft revision'}).click();
    dialog = page.getByRole('dialog', {name: 'Pack Person — Reference Pack', exact: true});
    await dialog.getByRole('button', {name: 'Approve all five views'}).waitFor();
    assert.deepEqual(Object.keys(saves[0].images), views);
    assert.ok(await dialog.evaluate(el => el.scrollWidth <= el.clientWidth), 'Mobile dialog fits');
    await dialog.getByRole('button', {name: 'Approve all five views'}).click();
    await dialog.getByRole('button', {name: 'Unapprove revision'}).waitFor();
    await dialog.getByRole('button', {name: 'New revision from this one'}).click();
    await dialog.getByLabel('Face close-up', {exact: true}).setInputFiles({name: 'replacement.png', mimeType: 'image/png', buffer: png});
    await dialog.getByLabel('Revision label').fill('Face correction');
    await dialog.getByRole('button', {name: 'Save draft revision'}).click();
    await dialog.getByRole('button', {name: 'Approve all five views'}).waitFor();
    assert.deepEqual(Object.keys(saves[1].images), ['face_closeup']);
    assert.equal(saves[1].base_revision_id, 'revision1');
    assert.equal(character.reference_pack.revisions[0].approved, true);
    assert.equal(character.reference_pack.revisions[1].approved, false);
    await dialog.getByRole('button', {name: 'Approve all five views'}).click();
    await dialog.getByRole('button', {name: 'Unapprove revision'}).waitFor();
    if (process.env.MAESTRO_SCREENSHOTS) {
      fs.mkdirSync(process.env.MAESTRO_SCREENSHOTS, {recursive: true});
      await page.screenshot({path: path.join(process.env.MAESTRO_SCREENSHOTS, 'pack-mobile.png')});
    }
    await page.setViewportSize({width: 1440, height: 1000});
    assert.ok(await dialog.evaluate(el => el.scrollWidth <= el.clientWidth), 'Desktop dialog fits');
    if (process.env.MAESTRO_SCREENSHOTS) await page.screenshot({path: path.join(process.env.MAESTRO_SCREENSHOTS, 'pack-desktop.png')});
    await dialog.getByRole('button', {name: 'Done', exact: true}).click();

    for (const scope of ['studio', 'director']) {
      await page.goto(`http://maestro.test/?scope=${scope}`);
      await page.evaluate(scope => {
        const ref = {id: 'scene', type: 'image', path: '/scene.png', filename: 'scene.png', image_intent: 'scene'};
        if (scope === 'studio') window.store.setState(s => ({params: {...s.params, minimax_h3_references: [ref]}}));
        else window.store.setState({directorH3References: [ref]});
      }, scope);
      const trigger = page.getByRole('button', {name: 'Characters', exact: true});
      if (await trigger.getAttribute('aria-expanded') !== 'true') await trigger.click();
      const library = page.getByRole('dialog', {name: 'Characters', exact: true});
      const add = library.getByRole('button', {name: 'Add Pack Person to references', exact: true});
      assert.equal(await add.isDisabled(), true, 'Multiple approved revisions require explicit choice');
      await library.getByLabel('Revision for Pack Person').selectOption('revision1');
      await add.click();
      const readRefs = () => page.evaluate(scope => scope === 'studio' ? window.store.getState().params.minimax_h3_references : window.store.getState().directorH3References, scope);
      const refs = await readRefs();
      assert.equal(refs.length, 6); assert.equal(refs[0].id, 'scene');
      assert.deepEqual(refs.slice(1).map(r => r.reference_pack_view), views);
      assert.ok(refs.slice(1).every(r => r.reference_pack_revision_id === 'revision1' && r.library_character_id === character.id));
      // Try with five occupied image slots: reject the whole pack without touching existing inputs.
      await page.evaluate(scope => {
        const refs = Array.from({length: 5}, (_, i) => ({id: `other${i}`, type: 'image', path: `/other${i}.png`, filename: 'other.png'}));
        if (scope === 'studio') window.store.setState(s => ({params: {...s.params, minimax_h3_references: refs}}));
        else window.store.setState({directorH3References: refs});
      }, scope);
      await add.click();
      await library.getByRole('alert').filter({hasText: 'exceed'}).waitFor();
      assert.equal((await readRefs()).length, 5, 'Capacity failure is atomic');
      // The pack editor nests inside the library without dismissing it.
      await library.getByRole('button', {name: 'Pack views & revisions'}).click();
      const pack = page.getByRole('dialog', {name: 'Pack Person — Reference Pack', exact: true});
      await pack.waitFor(); await pack.getByRole('button', {name: 'Done', exact: true}).click();
      await library.waitFor();
    }
    assert.deepEqual(errors, []);
    console.log('Reference Packs: import, review, inherited revision, mobile/desktop, explicit selection, Studio/Director insertion, preserved other references, atomic limits, nested editor passed.');
  } finally { await browser.close(); }
})().catch(e => {console.error(e); process.exitCode = 1});
