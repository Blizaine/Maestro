// LoRA aliases stay presentation-only across Studio, Director, and My LoRAs.
// Every API request is mocked; no live metadata or model is changed.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const playwright = require(process.env.MAESTRO_PLAYWRIGHT ||
  'C:/Users/bliza/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

const orbit = 'orbit_style.safetensors';
const alpha = 'alpha_scene.safetensors';
const mysticV1 = 'MysticXXX_MMH3-V1.safetensors';
const mysticV4 = 'MysticXXX_MMH3-V4.safetensors';
const mysticRef2va = 'MysticXXX_MMH3-V4-ref2va.safetensors';
const dayMs = 24 * 60 * 60 * 1000;
const dateFixtures = {
  [orbit]: {released_at: new Date(Date.now() - 4 * dayMs).toISOString(), downloaded_at: new Date(Date.now() - dayMs).toISOString()},
  [alpha]: {released_at: 'not-a-date', downloaded_at: new Date(Date.now() - 10 * dayMs).toISOString()},
};

(async () => {
  const bundle = await esbuild.build({stdin: {contents: `
    import React from 'react'; import {createRoot} from 'react-dom/client';
    import {useStore} from './src/stores/useStore';
    import {LoraSelector} from './src/components/SettingsDrawer/LoraSelector';
    import {DirectorLoraSelector} from './src/components/SettingsDrawer/DirectorLoraSelector';
    import {LoraBrowser} from './src/components/LoraBrowser/LoraBrowser';
    window.store=useStore;
    const pickerRoot=createRoot(document.getElementById('pickers'));
    window.mountPickers=()=>{
      const state=useStore.getState();
      useStore.setState({generationMode:'video',modelOptions:{guidance_max_phases:1},
        params:{...state.params,model_type:'test_video',activated_loras:['${orbit}']},
        availableLoras:['${orbit}','${alpha}','${mysticV1}','${mysticV4}','${mysticRef2va}'],lorasLoading:false,loraWeights:{'${orbit}':[0.73]},
        savedLoraPerMode:{...state.savedLoraPerMode,video:{activated_loras:['${orbit}'],loras_multipliers:'1.00',
          loraWeights:{'${orbit}':[0.73]},availableLoras:['${orbit}','${alpha}','${mysticV1}','${mysticV4}','${mysticRef2va}']}},loraPickerSort:'name'});
      pickerRoot.render(<div><LoraSelector/><DirectorLoraSelector mode="video" modelType="test_video"/></div>);
    };
    window.mountBrowser=()=>{
      useStore.setState({loraBrowserOpen:true,loraBrowserArch:null,loraBrowserDefaultDir:null,
        civitSearchResults:[],civitSearchCursor:null,civitSelectedModel:null});
      window.browserRoot=createRoot(document.getElementById('library'));
      window.browserRoot.render(<LoraBrowser/>);
    };
  `, resolveDir: path.join(root, 'ui'), loader: 'tsx'}, bundle: true, write: false,
    jsx: 'automatic', define: {'process.env.NODE_ENV': '"development"'}, logLevel: 'silent'});
  const assets = path.join(root, 'ui/dist/assets');
  const cssName = fs.readdirSync(assets).find(name => name.endsWith('.css'));
  assert.ok(cssName, 'UI build CSS is present');
  const css = fs.readFileSync(path.join(assets, cssName), 'utf8');
  const browser = await playwright.chromium.launch({headless: true,
    ...(process.platform === 'win32' ? {executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'} : {})});
  try {
    const page = await browser.newPage({viewport: {width: 1280, height: 900}});
    page.setDefaultTimeout(6000);
    const errors = [], nameUpdates = [], civitaiModelReads = [];
    page.on('console', message => { if (message.type() === 'error') errors.push(`console: ${message.text()}`); });
    const aliases = new Map();
    const automaticNames = new Map([[orbit, 'Orbit Style'], [alpha, 'Alpha Scene'],
      [mysticV1, 'Mystic XXX'], [mysticV4, 'Mystic XXX'], [mysticRef2va, 'Mystic XXX']]);
    const versionLabels = new Map([[mysticV1, 'V1'], [mysticV4, 'V4'], [mysticRef2va, 'V4 · ref2va']]);
    const recordFor = filename => {
      const override = aliases.get(filename) || null;
      return {filename, lora_id: `local:${filename}`, display_name: override || automaticNames.get(filename),
        display_name_override: override, suggested_name: automaticNames.get(filename), version_label: versionLabels.get(filename) || null, trained_words: [],
        preview_url: null, civitai_model_id: filename === orbit ? 42 : null, recommended_weights: null,
        managed: false, has_guide: false, nsfw: false, ...(dateFixtures[filename] || {})};
    };
    const installed = () => [
      ...[...automaticNames.keys()].map(filename => ({...recordFor(filename), directory: 'test-loras',
        linked: false, base_model: 'Test Base', hf_repo_id: null, size_bytes: 100, update_status: 'local'})),
      {...recordFor(mysticV4), directory: 'archive-loras', version_label: 'V4 · archived', linked: false,
        base_model: 'Test Base', hf_repo_id: null, size_bytes: 100, downloaded_at: null, released_at: null, update_status: 'local'},
    ];
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/*', async route => {
      const request = route.request(), url = new URL(request.url()), endpoint = url.pathname;
      const json = body => route.fulfill({json: body});
      if (request.method() === 'PUT' && endpoint === '/api/v1/loras/display-name') {
        const body = request.postDataJSON();
        nameUpdates.push(body);
        aliases.set(body.filename, body.display_name);
        return json(recordFor(body.filename));
      }
      if (endpoint === '/api/v1/loras/installed') return json({loras: installed()});
      if (endpoint === '/api/v1/loras/test_video/details') return json({loras: [...automaticNames.keys()].map(recordFor), guidance_max_phases: 1});
      if (endpoint === '/api/v1/loras/test_video') return json({loras: [...automaticNames.keys()], guidance_max_phases: 1});
      if (endpoint.endsWith('/guide')) return json({guide: null});
      if (endpoint === '/api/v1/presets') return json({presets: []});
      if (endpoint === '/api/v1/civitai/base-models') return json({filters: []});
      if (endpoint === '/api/v1/civitai/downloads') return json({downloads: []});
      if (endpoint === '/api/v1/civitai/search') return json({items: [], metadata: {}});
      if (endpoint === '/api/v1/civitai/model/42') { civitaiModelReads.push(endpoint); return json({id: 42}); }
      if (endpoint.startsWith('/api/')) return json({items: [], loras: [], directories: [], filters: [], downloads: [], presets: []});
      if (endpoint === '/') return route.fulfill({contentType: 'text/html', body:
        '<meta name="viewport" content="width=device-width,initial-scale=1"><div id="pickers" style="width:340px;max-width:100vw"></div><div id="library"></div>'});
      return route.fulfill({status: 404, body: ''});
    });
    await page.goto('http://lora-display-names.test');
    await page.addStyleTag({content: css});
    await page.addScriptTag({content: bundle.outputFiles[0].text});
    await page.evaluate(() => window.mountPickers());
    try {
      await page.getByText('Alpha Scene', {exact: true}).first().waitFor();
    } catch (error) {
      console.error('Picker debug:', await page.locator('#pickers').innerText(), errors);
      throw error;
    }

    const pickerSorts = await page.evaluate(() => [...document.querySelectorAll('div[class*="max-h-"]')]
      .filter(list => list.querySelector('button[title^="File:"]'))
      .map(list => [...list.querySelectorAll('button[title^="File:"]')].map(button => button.innerText.trim().replace(/\s+/g, ' '))));
    assert.equal(pickerSorts.length, 2, 'Studio and Director each render an available-LoRA list');
    for (const names of pickerSorts) {
      assert.ok(names[0].includes('Alpha Scene') && names.at(-1).includes('Orbit Style'),
        `both pickers sort by resolved display name: ${names.join(', ')}`);
      assert.deepEqual(names.filter(name => name.startsWith('Mystic XXX')),
        ['Mystic XXX V1', 'Mystic XXX V4', 'Mystic XXX V4 · ref2va'],
        'same-name releases remain visibly distinct and version-sorted');
    }
    assert.equal(await page.locator('button[title="File: ' + orbit + '"]').count(), 2,
      'the original filename remains available in picker tooltips');

    const releasedDate = new Date(dateFixtures[orbit].released_at).toLocaleDateString();
    const downloadedDate = new Date(dateFixtures[alpha].downloaded_at).toLocaleDateString();
    const orbitAgeChips = await page.locator('#pickers span[title^="Released "]').evaluateAll(nodes => nodes.map(node => ({title: node.title, age: node.textContent.trim()})));
    const alphaAgeChips = await page.locator('#pickers span[title^="Downloaded "]').evaluateAll(nodes => nodes.map(node => ({title: node.title, age: node.textContent.trim()})));
    assert.equal(orbitAgeChips.length, 2, 'Studio and Director both display the valid release date');
    assert.ok(orbitAgeChips.every(chip => chip.title === `Released ${releasedDate}` && chip.age === '4d'),
      'picker age and tooltip use the valid release date even when the download is newer');
    assert.equal(alphaAgeChips.length, 2, 'Studio and Director both fall back from an invalid release date');
    assert.ok(alphaAgeChips.every(chip => chip.title === `Downloaded ${downloadedDate}` && chip.age === '1w'),
      'picker age and tooltip use a valid download date when release metadata is malformed');

    await page.locator('#pickers button[title^="Sorted by name. Click to sort by release date"]').first().click();
    const newestPickerRows = await page.evaluate(() => [...document.querySelectorAll('div[class*="max-h-"]')]
      .filter(list => list.querySelector('button[title^="File:"]'))
      .map(list => [...list.querySelectorAll('button[title^="File:"]')].map(button => button.title.slice('File: '.length))));
    assert.equal(newestPickerRows.length, 2);
    for (const filenames of newestPickerRows) {
      assert.deepEqual(filenames.slice(0, 2), [orbit, alpha],
        'newest-first picker order uses release first and valid download fallback');
      assert.ok(filenames.slice(2).every(filename => filename.startsWith('MysticXXX_')),
        'entries without either valid date sort after dated entries');
    }
    await page.locator('#pickers button[title^="Sorted by release date"]').first().click();

    const studioSearch = page.getByPlaceholder('Search LoRAs...').nth(0);
    await page.setViewportSize({width: 390, height: 844});
    await studioSearch.fill('ref2va');
    assert.equal(await page.locator(`button[title="File: ${mysticRef2va}"]`).count(), 2,
      'variant labels are searchable in both pickers');
    await studioSearch.fill('');

    const mysticRow = filename => page.locator(`#pickers button[title="File: ${filename}"]`).first().locator('xpath=..');
    await mysticRow(mysticV4).locator('button[aria-label^="Edit display name for"]').click();
    await mysticRow(mysticV4).getByLabel(`Display name for ${mysticV4}`).fill('Mystic Alias');
    if (process.env.MAESTRO_SCREENSHOT_DIR) {
      fs.mkdirSync(process.env.MAESTRO_SCREENSHOT_DIR, {recursive: true});
      await page.screenshot({path: path.join(process.env.MAESTRO_SCREENSHOT_DIR, 'mobile-picker-version-name.png')});
    }
    await mysticRow(mysticV4).locator('button[aria-label="Save LoRA display name"]').click();
    const mysticNameUpdate = nameUpdates.at(-1);
    assert.equal(mysticNameUpdate.filename, mysticV4, 'release aliases still target the unchanged filename');
    assert.equal(mysticNameUpdate.display_name, 'Mystic Alias');
    await page.waitForFunction(filename => {
      const rows = [...document.querySelectorAll(`button[title="File: ${filename}"]`)];
      return rows.length === 2 && rows.every(row => row.innerText.includes('Mystic Alias') && row.innerText.includes('V4'));
    }, mysticV4);
    for (const row of await page.locator(`#pickers button[title="File: ${mysticV4}"]`).all()) {
      const text = (await row.innerText()).replace(/\s+/g, ' ');
      assert.ok(text.includes('Mystic Alias') && text.includes('V4'), `renamed release keeps its version label: ${text}`);
    }
    await mysticRow(mysticV4).locator('button[aria-label^="Edit display name for"]').click();
    await page.getByRole('button', {name: 'Reset automatic LoRA name'}).first().click();
    const mysticReset = nameUpdates.at(-1);
    assert.equal(mysticReset.filename, mysticV4);
    assert.equal(mysticReset.display_name, null, 'release alias resets to its automatic friendly name');
    await page.locator(`#pickers button[title="File: ${mysticV4}"]`).first().getByText('Mystic XXX', {exact: true}).waitFor();

    await studioSearch.fill(orbit);
    assert.equal(await page.locator(`button[title="File: ${orbit}"]`).count(), 2,
      'search still matches original filename rows when the visible name differs');
    await studioSearch.fill('');

    await page.getByRole('button', {name: 'Edit display name for Orbit Style'}).first().click();
    const editorInput = page.getByLabel(`Display name for ${orbit}`).first();
    const editorBounds = await editorInput.evaluate(input => {
      const bounds = input.closest('.relative.inline-flex').getBoundingClientRect();
      const container = document.querySelector('#pickers').getBoundingClientRect();
      return {left: bounds.left, right: bounds.right, containerLeft: container.left, containerRight: container.right};
    });
    assert.ok(editorBounds.left >= editorBounds.containerLeft && editorBounds.right <= editorBounds.containerRight,
      'the complete inline editor stays inside the narrow mobile sidecar width');
    await editorInput.fill('Orbit Hero');
    if (process.env.MAESTRO_SCREENSHOT_DIR) {
      fs.mkdirSync(process.env.MAESTRO_SCREENSHOT_DIR, {recursive: true});
      await page.screenshot({path: path.join(process.env.MAESTRO_SCREENSHOT_DIR, 'mobile-picker-rename.png')});
    }
    await page.getByRole('button', {name: 'Save LoRA display name'}).click();
    await page.getByText('Orbit Hero', {exact: true}).nth(1).waitFor();
    assert.deepEqual(nameUpdates.at(-1), {filename: orbit, display_name: 'Orbit Hero', model_type: 'test_video'},
      'Studio rename sends the unchanged filename and model scope');
    assert.deepEqual(await page.evaluate(() => window.store.getState().params.activated_loras), [orbit],
      'editing the display name does not toggle or rename the activated filename key');
    assert.deepEqual(await page.evaluate(filename => window.store.getState().loraWeights[filename], orbit), [0.73],
      'refreshing a display name preserves the selected LoRA weight');

    await page.getByRole('button', {name: 'Edit display name for Orbit Hero'}).nth(1).click();
    await page.getByRole('button', {name: 'Reset automatic LoRA name'}).click();
    await page.getByText('Orbit Style', {exact: true}).nth(1).waitFor();
    assert.equal(nameUpdates.at(-1).display_name, null, 'reset explicitly requests the automatic name');
    assert.equal(nameUpdates.at(-1).filename, orbit, 'reset remains keyed by the source filename');

    await page.evaluate(() => window.mountBrowser());
    await page.getByRole('heading', {name: 'Model Browser'}).waitFor();
    await page.getByRole('button', {name: 'My LoRAs'}).click();
    await page.locator('#library').getByText('Alpha Scene', {exact: true}).waitFor();
    const libraryNames = await page.locator('#library [title^="File:"]').evaluateAll(nodes => nodes.map(node => node.textContent.trim()));
    assert.ok(libraryNames[0].includes('Alpha Scene') && libraryNames.some(name => name.includes('Orbit Style')),
      `My LoRAs sorts alphabetically by display name: ${libraryNames.join(', ')}`);
    const libraryMysticRows = await page.locator('#library [title^="File: MysticXXX"]').evaluateAll(nodes => nodes.map(node => node.parentElement.parentElement.innerText.trim().replace(/\s+/g, ' ')));
    assert.ok(['V1', 'V4', 'V4 · ref2va', 'V4 · archived'].every(label => libraryMysticRows.some(text => text.includes('Mystic XXX') && text.includes(label))),
      'My LoRAs cards show release labels and keep same-filename entries scoped by directory');
    const librarySort = page.locator('#library select[title="Sort installed LoRAs"]');
    await librarySort.selectOption('released');
    const datedCardOrder = await page.evaluate(({orbitFilename, alphaFilename}) => {
      const grid = [...document.querySelectorAll('#library .grid')].find(node =>
        node.querySelector(`[title="File: ${orbitFilename}"]`) && node.querySelector(`[title="File: ${alphaFilename}"]`));
      const cardFor = filename => grid?.querySelector(`[title="File: ${filename}"]`)?.closest('div.relative.rounded-lg.border');
      return {orbitIndex: cardFor(orbitFilename) ? [...grid.children].indexOf(cardFor(orbitFilename)) : -1,
        alphaIndex: cardFor(alphaFilename) ? [...grid.children].indexOf(cardFor(alphaFilename)) : -1};
    }, {orbitFilename: orbit, alphaFilename: alpha});
    assert.ok(datedCardOrder.orbitIndex >= 0 && datedCardOrder.orbitIndex < datedCardOrder.alphaIndex,
      'My LoRAs newest-release sort uses valid release first and download fallback');
    const orbitLibraryDate = await page.locator(`#library [title="Released ${releasedDate}"]`).first().textContent();
    const alphaLibraryDate = await page.locator(`#library [title="Downloaded ${downloadedDate}"]`).first().textContent();
    assert.equal(orbitLibraryDate?.trim(), `Released ${releasedDate}`);
    assert.equal(alphaLibraryDate?.trim(), `Downloaded ${downloadedDate}`,
      'My LoRAs labels the actual fallback source and does not call it Updated or Added');
    assert.equal(await page.locator('#library [title^="Updated "]').count(), 0,
      'guide metadata write time is not presented as a LoRA update date');
    await page.getByPlaceholder('Search CivitAI...').fill(alpha);
    await page.locator('#library').getByText('Alpha Scene', {exact: true}).waitFor();
    assert.ok(await page.locator('#library [title="File: ' + alpha + '"]').count() > 0,
      'My LoRAs search matches the original filename');
    await page.getByPlaceholder('Search CivitAI...').fill('');

    const library = page.locator('#library');
    const libraryMysticRow = library.locator(`[title="File: ${mysticV4}"]`).first().locator('xpath=..');
    await libraryMysticRow.locator('button[aria-label^="Edit display name for"]').click();
    await libraryMysticRow.getByLabel(`Display name for ${mysticV4}`).fill('Library Mystic');
    if (process.env.MAESTRO_SCREENSHOT_DIR) {
      await page.screenshot({path: path.join(process.env.MAESTRO_SCREENSHOT_DIR, 'mobile-my-loras-version-name.png')});
    }
    await libraryMysticRow.locator('button[aria-label="Save LoRA display name"]').click();
    await library.getByText('Library Mystic', {exact: true}).first().waitFor();
    const libraryMysticText = await library.locator(`[title="File: ${mysticV4}"]`).first().locator('xpath=../..').innerText();
    assert.ok(libraryMysticText.includes('Library Mystic') && libraryMysticText.includes('V4'),
      `My LoRAs retains the version label when displaying a custom alias: ${libraryMysticText}`);
    await page.locator(`#pickers button[title="File: ${mysticV4}"]`).first().getByText('Library Mystic', {exact: true}).waitFor();
    assert.deepEqual(nameUpdates.at(-1), {filename: mysticV4, display_name: 'Library Mystic', directory: 'test-loras'},
      'My LoRAs rename sends the unchanged filename and directory scope');
    await library.locator(`[title="File: ${mysticV4}"]`).first().locator('xpath=../..').locator('button[aria-label^="Edit display name for"]').click();
    await library.getByRole('button', {name: 'Reset automatic LoRA name'}).click();
    assert.equal(nameUpdates.at(-1).filename, mysticV4);
    assert.equal(nameUpdates.at(-1).display_name, null);
    await library.locator(`[title="File: ${mysticV4}"]`).first().locator('xpath=../..').getByText('Mystic XXX', {exact: true}).waitFor();
    const resetMysticText = await library.locator(`[title="File: ${mysticV4}"]`).first().locator('xpath=../..').innerText();
    assert.ok(resetMysticText.includes('Mystic XXX') && resetMysticText.includes('V4'),
      'resetting a My LoRAs alias restores the friendly name without losing its version label');
    assert.deepEqual(civitaiModelReads, [], 'rename controls do not activate the parent CivitAI card');
    await page.getByRole('button', {name: 'Close Model Browser'}).click();
    assert.deepEqual(errors, [], 'alias editing has no browser errors');
    console.log('LoRA display names sort/search across pickers and library; aliases/reset preserve filename keys and scope');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
