// Uses the running app's read-only model catalogue. All browser requests,
// settings changes, uploads and generations are intercepted in an isolated origin.
// node tests/ui/sidebar_redesign.cjs http://127.0.0.1:<Maestro port>
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {assertPromptStability} = require('./prompt_stability.cjs');
const root = path.resolve(__dirname, '../..');
const base = process.argv[2];
if (!base) throw new Error('Pass the running Maestro URL; browser actions never reach it.');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const playwright = require(process.env.MAESTRO_PLAYWRIGHT || 'C:/Users/bliza/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const output = path.join(root, '.codex-tmp/sidebar-validation');
fs.mkdirSync(output, {recursive: true});
const read = async endpoint => {
  const response = await fetch(base + endpoint);
  if (!response.ok) throw new Error(endpoint + ': ' + response.status);
  return response.json();
};

(async () => {
  const catalogue = await read('/api/v1/models');
  const ids = ['minimax_h3_ref2va_fused_turbo', 'minimax_h3_ref2va', 'minimax_h3_fused_turbo', 'minimax_h3', 'viggle_animate', 'flux2_klein_9b', 'minimax_h3_voice_audio'];
  const options = Object.fromEntries(await Promise.all(ids.map(async id => [id, await read('/api/v1/model-options/' + id)])));
  const bundle = await esbuild.build({stdin: {contents: [
    "import React from 'react'; import {createRoot} from 'react-dom/client';",
    "import {useStore} from './src/stores/useStore'; import {Sidebar} from './src/components/Sidebar/Sidebar';",
    "import {applyThemePrefs, FAMILIES} from './src/lib/theme';",
    "window.store = useStore; window.themes = {applyThemePrefs, FAMILIES}; window.baseParams = {...useStore.getState().params};",
    "window.mount = () => {window.reactRoot = createRoot(document.getElementById('root')); window.reactRoot.render(<React.StrictMode><Sidebar/></React.StrictMode>);};",
  ].join('\n'), resolveDir: path.join(root, 'ui'), loader: 'tsx'}, bundle: true, write: false,
    jsx: 'automatic', define: {'process.env.NODE_ENV': '"development"'}, logLevel: 'silent'});
  const assets = path.join(root, 'ui/dist/assets');
  const css = fs.readFileSync(path.join(assets, fs.readdirSync(assets).find(name => name.endsWith('.css'))), 'utf8');
  // Headless Chromium normally hides scrollbars, concealing width/reflow bugs.
  const browser = await playwright.chromium.launch({headless: true, ignoreDefaultArgs: ['--hide-scrollbars'],
    ...(process.platform === 'win32' ? {executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'} : {})});
  try {
    const page = await browser.newPage({viewport: {width: 1360, height: 900}});
    const errors = [], requests = [];
    page.on('pageerror', error => { errors.push(error.message); console.error(error.stack); });
    const character = {id: 'blaine', name: 'Blaine', visual: {type: 'image', path: '/uploads/portrait.png', url: '/picture.svg', thumbnail_url: '/picture.svg'},
      voice: {path: '/uploads/voice.wav', url: '/voice.wav', duration_seconds: 3}};
    const characters = [character, ...Array.from({length: 15}, (_, n) => ({...character, id: 'person' + n, name: 'Person ' + n}))];
    const stats = {gpu: {available: true, name: 'Test GPU', percent: 0, vram_used_gb: 2, vram_total_gb: 24}, ram: {used_gb: 12, total_gb: 128, percent: 10}, cpu: {percent: 1}, model: {loaded: false}};
    await page.route('**/*', async route => {
      const url = new URL(route.request().url()), endpoint = url.pathname;
      const json = body => route.fulfill({contentType: 'application/json', body: JSON.stringify(body)});
      if (endpoint.includes('/model-options/')) return json(options[decodeURIComponent(endpoint.split('/').pop())] || {});
      if (endpoint === '/api/v1/models') return json(catalogue);
      if (endpoint === '/api/v1/system-stats') return json(stats);
      if (endpoint === '/api/v1/characters') return json({characters});
      if (endpoint.includes('/media-flow/capabilities')) return json({neural_rendering: {available: false, reason: 'Isolated test'}, frame_generation: {available: false, factors: [], reason: 'Isolated test'}});
      if (endpoint.includes('/upload')) return json({path: '/uploads/reference.png', url: '/picture.svg', duration_seconds: 3});
      if (endpoint === '/api/v1/generate') {
        requests.push(route.request().postDataJSON());
        return json({job_id: String(requests.length), status: 'held'});
      }
      if (endpoint.includes('/status/')) return json({status: 'completed', output_files: []});
      if (endpoint === '/api/v1/outputs') return json({outputs: [], total: 0});
      if (endpoint.startsWith('/api/')) return json({presets: [], loras: [], recipes: [], items: [], downloads: [], configured: true});
      if (endpoint === '/picture.svg') return route.fulfill({contentType: 'image/svg+xml', body: '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="400"><rect width="400" height="400" fill="#334155"/><circle cx="200" cy="140" r="90" fill="#c9a280"/><path d="M30 400Q200 90 370 400" fill="#8099aa"/></svg>'});
      if (endpoint !== '/') return route.fulfill({status: 404, body: ''});
      return route.fulfill({contentType: 'text/html', body: '<meta name="viewport" content="width=device-width,initial-scale=1"><div id="root" style="height:100dvh;display:flex"></div>'});
    });
    await page.goto('http://sidebar.test');
    await page.addStyleTag({content: css});
    await page.addScriptTag({content: bundle.outputFiles[0].text});
    await page.evaluate(({catalogue, options, stats}) => {
      window.options = options;
      window.resetFixture = (id = 'minimax_h3_ref2va_fused_turbo', mode = 'video', workflow = 'references') => {
        const options = window.options[id];
        window.store.setState({models: catalogue.models, families: catalogue.families, enabledModels: new Set(catalogue.models.map(model => model.model_type)),
          modelOptions: options, generationMode: mode, sidebarMode: 'studio', sidebarOpen: true,
          studioVideoWorkflow: workflow, studioImageWorkflow: 'generate', audioSubMode: 'speech',
          selectedModelPerMode: {[mode]: id}, durationSeconds: 124 / 24, slidingWindowSeconds: 243 / 24, slidingWindowOverlap: 18,
          slidingWindowLocked: false, h3WindowOverrides: {}, systemStats: stats, startImage: null, endImage: null, imageRefs: [],
          h3WindowPlan: null, spatialUpsampling: '', filmGrainIntensity: 0, jobs: [],
          params: {...window.baseParams, model_type: id, resolution: '864x480', prompt: 'A calm scene.', image_mode: workflow === 'extend' ? 3 : 0,
            _duration_planning_mode: 'duration', minimax_h3_sequence_prompt_mode: 'manual', minimax_h3_window_storyboard: false, minimax_h3_references: [],
            num_inference_steps: options.default_num_inference_steps || 4, guidance_scale: 1, seed: 42},
          resolutionPreset: '480p', aspectRatio: '16:9'});
      };
      localStorage.setItem('hwbar_collapsed', '1');
      window.resetFixture(); window.mount();
    }, {catalogue, options, stats});
    const pause = () => page.waitForTimeout(150);
    const sidebar = page.locator('.maestro-sidebar');
    const controls = page.getByTestId('studio-controls-scroll');
    const durationChip = () => sidebar.getByRole('button', {name: /^Duration:/});
    await sidebar.getByRole('button', {name: /Characters/}).waitFor();
    await pause();
    assert.deepEqual(errors, [], 'StrictMode renders the full sidebar without a loop');
    await assertPromptStability(page, sidebar);

    // Long workflow lists overlay the editor; the full catalogue remains reachable.
    const footerBeforeWorkflow = await page.getByTestId('studio-generate-bar').boundingBox();
    await sidebar.getByRole('button', {name: /^Workflow:/}).click();
    const workflows = page.getByRole('dialog', {name: 'Choose a workflow'});
    for (const label of ['Frames', 'References', 'Extend', 'Blend', 'Animate', 'Retake', 'Prompt Edit', 'Outpaint', 'Repaint', 'Recast']) {
      assert.equal(await workflows.getByRole('button', {name: new RegExp('^' + label + ' ')}).count(), 1, 'Workflow remains available: ' + label);
    }
    assert.deepEqual(await page.getByTestId('studio-generate-bar').boundingBox(), footerBeforeWorkflow, 'Workflow menu cannot push Generate offscreen');
    await workflows.getByRole('button', {name: /^References /}).click();

    // Full backend-provided option lists and selected values, including custom tiers.
    await sidebar.getByRole('button', {name: /^Resolution:/}).click();
    for (const preset of options[ids[0]].resolution_preset_order || ['480p', '540p', '720p', '1080p']) {
      const label = preset === 'auto' ? 'Auto' : options[ids[0]].resolution_presets?.[preset]?.label || preset;
      assert.equal(await page.getByRole('dialog', {name: 'Resolution', exact: true}).getByRole('button', {name: label, exact: true}).count(), 1, 'Full resolution option: ' + label);
    }
    await sidebar.getByRole('button', {name: /^Aspect ratio:/}).click();
    assert.equal(await page.getByRole('dialog', {name: 'Resolution', exact: true}).isVisible(), false, 'Switching settings opens only one overlay');
    for (const ratio of ['16:9', '9:16', '1:1', '4:3', '3:4']) assert.equal(await page.getByRole('dialog', {name: 'Aspect ratio', exact: true}).locator('button[aria-pressed]').filter({hasText: ratio}).count(), 1);
    await page.getByRole('dialog', {name: 'Aspect ratio', exact: true}).getByRole('button', {name: /3:4$/}).click();
    await sidebar.getByRole('button', {name: /^Aspect ratio:/}).click();
    assert.match(await sidebar.getByRole('button', {name: /^Aspect ratio:/}).innerText(), /3:4/);
    await durationChip().click();
    const native = page.getByRole('dialog', {name: 'Duration & windows'}).getByRole('slider', {name: 'Duration model duration'});
    await native.focus(); await native.press('ArrowRight');
    assert.match(await durationChip().innerText(), /5\.9/);
    await native.press('End');
    assert.ok(await page.evaluate(() => window.store.getState().durationSeconds <= 300));
    await page.getByRole('dialog', {name: 'Duration & windows'}).getByRole('button', {name: '60m', exact: true}).click();
    assert.ok(await page.evaluate(() => window.store.getState().durationSeconds > 3590));
    await durationChip().click();
    await page.evaluate(() => window.store.getState().setDurationSeconds(124 / 24));
    await pause();
    assert.equal(await page.evaluate(() => window.store.getState().params.sliding_window_size), 124, 'Collapsed duration still normalizes window size');
    console.log('Full resolution/aspect lists, native duration steps, 5m slider, 60m preset and collapsed window sizing passed');

    // Saved character adds one grouped tile with both media entries.
    await sidebar.getByRole('button', {name: /Characters/}).click();
    let dialog = page.getByRole('dialog', {name: 'Characters', exact: true});
    assert.ok((await dialog.boundingBox()).x > (await sidebar.boundingBox()).width, 'Desktop character library opens beside the sidebar');
    await page.screenshot({path: path.join(output, 'desktop-characters.png')});
    await dialog.getByRole('button', {name: 'Add Blaine to references'}).click();
    await dialog.getByRole('button', {name: 'Close Characters', exact: true}).click();
    assert.equal(await sidebar.locator('.media-input-card').count(), 1);
    assert.equal(await page.evaluate(() => window.store.getState().params.minimax_h3_references.length), 2);
    assert.equal(await sidebar.getByRole('button', {name: 'Add reference', exact: true}).count(), 1);
    await sidebar.getByLabel('Add reference files').setInputFiles({name: 'scene.png', mimeType: 'image/png', buffer: Buffer.from('test')});
    await pause();
    assert.equal(await sidebar.locator('.media-input-card').count(), 2);
    assert.equal(await sidebar.getByRole('button', {name: 'Add reference', exact: true}).count(), 1);
    await sidebar.locator('.media-input-card').nth(1).getByRole('button', {name: /^Edit /}).click();
    await page.getByLabel('Picture 2 use').selectOption('style');
    await page.getByRole('button', {name: 'Move Picture 2 earlier'}).click();
    assert.equal(await page.evaluate(() => window.store.getState().params.minimax_h3_references[0].image_intent), 'style');
    assert.equal(await page.evaluate(() => window.store.getState().params.minimax_h3_references.slice(1).every(ref => ref.library_character_id === 'blaine')), true);
    await page.getByRole('dialog', {name: /settings$/}).press('Escape');
    await page.evaluate(() => {
      const s = window.store.getState();
      window.store.setState({modelOptions: {...s.modelOptions, omni_reference_limits: {image: 2, video: 0, audio: 1, total: 3}}});
    });
    assert.equal(await sidebar.getByRole('button', {name: 'Add reference', exact: true}).count(), 0, 'Trailing tile stops at the combined model limit');
    await sidebar.getByRole('button', {name: 'Remove Picture 1', exact: true}).click();
    assert.equal(await sidebar.getByRole('button', {name: 'Add reference', exact: true}).count(), 1, 'Removing media frees a slot');
    await sidebar.getByLabel('Add reference files').setInputFiles({name: 'scene.png', mimeType: 'image/png', buffer: Buffer.from('test')});
    await pause();
    await sidebar.locator('.media-input-card').nth(1).getByRole('button', {name: /^Edit /}).click();
    await page.getByLabel('Picture 2 use').selectOption('style');
    await page.getByLabel('Replace Picture 2', {exact: true}).setInputFiles({name: 'replacement.png', mimeType: 'image/png', buffer: Buffer.from('test')});
    await pause();
    assert.equal(await page.evaluate(() => window.store.getState().params.minimax_h3_references[2].image_intent), 'style', 'Replace retains the role and order');
    assert.equal(await page.evaluate(() => window.store.getState().params.minimax_h3_references[2].filename), 'replacement.png');
    await page.getByRole('dialog', {name: /settings$/}).press('Escape');
    await page.evaluate(() => window.store.setState({modelOptions: window.options.minimax_h3_ref2va_fused_turbo}));
    console.log('Character appearance/voice grouping, uploads, trailing add tile, roles and accessible reference ordering passed');

    // Expansion preserves the actual textarea and its selection; it is not a second editor.
    const script = Array.from({length: 30}, (_, n) => 'Window ' + (n + 1) + ': Blaine <d>This is my tutorial dialogue.</d>').join('\n');
    const prompt = sidebar.getByRole('textbox', {name: 'Generation prompt'});
    await prompt.fill(script);
    await prompt.evaluate(node => {window.originalTextarea = node; node.setSelectionRange(10, 25)});
    const generateY = (await page.getByTestId('studio-generate-bar').boundingBox()).y;
    await controls.evaluate(node => {node.scrollTop = node.scrollHeight});
    assert.equal((await page.getByTestId('studio-generate-bar').boundingBox()).y, generateY);
    assert.ok((await prompt.boundingBox()).height >= 240, 'Prompt fills the desktop composition area');
    assert.ok(await prompt.evaluate(node => node.scrollHeight > node.clientHeight), 'Long scripts scroll inside the editor');
    await sidebar.getByRole('button', {name: 'Expand prompt editor'}).click();
    assert.equal(await page.evaluate(() => window.originalTextarea === document.querySelector('[aria-label="Generation prompt"]')), true);
    await page.getByRole('dialog', {name: 'Expanded prompt editor'}).getByRole('button', {name: 'Done', exact: true}).click();
    assert.equal(await prompt.inputValue(), script);
    assert.equal(await prompt.evaluate(node => node.selectionStart), 10);
    await prompt.fill('A calm scene.');
    console.log('Long prompt, editor identity/selection and anchored Generate passed');

    // Reviewed H3 per-window prompts survive the compact/expanded presentation.
    await page.evaluate(() => {
      const s = window.store.getState();
      s.setParam('minimax_h3_sequence_prompt_mode', 'auto'); s.setDurationSeconds(30);
    });
    await pause();
    await page.evaluate(() => {
      const s = window.store.getState();
      const windows = [1, 2, 3].map(index => ({index, title: 'Tutorial part ' + index,
        start_frame: (index - 1) * 240, end_frame: index * 240, start_seconds: (index - 1) * 10, end_seconds: index * 10,
        opening_state: '', closing_state: '', prompt: 'Exact window prompt ' + index}));
      window.store.setState({h3WindowPlan: {source_prompt: s.params.prompt, signature: 'reviewed-plan', plan_kind: 'reference_sequence',
        planned_by: 'llm', total_frames: s.params.video_length, window_frames: s.params.sliding_window_size, window_count: 3,
        resolution: s.params.resolution, model_type: s.params.model_type, windows, window_prompts: windows.map(window => window.prompt)}});
    });
    await sidebar.getByRole('button', {name: /Exact H3 prompts/}).click();
    let writing = page.getByRole('dialog', {name: 'Expanded prompt editor'});
    await writing.locator('textarea[title]').nth(1).fill('My revised second-window dialogue.');
    await writing.getByRole('button', {name: 'Done', exact: true}).click();
    assert.equal(await page.evaluate(() => window.store.getState().h3WindowPlan.windows[1].prompt), 'My revised second-window dialogue.');
    await sidebar.getByRole('button', {name: 'Expand prompt editor'}).click();
    writing = page.getByRole('dialog', {name: 'Expanded prompt editor'});
    assert.equal(await writing.locator('textarea[title]').nth(1).inputValue(), 'My revised second-window dialogue.');
    await writing.press('Escape');
    await page.evaluate(() => {
      const s = window.store.getState(); s.clearH3WindowPlan(); s.setParam('minimax_h3_sequence_prompt_mode', 'manual'); s.setDurationSeconds(124 / 24);
    });
    console.log('Exact H3 window-plan review, editing and retention passed');

    await controls.evaluate(node => {node.scrollTop = 0});
    const advancedTrigger = sidebar.getByRole('button', {name: /^Advanced settings/});
    const advanced = page.getByRole('dialog', {name: 'Advanced settings', exact: true});
    const advancedSection = key => advanced.getByTestId('advanced-' + key);
    const disclose = async (key, expanded = true) => {
      const section = advancedSection(key);
      if (await section.evaluate(node => node.open) !== expanded) await section.locator(':scope > summary').click();
      assert.equal(await section.evaluate(node => node.open), expanded);
    };
    const assertSettingsOverlay = async () => {
      const triggerBox = await advancedTrigger.boundingBox(), panelBox = await advanced.boundingBox();
      const viewport = page.viewportSize();
      assert.ok(triggerBox.height <= 44, 'Settings indicators keep their compact height');
      assert.ok(panelBox.x >= 0 && panelBox.x + panelBox.width <= viewport.width && panelBox.y >= 0 && panelBox.y + panelBox.height <= viewport.height, 'Settings overlay fits the viewport');
      if (viewport.width >= 768) assert.ok(panelBox.y + panelBox.height < triggerBox.y, 'Desktop settings open upward from the indicator strip');
      assert.equal(await advancedTrigger.getAttribute('aria-controls'), await advanced.getAttribute('id'));
    };
    const dockBeforeAdvanced = await page.getByTestId('studio-generate-bar').boundingBox();
    await advancedTrigger.click();
    const promptBeforeAdvanced = await prompt.boundingBox();
    await assertSettingsOverlay();
    assert.deepEqual(await page.getByTestId('studio-generate-bar').boundingBox(), dockBeforeAdvanced, 'Expanding Advanced leaves Generate anchored');
    assert.deepEqual(await prompt.boundingBox(), promptBeforeAdvanced, 'Opening Advanced does not move the prompt');
    for (const key of ['loras', 'performance', 'finishing', 'generation']) {
      assert.equal(await advancedSection(key).evaluate(node => node.open), false, 'Advanced initially shows collapsed section headings');
    }
    await disclose('finishing');
    await advanced.getByRole('checkbox', {name: 'Refine faces after generation'}).check();
    await disclose('performance');
    assert.equal(await advanced.getByRole('checkbox', {name: 'Refine faces after generation'}).isVisible(), true, 'Sections expand independently');
    await disclose('finishing', false);
    assert.equal(await advanced.getByRole('checkbox', {name: 'Refine faces after generation'}).isVisible(), false);
    await advanced.getByLabel('Reference detail', {exact: true}).selectOption('max');
    assert.equal(await page.evaluate(() => window.store.getState().params.minimax_h3_reference_detail), 'max');
    await disclose('loras');
    await advanced.getByRole('button', {name: 'Save Current'}).click();
    await advanced.getByPlaceholder('Preset name...').fill('Unfinished setup');
    await disclose('loras', false);
    assert.equal(await advanced.getByPlaceholder('Preset name...').isVisible(), false);
    await advancedSection('loras').locator(':scope > summary').press('Enter');
    assert.equal(await advanced.getByPlaceholder('Preset name...').inputValue(), 'Unfinished setup', 'Keyboard expansion retains the draft');
    await advancedTrigger.click();
    assert.equal(await advanced.isVisible(), false, 'Pressing Advanced again collapses it');
    await advancedTrigger.click();
    assert.equal(await advancedSection('loras').evaluate(node => node.open), true);
    assert.equal(await advanced.getByPlaceholder('Preset name...').inputValue(), 'Unfinished setup', 'Collapsing preserves an unsaved preset draft');
    assert.equal(await page.evaluate(() => window.store.getState().params.minimax_h3_reference_detail), 'max', 'Reference preparation choice survives closing the overlay');
    await advanced.getByPlaceholder('Preset name...').press('Escape');
    assert.equal(await advanced.isVisible(), false);
    assert.equal(await advancedTrigger.evaluate(node => document.activeElement === node), true, 'Escape restores the Advanced trigger');
    assert.match(await advancedTrigger.getAttribute('title'), /Face refinement/);
    await sidebar.getByRole('button', {name: /Characters/}).click();
    await page.getByRole('dialog', {name: 'Characters', exact: true}).getByRole('button', {name: /Face refinement & character mapping/}).click();
    assert.equal(await advancedSection('finishing').evaluate(node => node.open), true, 'Character shortcut expands Finishing');
    assert.equal(await advancedSection('loras').evaluate(node => node.open), false, 'Shortcut brings finishing into view');
    assert.equal(await advanced.getByRole('checkbox', {name: 'Refine faces after generation'}).isChecked(), true);
    await pause();
    assert.equal(await advanced.evaluate(node => document.activeElement === node), true, 'Character shortcut focuses Finishing in the settings overlay');
    assert.equal(await page.getByRole('dialog', {name: 'Characters', exact: true}).isVisible(), false);
    await advanced.getByRole('button', {name: 'Close Advanced settings'}).click();
    console.log('Settings overlays, drafts, active indicators and shared face-refinement settings passed');

    // Real submission routing, intercepted before it can create any work.
    await page.evaluate(() => window.store.getState().setParam('face_refiner', {enabled: false}));
    await sidebar.getByRole('button', {name: 'Add current Studio settings to the queue'}).click();
    await pause();
    assert.equal(requests.at(-1)._queue_mode, 'held');
    await sidebar.getByRole('button', {name: 'Generate', exact: true}).click();
    await pause();
    assert.notEqual(requests.at(-1)._queue_mode, 'held');
    assert.equal(requests.at(-1).minimax_h3_references.length, 3);
    await sidebar.getByRole('button', {name: 'Recipes and model browser'}).click();
    await page.getByRole('dialog', {name: 'Studio libraries'}).getByRole('button', {name: 'Open recipes'}).click();
    assert.equal(await page.evaluate(() => window.store.getState().recipesOpen), true);
    await sidebar.getByRole('button', {name: 'Recipes and model browser'}).click();
    await page.getByRole('dialog', {name: 'Studio libraries'}).getByRole('button', {name: 'Open model browser'}).click();
    assert.equal(await page.evaluate(() => window.store.getState().loraBrowserOpen), true);
    await sidebar.getByRole('button', {name: 'Choose model'}).click();
    await page.getByRole('dialog', {name: 'Choose a model'}).getByRole('button', {name: 'Browse models, LoRAs & characters'}).click();
    assert.equal(await page.evaluate(() => window.store.getState().loraBrowserOpen), true, 'Model picker also exposes the browser');
    console.log('Generate vs held queue payloads, reference preservation, Recipes and Browser routing passed');

    // All six palettes at desktop and mobile; the actual theme function is used.
    for (const viewport of [{width: 1360, height: 900}, {width: 767, height: 844}, {width: 440, height: 844}, {width: 390, height: 844}, {width: 320, height: 568}]) {
      await page.setViewportSize(viewport);
      await page.evaluate(() => window.store.setState({sidebarOpen: true}));
      for (const family of ['default', 'golden-hour', 'onyx']) for (const mode of ['dark', 'light']) {
        await page.evaluate(prefs => window.themes.applyThemePrefs(prefs), {family, mode});
        await controls.evaluate(node => {node.scrollTop = 0});
        await pause();
        assert.ok(await sidebar.evaluate(node => node.scrollWidth <= node.clientWidth + 1), 'Sidebar has no horizontal overflow');
        assert.ok(await controls.evaluate(node => node.scrollWidth <= node.clientWidth + 1), 'Controls fit width ' + viewport.width);
        assert.ok(await page.getByTestId('studio-settings-strip').evaluate(node => node.scrollWidth <= node.clientWidth + 1), 'Every settings indicator fits width ' + viewport.width);
        assert.ok(await page.getByTestId('studio-settings-strip').evaluate(node => {
          const buttons = [...node.querySelectorAll('button')].map(button => button.getBoundingClientRect()).filter(box => box.width > 0);
          return buttons.every((a, index) => buttons.slice(index + 1).every(b =>
            a.right <= b.left || b.right <= a.left || a.bottom <= b.top || b.bottom <= a.top));
        }), 'Characters and output settings never overlap at width ' + viewport.width);
        assert.equal(await sidebar.locator('.studio-advanced-label').isVisible(), viewport.width >= 768, 'Advanced compacts to the sidebar width, including wider mobile viewports');
        const toolbarBounds = await sidebar.getByRole('group', {name: 'Prompt controls', exact: true}).boundingBox();
        const promptBounds = await prompt.boundingBox();
        const writingModeBounds = await sidebar.getByRole('combobox', {name: 'Prompt writing mode'}).boundingBox();
        assert.ok(writingModeBounds.y >= promptBounds.y + promptBounds.height && writingModeBounds.y >= toolbarBounds.y, 'Prompt mode stays below the text');
        const expandBounds = await sidebar.getByRole('button', {name: 'Expand prompt editor'}).boundingBox();
        assert.ok(Math.abs(expandBounds.x + expandBounds.width - toolbarBounds.x - toolbarBounds.width) < 1, 'Prompt tools align to the bottom-right');
        const stripBounds = await page.getByTestId('studio-settings-strip').boundingBox();
        const settingsBounds = await page.getByTestId('studio-output-settings').boundingBox();
        assert.ok(Math.abs(settingsBounds.x + settingsBounds.width - stripBounds.x - stripBounds.width) < 1, 'Settings group aligns to the right edge');
        assert.ok(await page.getByTestId('studio-output-settings').evaluate(node => {
          const boxes = [...node.children].map(child => child.getBoundingClientRect());
          return boxes.slice(1).every((box, i) => Math.abs(box.left - boxes[i].right - 4) < 1);
        }), 'Settings keep compact, even gaps instead of spreading out');
        const modelBox = await sidebar.getByRole('button', {name: 'Choose model'}).boundingBox();
        const bar = await page.getByTestId('studio-generate-bar').boundingBox();
        assert.ok(modelBox.y >= bar.y && modelBox.y + modelBox.height <= bar.y + bar.height, 'Model selector is beside Generate');
        assert.ok(bar.y >= 0 && bar.y + bar.height <= viewport.height, 'Generate stays in viewport');
        await page.screenshot({path: path.join(output, viewport.width + '-' + family + '-' + mode + '.png')});
        await advancedTrigger.click();
        await assertSettingsOverlay();
        await disclose('generation');
        assert.ok(await advanced.evaluate(node => node.scrollWidth <= node.clientWidth + 1), 'Advanced content fits width ' + viewport.width);
        const beforeScroll = await page.getByTestId('studio-generate-bar').boundingBox();
        await advanced.locator(':scope > div').last().evaluate(node => {node.scrollTop = node.scrollHeight});
        assert.deepEqual(await page.getByTestId('studio-generate-bar').boundingBox(), beforeScroll, 'Scrolling Advanced leaves Generate in place');
        await page.screenshot({path: path.join(output, viewport.width + '-' + family + '-' + mode + '-advanced.png')});
        await advanced.getByRole('button', {name: 'Close Advanced settings'}).click();
      }
    }
    await sidebar.getByRole('button', {name: /Characters/}).click();
    dialog = page.getByRole('dialog', {name: 'Characters', exact: true});
    await dialog.getByRole('button', {name: 'Add Person 14 to references'}).scrollIntoViewIfNeeded();
    assert.ok(await dialog.evaluate(node => node.scrollWidth <= node.clientWidth + 1));
    await page.screenshot({path: path.join(output, 'mobile-characters.png')});
    await dialog.getByRole('button', {name: 'Close Characters'}).click();
    await prompt.focus();
    await page.evaluate(() => {
      Object.defineProperty(window.visualViewport, 'height', {configurable: true, value: 370});
      window.visualViewport.dispatchEvent(new Event('resize'));
    });
    await pause();
    let bar = await page.getByTestId('studio-generate-bar').boundingBox();
    assert.ok(bar.y + bar.height <= 371, 'Generate follows the visual viewport when the keyboard opens');
    assert.equal(await sidebar.locator('.studio-hardware').isVisible(), false);
    assert.ok((await prompt.boundingBox()).height >= 100, 'Typing has a usable prompt area above the keyboard');
    assert.equal(await page.evaluate(() => localStorage.getItem('hwbar_collapsed')), '1', 'Keyboard does not change the saved hardware preference');
    await advancedTrigger.click();
    const keyboardOverlay = await advanced.boundingBox();
    assert.ok(keyboardOverlay.y >= 0 && keyboardOverlay.y + keyboardOverlay.height <= 370, 'Settings sheet follows the keyboard viewport');
    await advanced.getByRole('button', {name: 'Close Advanced settings'}).click();
    await sidebar.getByRole('button', {name: 'Expand prompt editor'}).click();
    bar = await page.getByRole('dialog', {name: 'Expanded prompt editor'}).getByRole('button', {name: 'Done', exact: true}).boundingBox();
    assert.ok(bar.y + bar.height <= 371, 'Expanded editor follows the available visual viewport too');
    await page.getByRole('dialog', {name: 'Expanded prompt editor'}).press('Escape');
    await page.evaluate(() => { delete window.visualViewport.height; window.visualViewport.dispatchEvent(new Event('resize')); });
    console.log('Six theme variants at 1360px, 767px, 440px, 390px and 320px: no overlapping indicators, bottom prompt tools, mobile character scrolling and simulated keyboard viewport passed');

    // Recreate the original Extend model transition against real model metadata.
    await page.setViewportSize({width: 1360, height: 900});
    await page.evaluate(() => window.resetFixture('minimax_h3_fused_turbo', 'video', 'extend'));
    await sidebar.getByRole('button', {name: 'Choose model'}).click();
    const fullName = catalogue.models.find(model => model.model_type === 'minimax_h3').name;
    await page.getByRole('dialog', {name: 'Choose a model'}).getByRole('button', {name: fullName, exact: false}).first().click();
    await pause();
    assert.equal(await page.evaluate(() => window.store.getState().params.model_type), 'minimax_h3');
    assert.deepEqual(errors, [], 'Extend switch must not reproduce React #185');
    for (const width of [1360, 390, 320]) {
      await page.setViewportSize({width, height: 844});
      await page.evaluate(() => window.resetFixture('minimax_h3', 'video', 'frames'));
      await pause();
      const frameTiles = await controls.locator('.studio-media-grid').evaluate(node => [...node.children].map(child => {
        const bounds = child.getBoundingClientRect();
        return {x: bounds.x, y: bounds.y, width: bounds.width, height: bounds.height};
      }));
      assert.ok(frameTiles.length >= 3);
      assert.ok(frameTiles.slice(0, 3).every(tile => tile.y === frameTiles[0].y), 'Frames inputs are three across at ' + width);
      assert.ok(frameTiles.slice(0, 3).every(tile => Math.abs(tile.width - frameTiles[0].width) < 1 && tile.height === frameTiles[0].height));
      await page.screenshot({path: path.join(output, width + '-frames-tiles.png')});
      await page.evaluate(() => window.resetFixture('minimax_h3_ref2va', 'video', 'references'));
      await pause();
      const referenceTile = await sidebar.getByRole('button', {name: 'Add reference', exact: true}).evaluate(node => {
        const bounds = node.parentElement.getBoundingClientRect(); return {width: bounds.width, height: bounds.height};
      });
      assert.ok(Math.abs(referenceTile.width - frameTiles[0].width) < 1 && referenceTile.height === frameTiles[0].height, 'Frames and Reference share compact tile dimensions at ' + width);
    }
    await page.setViewportSize({width: 1360, height: 900});
    for (const [id, mode, workflow] of [['minimax_h3', 'video', 'frames'], ['viggle_animate', 'video', 'animate'], ['flux2_klein_9b', 'image', 'frames'], ['minimax_h3_voice_audio', 'audio', 'frames']]) {
      await page.evaluate(args => window.resetFixture(...args), [id, mode, workflow]);
      await pause();
      assert.ok(await sidebar.innerText());
      assert.ok(await sidebar.getByRole('button', {name: 'Choose model'}).evaluate(node => node.parentElement.getBoundingClientRect().height < 70), 'Closed model control does not stretch the column');
      assert.ok(await controls.evaluate(node => node.scrollWidth <= node.clientWidth + 1), id + ' fits');
      await page.screenshot({path: path.join(output, id + '.png')});
      if (mode === 'image' || mode === 'audio') {
        await advancedTrigger.click();
        assert.equal(await advancedSection('generation').isVisible(), true, 'Applicable generation controls remain available');
        assert.equal(await advancedSection('loras').isVisible(), true, 'Presets remain available even without installed LoRAs');
        if (mode === 'image') assert.equal(await advancedSection('performance').isVisible(), false, 'Flux has no empty Performance section');
        if (mode === 'audio') assert.equal(await advancedSection('finishing').isVisible(), false, 'Speech has no empty Finishing section');
        await page.screenshot({path: path.join(output, id + '-advanced.png')});
        await advanced.press('Escape');
        await sidebar.getByRole('button', {name: 'Characters', exact: true}).click();
        const library = page.getByRole('dialog', {name: mode === 'image' ? 'Choose a character' : 'Voice characters'});
        assert.ok((await library.boundingBox()).x > (await sidebar.boundingBox()).width, mode + ' characters use the desktop side library');
        await library.press('Escape');
        if (mode === 'audio') {
          const speechTools = sidebar.getByRole('button', {name: 'Speech enhancement options'});
          await speechTools.click();
          const speechChoice = sidebar.getByRole('button', {name: /Write Speech/}).first();
          assert.ok((await speechChoice.boundingBox()).y < (await speechTools.boundingBox()).y, 'Speech enhancement menu opens upward from the bottom toolbar');
          assert.ok((await speechChoice.boundingBox()).y >= 0, 'Speech choices fit the viewport');
          await speechTools.click();
        }
      }
    }
    assert.deepEqual(errors, []);
    console.log('Extend full-model transition plus Frames, Viggle, Image and H3 Speech rendered without errors');
    console.log('Sidebar regression checks passed. Screenshots: ' + output);
  } catch (error) {
    for (const context of browser.contexts()) for (const page of context.pages()) await page.screenshot({path: path.join(output, 'failure.png')}).catch(() => {});
    throw error;
  } finally { await browser.close(); }
})().catch(error => {console.error(error); process.exitCode = 1;});
