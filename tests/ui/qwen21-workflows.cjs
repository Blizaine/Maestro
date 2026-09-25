// Real React + Zustand Qwen 2.1 regression with every browser request mocked.
// Run: node tests/ui/qwen21-workflows.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const {chromium} = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');

const qwenId = 'qwen_image_21_7B';
const fluxId = 'flux2_klein_9b';
const ratios = ['21:9', '16:9', '9:16', '1:1', '4:3', '3:4'];
const qwen2k = {
  auto: 'auto_2k',
  '21:9': '3136x1344', '16:9': '2752x1536', '9:16': '1536x2752',
  '1:1': '2048x2048', '4:3': '2400x1792', '3:4': '1792x2400',
};
const flux720 = {
  '21:9': '1200x520', '16:9': '1024x576', '9:16': '576x1024',
  '1:1': '720x720', '4:3': '960x720', '3:4': '720x960',
};

const qwenDefaults = JSON.parse(fs.readFileSync(path.join(root, 'app/defaults', `${qwenId}.json`), 'utf8'));
const fluxDefaults = JSON.parse(fs.readFileSync(path.join(root, 'app/defaults', `${fluxId}.json`), 'utf8'));
const qwenModel = {
  ...qwenDefaults.model, model_type: qwenId, family: 'qwen',
  architecture: qwenId, supports_image_edit: true,
  supports_image_inpaint: true, supports_image_outpaint: true,
};
const fluxModel = {
  ...fluxDefaults.model, model_type: fluxId, family: 'flux2',
  architecture: fluxId, supports_image_edit: true,
  supports_image_inpaint: true, supports_image_outpaint: true,
};
const refChoices = {choices: [['None', ''], ['Main image / landscape first', 'KI'], ['Reference images', 'I']],
  letters_filter: 'KI', default: 'I'};
const qwenOptions = {
  model_type: qwenId, architecture: qwenId, fps: 24, guidance_max_phases: 1,
  default_num_inference_steps: 40, default_guidance_scale: 4,
  image_outputs: true, supports_image_edit: true, max_image_refs: 10,
  supports_auto_aspect: true,
  image_ref_choices: refChoices, image_ref_inpaint: true,
  inpaint_video_prompt_type: 'VAG',
  model_modes: {label: 'Inpainting Method', image_modes: [2], default: 0, choices: [
    ['Masked Denoising', 0], ['LanPaint (2 steps)', 2], ['LanPaint (5 steps)', 3],
    ['LanPaint (10 steps)', 4], ['LanPaint (15 steps)', 5],
  ]},
  guide_preprocessing: {
    selection: ['', 'PV', 'DV', 'EV', 'SV', 'CV', 'V'],
    labels: {V: 'Control Image'}, default: '',
  },
  qwen21_acceleration_profiles: {
    viggle_v01: {label: 'Viggle Turbo v0.1 (4 steps)', steps: 4, guidance: 1},
    viggle_v02: {label: 'Viggle Turbo v0.2 (5 steps)', steps: 5, guidance: 1},
    viggle_v021: {label: 'Viggle Turbo v0.2.1 (6 steps)', steps: 6, guidance: 1},
  },
  resolution_preset_order: ['auto', '480p', '540p', '720p', '1080p', '2k'],
  resolution_presets: {
    '2k': {label: '2K (native)', values: qwen2k},
  },
};
const fluxOptions = {
  model_type: fluxId, architecture: fluxId, fps: 24, guidance_max_phases: 1,
  default_num_inference_steps: 4, default_guidance_scale: 1,
  image_outputs: true, supports_image_edit: true, max_image_refs: 10,
  image_ref_choices: refChoices,
  resolution_preset_order: ['720p'],
  resolution_presets: {'720p': {label: '720p', values: flux720}},
};

(async () => {
  const fixture = {qwenId, fluxId, models: [fluxModel, qwenModel],
    families: [{id: 'flux2', label: 'Flux 2', order: 100}, {id: 'qwen', label: 'Qwen', order: 110}],
    qwenOptions, fluxOptions, qwenDefaults, fluxDefaults};
  const bundle = await esbuild.build({
    stdin: {contents: `
      import React from 'react';
      import {createRoot} from 'react-dom/client';
      import {useStore} from './src/stores/useStore';
      import {ModelSelector} from './src/components/Sidebar/ModelSelector';
      import {ImageWorkflowSelector} from './src/components/Sidebar/ImageWorkflowSelector';
      import {ImageWorkflowControls} from './src/components/Sidebar/ImageWorkflowControls';
      import {ImageRefSection} from './src/components/Sidebar/ImageRefSection';
      import {ControlVideoSection} from './src/components/Sidebar/ControlVideoSection';
      import {Qwen21Controls} from './src/components/Sidebar/Qwen21Controls';
      import {AspectRatioGrid} from './src/components/Sidebar/AspectRatioGrid';
      import {ResolutionPresets} from './src/components/Sidebar/ResolutionPresets';
      import {GenerateButton} from './src/components/Sidebar/GenerateButton';
      const fixture = ${JSON.stringify(fixture)};
      window.store = useStore;
      window.seed = () => {
        const state = useStore.getState();
        useStore.setState({
          models: fixture.models,
          families: fixture.families,
          enabledModels: new Set([fixture.qwenId, fixture.fluxId]),
          generationMode: 'image',
          studioImageWorkflow: 'generate',
          selectedModelPerMode: {image: fixture.qwenId},
          modelOptions: fixture.qwenOptions,
          modelOptionsLoading: false,
          resolutionPreset: '720p', aspectRatio: '16:9',
          imageRefs: [], imageRefType: '', imageWorkflowSourceFile: null,
          imageWorkflowSourcePath: '', imageWorkflowSourceUrl: '',
          imageWorkflowMaskFile: null, imageWorkflowMaskPath: '', imageWorkflowMaskUrl: '',
          jobs: [], isGenerating: false,
          params: {...state.params, ...fixture.qwenDefaults,
            model_type: fixture.qwenId, prompt: 'A blue ceramic bird on a white plinth.',
            image_mode: 1, video_prompt_type: 'I', sample_solver: 'default',
            num_inference_steps: 40, guidance_scale: 4, custom_settings: {},
            image_guide: undefined, image_mask: undefined, image_refs: undefined,
            model_mode: 0, activated_loras: [], loras_multipliers: ''},
        });
      };
      window.mount = () => {
        window.root = createRoot(document.getElementById('root'));
        window.root.render(<main>
          <ModelSelector/>
          <ImageWorkflowSelector/>
          <Qwen21Controls/>
          <AspectRatioGrid/>
          <ResolutionPresets/>
          <ControlVideoSection/>
          <ImageWorkflowControls/>
          <ImageRefSection/>
          <GenerateButton stretch/>
        </main>);
      };
    `, resolveDir: path.join(root, 'ui'), loader: 'tsx'},
    bundle: true, write: false, jsx: 'automatic', logLevel: 'silent',
    define: {'process.env.NODE_ENV': '"development"'},
  });
  const browser = await chromium.launch({headless: true, ...(process.platform === 'win32'
    ? {executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'} : {})});
  const submissions = [];
  const uploads = [];
  const externalAttempts = [];
  const requestedUrls = [];
  try {
    const context = await browser.newContext({viewport: {width: 900, height: 1100}});
    const page = await context.newPage();
    const pageErrors = [];
    page.on('pageerror', error => pageErrors.push(error.message));
    page.on('request', request => requestedUrls.push(request.url()));
    let prefs = {configured: true, generation_mode: 'image', studio_image_workflow: 'generate',
      selected_model_per_mode: {image: qwenId}};
    await page.route('**/*', async route => {
      const request = route.request();
      const url = new URL(request.url());
      if (url.hostname !== 'qwen21.test') {
        externalAttempts.push(request.url());
        return route.abort();
      }
      const key = url.pathname;
      if (key === '/') return route.fulfill({contentType: 'text/html', body: '<div id="root"></div>'});
      if (key === '/api/v1/studio-preferences') {
        if (request.method() === 'PUT') prefs = {...prefs, ...request.postDataJSON(), configured: true};
        return route.fulfill({json: prefs});
      }
      if (key.startsWith('/api/v1/model-options/')) {
        const modelType = decodeURIComponent(key.split('/').pop());
        return route.fulfill({json: modelType === fluxId ? fluxOptions : qwenOptions});
      }
      if (key.startsWith('/api/v1/defaults/')) {
        const modelType = decodeURIComponent(key.split('/').pop());
        return route.fulfill({json: modelType === fluxId ? fluxDefaults : qwenDefaults});
      }
      if (key.startsWith('/api/v1/loras/')) return route.fulfill({json: {loras: [], guidance_max_phases: 1}});
      if (key === '/api/v1/upload' && request.method() === 'POST') {
        const index = uploads.length;
        const fileNames = ['control.png', 'source.png', 'mask.png', 'optional-reference.png'];
        const path = `/uploads/${fileNames[index] || `reference-${index}.png`}`;
        uploads.push({path, index});
        return route.fulfill({json: {filename: path.split('/').pop(), path, url: `http://qwen21.test${path}`}});
      }
      if (key === '/api/v1/generate' && request.method() === 'POST') {
        submissions.push(request.postDataJSON());
        return route.fulfill({json: {job_id: `mock-job-${submissions.length}`, status: 'held'}});
      }
      // Unknown app endpoints stay on this fake origin and return inert data.
      return route.fulfill({json: {}});
    });
    await page.goto('http://qwen21.test');
    await page.addScriptTag({content: bundle.outputFiles[0].text});
    await page.evaluate(() => {window.seed(); window.mount();});
    await page.getByLabel('Image acceleration').waitFor();

    for (const [profile, steps] of [['viggle_v01', 4], ['viggle_v02', 5], ['viggle_v021', 6]]) {
      await page.getByLabel('Image acceleration').selectOption(profile);
      assert.deepEqual(await page.evaluate(() => {
        const params = window.store.getState().params;
        return {steps: params.num_inference_steps, cfg: params.guidance_scale};
      }), {steps, cfg: 1}, `${profile} selects its step count and CFG 1`);
    }
    await page.getByLabel('Image acceleration').selectOption('default');
    assert.deepEqual(await page.evaluate(() => {
      const params = window.store.getState().params;
      return {steps: params.num_inference_steps, cfg: params.guidance_scale};
    }), {steps: 40, cfg: 4}, 'Base model restores 40 steps and CFG 4');

    const submit = async () => {
      const count = submissions.length;
      const response = page.waitForResponse(res => new URL(res.url()).pathname === '/api/v1/generate');
      await page.getByRole('button', {name: 'Add current Studio settings to the queue'}).click();
      await response;
      assert.equal(submissions.length, count + 1, 'The UI submits one mocked generation request');
      return submissions.at(-1);
    };

    for (const ratio of ratios) {
      await page.getByRole('button', {name: ratio, exact: true}).click();
      await page.getByRole('button', {name: '2K (native)', exact: true}).click();
      assert.equal(await page.evaluate(() => window.store.getState().params.resolution), qwen2k[ratio],
        `${ratio} selects its own native 2K canvas`);
      const payload = await submit();
      assert.equal(payload.resolution, qwen2k[ratio], `${ratio} native 2K reaches the generation payload`);
    }
    await page.locator('button[aria-label="Auto"]').click();
    assert.equal(await page.evaluate(() => window.store.getState().params.resolution), qwen2k.auto,
      'Auto aspect resolves to the native 2K automatic canvas');
    const auto2kPayload = await submit();
    assert.equal(auto2kPayload.resolution, qwen2k.auto, 'Native 2K auto canvas reaches the generation payload');

    // The control-image workflow is independent from optional references.
    await page.getByRole('button', {name: '16:9', exact: true}).click();
    await page.getByRole('button', {name: '720p', exact: true}).click();
    await page.locator('#root select').nth(1).selectOption('V');
    const controlFileChooser = page.waitForEvent('filechooser');
    await page.getByText('Drop control image (.png, .jpg, .webp)', {exact: true}).click();
    await (await controlFileChooser).setFiles({name: 'control.png', mimeType: 'image/png', buffer: tinyPng()});
    await page.waitForFunction(() => window.store.getState().params.image_guide === '/uploads/control.png');
    await page.getByText('Up to 9 reference images. The source/control image is image 1.', {exact: true}).waitFor();
    const controlPayload = await submit();
    assert.equal(controlPayload.image_guide, '/uploads/control.png');
    assert.equal(controlPayload.video_prompt_type, 'V', 'Control-image transfer survives without reference images');
    assert.equal(controlPayload.image_refs, undefined, 'No synthetic reference image is attached');
    await page.evaluate(() => window.store.setState({
      imageRefs: Array.from({length: 10}, (_, index) => new File(['fixture'], `existing-${index + 1}.png`, {type: 'image/png'})),
      imageRefType: 'I',
    }));
    const overflowWarning = page.locator('#root p').filter({hasText: 'Remove 1 reference image(s) before generating.'});
    await overflowWarning.waitFor();
    assert.match(await overflowWarning.innerText(), /The source\/control image is image 1\./);
    await page.evaluate(() => window.store.setState({imageRefs: [], imageRefType: ''}));

    // Enter inpaint through the workflow picker and use its real source/mask inputs.
    await page.locator('button[title="Create from text or edit supplied images"]').click();
    await page.getByRole('dialog', {name: 'Choose a workflow'}).getByRole('button', {name: /Inpaint/}).click();
    const sourceChooser = page.waitForEvent('filechooser');
    await page.getByRole('button', {name: 'Upload an image', exact: true}).click();
    await (await sourceChooser).setFiles({name: 'source.png', mimeType: 'image/png', buffer: tinyPng()});
    await page.waitForFunction(() => window.store.getState().imageWorkflowSourcePath === '/uploads/source.png');
    const maskChooser = page.waitForEvent('filechooser');
    await page.getByRole('button', {name: 'Upload a black-and-white mask', exact: true}).click();
    await (await maskChooser).setFiles({name: 'mask.png', mimeType: 'image/png', buffer: tinyPng()});
    await page.waitForFunction(() => window.store.getState().imageWorkflowMaskPath === '/uploads/mask.png');
    await page.locator('#root select').nth(1).selectOption('2');
    await page.getByRole('checkbox', {name: /Cache reference attention/}).check();
    await page.getByLabel('Add reference files').setInputFiles({
      name: 'optional-reference.png', mimeType: 'image/png', buffer: tinyPng(),
    });
    await page.waitForFunction(() => window.store.getState().imageRefs.length === 1
      && ['KI', 'I'].includes(window.store.getState().imageRefType));
    await page.getByRole('img', {name: 'Ref 2'}).waitFor();
    await page.getByRole('button', {name: 'People / Objects', exact: true}).click();
    const inpaintPayload = await submit();
    assert.equal(inpaintPayload._studio_image_workflow, 'inpaint');
    assert.equal(inpaintPayload.image_mode, 2);
    assert.equal(inpaintPayload.image_guide, '/uploads/source.png');
    assert.equal(inpaintPayload.image_mask, '/uploads/mask.png');
    assert.deepEqual(inpaintPayload.image_refs, ['/uploads/optional-reference.png']);
    assert.equal(inpaintPayload.video_prompt_type, 'VAGI');
    assert.equal(inpaintPayload.model_mode, 2, 'LanPaint method reaches the payload');
    assert.equal(inpaintPayload.custom_settings.qwen21_kv_cache, 'Enabled', 'Cache preference reaches the payload');

    // Clear image-workflow state, then switch models in the actual selector.
    await page.getByRole('button', {name: '2K (native)', exact: true}).click();
    await page.evaluate(() => {
      const state = window.store.getState();
      window.store.setState({imageRefs: [], imageRefType: '',
        imageWorkflowSourceFile: null, imageWorkflowSourcePath: '', imageWorkflowSourceUrl: '',
        imageWorkflowMaskFile: null, imageWorkflowMaskPath: '', imageWorkflowMaskUrl: '',
        params: {...state.params, video_prompt_type: '', image_guide: undefined, image_mask: undefined, image_refs: undefined}});
    });
    await page.locator('button[title="Regenerate only a masked area"]').click();
    await page.getByRole('dialog', {name: 'Choose a workflow'}).getByRole('button', {name: /^Generate/}).click();
    await page.getByRole('button', {name: 'Choose model', exact: true}).click();
    await page.getByRole('button', {name: 'Flux 2 Klein 9B', exact: true}).click();
    await page.waitForFunction(id => window.store.getState().params.model_type === id
      && !window.store.getState().modelOptionsLoading
      && window.store.getState().modelOptions?.model_type === id, fluxId);
    assert.equal(await page.evaluate(() => window.store.getState().resolutionPreset), '720p',
      'A model without native 2K resets the selected preset to its supported tier');
    const fluxPayload = await submit();
    assert.equal(fluxPayload.model_type, fluxId);
    assert.equal(fluxPayload.resolution, flux720['16:9'], 'Switched model submits its own 720p per-aspect value');
    assert.equal(Object.values(qwen2k).includes(fluxPayload.resolution), false, 'No stale Qwen 2K canvas reaches Flux');
    assert.equal(await page.getByRole('button', {name: '2K (native)', exact: true}).count(), 0,
      'Unsupported 2K preset disappears from the switched model controls');

    assert.deepEqual(pageErrors, [], pageErrors.join('\n'));
    assert.deepEqual(externalAttempts, [], 'Every request is intercepted on the fake origin');
    assert.ok(requestedUrls.every(url => new URL(url).hostname === 'qwen21.test'), 'No request reaches a real app server');
    assert.ok(!requestedUrls.some(url => /\/gpu(?:\/|\?|$)|\/cuda(?:\/|\?|$)/i.test(url)), 'No GPU endpoint is requested');
    assert.equal(submissions.length, ratios.length + 4, 'Six aspect payloads, Auto 2K, control, inpaint, and model-switch submissions were captured');
    console.log('Qwen 2.1 UI workflows: profiles, six native 2K payloads, control-only image, inpaint assets/options, and non-2K model fallback passed.');
    await context.close();
  } finally {
    await browser.close();
  }
})().catch(error => {console.error(error); process.exitCode = 1;});

function tinyPng() {
  return Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/pXcAAAAASUVORK5CYII=', 'base64');
}
