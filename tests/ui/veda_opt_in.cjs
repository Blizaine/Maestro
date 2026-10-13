// Real controls and store submissions on an isolated, fully intercepted origin.
// node tests/ui/veda_opt_in.cjs (MAESTRO_PLAYWRIGHT may point to Playwright).
const assert = require('node:assert/strict');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const {chromium} = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');
const ordinary = ['minimax_h3', 'minimax_h3_ref2va'];
const unsupported = 'minimax_h3_ref2va_full';
const fused = 'minimax_h3_ref2va_fused_turbo';
const vedaCapable = id => ordinary.includes(id) || id === unsupported || id === fused;
const optionsFor = id => ({
  model_type: id, architecture: id.includes('ref2va') ? 'minimax_h3_ref2va' : 'minimax_h3',
  fps: 24, frames_minimum: 124, frames_maximum: 345, frames_steps: 17,
  default_num_inference_steps: id === fused ? 4 : 20, default_guidance_scale: 1,
  minimax_h3_fused_turbo: id === fused, minimax_h3_full_checkpoint: id.endsWith('_full'),
  omni_reference: id.includes('ref2va'),
  sla_attention: true, sla_attention_default: id === fused, sla_attention_experimental: id !== fused,
  sla_attention_status: {installed: true, supported: true},
  sol_attention: id !== fused, sol_attention_status: {installed: true, supported: true},
  veda_attention: vedaCapable(id), veda_attention_experimental: vedaCapable(id),
  veda_attention_status: {installed: false, supported: id !== unsupported,
    reason: id === unsupported ? 'Veda requires a supported GPU.' : null,
    capability: id === unsupported ? 'unsupported' : 'supported', triton_version: '3.4.0'},
  first_block_cache: id !== fused,
});
const models = [...ordinary, unsupported, fused].map(id => ({...optionsFor(id), name: id,
  family: 'minimax_h3', is_downloaded: true, is_t2v: true, is_i2v: true}));

(async () => {
  const bundle = await esbuild.build({stdin: {
    contents: `import React from 'react'; import {createRoot} from 'react-dom/client';
      import {useStore} from './src/stores/useStore';
      import {MiniMaxH3Optimizations} from './src/components/Sidebar/MiniMaxH3Optimizations';
      import {DirectorH3Optimizations} from './src/components/Sidebar/DirectorH3Optimizations';
      const root=createRoot(document.getElementById('root')); window.store=useStore;
      window.showStudio=()=>{useStore.setState({sidebarMode:'studio'});root.render(React.createElement('div',{'data-testid':'studio'},React.createElement(MiniMaxH3Optimizations)))};
      window.showDirector=()=>{useStore.setState({sidebarMode:'director'});root.render(React.createElement('div',{'data-testid':'director'},React.createElement(DirectorH3Optimizations)))};`,
    resolveDir: path.join(root, 'ui'), loader: 'tsx',
  }, bundle: true, write: false, jsx: 'automatic', define: {'process.env.NODE_ENV': '"development"'}, logLevel: 'silent'});
  const browser = await chromium.launch({headless: true, args: ['--disable-gpu'], ...(process.platform === 'win32'
    ? {executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'} : {})});
  try {
    const page = await browser.newPage();
    const errors = [], generations = [], pipelines = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/*', route => {
      const request = route.request(), key = new URL(request.url()).pathname;
      if (key === '/') return route.fulfill({contentType: 'text/html', body: '<div id="root"></div>'});
      let body = {};
      if (key.startsWith('/api/v1/model-options/')) body = optionsFor(decodeURIComponent(key.split('/').pop()));
      if (key.startsWith('/api/v1/defaults/')) body = {num_inference_steps: key.endsWith(fused) ? 4 : 20, guidance_scale: 1};
      if (key === '/api/v1/generate') {
        generations.push(request.postDataJSON());
        body = {job_id: `held-${generations.length}`, status: 'held'};
      }
      if (key === '/api/v1/director/queue' && request.method() === 'POST') {
        pipelines.push(request.postDataJSON().params);
        body = {entries: [], running: false};
      }
      return route.fulfill({json: body});
    });
    await page.goto('http://veda-ui.test');
    await page.addScriptTag({content: bundle.outputFiles[0].text});
    const seed = async id => page.evaluate(({id, models, options}) => {
      const current = window.store.getState();
      window.store.setState({models, modelsLoaded: true, enabledModels: new Set(models.map(model => model.model_type)),
        modelVisibilityOverrides: Object.fromEntries(['video:frames', 'video:references'].map(scope =>
          [scope, Object.fromEntries(models.map(model => [model.model_type, true]))])),
        modelOptions: options,
        generationMode: 'video', studioVideoWorkflow: id.includes('ref2va') ? 'references' : 'frames',
        selectedModelPerMode: {video: id}, savedParamsPerMode: {}, savedLoraPerMode: {}, activeWorkspace: 'ui-test',
        jobs: [], directorH3SolModeByModel: {}, directorH3SlaModeByModel: {}, directorH3VedaModeByModel: {},
        directorH3TurboModeByModel: {}, directorH3FirstBlockCacheByModel: {}, directorVideoInferenceStepsByModel: {},
        directorH3References: [], directorSceneDescription: 'A traveler looks toward a mountain.', directorAudioPath: null,
        selectedOutputMeta: null,
        params: {...current.params, model_type: id, prompt: 'A traveler looks toward a mountain.',
          seed: 1234, resolution: '1280x704', video_length: 124, sliding_window_size: 124,
          num_inference_steps: id === 'minimax_h3_ref2va_fused_turbo' ? 4 : 20,
          image_mode: 0, guidance_scale: 1, activated_loras: [], loras_multipliers: '',
          override_attention: '', minimax_h3_turbo_mode: false, skip_steps_cache_type: '',
          minimax_h3_references: id.includes('ref2va')
            ? [{id: 'traveler', type: 'image', path: '/fixture/traveler.png', filename: 'traveler.png'}] : []}});
    }, {id, models, options: optionsFor(id)});
    const submitStudio = async expected => {
      const count = generations.length;
      await page.evaluate(() => window.store.getState().startGeneration('queue'));
      assert.equal(generations.length, count + 1, 'Studio submits a held generation');
      assert.equal(generations.at(-1).override_attention, expected);
    };
    const submitDirector = async expected => {
      const count = pipelines.length;
      await page.evaluate(() => window.store.getState().startDirectorPipeline('queue'));
      assert.equal(pipelines.length, count + 1, 'Director submits a held queue request');
      assert.equal(pipelines.at(-1).video_params.override_attention, expected);
    };
    const vedaLabel = /Veda Sparse Attention — Experimental/;

    await seed(ordinary[0]);
    await page.evaluate(() => window.store.getState().loadModelOptions(window.store.getState().params.model_type));
    await page.evaluate(() => window.showStudio());
    await page.getByTestId('studio').waitFor();
    let veda = page.getByRole('checkbox', {name: vedaLabel});
    await veda.waitFor();
    assert.equal(await veda.isChecked(), false, 'Studio Veda starts off');
    await submitStudio('');
    await veda.check();
    assert.equal(await page.evaluate(() => window.store.getState().h3OptimizationPreferences.override_attention), 'veda');
    assert.equal(await page.evaluate(() => window.store.getState().params.override_attention), 'veda');
    await submitStudio('veda');
    await page.getByRole('checkbox', {name: /SLA Sparse Attention/}).check();
    assert.equal(await veda.isChecked(), false, 'SLA replaces Veda in Studio');
    await submitStudio('sla');
    await page.getByRole('checkbox', {name: /Sol Engine/}).check();
    assert.equal(await veda.isChecked(), false, 'Sol replaces Veda in Studio');
    await submitStudio('sol');

    await page.evaluate(() => window.showDirector());
    await page.getByTestId('director').waitFor();
    veda = page.getByRole('checkbox', {name: vedaLabel});
    await veda.waitFor();
    assert.equal(await veda.isChecked(), false, 'Director Veda starts off independently of Studio');
    await veda.check();
    assert.equal(await page.evaluate(() => window.store.getState().directorH3VedaModeByModel['minimax_h3']), true);
    assert.equal(await page.evaluate(() => window.store.getState().directorH3SlaModeByModel['minimax_h3']), false);
    assert.equal(await page.evaluate(() => window.store.getState().directorH3SolModeByModel['minimax_h3']), false);
    assert.equal(await page.evaluate(() => window.store.getState().directorH3VedaModeByModel['minimax_h3']), true);
    await submitDirector('veda');
    assert.equal(pipelines.at(-1).director_ui_snapshot.directorH3VedaModeByModel['minimax_h3'], true,
      'Director project snapshot preserves the per-model Veda preference');
    assert.equal(await page.evaluate(() => window.store.getState().params.override_attention), 'sol',
      'Director leaves Studio attention unchanged');

    await page.getByRole('checkbox', {name: /SLA Sparse Attention/}).check();
    assert.equal(await veda.isChecked(), false, 'SLA replaces Veda in Director');
    await submitDirector('sla');
    await veda.check();
    assert.equal(await page.getByRole('checkbox', {name: /SLA Sparse Attention/}).isChecked(), false);
    await page.getByRole('checkbox', {name: /^Sol Engine/}).check();
    assert.equal(await veda.isChecked(), false, 'Sol replaces Veda in Director');
    await veda.check();
    assert.equal(await page.getByRole('checkbox', {name: /^Sol Engine/}).isChecked(), false);
    await submitDirector('veda');

    await page.evaluate(id => window.store.setState({selectedModelPerMode: {
      ...window.store.getState().selectedModelPerMode, video: id,
    }}), ordinary[1]);
    await page.waitForFunction(id => window.store.getState().selectedModelPerMode.video === id, ordinary[1]);
    await page.getByRole('checkbox', {name: vedaLabel}).waitFor();
    assert.equal(await page.getByRole('checkbox', {name: vedaLabel}).isChecked(), false,
      'Director preference is isolated by model');
    await page.evaluate(id => window.store.setState({selectedModelPerMode: {
      ...window.store.getState().selectedModelPerMode, video: id,
    }}), ordinary[0]);
    await page.waitForFunction(id => window.store.getState().selectedModelPerMode.video === id, ordinary[0]);
    await page.waitForFunction(() => document.querySelector('input[type=checkbox]:checked') !== null);
    assert.equal(await page.getByRole('checkbox', {name: vedaLabel}).isChecked(), true,
      'Director restores the preference for the original model');

    await page.evaluate(() => window.store.setState({selectedModelPerMode: {
      ...window.store.getState().selectedModelPerMode, video: 'minimax_h3_ref2va_full',
    }, directorH3VedaModeByModel: {'minimax_h3_ref2va_full': true}}));
    const unsupportedVeda = page.getByRole('checkbox', {name: vedaLabel});
    await unsupportedVeda.waitFor();
    await page.waitForFunction(() => window.store.getState().directorH3VedaModeByModel['minimax_h3_ref2va_full'] === false);
    assert.equal(await unsupportedVeda.isDisabled(), true);
    assert.match(await page.getByText('Veda requires a supported GPU.').textContent(), /supported GPU/);

    await seed(ordinary[0]);
    await page.evaluate(() => window.store.getState().loadModelOptions(window.store.getState().params.model_type));
    await page.evaluate(() => window.showStudio());
    await page.getByTestId('studio').waitFor();
    await page.getByRole('checkbox', {name: vedaLabel}).check();
    await page.evaluate(id => window.store.getState().selectModel(id), unsupported);
    await page.waitForFunction(id => window.store.getState().modelOptions?.model_type === id, unsupported);
    assert.equal(await page.evaluate(() => window.store.getState().params.override_attention), '',
      'Studio clears stale Veda when a newly selected model cannot run it');

    await page.evaluate(async id => {
      window.store.setState({selectedOutputMeta: {source: 'sidecar', params: {
        model_type: id, override_attention: 'veda', prompt: 'A restored scene', seed: 42,
        resolution: '1280x704', video_length: 124, num_inference_steps: 20,
      }}});
      await window.store.getState().loadSettingsFromOutput();
    }, ordinary[0]);
    assert.equal(await page.evaluate(() => window.store.getState().params.override_attention), 'veda',
      'Metadata restore retains Veda on a supported Full/Pruned model');
    await page.evaluate(async id => {
      window.store.setState({selectedOutputMeta: {source: 'sidecar', params: {
        model_type: id, override_attention: 'veda', prompt: 'An unsupported restored scene', seed: 43,
        resolution: '1280x704', video_length: 124, num_inference_steps: 20,
      }}});
      await window.store.getState().loadSettingsFromOutput();
    }, unsupported);
    assert.equal(await page.evaluate(() => window.store.getState().params.override_attention), '',
      'Metadata restore clears Veda when the runtime reports unsupported hardware');

    await seed(fused);
    await page.evaluate(() => window.store.getState().loadModelOptions(window.store.getState().params.model_type));
    await page.evaluate(() => window.showStudio());
    await page.getByTestId('studio').waitFor();
    const fusedSla = page.getByRole('checkbox', {name: /^SLA Sparse Attention/});
    await fusedSla.waitFor();
    assert.equal(await fusedSla.isChecked(), true, 'Fused retains its default-on SLA recipe');
    await submitStudio('sla');
    const fusedVeda = page.getByRole('checkbox', {name: vedaLabel});
    await fusedVeda.waitFor();
    assert.equal(await fusedVeda.isChecked(), false, 'Fused Veda remains opt-in');
    await fusedVeda.check();
    assert.equal(await fusedSla.isChecked(), false, 'Fused Veda replaces SLA');
    await submitStudio('veda');
    await page.evaluate(() => window.store.getState().loadModelOptions(window.store.getState().params.model_type));
    assert.equal(await fusedVeda.isChecked(), true, 'Fused Veda survives an option refresh');
    await submitStudio('veda');
    await fusedSla.check();
    assert.equal(await fusedVeda.isChecked(), false, 'Fused SLA replaces Veda');
    await fusedSla.uncheck();
    await submitStudio('sdpa');
    await page.evaluate(id => window.store.getState().selectModel(id), ordinary[1]);
    await page.waitForFunction(id => window.store.getState().modelOptions?.model_type === id, ordinary[1]);
    assert.equal(await page.evaluate(() => window.store.getState().params.override_attention), '',
      'Fused SDPA opt-out does not leak into an ordinary model');

    await page.evaluate(id => {
      window.store.setState({selectedOutputMeta: {source: 'sidecar', params: {
        model_type: id, override_attention: 'veda', prompt: 'A restored fused scene', seed: 42,
        resolution: '1280x704', video_length: 124, num_inference_steps: 4,
      }}});
      return window.store.getState().loadSettingsFromOutput();
    }, fused);
    assert.equal(await page.evaluate(() => window.store.getState().params.override_attention), 'veda',
      'Fused output settings restore Veda instead of silently substituting SLA');

    await seed(fused);
    await page.evaluate(() => window.showDirector());
    await page.getByTestId('director').waitFor();
    const directorFusedVeda = page.getByRole('checkbox', {name: vedaLabel});
    await directorFusedVeda.waitFor();
    assert.equal(await directorFusedVeda.isChecked(), false, 'Director Fused Veda remains opt-in');
    await directorFusedVeda.check();
    await submitDirector('veda');
    assert.equal(pipelines.at(-1).video_params.num_inference_steps, 4, 'Fused Veda keeps its four-step recipe');
    await page.getByRole('checkbox', {name: /^SLA Sparse Attention/}).check();
    assert.equal(await directorFusedVeda.isChecked(), false, 'Director Fused SLA replaces Veda');
    await submitDirector('sla');
    assert.deepEqual(errors, []);
    console.log('Veda: Studio and Director opt-in, exclusivity, per-model persistence, unsupported handling, metadata restore, and Fused defaults passed.');
  } finally { await browser.close(); }
})().catch(error => {console.error(error); process.exitCode = 1;});
