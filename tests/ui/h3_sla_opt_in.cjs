// Real controls and store submissions on an isolated, fully intercepted origin.
// node tests/ui/h3_sla_opt_in.cjs (MAESTRO_PLAYWRIGHT may point to Playwright).
const assert = require('node:assert/strict');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const {chromium} = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');
const ordinary = ['minimax_h3', 'minimax_h3_ref2va', 'minimax_h3_full', 'minimax_h3_ref2va_full'];
const fused = 'minimax_h3_ref2va_fused_turbo';
const optionsFor = id => ({
  model_type: id, architecture: id.includes('ref2va') ? 'minimax_h3_ref2va' : 'minimax_h3',
  fps: 24, frames_minimum: 124, frames_maximum: 345, frames_steps: 17,
  default_num_inference_steps: id === fused ? 4 : 20, default_guidance_scale: 1,
  minimax_h3_fused_turbo: id === fused, minimax_h3_full_checkpoint: id.endsWith('_full'),
  omni_reference: id.includes('ref2va'),
  sla_attention: true, sla_attention_default: id === fused, sla_attention_experimental: id !== fused,
  sla_attention_status: {supported: true}, sol_attention: id !== fused,
  sol_attention_status: {supported: true}, first_block_cache: id !== fused,
});
const models = [...ordinary, fused].map(id => ({...optionsFor(id), name: id, family: 'minimax_h3',
  is_downloaded: true, is_t2v: true, is_i2v: true}));

(async () => {
  const bundle = await esbuild.build({stdin: {
    contents: `import React from 'react'; import {createRoot} from 'react-dom/client';
      import {useStore} from './src/stores/useStore';
      import {MiniMaxH3Optimizations} from './src/components/Sidebar/MiniMaxH3Optimizations';
      import {DirectorH3Optimizations} from './src/components/Sidebar/DirectorH3Optimizations';
      import {useAdvancedActiveItems} from './src/components/Sidebar/AdvancedSettings';
      import {directorH3SlaEnabled} from './src/lib/h3Attention';
      window.store=useStore; window.slaSelected=directorH3SlaEnabled;
      const root=createRoot(document.getElementById('root'));
      function ActiveItems(){return React.createElement('output',{'data-testid':'advanced-items'},useAdvancedActiveItems().join(', '))}
      window.showStudio=()=>{useStore.setState({sidebarMode:'studio'});root.render(React.createElement('div',{'data-testid':'studio'},React.createElement(MiniMaxH3Optimizations),React.createElement(ActiveItems)))};
      window.showDirector=()=>{useStore.setState({sidebarMode:'director'});root.render(React.createElement('div',{'data-testid':'director'},React.createElement(DirectorH3Optimizations),React.createElement(ActiveItems)))};`,
    resolveDir: path.join(root, 'ui'), loader: 'tsx',
  }, bundle: true, write: false, jsx: 'automatic', define: {'process.env.NODE_ENV': '"development"'}, logLevel: 'silent'});
  const browser = await chromium.launch({headless: true, ...(process.platform === 'win32'
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
        body = {job_id: `test-${generations.length}`, status: 'held'};
      }
      if (key === '/api/v1/director/queue' && request.method() === 'POST') {
        pipelines.push(request.postDataJSON().params);
        body = {entries: [], running: false};
      }
      return route.fulfill({json: body});
    });
    await page.goto('http://h3-sla.test');
    await page.addScriptTag({content: bundle.outputFiles[0].text});
    const seed = async id => page.evaluate(({id, models, options}) => {
      const state = window.store.getState();
      window.store.setState({models, modelsLoaded: true, enabledModels: new Set(models.map(m => m.model_type)),
        modelVisibilityOverrides: Object.fromEntries(['video:frames', 'video:references'].map(scope =>
          [scope, Object.fromEntries(models.map(model => [model.model_type, true]))])),
        modelOptions: options, generationMode: 'video', studioVideoWorkflow: id.includes('ref2va') ? 'references' : 'frames',
        selectedModelPerMode: {video: id}, savedParamsPerMode: {}, savedLoraPerMode: {}, activeWorkspace: 'isolated-test',
        jobs: [], directorH3SolModeByModel: {}, directorH3SlaModeByModel: {}, directorH3TurboModeByModel: {},
        directorH3FirstBlockCacheByModel: {}, directorVideoInferenceStepsByModel: {}, directorH3References: [],
        directorSceneDescription: 'A traveler looks toward a mountain.', directorAudioPath: null,
        params: {...state.params, model_type: id, prompt: 'A traveler looks toward a mountain.',
          seed: 1234, resolution: '1280x704', video_length: 124, sliding_window_size: 124,
          num_inference_steps: options.default_num_inference_steps, image_mode: 0, guidance_scale: 1,
          activated_loras: [], loras_multipliers: '', override_attention: '', minimax_h3_turbo_mode: false,
          skip_steps_cache_type: '', minimax_h3_references: id.includes('ref2va')
            ? [{id: 'traveler', type: 'image', path: '/fixture/traveler.png', filename: 'traveler.png'}] : []}});
    }, {id, models, options: optionsFor(id)});
    const submitStudio = async expected => {
      const count = generations.length;
      await page.evaluate(() => window.store.getState().startGeneration('queue'));
      assert.equal(generations.length, count + 1, 'Studio submits a held generation: ' + JSON.stringify(
        await page.evaluate(() => ({jobs: window.store.getState().jobs, error: window.store.getState().promptEnhanceError}))));
      assert.equal(generations.at(-1).override_attention, expected);
    };
    const submitDirector = async expected => {
      const count = pipelines.length;
      await page.evaluate(() => window.store.getState().startDirectorPipeline('queue'));
      assert.equal(pipelines.length, count + 1, 'Director queues the actual request');
      assert.equal(pipelines.at(-1).video_params.override_attention, expected);
    };
    for (const id of ordinary) {
      await seed(id);
      await page.evaluate(() => window.showStudio());
      await page.getByTestId('studio').waitFor();
      const sla = page.getByRole('checkbox', {name: /SLA Sparse Attention — Experimental/});
      await sla.waitFor();
      assert.equal(await sla.isChecked(), false);
      await submitStudio('');
      await sla.check();
      await page.evaluate(id => window.store.getState().loadModelOptions(id), id);
      assert.equal(await sla.isChecked(), true, 'Reloading options preserves the explicit opt-in');
      await submitStudio('sla');
      await sla.uncheck();
      await submitStudio('');
      await page.getByRole('checkbox', {name: /Sol Engine/}).check();
      await submitStudio('sol');
      assert.equal(await sla.isChecked(), false);

      await page.evaluate(() => window.showDirector());
      await page.getByTestId('director').waitFor();
      await sla.waitFor();
      assert.equal(await sla.isChecked(), false, 'Director starts with SLA off independently of Studio');
      await submitDirector('');
      await sla.check();
      await submitDirector('sla');
      assert.equal(await page.getByTestId('advanced-items').textContent(), 'H3 SLA');
      assert.equal(pipelines.at(-1).director_ui_snapshot.directorH3SlaModeByModel[id], true,
        'Saved project retains SLA: ' + JSON.stringify({id, video: pipelines.at(-1).video_model,
          sla: pipelines.at(-1).director_ui_snapshot.directorH3SlaModeByModel}));
      assert.equal(await page.evaluate(() => window.store.getState().params.override_attention), 'sol', 'Director leaves Studio unchanged');
      await page.getByRole('checkbox', {name: /^Sol Engine/}).check();
      assert.equal(await sla.isChecked(), false);
      await submitDirector('sol');
      assert.equal(await page.getByTestId('advanced-items').textContent(), 'H3 Sol Engine');
      await sla.check();
      assert.equal(await page.getByRole('checkbox', {name: /^Sol Engine/}).isChecked(), false);
      await sla.uncheck();
      await submitDirector('');
      assert.equal(pipelines.at(-1).video_params.num_inference_steps, 20);
    }
    await seed(fused);
    await page.evaluate(() => window.store.getState().loadModelOptions(window.store.getState().params.model_type));
    await page.evaluate(() => window.showStudio());
    await page.getByTestId('studio').waitFor();
    const fusedSla = page.getByRole('checkbox', {name: /^SLA Sparse Attention/});
    await fusedSla.waitFor();
    assert.equal(await fusedSla.isChecked(), true, 'Fused retains its default-on recipe');
    await submitStudio('sla');
    await page.evaluate(id => window.store.getState().selectModel(id), ordinary[1]);
    await page.waitForFunction(id => window.store.getState().modelOptions.model_type === id, ordinary[1]);
    assert.equal(await fusedSla.isChecked(), false, 'Switching from Fused does not opt into experimental Pruned SLA');
    await page.evaluate(id => window.store.getState().selectModel(id), fused);
    await page.waitForFunction(id => window.store.getState().modelOptions.model_type === id, fused);
    assert.equal(await fusedSla.isChecked(), true);
    await fusedSla.uncheck();
    await submitStudio('sdpa');
    await page.evaluate(id => window.store.getState().selectModel(id), ordinary[1]);
    await page.waitForFunction(id => window.store.getState().modelOptions.model_type === id, ordinary[1]);
    assert.equal(await page.evaluate(() => window.store.getState().params.override_attention), '',
      'Turning Fused SLA off does not force SDPA on a different Pruned checkpoint');
    await page.evaluate(id => window.store.getState().selectModel(id), fused);
    await page.waitForFunction(id => window.store.getState().modelOptions.model_type === id, fused);
    await page.evaluate(() => window.showDirector());
    await page.getByTestId('director').waitFor();
    await fusedSla.waitFor();
    assert.equal(await fusedSla.isChecked(), true);
    assert.equal(await page.getByTestId('advanced-items').textContent(), 'H3 SLA');
    await fusedSla.uncheck();
    await submitDirector('sdpa');
    for (const enabled of [true, false]) {
      assert.equal(await page.evaluate(({id, options, enabled}) =>
        window.slaSelected(options, id, {}, {[id]: enabled}), {id: fused, options: optionsFor(fused), enabled}), enabled,
      'Legacy Fused SLA preference is preserved');
    }
    assert.equal(await page.evaluate(options => window.slaSelected(options, options.model_type, {}, {[options.model_type]: true}), optionsFor(ordinary[0])), false,
      'Legacy Pruned Sol opt-in does not enable SLA');
    await page.evaluate(() => window.store.setState({modelOptions: {architecture: 'minimax_h3', sla_attention: false, sol_attention: false}}));
    await page.evaluate(() => window.showStudio());
    await page.getByRole('checkbox', {name: /SLA Sparse Attention/}).waitFor({state: 'hidden'});
    assert.deepEqual(errors, []);
    console.log('Full/Pruned Frames/References: Studio and Director default-off, opt-in/out payloads, Sol exclusivity, saved choice, Fused defaults and legacy preferences passed.');
  } finally { await browser.close(); }
})().catch(error => {console.error(error); process.exitCode = 1;});
