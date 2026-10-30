// Pika Flex cache normalization preserves durable H3 preferences.
// Run: node tests/ui/pika_cache_compat.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ts = require('../../ui/node_modules/typescript');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const playwrightPath = process.env.MAESTRO_PLAYWRIGHT
  || (() => {
    try {
      return require.resolve('playwright', {paths: [path.join(root, 'ui')]});
    } catch (error) {
      if (error.code !== 'MODULE_NOT_FOUND') throw error;
      return require.resolve('playwright');
    }
  })();
const playwright = require(playwrightPath);

const helperPath = path.join(root, 'ui/src/lib/h3Memory.ts');
const helperSource = fs.readFileSync(helperPath, 'utf8');
const compiled = ts.transpileModule(helperSource, {
  compilerOptions: {module: ts.ModuleKind.CommonJS},
}).outputText;
const context = {exports: {}};
vm.runInNewContext(compiled, context);
const {isPikaFlexModelType, clearPikaFlexCacheSettings} = context.exports;

const savedPreferences = {
  skip_steps_cache_type: 'first_block',
  skip_steps_multiplier: 0.04,
  skip_steps_start_step_perc: 25,
};

for (const modelType of ['minimax_h3_pika_flex', 'minimax_h3_ref2va_pika_flex']) {
  const request = {
    model_type: modelType,
    ...savedPreferences,
    override_attention: 'sol',
    activated_loras: ['explicit-extra.safetensors'],
  };
  assert.equal(isPikaFlexModelType(modelType, {
    model_type: 'previous_h3', first_block_cache: true,
  }), true, 'The selected Frames/References model wins over stale model options');
  clearPikaFlexCacheSettings(modelType, request, {
    model_type: 'previous_h3', first_block_cache: true,
  });
  assert.equal(request.skip_steps_cache_type, '');
  assert.equal(Object.hasOwn(request, 'skip_steps_multiplier'), false);
  assert.equal(Object.hasOwn(request, 'skip_steps_start_step_perc'), false);
  assert.equal(request.override_attention, 'sol', 'Pika attention remains available');
  assert.deepEqual(request.activated_loras, ['explicit-extra.safetensors'],
    'Cache normalization does not silently accept or strip additional LoRAs');
  assert.equal(savedPreferences.skip_steps_cache_type, 'first_block',
    'The saved H3 preference is not mutated when cleaning a request copy');
}

const ordinaryH3 = {
  model_type: 'ordinary_h3',
  skip_steps_cache_type: 'first_block',
  skip_steps_multiplier: 0.08,
  skip_steps_start_step_perc: 30,
};
clearPikaFlexCacheSettings('ordinary_h3', ordinaryH3, {
  model_type: 'stale_pika_options', minimax_h3_pika_flex: true,
});
assert.equal(ordinaryH3.skip_steps_cache_type, 'first_block',
  'A stale options response cannot clear cache for a different selected model');
assert.equal(ordinaryH3.skip_steps_multiplier, 0.08);

assert.equal(isPikaFlexModelType('pika_alias', {
  model_type: 'pika_alias', minimax_h3_pika_flex: true,
}), true, 'A matching model-options capability covers aliases');
assert.equal(isPikaFlexModelType('ordinary_h3', {
  model_type: 'pika_alias', minimax_h3_pika_flex: true,
}), false, 'A mismatched Pika capability cannot override selected model identity');

const storeSource = fs.readFileSync(path.join(root, 'ui/src/stores/useStore.ts'), 'utf8');
const optimizerSource = fs.readFileSync(
  path.join(root, 'ui/src/components/Sidebar/MiniMaxH3Optimizations.tsx'), 'utf8',
);
for (const [filename, source] of [
  ['useStore.ts', storeSource],
  ['MiniMaxH3Optimizations.tsx', optimizerSource],
]) {
  const diagnostics = ts.transpileModule(source, {
    fileName: filename,
    reportDiagnostics: true,
    compilerOptions: {
      module: ts.ModuleKind.ESNext,
      target: ts.ScriptTarget.ES2022,
      jsx: ts.JsxEmit.ReactJSX,
    },
  }).diagnostics || [];
  const errors = diagnostics.filter(item => item.category === ts.DiagnosticCategory.Error);
  assert.deepEqual(errors, [], `${filename} transpiles: ${errors.map(item => item.messageText).join('; ')}`);
}
assert.match(storeSource, /if \(isPikaFlexModelType\(params\.model_type, state\.modelOptions\)\)[\s\S]{0,220}clearPikaFlexCacheSettings/,
  'The final submit sanitizer uses selected model identity during stale-options races');
assert.match(storeSource, /clearPikaFlexCacheSettings\(\s*initialModelType,/,
  'Startup hydration filters durable cache preferences for Pika');
assert.match(storeSource, /clearPikaFlexCacheSettings\(modelType, newParams, restoredModelOptions\)/,
  'Use Settings restoration filters Pika cache state after loading model capabilities');
assert.match(storeSource, /activeState\.h3OptimizationPreferences\.skip_steps_cache_type === 'first_block'/,
  'Compatible H3 model options can restore the still-saved preference');
assert.match(optimizerSource, /const cacheEnabled = !pikaFlex && params\.skip_steps_cache_type === 'first_block'/,
  'The optimizer count excludes cache when the selected model is Pika');
assert.match(optimizerSource, /!pikaFlex && modelOptions\?\.model_type === params\.model_type && modelOptions\.first_block_cache/,
  'Stale options cannot expose the unsupported cache toggle');
assert.match(optimizerSource, /const option = pikaFlex \? undefined : modelOptions\?\.minimax_h3_turbo/,
  'Stale Turbo metadata cannot expose Pika-incompatible adapter controls');

const modelOptionsFor = modelType => ({
  model_type: modelType,
  architecture: 'minimax_h3',
  fps: 24,
  frames_minimum: 124,
  frames_maximum: 345,
  frames_steps: 17,
  sliding_window: true,
  video_continuation: true,
  sliding_window_defaults: {
    window_min: 124, window_max: 345, window_step: 17,
    window_default: 124, overlap_min: 1, overlap_max: 103,
    overlap_step: 17, overlap_default: 18, discard_last_frames: 0,
  },
  first_block_cache: true,
  skip_steps_multiplier_choices: [['4%', 0.04]],
  default_skip_steps_multiplier: 0.04,
  default_skip_steps_start_step_perc: 25,
  sol_attention: true,
  sol_attention_status: {supported: true},
  sla_attention: true,
  sla_attention_default: false,
  sla_attention_status: {supported: true},
  minimax_h3_turbo: {
    preset_id: 'legacy-test', filename: 'legacy-test.safetensors',
    steps: 8, weight: 1, default_enabled: false,
  },
  minimax_h3_pika_flex: false,
  omni_reference: false,
  t2v_class: true,
  i2v_class: true,
  returns_audio: true,
  lock_inference_steps: false,
  lock_guidance_scale: false,
  guidance_max_phases: 1,
  default_num_inference_steps: 6,
  resolution_preset_order: ['480p'],
  resolution_presets: {'480p': {label: '480p', values: {'16:9': '832x480'}}},
});

(async () => {
  const bundle = await esbuild.build({stdin: {contents: `
    import React from 'react';
    import {createRoot} from 'react-dom/client';
    import {useStore} from './src/stores/useStore';
    import {MiniMaxH3Optimizations} from './src/components/Sidebar/MiniMaxH3Optimizations';
    window.store = useStore;
    window.mount = () => {
      window.root = createRoot(document.getElementById('root'));
      window.root.render(React.createElement(MiniMaxH3Optimizations));
    };
  `, resolveDir: path.join(root, 'ui'), loader: 'tsx'}, bundle: true, write: false,
    jsx: 'automatic', define: {'process.env.NODE_ENV': '"development"'}, logLevel: 'silent'});
  const browser = await playwright.chromium.launch({headless: true,
    ...(process.env.MAESTRO_CHROME || process.platform === 'win32'
      ? {executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'} : {})});
  try {
    const page = await browser.newPage();
    const errors = [];
    const submissions = [];
    const optionsByType = {};
    let holdPikaOptions = false;
    let releasePikaOptions;
    let notifyPikaOptionsRequest;
    const pikaOptionsRequestSeen = new Promise(resolve => { notifyPikaOptionsRequest = resolve; });
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/*', route => route.fulfill({contentType: 'text/html', body: '<div id="root"></div>'}));
    await page.route('**/api/v1/model-options/*', async route => {
      const modelType = decodeURIComponent(new URL(route.request().url()).pathname.split('/').pop());
      if (holdPikaOptions && modelType === 'minimax_h3_pika_flex') {
        notifyPikaOptionsRequest();
        await new Promise(resolve => { releasePikaOptions = resolve; });
      }
      return route.fulfill({json: optionsByType[modelType] || {detail: 'Unknown test model'}});
    });
    await page.route('**/api/v1/loras/*', route => route.fulfill({json: {loras: []}}));
    await page.route('**/api/v1/studio-preferences', route => route.fulfill({json: {}}));
    await page.route('**/api/v1/generate', route => {
      submissions.push(route.request().postDataJSON());
      return route.fulfill({status: 400, json: {detail: 'Captured by isolated cache test'}});
    });
    await page.goto('http://pika-cache-compat.test');
    await page.addScriptTag({content: bundle.outputFiles[0].text});

    const previousModel = 'ordinary_h3_cache_test';
    const pikaModel = 'minimax_h3_pika_flex';
    optionsByType[previousModel] = modelOptionsFor(previousModel);
    optionsByType[pikaModel] = {
      ...modelOptionsFor(pikaModel),
      minimax_h3_pika_flex: true,
      first_block_cache: false,
      loras_disabled: true,
      minimax_h3_turbo: {
        preset_id: 'stale-api-recipe', filename: 'stale-api-recipe.safetensors',
        steps: 8, weight: 1, default_enabled: true,
      },
    };
    await page.evaluate(({previousModel, pikaModel, previousOptions}) => {
      const store = window.store;
      const state = store.getState();
      const previous = {
        model_type: previousModel, name: previousModel, family: 'minimax_h3',
        architecture: 'minimax_h3', fps: 24, is_i2v: true, is_t2v: true,
        guidance_max_phases: 1, is_downloaded: true,
      };
      const pika = {
        model_type: pikaModel, name: pikaModel, family: 'minimax_h3',
        architecture: 'minimax_h3', fps: 24, is_i2v: true, is_t2v: true,
        guidance_max_phases: 1, is_downloaded: true,
      };
      store.setState({
        generationMode: 'video', studioVideoWorkflow: 'frames',
        studioVideoCreateRoute: 'auto', studioVideoEffectiveCreateRoute: 'generate',
        models: [previous, pika], modelsLoaded: true,
        enabledModels: new Set([previousModel, pikaModel]),
        selectedModelPerMode: {...state.selectedModelPerMode, video: previousModel},
        modelOptions: previousOptions, modelOptionsLoading: false,
        h3OptimizationPreferences: {
          override_attention: 'sol', skip_steps_cache_type: 'first_block',
          skip_steps_multiplier: 0.04, skip_steps_start_step_perc: 25,
        },
        durationSeconds: 124 / 24, slidingWindowSeconds: 124 / 24,
        slidingWindowLocked: false, slidingWindowOverlap: 18,
        startImage: null, endImage: null, imageRefs: [], clips: [], jobs: [],
        isGenerating: false, promptEnhanceError: null,
        params: {
          ...state.params,
          model_type: previousModel, prompt: 'A quiet garden at sunrise.',
          image_mode: 0, resolution: '832x480', video_length: 124,
          sliding_window_size: 124, sliding_window_overlap: 18,
          sliding_window_discard_last_frames: 0,
          num_inference_steps: 6, guidance_scale: 1, flow_shift: 12,
          audio_flow_shift: 3, sample_solver: 'euler',
          activated_loras: [], loras_multipliers: '',
          skip_steps_cache_type: 'first_block', skip_steps_multiplier: 0.04,
          skip_steps_start_step_perc: 25, override_attention: '',
          minimax_h3_turbo_mode: false, custom_settings: {},
        },
      });
    }, {previousModel, pikaModel, previousOptions: optionsByType[previousModel]});

    holdPikaOptions = true;
    await page.evaluate(modelType => window.store.getState().selectModel(modelType), pikaModel);
    await pikaOptionsRequestSeen;
    await page.waitForFunction(modelType => window.store.getState().params.model_type === modelType
      && window.store.getState().modelOptionsLoading, pikaModel);
    const switched = await page.evaluate(() => {
      const state = window.store.getState();
      return {
        model: state.params.model_type,
        optionsModel: state.modelOptions?.model_type,
        cache: state.params.skip_steps_cache_type,
        multiplier: state.params.skip_steps_multiplier,
        preference: state.h3OptimizationPreferences.skip_steps_cache_type,
        turbo: state.params.minimax_h3_turbo_mode,
      };
    });
    assert.deepEqual(switched, {
      model: pikaModel, optionsModel: previousModel, cache: '',
      multiplier: undefined, preference: 'first_block', turbo: false,
    }, 'Pika model identity clears runtime cache while options are stale');

    await page.evaluate(() => window.mount());
    assert.equal(await page.getByText('First Block Cache', {exact: true}).count(), 0,
      'Stale capabilities cannot expose First Block Cache for Pika');
    assert.equal(await page.getByText('Turbo', {exact: true}).count(), 0,
      'Stale Turbo metadata cannot expose the unsupported adapter');
    assert.equal(await page.getByText('Optional', {exact: true}).count(), 1,
      'The hidden cache is not counted as an active optimization');

    assert.equal(submissions.length, 0,
      'Switching models and rendering controls never submits generation');

    holdPikaOptions = false;
    releasePikaOptions();
    await page.waitForFunction(() => !window.store.getState().modelOptionsLoading
      && window.store.getState().modelOptions?.model_type === 'minimax_h3_pika_flex');
    const loadedPika = await page.evaluate(() => {
      const state = window.store.getState();
      return {
        cache: state.params.skip_steps_cache_type,
        preference: state.h3OptimizationPreferences.skip_steps_cache_type,
        turboOption: state.modelOptions?.minimax_h3_turbo,
        sol: state.modelOptions?.sol_attention,
        sla: state.modelOptions?.sla_attention,
      };
    });
    assert.equal(loadedPika.cache, '');
    assert.equal(loadedPika.preference, 'first_block');
    assert.equal(loadedPika.turboOption, undefined,
      'Pika capability normalization ignores a stale Turbo option');
    assert.equal(loadedPika.sol, true, 'Pika Sol remains available');
    assert.equal(loadedPika.sla, true, 'Pika SLA remains available');

    // Simulate a restored hidden cache with legacy capability metadata. The
    // final request guard must use selected-model identity, not that metadata.
    await page.evaluate(() => {
      const state = window.store.getState();
      window.store.setState({
        promptEnhanceError: null,
        modelOptions: {...state.modelOptions, first_block_cache: true},
        params: {...state.params, skip_steps_cache_type: 'first_block',
          skip_steps_multiplier: 0.04, skip_steps_start_step_perc: 25},
      });
    });
    await page.evaluate(async () => window.store.getState().startGeneration());
    const admissionState = await page.evaluate(() => ({
      error: window.store.getState().promptEnhanceError,
      jobs: window.store.getState().jobs,
    }));
    assert.equal(submissions.length, 1,
      `The restored-state path reaches the mocked endpoint: ${JSON.stringify(admissionState)}`);
    assert.equal(submissions[0].model_type, pikaModel);
    assert.equal(submissions[0].skip_steps_cache_type, '',
      'The final request guard disables First Block based on selected model identity');
    assert.deepEqual(submissions[0].activated_loras || [], [],
      'The Pika switch does not inherit ordinary model LoRAs');
    assert.equal(await page.evaluate(() =>
      window.store.getState().h3OptimizationPreferences.skip_steps_cache_type), 'first_block',
      'Submitting with Pika does not overwrite the saved H3 preference');

    await page.evaluate(modelType => window.store.getState().selectModel(modelType), previousModel);
    await page.waitForFunction(modelType => !window.store.getState().modelOptionsLoading
      && window.store.getState().modelOptions?.model_type === modelType, previousModel);
    const restored = await page.evaluate(() => ({
      cache: window.store.getState().params.skip_steps_cache_type,
      preference: window.store.getState().h3OptimizationPreferences.skip_steps_cache_type,
    }));
    assert.deepEqual(restored, {cache: 'first_block', preference: 'first_block'},
      'Returning to compatible H3 restores the saved cache choice');
    assert.deepEqual(errors, [], `Browser errors: ${errors.join('; ')}`);
    await page.evaluate(() => window.root.unmount());
  } finally {
    await browser.close();
  }
  console.log('Pika cache stale-options browser regression passed.');
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
