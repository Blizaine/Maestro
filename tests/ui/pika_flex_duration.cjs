// Pika H3 Flex duration stays independent of stale H3 windows; requests are mocked.
// Run: node tests/ui/pika_flex_duration.cjs
const assert = require('node:assert/strict');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const playwright = require(process.env.MAESTRO_PLAYWRIGHT
  || 'C:/Users/bliza/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

const modelOptionsFor = (modelType, reference = false) => ({
  model_type: modelType,
  architecture: 'minimax_h3',
  fps: 24,
  frames_minimum: 124,
  frames_maximum: 362,
  frames_steps: 17,
  sliding_window: false,
  sliding_window_defaults: null,
  sliding_window_memory_policy: null,
  omni_sequence_memory_policy: null,
  minimax_h3_pika_flex: true,
  omni_reference: reference,
  t2v_class: true,
  i2v_class: !reference,
  returns_audio: true,
  audio_only: false,
  lock_inference_steps: true,
  lock_guidance_scale: true,
  lock_guidance_phases: true,
  inference_steps_min: 6,
  inference_steps_max: 6,
  guidance_max_phases: 1,
  default_num_inference_steps: 6,
  resolution_preset_order: ['480p', '768p'],
  resolution_presets: {
    '480p': {label: '480p', values: {'16:9': '832x480'}},
    '768p': {label: '768p', values: {'16:9': '1344x768'}},
  },
});

const modelDefFor = (modelType, reference = false) => ({
  model_type: modelType,
  name: modelType,
  family: 'minimax_h3',
  architecture: 'minimax_h3',
  is_i2v: !reference,
  is_t2v: true,
  fps: 24,
  guidance_max_phases: 1,
  omni_reference: reference,
  is_downloaded: true,
});

(async () => {
  const bundle = await esbuild.build({stdin: {contents: `
    import React from 'react';
    import {createRoot} from 'react-dom/client';
    import {useStore} from './src/stores/useStore';
    import {DurationSlider, WindowSettings} from './src/components/Sidebar/DurationSlider';
    window.store = useStore;
    window.mount = () => {
      window.root = createRoot(document.getElementById('root'));
      window.root.render(<><DurationSlider includeWindowSettings/><WindowSettings/></>);
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
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/*', route => route.fulfill({contentType: 'text/html', body: '<div id="root"></div>'}));
    await page.route('**/api/v1/model-options/*', route => {
      const modelType = decodeURIComponent(new URL(route.request().url()).pathname.split('/').pop());
      const options = optionsByType[modelType];
      return options
        ? route.fulfill({json: options})
        : route.fulfill({status: 404, json: {detail: 'Unknown test model'}});
    });
    await page.route('**/api/v1/loras/*', route => route.fulfill({json: {loras: [], guidance_max_phases: 1}}));
    await page.route('**/api/v1/generate', route => {
      submissions.push(route.request().postDataJSON());
      return route.fulfill({status: 400, json: {detail: 'Captured by isolated duration test'}});
    });
    await page.goto('http://pika-flex-duration.test');
    await page.addScriptTag({content: bundle.outputFiles[0].text});

    const configure = async ({modelType, reference = false, resolution, ordinary = false}) => {
      const options = ordinary ? {
        ...modelOptionsFor(modelType, false),
        minimax_h3_pika_flex: false,
        frames_maximum: 345,
        sliding_window: true,
        sliding_window_defaults: {
          window_min: 124, window_max: 345, window_step: 17,
          window_default: 243, overlap_default: 18, discard_last_frames: 0,
        },
      } : modelOptionsFor(modelType, reference);
      optionsByType[modelType] = options;
      await page.evaluate(({modelType, reference, resolution, options}) => {
        const state = window.store.getState();
        const references = reference
          ? [{type: 'image', path: '/uploads/ref.png', filename: 'ref.png', role: 'Traveler'}]
          : [];
        const model = {...(
          reference ? {omni_reference: true} : {}
        ),
          model_type: modelType, name: modelType, family: 'minimax_h3', architecture: 'minimax_h3',
          is_i2v: !reference, is_t2v: true, fps: 24, guidance_max_phases: 1, is_downloaded: true,
        };
        window.store.setState({
          generationMode: 'video',
          studioVideoWorkflow: reference ? 'references' : 'frames',
          studioVideoCreateRoute: 'auto',
          studioVideoEffectiveCreateRoute: reference ? 'omni' : 'generate',
          modelOptions: null,
          modelOptionsLoading: false,
          models: [model],
          modelsLoaded: true,
          enabledModels: new Set([modelType]),
          selectedModelPerMode: {...state.selectedModelPerMode, video: modelType},
          resolutionPreset: resolution === '1344x768' ? '768p' : '480p',
          aspectRatio: '16:9',
          durationSeconds: 362 / 24,
          slidingWindowSeconds: 243 / 24,
          slidingWindowLocked: true,
          slidingWindowOverlap: 18,
          h3WindowOverrides: {},
          systemStats: {gpu: {vram_total_gb: 48}, ram: {total_gb: 128}},
          startImage: null,
          endImage: null,
          imageRefs: [],
          jobs: [],
          isGenerating: false,
          params: {
            ...state.params,
            model_type: modelType,
            resolution,
            image_mode: 0,
            prompt: 'A traveler crosses a quiet garden at sunrise.',
            video_length: 362,
            sliding_window_size: 243,
            sliding_window_overlap: 18,
            sliding_window_discard_last_frames: 0,
            sliding_window_memory_override: true,
            minimax_h3_extended_duration: true,
            minimax_h3_multi_window: true,
            minimax_h3_reference_sequence: true,
            minimax_h3_sequence_memory_override: true,
            minimax_h3_references: references,
            image_start: '', image_end: '', image_prompt_type: '',
            video_guide: '', audio_guide: '',
            _duration_planning_mode: 'duration',
          },
        });
        return window.store.getState().loadModelOptions(modelType);
      }, {modelType, reference, resolution, options});
      await page.waitForFunction(modelType => !window.store.getState().modelOptionsLoading
        && window.store.getState().modelOptions?.model_type === modelType, modelType);
    };

    await page.evaluate(() => window.mount());

    for (const scenario of [
      {modelType: 'pika_flex_frames_test', reference: false, resolution: '832x480'},
      {modelType: 'pika_flex_references_test', reference: true, resolution: '1344x768'},
    ]) {
      submissions.length = 0;
      await configure(scenario);
      const hydrated = await page.evaluate(() => {
        const state = window.store.getState();
        return {
          frames: state.params.video_length,
          window: state.params.sliding_window_size,
          duration: state.durationSeconds,
          windowSeconds: state.slidingWindowSeconds,
          locked: state.slidingWindowLocked,
          sequence: !!(state.params.minimax_h3_multi_window || state.params.minimax_h3_reference_sequence),
          extended: state.params.minimax_h3_extended_duration,
        };
      });
      assert.deepEqual(hydrated, {
        frames: 362, window: 362, duration: 362 / 24, windowSeconds: 362 / 24,
        locked: false, sequence: false, extended: false,
      }, 'Pika model-option hydration resets stale H3 window state without shortening duration');
      const durationState = await page.evaluate(() => {
        const store = window.store;
        const state = store.getState();
        store.setState({params: {
          ...state.params,
          multi_prompts_gen_type: 3,
          minimax_h3_multi_window: true,
          minimax_h3_reference_sequence: true,
          minimax_h3_extended_duration: true,
          sliding_window_memory_override: true,
          per_clip_frames: [243, 243],
          per_clip_minimax_h3_references: [[{type: 'image', path: '/uploads/stale.png'}]],
          minimax_h3_sequence_clip_frames: 243,
          minimax_h3_sequence_continuity: true,
          minimax_h3_sequence_memory_override: true,
          minimax_h3_sequence_prompt_mode: 'manual',
          minimax_h3_window_storyboard: false,
          h3_window_prompts: ['stale first', 'stale second'],
          h3_window_plan_signature: 'stale-plan',
          h3_window_plan: {windows: [{prompt: 'stale'}]},
          _h3_window_plan_reviewed: true,
          _reviewed_h3_runtime_geometry: {video_length: 486},
          _omni_sequence_continuity: true,
          _omni_sequence_target_frames: 486,
        }});
        store.getState().setDurationSeconds(362 / 24);
        const params = store.getState().params;
        return {
          promptMode: params.multi_prompts_gen_type,
          flags: [params.minimax_h3_multi_window, params.minimax_h3_reference_sequence,
            params.minimax_h3_extended_duration, params.sliding_window_memory_override],
          staleFieldsRemain: [
            'per_clip_frames', 'per_clip_minimax_h3_references',
            'minimax_h3_sequence_clip_frames', 'minimax_h3_sequence_continuity',
            'minimax_h3_sequence_memory_override', 'minimax_h3_sequence_prompt_mode',
            'minimax_h3_window_storyboard', 'h3_window_prompts',
            'h3_window_plan_signature', 'h3_window_plan', '_h3_window_plan_reviewed',
            '_reviewed_h3_runtime_geometry', '_omni_sequence_continuity',
            '_omni_sequence_target_frames',
          ].filter(key => Object.prototype.hasOwnProperty.call(params, key)),
        };
      });
      assert.deepEqual(durationState, {
        promptMode: 2, flags: [false, false, false, false], staleFieldsRemain: [],
      },
        'Pika duration changes clear stale multi-clip and H3 sequence metadata');
      assert.equal(await page.evaluate(() => window.store.getState().params.video_length), 362);
      assert.equal(await page.getByRole('slider', {name: 'Duration model duration'}).getAttribute('max'), '14');
      assert.equal(await page.getByRole('slider', {name: 'Duration model duration'}).getAttribute('aria-valuetext'), '15.1s');
      assert.equal(await page.getByRole('slider', {name: 'Window length', exact: true}).count(), 0);
      assert.equal(await page.getByRole('button', {name: 'Window', exact: true}).count(), 0);
      assert.equal(await page.getByRole('checkbox', {name: /Allow 30s clips/}).count(), 0);

      await page.evaluate(async () => {
        const store = window.store;
        const state = store.getState();
        store.setState({params: {
          ...state.params,
          multi_prompts_gen_type: 3,
          minimax_h3_multi_window: true,
          minimax_h3_reference_sequence: true,
          minimax_h3_extended_duration: true,
          sliding_window_memory_override: true,
          per_clip_frames: [243, 243],
          per_clip_minimax_h3_references: [[{type: 'image', path: '/uploads/stale.png'}]],
          minimax_h3_sequence_clip_frames: 243,
          minimax_h3_sequence_continuity: true,
          h3_window_prompts: ['stale first', 'stale second'],
          h3_window_plan_signature: 'stale-plan',
          h3_window_plan: {windows: [{prompt: 'stale'}]},
          _reviewed_h3_runtime_geometry: {video_length: 486},
        }});
        await store.getState().startGeneration();
      });
      assert.equal(submissions.length, 1, 'The mock API captures one request');
      assert.equal(submissions[0].video_length, 362, 'The UI request retains the full native duration');
      assert.equal(submissions[0].sliding_window_size, 362, 'The inert window field matches duration');
      assert.equal(submissions[0].resolution, scenario.resolution, 'Exact Pika canvas bucket is retained');
      assert.equal(submissions[0].minimax_h3_multi_window, false);
      assert.equal(submissions[0].minimax_h3_reference_sequence ?? false, false);
      assert.equal(submissions[0].sliding_window_memory_override, false);
      assert.equal(submissions[0].minimax_h3_extended_duration, false);
      assert.equal(submissions[0].multi_prompts_gen_type, 2);
      for (const key of [
        'per_clip_frames', 'per_clip_minimax_h3_references',
        'minimax_h3_sequence_clip_frames', 'minimax_h3_sequence_continuity',
        'h3_window_prompts', 'h3_window_plan_signature', 'h3_window_plan',
        '_reviewed_h3_runtime_geometry',
      ]) {
        assert.equal(Object.prototype.hasOwnProperty.call(submissions[0], key), false,
          `The Pika request drops stale ${key}`);
      }
    }

    const ordinaryModel = 'ordinary_h3_duration_test';
    submissions.length = 0;
    await page.evaluate(() => window.root.unmount());
    await page.evaluate(modelType => {
      const state = window.store.getState();
      const model = {
        model_type: modelType, name: modelType, family: 'minimax_h3', architecture: 'minimax_h3',
        is_i2v: true, is_t2v: true, fps: 24, guidance_max_phases: 1, is_downloaded: true,
      };
      window.store.setState({
        generationMode: 'video', studioVideoWorkflow: 'frames',
        studioVideoCreateRoute: 'auto', studioVideoEffectiveCreateRoute: 'generate',
        models: [model], enabledModels: new Set([modelType]), modelOptions: {
          model_type: modelType, architecture: 'minimax_h3', fps: 24,
          frames_minimum: 124, frames_maximum: 345, frames_steps: 17,
          sliding_window: true, omni_reference: false,
          sliding_window_defaults: {window_min:124, window_max:345, window_step:17,
            window_default:243, overlap_default:18, discard_last_frames:0},
          sliding_window_memory_policy: null,
        },
        selectedModelPerMode: {...state.selectedModelPerMode, video: modelType},
        durationSeconds: 362 / 24, slidingWindowSeconds: 243 / 24, slidingWindowLocked: true,
        studioVideoWorkflow: 'frames', jobs: [], isGenerating: false,
        params: {...state.params, model_type: modelType, resolution:'832x480', image_mode:0,
          prompt:'A traveler crosses a quiet garden at sunrise.', video_length:362,
          sliding_window_size:243, minimax_h3_multi_window:false,
          minimax_h3_reference_sequence:false, minimax_h3_extended_duration:false},
      });
    }, ordinaryModel);
    await page.evaluate(async () => window.store.getState().startGeneration());
    const ordinaryState = await page.evaluate(() => {
      const state = window.store.getState();
      return {
        modelType: state.modelOptions?.model_type,
        prompt: state.params.prompt,
        error: state.promptEnhanceError,
        loading: state.isGenerating,
        paramsModelType: state.params.model_type,
      };
    });
    assert.equal(submissions.length, 1, `Ordinary H3 request was not captured: ${JSON.stringify(ordinaryState)}`);
    const ordinary = submissions[0];
    assert.equal(ordinary.model_type, ordinaryModel);
    assert.equal(ordinary.video_length, 243, 'Ordinary H3 keeps its legacy one-window clamp');

    assert.deepEqual(errors, [], 'The duration controls render without UI errors');
    console.log('Pika Flex Frames/References keep 362 frames through hydration and generation; ordinary H3 remains window-capped');
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
