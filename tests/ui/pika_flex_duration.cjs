// Pika H3 Flex duration stays independent of stale H3 windows; requests are mocked.
// Run: node tests/ui/pika_flex_duration.cjs
const assert = require('node:assert/strict');
const path = require('node:path');
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

const modelOptionsFor = (modelType, reference = false, rolling = false) => ({
  model_type: modelType,
  architecture: 'minimax_h3',
  fps: 24,
  frames_minimum: 124,
  frames_maximum: 362,
  frames_steps: 17,
  sliding_window: rolling,
  video_continuation: rolling,
  sliding_window_audio_history: rolling,
  sliding_window_defaults: rolling ? {
    window_min: 124, window_max: 362, window_step: 17,
    window_default: 243, overlap_min: 1, overlap_max: 103,
    overlap_step: 17, overlap_default: 18, discard_last_frames: 0,
  } : null,
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
  resolution_preset_order: ['480p', '540p', '720p', '768p', '1080p'],
  resolution_presets: {
    '480p': {label: '480p', values: {'16:9': '832x480'}},
    '540p': {label: '540p', values: {'16:9': '960x544'}},
    '720p': {label: '720p', values: {'16:9': '1280x704'}},
    '768p': {label: '768p', values: {'16:9': '1344x768'}},
    '1080p': {label: '1080p', experimental: true, values: {'16:9': '1920x1088'}},
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
    import {DurationSlider} from './src/components/Sidebar/DurationSlider';
    import {ResolutionPresets} from './src/components/Sidebar/ResolutionPresets';
    window.store = useStore;
    window.mount = () => {
      window.root = createRoot(document.getElementById('root'));
      window.root.render(<><DurationSlider includeWindowSettings/><ResolutionPresets/></>);
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

    const configure = async ({modelType, reference = false, resolution, ordinary = false, rolling = false}) => {
      const options = ordinary ? {
        ...modelOptionsFor(modelType, false),
        minimax_h3_pika_flex: false,
        frames_maximum: 345,
        sliding_window: true,
        sliding_window_defaults: {
          window_min: 124, window_max: 345, window_step: 17,
          window_default: 243, overlap_default: 18, discard_last_frames: 0,
        },
      } : modelOptionsFor(modelType, reference, rolling);
      optionsByType[modelType] = options;
      await page.evaluate(({modelType, reference, resolution, options, ordinary, rolling}) => {
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
          resolutionPreset: ({'960x544': '540p', '1280x704': '720p',
            '1344x768': '768p', '1920x1088': '1080p'})[resolution] || '480p',
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
            sliding_window_memory_override: !rolling && !ordinary,
            minimax_h3_multi_window: !rolling && !ordinary,
            minimax_h3_reference_sequence: !rolling && !ordinary,
            minimax_h3_sequence_continuity: true,
            minimax_h3_sequence_prompt_mode: 'auto',
            minimax_h3_window_storyboard: true,
            num_inference_steps: 6,
            minimax_h3_extended_duration: !rolling && !ordinary,
            minimax_h3_sequence_memory_override: !rolling && !ordinary,
            minimax_h3_references: references,
            image_start: '', image_end: '', image_prompt_type: '',
            video_guide: '', audio_guide: '',
            _duration_planning_mode: 'duration',
          },
        });
        return window.store.getState().loadModelOptions(modelType);
      }, {modelType, reference, resolution, options, ordinary, rolling});
      await page.waitForFunction(modelType => !window.store.getState().modelOptionsLoading
        && window.store.getState().modelOptions?.model_type === modelType, modelType);
    };

    await page.evaluate(() => window.mount());

    for (const scenario of [
      {modelType: 'pika_flex_frames_test', reference: false, resolution: '832x480'},
      {modelType: 'pika_flex_references_test', reference: true, resolution: '1344x768'},
      ...['960x544', '1280x704', '1920x1088'].flatMap(resolution => [
        {modelType: `pika_flex_frames_${resolution}_test`, reference: false, resolution},
        {modelType: `pika_flex_references_${resolution}_test`, reference: true, resolution},
      ]),
    ]) {
      submissions.length = 0;
      await configure(scenario);
      for (const tier of ['480p', '540p', '720p', '768p', '1080p']) {
        assert.equal(await page.getByRole('button', {name: tier, exact: true}).count(), 1,
          `Both Pika modes expose the ${tier} resolution choice`);
      }
      const selectedTier = await page.evaluate(() => window.store.getState().resolutionPreset);
      await page.getByRole('button', {name: selectedTier, exact: true}).click();
      assert.equal(await page.evaluate(() => window.store.getState().params.resolution),
        scenario.resolution, 'Selecting a resolution retains its exact aligned canvas');
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

    await page.evaluate(() => window.mount());
    const referenceRecord = {type: 'image', path: '/uploads/ref.png', filename: 'ref.png', role: 'Traveler'};
    const rollingScenarios = [
      {modelType: 'pika_flex_frames_rolling_test', reference: false, resolution: '832x480'},
      {modelType: 'pika_flex_references_rolling_test', reference: true, resolution: '1344x768'},
    ];
    const setRollingTimeline = async ({reference, frames}) => {
      const windowSlider = page.getByRole('slider', {name: 'Window length', exact: true});
      assert.equal(await windowSlider.count(), 1, 'The rolling backend exposes Window Settings');
      assert.equal(await windowSlider.isVisible(), true, 'Window Length is visible for rolling Pika Flex');
      assert.ok(Number(await page.getByRole('slider', {name: 'Duration model duration'}).getAttribute('max')) > 14,
        'The duration slider exposes long-form durations');
      await windowSlider.focus();
      await windowSlider.press('Home');
      await page.getByRole('button', {name: 'Time', exact: true}).click();
      const timecode = page.getByRole('textbox', {name: 'Duration timecode', exact: true});
      await timecode.fill(String(frames / 24));
      await timecode.press('Enter');
      await page.waitForFunction(({frames: expectedFrames, reference: isReference}) => {
        const state = window.store.getState();
        return state.params.video_length === expectedFrames
          && state.params.sliding_window_size === 124
          && state.durationSeconds === expectedFrames / 24
          && state.slidingWindowSeconds === 124 / 24
          && state.slidingWindowLocked === true
          && state.params[isReference ? 'minimax_h3_reference_sequence' : 'minimax_h3_multi_window'] === true;
      }, {frames, reference});
      const state = await page.evaluate(() => {
        const current = window.store.getState();
        return {
          frames: current.params.video_length,
          window: current.params.sliding_window_size,
          duration: current.durationSeconds,
          windowSeconds: current.slidingWindowSeconds,
          locked: current.slidingWindowLocked,
          overlap: current.params.sliding_window_overlap,
          sequence: current.params.minimax_h3_reference_sequence === true
            || current.params.minimax_h3_multi_window === true,
          steps: current.params.num_inference_steps,
        };
      });
      assert.deepEqual(state, {
        frames, window: 124, duration: frames / 24, windowSeconds: 124 / 24,
        locked: true, overlap: 18, sequence: true, steps: 6,
      }, 'The locked native window preserves the exact non-grid total duration');
      assert.ok(await page.getByText('18f (0.8s)', {exact: true}).isVisible(),
        'Window overlap settings remain visible at the 18-frame default');
      assert.match(await page.locator('#root').innerText(), reference
        ? /2 native Omni windows/
        : /2 windows of 5\.2s/,
      'The duration controls report two rolling windows');
    };
    const restoreFromSidecar = async ({modelType, params, name}) => {
      await page.evaluate(async ({modelType, params, name}) => {
        const state = window.store.getState();
        window.store.setState({
          selectedOutputMeta: {source: 'sidecar', params},
          selectedOutput: 0,
          outputs: [{name, workspace: state.activeWorkspace, path: '/outputs/' + name,
            url: '/' + name, type: 'video', mode: 'video', size: 1, created_at: 1}],
        });
        await window.store.getState().loadSettingsFromOutput();
      }, {modelType, params, name});
      await page.waitForFunction(expectedModel => window.store.getState().params.model_type === expectedModel,
        modelType);
    };

    for (const scenario of rollingScenarios) {
      submissions.length = 0;
      await configure({...scenario, rolling: true});
      await page.evaluate(() => {
        const state = window.store.getState();
        window.store.setState({params: {...state.params,
          prompt: 'A traveler crosses a quiet garden at sunrise.\nThe traveler reaches the far garden gate.'}});
      });
      const capabilities = await page.evaluate(() => ({
        slidingWindow: window.store.getState().modelOptions?.sliding_window,
        continuation: window.store.getState().modelOptions?.video_continuation,
        audioHistory: window.store.getState().modelOptions?.sliding_window_audio_history,
        maximum: window.store.getState().modelOptions?.frames_maximum,
        minimum: window.store.getState().modelOptions?.frames_minimum,
        step: window.store.getState().modelOptions?.frames_steps,
      }));
      assert.deepEqual(capabilities, {
        slidingWindow: true, continuation: true, audioHistory: true,
        maximum: 362, minimum: 124, step: 17,
      }, 'The mock advertises the new native H3 motion and audio continuation backend');
      await setRollingTimeline({reference: scenario.reference, frames: 230});
      submissions.length = 0;
      await page.evaluate(async () => window.store.getState().startGeneration());
      const generationState = await page.evaluate(() => {
        const state = window.store.getState();
        return {modelType: state.params.model_type, route: state.studioVideoEffectiveCreateRoute,
          error: state.promptEnhanceError, sequence: state.params.minimax_h3_multi_window,
          referenceSequence: state.params.minimax_h3_reference_sequence,
          workflow: state.studioVideoWorkflow, frames: state.params.video_length,
          window: state.params.sliding_window_size};
      });
      assert.equal(submissions.length, 1,
        'The rolling Pika request reaches the mocked generation endpoint: ' + JSON.stringify(generationState));
      const submitted = submissions[0];
      assert.equal(submitted.video_length, 230, 'Rolling generation retains the requested total frames');
      assert.equal(submitted.sliding_window_size, 124, 'Rolling generation uses the locked native window');
      assert.equal(submitted.sliding_window_overlap, 18, 'Rolling generation carries the configured overlap');
      assert.equal(submitted.num_inference_steps, 6, 'Fixed six-step Pika settings remain unchanged');
      assert.equal(submitted.minimax_h3_multi_window ?? false, !scenario.reference,
        'Frames uses the H3 multi-window sequence flag');
      assert.equal(submitted.minimax_h3_reference_sequence ?? false, scenario.reference,
        'References uses the H3 Omni sequence flag');
      if (scenario.reference) {
        assert.equal(submitted.minimax_h3_sequence_continuity, true,
          'Reference windows use native motion and audio continuation');
        assert.equal(submitted.minimax_h3_sequence_clip_frames, 124,
          'References retain the selected native clip size');
        assert.deepEqual(submitted.minimax_h3_references, [referenceRecord],
          'The canonical reference manifest survives sequence generation');
      }

      await restoreFromSidecar({
        modelType: scenario.modelType,
        params: submitted,
        name: scenario.modelType + '-sequence.mp4',
      });
      const restored = await page.evaluate(() => {
        const state = window.store.getState();
        return {
          frames: state.params.video_length,
          window: state.params.sliding_window_size,
          overlap: state.params.sliding_window_overlap,
          duration: state.durationSeconds,
          windowSeconds: state.slidingWindowSeconds,
          locked: state.slidingWindowLocked,
          framesSequence: state.params.minimax_h3_multi_window === true,
          referenceSequence: state.params.minimax_h3_reference_sequence === true,
          steps: state.params.num_inference_steps,
          references: state.params.minimax_h3_references,
        };
      });
      assert.equal(restored.frames, 230, 'A sequence sidecar restores its exact total frame count');
      assert.equal(restored.window, 124, 'A sequence sidecar restores its native window');
      assert.equal(restored.overlap, 18, 'A sequence sidecar restores its overlap');
      assert.equal(restored.duration, 230 / 24, 'Restored duration does not round onto the native grid');
      assert.equal(restored.windowSeconds, 124 / 24);
      assert.equal(restored.locked, true);
      assert.equal(restored.framesSequence, !scenario.reference);
      assert.equal(restored.referenceSequence, scenario.reference);
      assert.equal(restored.steps, 6);
      if (scenario.reference) {
        assert.deepEqual(restored.references, [referenceRecord],
          'The canonical reference manifest survives sidecar restoration');

        // 231 frames is also off the 17-frame native grid and catches rounding
        // loss that happens to leave 230 unchanged.
        await restoreFromSidecar({
          modelType: scenario.modelType,
          params: {...submitted, video_length: 231, duration_seconds: 231 / 24},
          name: scenario.modelType + '-sequence-231.mp4',
        });
        const restored231 = await page.evaluate(() => ({
          frames: window.store.getState().params.video_length,
          duration: window.store.getState().durationSeconds,
          window: window.store.getState().params.sliding_window_size,
          overlap: window.store.getState().params.sliding_window_overlap,
          sequence: window.store.getState().params.minimax_h3_reference_sequence,
        }));
        assert.deepEqual(restored231, {
          frames: 231, duration: 231 / 24, window: 124, overlap: 18, sequence: true,
        }, 'Reference sidecar restoration preserves a second non-grid total exactly');
      }
    }

    const rollingFrames = rollingScenarios[0];
    await configure({...rollingFrames, rolling: true});
    submissions.length = 0;
    await page.evaluate(async () => {
      const state = window.store.getState();
      window.store.setState({
        durationSeconds: 362 / 24,
        slidingWindowSeconds: 243 / 24,
        slidingWindowLocked: true,
        params: {...state.params,
          video_length: 362, sliding_window_size: 243, sliding_window_overlap: 18,
          minimax_h3_multi_window: false, minimax_h3_reference_sequence: false,
          minimax_h3_extended_duration: false, num_inference_steps: 6},
      });
      await window.store.getState().startGeneration();
    });
    assert.equal(submissions.length, 1, 'A native one-window Flex request is captured');
    assert.equal(submissions[0].video_length, 362,
      'Explicitly disabled sequence flags keep the full native duration');
    assert.equal(submissions[0].sliding_window_size, 362,
      'A stale locked 243-frame window cannot shorten the native one-window request');
    assert.equal(submissions[0].sliding_window_overlap, 0);
    assert.equal(submissions[0].minimax_h3_multi_window, false);
    assert.equal(submissions[0].minimax_h3_reference_sequence ?? false, false);
    assert.equal(submissions[0].num_inference_steps, 6);

    const legacyReferenceModel = rollingScenarios[1].modelType.replace('_rolling_test', '_test');
    await configure({modelType: legacyReferenceModel, reference: true, resolution: '1344x768'});
    await restoreFromSidecar({
      modelType: legacyReferenceModel,
      params: {
        model_type: legacyReferenceModel, prompt: 'A traveler crosses a quiet garden at sunrise.',
        video_length: 362, duration_seconds: 362 / 24, sliding_window_size: 243,
        sliding_window_overlap: 18, minimax_h3_multi_window: false,
        minimax_h3_reference_sequence: false, minimax_h3_references: [referenceRecord],
        num_inference_steps: 6, image_mode: 0, _studio_video_workflow: 'references',
      },
      name: legacyReferenceModel + '-single-pass.mp4',
    });
    const legacyRestored = await page.evaluate(() => {
      const state = window.store.getState();
      return {frames: state.params.video_length, window: state.params.sliding_window_size,
        overlap: state.params.sliding_window_overlap, duration: state.durationSeconds,
        sequence: state.params.minimax_h3_reference_sequence};
    });
    assert.deepEqual(legacyRestored, {
      frames: 362, window: 362, overlap: 0, duration: 362 / 24, sequence: false,
    }, 'The legacy single-pass sidecar still restores its full 362-frame native clip');

    assert.deepEqual(errors, [], 'The duration controls render without UI errors');
    console.log('Pika Flex: legacy single-pass, new Frames/References sequences, exact sidecar restores, fixed steps, and ordinary H3 compatibility passed');
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
