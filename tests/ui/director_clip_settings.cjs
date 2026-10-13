// Exercise the real gallery restore and Studio submission paths with all
// network requests intercepted. No render or LLM service is contacted.
// Run: node tests/ui/director_clip_settings.cjs
const assert = require('node:assert/strict');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const playwright = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');

const referencesModel = 'minimax_h3_ref2va_clip_restore_test';
const framesModel = 'minimax_h3_frames_clip_restore_test';
const previousFramesModel = 'minimax_h3_frames_previous_clip_restore_test';
const modelTypes = [referencesModel, framesModel, previousFramesModel];
const clipRestoreMemoryPolicy = {
  checkpoint: 'pruned',
  auto_resolution_pixels: {'864x480': 414720, '1280x720': 921600},
  resolution_bands: [{
    min_pixels: 0,
    vram_tiers: [{max_vram_gb: 24, frames: 243, fallback_resolution: '864x480'}],
  }],
};
const optionsFor = modelType => {
  const references = modelType === referencesModel;
  return {
    model_type: modelType,
    architecture: references ? 'minimax_h3_ref2va' : 'minimax_h3',
    fps: 24,
    frames_minimum: 124,
    frames_steps: 17,
    frames_maximum: 345,
    default_num_inference_steps: 20,
    default_guidance_scale: 4,
    guidance_max_phases: 1,
    omni_reference: references,
    sliding_window: true,
    sliding_window_memory_policy: clipRestoreMemoryPolicy,
    omni_sequence_memory_policy: clipRestoreMemoryPolicy,
    sliding_window_auto_prompt_pacing: true,
    sliding_window_defaults: {
      window_min: 124,
      window_max: 345,
      window_step: 17,
      window_default: 124,
      overlap_min: 1,
      overlap_max: 103,
      overlap_step: 17,
      overlap_offset: 1,
      overlap_default: 18,
      discard_last_frames: 0,
    },
    veda_attention: true,
    veda_attention_status: {supported: true},
    sla_attention: true,
    sla_attention_default: false,
  };
};

function output(name, workspace = 'director-archive') {
  return {
    name,
    workspace,
    path: `/outputs/${name}`,
    url: `/${name}`,
    type: 'video',
    mode: 'video',
    size: 1,
    created_at: 1,
  };
}

function clipMetadata({index, modelType, prompt, params = {}, uploads = {}, urls = {}}) {
  return {
    source: 'sidecar',
    director_pipeline_id: 'saved-director-project',
    director_artifact_kind: 'clip',
    director_clip_index: index,
    director_clip_settings_version: 1,
    director_clip_settings_provenance: 'output_name',
    upload_filenames: uploads,
    director_clip_asset_urls: urls,
    params: {
      model_type: modelType,
      prompt,
      seed: 8302,
      num_inference_steps: 19,
      guidance_scale: 1,
      resolution: '864x480',
      video_length: 345,
      duration_seconds: 14.375,
      fps: 24,
      sliding_window_size: 345,
      sliding_window_overlap: 18,
      sliding_window_discard_last_frames: 0,
      multi_prompts_gen_type: 0,
      image_mode: 0,
      repeat_generation: 1,
      activated_loras: [],
      loras_multipliers: '',
      override_attention: 'veda',
      minimax_h3_multi_window: true,
      minimax_h3_window_storyboard: true,
      minimax_h3_sequence_prompt_mode: 'creative',
      h3_window_prompts: ['stale window prompt'],
      h3_window_plan: {
        source_prompt: 'stale source prompt',
        plan_kind: 'sliding_window',
        windows: [{index: 1, prompt: 'stale planned window'}],
      },
      _h3_original_prompt: 'stale hidden source prompt',
      _tts_original_prompt: 'stale substituted prompt',
      custom_settings: {},
      _studio_video_workflow: modelType === referencesModel ? 'references' : 'frames',
      ...params,
    },
  };
}

(async () => {
  const bundle = await esbuild.build({
    stdin: {
      contents: `
        import {createElement} from 'react';
        import {createRoot} from 'react-dom/client';
        import {DurationSlider} from './src/components/Sidebar/DurationSlider';
        import {useStore, shouldEnhanceOnGeneration} from './src/stores/useStore';
        import {subscribeMaestroAlerts} from './src/lib/notifications';
        window.store = useStore;
        window.shouldEnhanceOnGeneration = shouldEnhanceOnGeneration;
        window.mountDurationSlider = (copies = 1) => {
          const container = document.createElement('div');
          container.id = 'duration-slider-test';
          document.body.appendChild(container);
          window.durationSliderRoot = createRoot(container);
          window.durationSliderRoot.render(createElement('div', {},
            Array.from({length: copies}, (_, key) => createElement(DurationSlider, {key, includeWindowSettings: true}))));
        };
        window.unmountDurationSlider = () => {
          window.durationSliderRoot?.unmount();
          document.getElementById('duration-slider-test')?.remove();
          window.durationSliderRoot = null;
        };
        window.alerts = [];
        subscribeMaestroAlerts(alert => window.alerts.push(alert));
      `,
      resolveDir: path.join(root, 'ui'),
      loader: 'tsx',
    },
    bundle: true,
    write: false,
    jsx: 'automatic',
    define: {'process.env.NODE_ENV': '"development"'},
    logLevel: 'silent',
  });
  const browser = await playwright.chromium.launch({
    headless: true,
    ...(process.env.MAESTRO_CHROME
      ? {executablePath: process.env.MAESTRO_CHROME}
      : process.platform === 'win32' ? {channel: 'chrome'} : {}),
  });

  try {
    const page = await browser.newPage();
    const fileRequests = [];
    const generationRequests = [];
    const directorLoads = [];
    const pageErrors = [];
    page.on('pageerror', error => pageErrors.push(error.message));

    await page.route('**/*', async route => {
      const request = route.request();
      const url = new URL(request.url());
      const json = (body, status = 200) => route.fulfill({
        status,
        contentType: 'application/json',
        body: JSON.stringify(body),
      });
      if (url.pathname === '/') {
        return route.fulfill({contentType: 'text/html', body: '<div id="root"></div>'});
      }
      if (url.pathname.startsWith('/api/v1/file/')) {
        fileRequests.push({pathname: url.pathname, workspace: url.searchParams.get('workspace')});
        return route.fulfill({status: 200, contentType: 'image/png', body: 'fixture image bytes'});
      }
      if (url.pathname.startsWith('/api/v1/model-options/')) {
        return json(optionsFor(decodeURIComponent(url.pathname.split('/').pop())));
      }
      if (url.pathname.startsWith('/api/v1/loras/')) return json({loras: [], guidance_max_phases: 1});
      if (url.pathname === '/api/v1/generate') {
        generationRequests.push(request.postDataJSON());
        return json({job_id: `clip-restore-${generationRequests.length}`, status: 'held'});
      }
      if (url.pathname.startsWith('/api/')) return json({jobs: [], outputs: [], items: [], configured: true});
      return route.fulfill({status: 404, body: ''});
    });

    await page.goto('http://director-clip-settings.test');
    await page.addScriptTag({content: bundle.outputFiles[0].text});

    const seed = (meta, media, modelType, extra = {}) => page.evaluate(
      ({meta, media, modelType, modelTypes, extra}) => {
        const store = window.store;
        const state = store.getState();
        const models = modelTypes.map(type => ({
          model_type: type,
          name: type,
          family: 'minimax_h3',
          architecture: type === 'minimax_h3_ref2va_clip_restore_test'
            ? 'minimax_h3_ref2va' : 'minimax_h3',
          is_i2v: true,
          is_t2v: true,
          guidance_max_phases: 1,
          fps: 24,
          is_downloaded: true,
        }));
        store.setState({
          outputs: [media],
          selectedOutput: 0,
          selectedOutputMeta: meta,
          mediaFilter: 'all',
          activeWorkspace: 'different-active-workspace',
          browsingAllFolders: false,
          browsingUploads: false,
          models,
          modelsLoaded: true,
          enabledModels: new Set(modelTypes),
          selectedModelPerMode: {video: modelType},
          restoredWindowSelectionKey: null,
          savedParamsPerMode: {},
          savedLoraPerMode: {},
          savedPromptPerMode: {},
          modelOptions: null,
          generationMode: 'video',
          sidebarMode: 'director',
          studioVideoWorkflow: 'references',
          studioVideoEffectiveCreateRoute: 'generate',
          durationSeconds: 5,
          slidingWindowSeconds: 5,
          slidingWindowOverlap: 18,
          slidingWindowLocked: false,
          startImage: null,
          endImage: null,
          imageRefs: [],
          imageRefType: 'I',
          clips: [],
          singlePromptMode: false,
          h3WindowPlan: null,
          jobs: [],
          isGenerating: false,
          enhanceOnGeneration: extra.enhanceOnGeneration ?? true,
          enhanceOnGenerationDefault: extra.enhanceOnGenerationDefault ?? true,
          servicesConfig: {nsfw_mode: false},
          params: {
            ...state.params,
            model_type: modelType,
            prompt: 'previous Studio prompt',
            seed: 1,
            image_mode: 0,
            video_length: 81,
            custom_settings: {},
          },
          loadDirectorFromPipeline: async id => {
            window.directorLoads.push(id);
            store.setState({sidebarMode: 'director'});
          },
        });
      },
      {meta, media, modelType, modelTypes, extra},
    );
    await page.evaluate(() => { window.directorLoads = []; });

    const prompt = '  Preserve this clip prompt exactly.\nKeep its second line.  ';
    const archiveWorkspace = 'director-archive';
    const clipZeroAudioPath = 'C:/Maestro/outputs/_director_clip_settings_slices/saved-director-project/clip-0.wav';
    const clipZero = clipMetadata({
      index: 0,
      modelType: referencesModel,
      prompt,
      params: {
        audio_guide: clipZeroAudioPath,
        audio_prompt_type: 'A',
        // Older Omni sidecars can carry stale frame flags and upload names.
        // The scoped Director recipe must only hydrate actual clip inputs.
        image_prompt_type: 'S',
        minimax_h3_references: [
          {id: 'hero', type: 'image', path: 'C:/Maestro/outputs/_director_assets/saved-director-project/ref/hero.png', filename: 'hero.png'},
          {id: 'continuity', type: 'image', path: 'C:/Maestro/outputs/_director_assets/saved-director-project/ref/continuity.png', filename: 'continuity.png', _maestro_generated_continuity: true},
        ],
      },
      uploads: {
        audio_guide: '_director_clip_settings_slices/saved-director-project/clip-0.wav',
        image_start: '_director_assets/saved-director-project/frames/stale-omni-anchor.png',
        minimax_h3_references: [
          '_director_assets/saved-director-project/ref/hero.png',
          '_director_assets/saved-director-project/ref/continuity.png',
        ],
      },
      urls: {
        audio_guide: '/api/v1/file/_director_clip_settings_slices/saved-director-project/clip-0.wav?workspace=director-archive',
        minimax_h3_references: [
          '/api/v1/file/_director_assets/saved-director-project/ref/hero.png?workspace=director-archive',
          '/api/v1/file/_director_assets/saved-director-project/ref/continuity.png?workspace=director-archive',
        ],
      },
    });
    await seed(clipZero, output('clip-0.mp4', archiveWorkspace), referencesModel);
    await page.evaluate(() => window.store.getState().loadSettingsFromOutput());
    await page.waitForFunction(() => window.store.getState().sidebarMode === 'studio');

    assert.equal(await page.evaluate(() => window.store.getState().params.prompt), prompt,
      'clip index 0 restores its exact prompt rather than a saved hidden source prompt');
    assert.equal(await page.evaluate(() => window.store.getState().params.seed), 8302);
    assert.equal(await page.evaluate(() => window.store.getState().params.num_inference_steps), 19);
    assert.equal(await page.evaluate(() => window.store.getState().params.override_attention), 'veda',
      'the historical Veda choice survives restore for a later attention-only A/B');
    assert.equal(await page.evaluate(() => window.store.getState().params.video_length), 345);
    assert.equal(await page.evaluate(() => window.store.getState().durationSeconds), 14.375,
      'the native 345-frame / 24-fps duration remains exact');
    assert.equal(await page.evaluate(() => window.store.getState().params.duration_seconds), 14.375);
    assert.equal(await page.evaluate(() => window.store.getState().params.multi_prompts_gen_type), 0);
    assert.equal(await page.evaluate(() => window.store.getState().params.h3_window_plan), undefined);
    assert.equal(await page.evaluate(() => window.store.getState().params.h3_window_prompts), undefined);
    assert.equal(await page.evaluate(() => window.store.getState().h3WindowPlan), null);
    assert.equal(await page.evaluate(() => window.store.getState().params.minimax_h3_window_storyboard), false);
    assert.equal(await page.evaluate(() => window.store.getState().params.minimax_h3_sequence_prompt_mode), 'manual');
    assert.deepEqual(await page.evaluate(() => window.store.getState().params.minimax_h3_references.map(ref => ref.url)), [
      '/api/v1/file/_director_assets/saved-director-project/ref/hero.png?workspace=director-archive',
      '/api/v1/file/_director_assets/saved-director-project/ref/continuity.png?workspace=director-archive',
    ], 'archive URLs retain nested paths and the selected output workspace');
    assert.equal(await page.evaluate(() => window.store.getState().params.minimax_h3_references.length), 2,
      'a real continuity reference used by this clip is not filtered out');
    assert.equal(await page.evaluate(() => window.store.getState().params.audio_guide), clipZeroAudioPath,
      'the exact scoped soundtrack slice path remains the Studio recipe');
    assert.equal(await page.evaluate(() => window.store.getState().audioGuideFilename), 'clip-0.wav',
      'the soundtrack label names the clip slice, not the original full song');
    assert.equal(await page.evaluate(() => window.store.getState().startImage), null,
      'a stale Omni start flag and upload filename cannot hydrate an unconditioned anchor');
    assert.ok(!fileRequests.some(request => request.pathname.endsWith('/frames/stale-omni-anchor.png')),
      'stale Omni start filenames do not trigger an archive asset fetch');
    assert.deepEqual(await page.evaluate(() => window.directorLoads), [],
      'an individual clip does not reopen the full Director project');
    assert.equal(await page.evaluate(() => window.store.getState().enhanceOnGeneration), false,
      'the one-shot enhancer is disabled for this reviewed prompt');
    assert.equal(await page.evaluate(() => window.store.getState().enhanceOnGenerationDefault), true,
      'the user’s sticky enhancement preference remains enabled for future prompts');
    assert.equal(await page.evaluate(() => window.shouldEnhanceOnGeneration(window.store.getState())), false);

    await page.evaluate(() => window.store.getState().startGeneration('queue'));
    assert.equal(generationRequests.length, 1, 'Studio submits the restored clip recipe');
    assert.equal(generationRequests[0].prompt, prompt);
    assert.equal(generationRequests[0].seed, 8302);
    assert.equal(generationRequests[0].num_inference_steps, 19);
    assert.equal(generationRequests[0].override_attention, 'veda');
    assert.equal(generationRequests[0].video_length, 345);
    assert.equal(generationRequests[0].duration_seconds, 14.375);
    assert.equal(generationRequests[0]._enhance_on_generation, undefined,
      'a sticky default cannot silently rewrite the restored prompt on submission');

    const startPath = 'C:/Maestro/outputs/_director_assets/saved-director-project/frames/start.png';
    const imageRefPath = 'C:/Maestro/outputs/_director_assets/saved-director-project/refs/look.png';
    const clipSeven = clipMetadata({
      index: 7,
      modelType: framesModel,
      prompt: 'Clip seven exact prompt.',
      params: {
        image_start: startPath,
        image_refs: [imageRefPath],
        audio_guide: undefined,
        minimax_h3_references: undefined,
        sliding_window_memory_override: true,
      },
      uploads: {
        image_start: '_director_assets/saved-director-project/frames/start.png',
        image_refs: ['_director_assets/saved-director-project/refs/look.png'],
      },
      urls: {
        image_start: '/api/v1/file/_director_assets/saved-director-project/frames/start.png?workspace=director-archive',
        image_refs: ['/api/v1/file/_director_assets/saved-director-project/refs/look.png?workspace=director-archive'],
      },
    });
    await seed(clipSeven, output('clip-7.mp4', archiveWorkspace), framesModel);
    await page.evaluate(() => window.store.setState({audioGuideFilename: 'whole-song.wav'}));

    // Mount the actual Studio duration controls on a prior H3 model and
    // resolution. The archived clip then changes both selectors while the
    // hardware profile recommends 243 frames (10.1s) for ordinary Auto use.
    const priorModelOptions = await page.evaluate(async modelType => {
      const response = await fetch(`/api/v1/model-options/${modelType}`);
      return response.json();
    }, previousFramesModel);
    await page.evaluate(({modelType, modelOptions}) => {
      const store = window.store;
      const state = store.getState();
      store.setState({
        modelOptions,
        selectedModelPerMode: {...state.selectedModelPerMode, video: modelType},
        systemStats: {gpu: {vram_total_gb: 16}, ram: {total_gb: 64}},
        durationSeconds: 14.375,
        slidingWindowSeconds: 14.375,
        slidingWindowLocked: false,
        params: {
          ...state.params,
          model_type: modelType,
          resolution: '1280x720',
          prompt: 'Create a 14.4 second scene.',
          _duration_planning_mode: 'duration',
          video_length: 345,
          duration_seconds: 14.375,
          sliding_window_size: 345,
          sliding_window_memory_override: false,
          minimax_h3_multi_window: true,
          minimax_h3_reference_sequence: false,
          override_attention: 'sla',
        },
      });
    }, {modelType: previousFramesModel, modelOptions: priorModelOptions});
    await page.evaluate(() => window.mountDurationSlider(2));
    await page.waitForFunction(() => document.querySelector('#duration-slider-test [aria-label="Window length"]'));
    await page.evaluate(() => window.store.getState().loadSettingsFromOutput());
    await page.waitForFunction(() => window.store.getState().startImage?.name === 'start.png'
      && window.store.getState().imageRefs.length === 1);
    await page.waitForFunction(() => {
      const state = window.store.getState();
      return state.modelOptions?.model_type === 'minimax_h3_frames_clip_restore_test'
        && state.restoredWindowSelectionKey === 'minimax_h3_frames_clip_restore_test::864x480'
        && state.slidingWindowLocked
        && state.slidingWindowSeconds === 345 / 24;
    });
    assert.equal(await page.evaluate(() => window.store.getState().params.prompt), 'Clip seven exact prompt.');
    assert.equal(await page.evaluate(() => window.store.getState().params.image_start), startPath);
    assert.equal(await page.evaluate(() => window.store.getState().startImage.name), 'start.png');
    assert.equal(await page.evaluate(() => window.store.getState().imageRefs[0].name), 'look.png');
    assert.equal(await page.evaluate(() => window.store.getState().audioGuideFilename), null,
      'a clip without soundtrack input clears the previously selected full-song label');
    assert.ok(fileRequests.some(request => request.pathname.endsWith('/frames/start.png')
      && request.workspace === archiveWorkspace), 'the nested start frame is hydrated from its archive workspace');
    assert.ok(fileRequests.some(request => request.pathname.endsWith('/refs/look.png')
      && request.workspace === archiveWorkspace), 'nested reference previews use the selected output workspace');
    assert.match(await page.locator('#duration-slider-test').innerText(), /1 window · 14\.4s final output/,
      'the mounted Studio duration control displays the restored 345-frame clip as one window');

    await page.evaluate(() => window.store.getState().setParam('override_attention', 'sla'));
    await page.waitForFunction(() => window.store.getState().params.override_attention === 'sla');
    assert.equal(await page.evaluate(() => window.store.getState().slidingWindowLocked), true,
      'changing the attention backend preserves the restored manual window');
    assert.equal(await page.evaluate(() => window.store.getState().slidingWindowSeconds), 345 / 24,
      'changing the attention backend does not replace the restored 345-frame window with the Auto profile');
    assert.match(await page.locator('#duration-slider-test').innerText(), /1 window · 14\.4s final output/,
      'Veda-to-SLA leaves the mounted clip at one window');

    // Match the reported References workflow too, keeping both mounted
    // duration panels alive across the model/resolution hydration.
    const referenceClipSeven = {
      ...clipZero,
      director_clip_index: 6,
      params: {
        ...clipZero.params,
        prompt: 'Reference clip seven exact prompt.',
        resolution: '1280x704',
        sliding_window_size: 345,
        sliding_window_memory_override: false,
        minimax_h3_reference_sequence: true,
        minimax_h3_sequence_clip_frames: 243,
        minimax_h3_sequence_memory_override: false,
      },
    };
    await page.evaluate(({metadata, media}) => window.store.setState({
      selectedOutputMeta: metadata, selectedOutput: 0, outputs: [media],
    }), {metadata: referenceClipSeven, media: output('reference-clip-7.mp4', archiveWorkspace)});
    await page.evaluate(() => window.store.getState().loadSettingsFromOutput());
    await page.waitForFunction(() => window.store.getState().params.model_type === 'minimax_h3_ref2va_clip_restore_test'
      && window.store.getState().slidingWindowSeconds === 345 / 24
      && window.store.getState().slidingWindowLocked);
    await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    assert.equal(await page.evaluate(() => window.store.getState().params.minimax_h3_reference_sequence), false);
    assert.equal(await page.evaluate(() => window.store.getState().params.minimax_h3_sequence_clip_frames), 345);
    assert.equal(await page.evaluate(() => window.store.getState().params.minimax_h3_sequence_memory_override), true);
    assert.equal(await page.evaluate(() => window.store.getState().params._duration_planning_mode), 'duration');
    assert.deepEqual(await page.evaluate(() => window.store.getState().h3WindowOverrides), {},
      'restoring a clip does not save a per-model hardware override');
    assert.equal((await page.locator('#duration-slider-test').innerText()).match(/1 window · 14\.4s final output/g)?.length, 2,
      'both mounted References duration panels retain the original single window');
    await page.evaluate(() => window.store.getState().setParam('override_attention', 'sla'));
    await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    assert.equal(await page.evaluate(() => window.store.getState().slidingWindowSeconds), 345 / 24);
    assert.equal(await page.evaluate(() => window.store.getState().durationSeconds), 345 / 24);
    await page.evaluate(() => window.store.getState().startGeneration('queue'));
    assert.equal(generationRequests.length, 2);
    const referenceRequest = generationRequests[1];
    assert.equal(referenceRequest.prompt, 'Reference clip seven exact prompt.');
    assert.equal(referenceRequest.seed, 8302);
    assert.equal(referenceRequest.override_attention, 'sla');
    assert.equal(referenceRequest.video_length, 345);
    assert.equal(referenceRequest.sliding_window_size, 345);
    assert.equal(referenceRequest.minimax_h3_reference_sequence, false,
      'the matching SLA request remains a single native References window');
    assert.equal(referenceRequest.restoredWindowSelectionKey, undefined,
      'transient UI restore ownership is not a generation setting');
    await page.evaluate(() => window.unmountDurationSlider());

    const autoModelOptions = await page.evaluate(async modelType => {
      const response = await fetch(`/api/v1/model-options/${modelType}`);
      return response.json();
    }, framesModel);
    await page.evaluate(({modelType, modelOptions}) => {
      const store = window.store;
      const state = store.getState();
      store.setState({
        modelOptions,
        restoredWindowSelectionKey: null,
        systemStats: {gpu: {vram_total_gb: 16}, ram: {total_gb: 64}},
        durationSeconds: 14.375,
        slidingWindowSeconds: 345 / 24,
        slidingWindowLocked: false,
        params: {
          ...state.params,
          model_type: modelType,
          resolution: '864x480',
          prompt: 'Create a 14.375 second scene.',
          _duration_planning_mode: 'auto',
          video_length: 345,
          duration_seconds: 14.375,
          sliding_window_size: 345,
          sliding_window_memory_override: false,
          minimax_h3_multi_window: false,
          minimax_h3_reference_sequence: false,
          override_attention: 'sla',
        },
      });
    }, {modelType: framesModel, modelOptions: autoModelOptions});
    await page.evaluate(() => window.mountDurationSlider());
    await page.waitForFunction(() => window.store.getState().slidingWindowSeconds === 243 / 24
      && window.store.getState().params.minimax_h3_multi_window === true);
    assert.equal(await page.evaluate(() => window.store.getState().slidingWindowLocked), false,
      'a user-created Auto duration remains unlocked');
    assert.match(await page.locator('#duration-slider-test').innerText(), /2 windows of up to 10\.1s/,
      'ordinary Auto still follows the 243-frame hardware recommendation');
    await page.evaluate(() => window.unmountDurationSlider());

    const beforeError = await page.evaluate(() => {
      const state = window.store.getState();
      return {sidebarMode: state.sidebarMode, params: state.params, durationSeconds: state.durationSeconds,
        startImage: state.startImage?.name, enhanceOnGeneration: state.enhanceOnGeneration};
    });
    const errorMetadata = {
      source: 'sidecar',
      director_pipeline_id: 'saved-director-project',
      director_artifact_kind: 'clip',
      director_clip_index: 7,
      director_clip_settings_error: 'The archived soundtrack slice is missing.',
      params: null,
    };
    await seed(errorMetadata, output('unsafe-clip.mp4', archiveWorkspace), framesModel,
      {enhanceOnGeneration: false, enhanceOnGenerationDefault: true});
    const beforeUnsafe = await page.evaluate(() => {
      const state = window.store.getState();
      return {sidebarMode: state.sidebarMode, params: state.params, durationSeconds: state.durationSeconds,
        startImage: state.startImage?.name, enhanceOnGeneration: state.enhanceOnGeneration};
    });
    await page.evaluate(() => window.store.getState().loadSettingsFromOutput());
    const afterUnsafe = await page.evaluate(() => {
      const state = window.store.getState();
      return {sidebarMode: state.sidebarMode, params: state.params, durationSeconds: state.durationSeconds,
        startImage: state.startImage?.name, enhanceOnGeneration: state.enhanceOnGeneration};
    });
    assert.deepEqual(afterUnsafe, beforeUnsafe, 'an unsafe recipe produces no Studio or Director mutation');
    const errorAlert = await page.evaluate(() => window.alerts.at(-1));
    assert.equal(errorAlert.category, 'failure');
    assert.equal(errorAlert.body, 'The archived soundtrack slice is missing.');

    const staleMultiClip = clipMetadata({
      index: 2,
      modelType: framesModel,
      prompt: 'Whole Director prompt\n---CLIP_BOUNDARY---\nAnother prompt',
      params: {multi_prompts_gen_type: 3, image_start: ['/a.png', '/b.png'], per_clip_frames: [24, 24]},
    });
    await seed(staleMultiClip, output('stale-clip.mp4', archiveWorkspace), framesModel,
      {enhanceOnGeneration: false, enhanceOnGenerationDefault: true});
    const beforeStale = await page.evaluate(() => ({
      sidebarMode: window.store.getState().sidebarMode,
      prompt: window.store.getState().params.prompt,
      generationMode: window.store.getState().generationMode,
    }));
    await page.evaluate(() => window.store.getState().loadSettingsFromOutput());
    assert.deepEqual(await page.evaluate(() => ({
      sidebarMode: window.store.getState().sidebarMode,
      prompt: window.store.getState().params.prompt,
      generationMode: window.store.getState().generationMode,
    })), beforeStale, 'a whole multi-clip recipe is refused without mutating the active Studio');
    assert.match(await page.evaluate(() => window.alerts.at(-1).body), /multi-clip recipe/);

    await seed({
      source: 'sidecar',
      director_pipeline_id: 'saved-director-project',
      director_artifact_kind: 'joined',
      params: {model_type: framesModel, prompt: 'joined output'},
    }, output('joined-final.mp4', archiveWorkspace), framesModel);
    await page.evaluate(() => window.store.getState().loadSettingsFromOutput());
    assert.deepEqual(await page.evaluate(() => window.directorLoads), ['saved-director-project'],
      'the joined final still opens its full Director project');
    assert.equal(await page.evaluate(() => window.store.getState().params.prompt), 'previous Studio prompt',
      'the joined final is not flattened into Studio settings');

    const ordinaryStudioMulti = {
      source: 'sidecar',
      // launch.py currently writes this incidental field for ordinary Studio
      // multi-clip outputs too; without a Director id it is not a Director clip.
      director_clip_index: 7,
      params: {
        model_type: framesModel,
        prompt: 'Ordinary first\nOrdinary second',
        seed: 77,
        video_length: 48,
        duration_seconds: 2,
        image_mode: 0,
        multi_prompts_gen_type: 3,
        image_start: ['/uploads/first.png', '/uploads/second.png'],
        per_clip_frames: [24, 24],
        activated_loras: [],
        custom_settings: {},
      },
    };
    await seed(ordinaryStudioMulti, output('ordinary-multiclip.mp4', archiveWorkspace), framesModel,
      {enhanceOnGeneration: false, enhanceOnGenerationDefault: true});
    await page.evaluate(() => window.store.getState().loadSettingsFromOutput());
    assert.equal(await page.evaluate(() => window.store.getState().sidebarMode), 'studio');
    assert.equal(await page.evaluate(() => window.store.getState().params.multi_prompts_gen_type), 3,
      'ordinary Studio multi-clip metadata keeps its existing restore semantics');
    assert.equal(await page.evaluate(() => window.store.getState().clips.length), 2);
    assert.deepEqual(await page.evaluate(() => window.directorLoads), ['saved-director-project']);

    assert.deepEqual(pageErrors, [], `Browser errors: ${pageErrors.join('; ')}`);
    console.log('Director clip Load Settings: indices 0/7, exact recipe and prompt, sticky enhancement override, archive asset URLs, visible safety errors, joined routing, and ordinary Studio multiclip compatibility passed');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
