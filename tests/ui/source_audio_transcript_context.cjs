// Run with: node tests/ui/source_audio_transcript_context.cjs
// Focused opt-in audio-context UI and Studio request regression.
const assert = require('node:assert/strict');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const playwright = require(process.env.MAESTRO_PLAYWRIGHT || path.join(root, 'ui/node_modules/playwright'));

(async () => {
  const bundle = await esbuild.build({stdin: {contents: `
    import React from 'react'; import {createRoot} from 'react-dom/client';
    import {useStore, hasStudioSourceAudioTranscriptDriver} from './src/stores/useStore';
    import {SourceAudioTranscriptControls} from './src/components/Sidebar/SourceAudioTranscriptControls';
    window.store = useStore; window.hasTranscriptDriver = hasStudioSourceAudioTranscriptDriver;
    const root = createRoot(document.getElementById('root'));
    window.mount = () => root.render(<SourceAudioTranscriptControls/>);
  `, resolveDir: path.join(root, 'ui'), loader: 'tsx'}, bundle: true, write: false,
    jsx: 'automatic', tsconfigRaw: {compilerOptions: {jsx: 'react-jsx'}},
    define: {'process.env.NODE_ENV': '"development"'}, logLevel: 'silent'});
  const browser = await playwright.chromium.launch({headless: true,
    ...(process.env.MAESTRO_CHROME ? {executablePath: process.env.MAESTRO_CHROME} : {})});
  try {
    const page = await browser.newPage();
    page.setDefaultTimeout(7000);
    const enhancementRequests = [], generationRequests = [], windowPlanRequests = [], sequencePlanRequests = [];
    await page.route('**/*', async route => {
      const url = new URL(route.request().url());
      const json = body => route.fulfill({contentType: 'application/json', body: JSON.stringify(body)});
      if (url.pathname === '/api/v1/llm/enhance-prompt') {
        enhancementRequests.push(route.request().postDataJSON());
        return json({original: 'A singer performs.', enhanced: 'A singer performs in close-up.'});
      }
      if (url.pathname === '/api/v1/llm/plan-h3-windows') {
        const body = route.request().postDataJSON();
        windowPlanRequests.push(body);
        const windows = [
          {index: 1, title: 'Opening', start_frame: 0, end_frame: body.window_frames,
            start_seconds: 0, end_seconds: body.window_frames / 24, prompt: 'The singer opens.'},
          {index: 2, title: 'Ending', start_frame: body.window_frames, end_frame: body.total_frames,
            start_seconds: body.window_frames / 24, end_seconds: body.total_frames / 24, prompt: 'The singer finishes.'},
        ];
        return json({...body, source_prompt: body.prompt, plan_kind: 'sliding_window', signature: 'h3-window-plan',
          planned_by: 'llm', effective_window_frames: body.window_frames, total_frames: body.total_frames,
          overlap_frames: body.overlap_frames, discard_frames: body.discard_frames,
          window_count: windows.length, windows, window_prompts: windows.map(window => window.prompt)});
      }
      if (url.pathname === '/api/v1/llm/plan-h3-sequence') {
        const body = route.request().postDataJSON();
        sequencePlanRequests.push(body);
        const windowFrames = body.sequence_clip_frames || 124;
        const windows = [
          {index: 1, title: 'Opening', start_frame: 0, end_frame: windowFrames,
            start_seconds: 0, end_seconds: windowFrames / 24, prompt: 'The subject enters.'},
          {index: 2, title: 'Ending', start_frame: windowFrames, end_frame: body.total_frames,
            start_seconds: windowFrames / 24, end_seconds: body.total_frames / 24, prompt: 'The subject exits.'},
        ];
        return json({...body, source_prompt: body.prompt, plan_kind: 'reference_sequence', signature: 'h3-sequence-plan',
          planned_by: 'llm', window_frames: windowFrames, effective_window_frames: windowFrames,
          total_frames: body.total_frames, window_count: windows.length,
          windows, window_prompts: windows.map(window => window.prompt)});
      }
      if (url.pathname === '/api/v1/generate') {
        generationRequests.push(route.request().postDataJSON());
        return json({job_id: `job-${generationRequests.length}`, status: 'queued'});
      }
      if (url.pathname === '/') return route.fulfill({contentType: 'text/html', body: '<div id="root"></div>'});
      if (url.pathname.startsWith('/api/')) return json({jobs: [], outputs: [], items: [], configured: true});
      return route.fulfill({status: 404, body: ''});
    });
    await page.goto('http://source-transcript.test');
    await page.addScriptTag({content: bundle.outputFiles[0].text});

    const state = () => page.evaluate(() => window.store.getState());
    await page.evaluate(() => {
      const store = window.store, s = store.getState();
      store.setState({generationMode: 'video', studioVideoWorkflow: 'frames',
        params: {...s.params, image_mode: 0, audio_guide: '/uploads/source.wav', audio_prompt_type: 'K',
          source_audio_transcript_in_prompt: false, source_audio_transcript_for_enhancement: false}});
      window.mount();
    });
    assert.equal(await page.getByRole('region', {name: 'Source audio transcript context'}).count(), 0,
      'Frames audio guide in control-video mode is not an exact soundtrack driver');
    assert.equal(await page.evaluate(() => window.hasTranscriptDriver({generationMode: 'video', studioVideoWorkflow: 'frames',
      params: {image_mode: 0, audio_guide: '/uploads/source.wav', audio_prompt_type: 'A'}})), true);
    for (const audioPromptType of ['AD', 'AK']) {
      assert.equal(await page.evaluate(audioPromptType => window.hasTranscriptDriver({generationMode: 'video',
        studioVideoWorkflow: 'frames', params: {image_mode: 0, audio_guide: '/uploads/source.wav', audio_prompt_type: audioPromptType}}), audioPromptType), true,
      `Frames mode ${audioPromptType} keeps the uploaded soundtrack as a transcript driver`);
    }
    assert.equal(await page.evaluate(() => window.hasTranscriptDriver({generationMode: 'video', studioVideoWorkflow: 'frames',
      params: {image_mode: 0, audio_guide: '/uploads/source.wav', audio_prompt_type: 'R'}})), false,
    'remux-only mode does not activate transcript context');
    await page.evaluate(() => window.store.setState(s => ({params: {...s.params, audio_prompt_type: 'AN'}})));
    const panel = page.getByRole('region', {name: 'Source audio transcript context'});
    await panel.waitFor();
    await page.getByRole('button', {name: 'Audio context'}).click();
    const inPrompts = page.getByRole('switch', {name: 'Include transcript in video prompts'});
    const forEnhancement = page.getByRole('switch', {name: 'Use transcript for prompt enhancement'});
    assert.equal(await inPrompts.isChecked(), false, 'the prompt-context opt-in starts off');
    assert.equal(await forEnhancement.isChecked(), false, 'the enhancement opt-in starts off');
    await page.evaluate(() => window.store.setState({isEnhancing: true}));
    assert.equal(await inPrompts.isDisabled(), true, 'transcript preferences stay stable during enhancement');
    assert.equal(await forEnhancement.isDisabled(), true, 'transcript preferences stay stable during enhancement');
    await page.evaluate(() => window.store.setState({isEnhancing: false}));
    await inPrompts.check();
    await forEnhancement.check();
    assert.equal((await state()).params.source_audio_transcript_in_prompt, true);
    assert.equal((await state()).params.source_audio_transcript_for_enhancement, true);
    assert.equal((await state()).savedParamsPerMode.video.source_audio_transcript_in_prompt, true,
      'the prompt-context opt-in is included in the saved video params');
    assert.equal((await state()).savedParamsPerMode.video.source_audio_transcript_for_enhancement, true,
      'the enhancement opt-in is included in the saved video params');
    await page.getByText(/Each window gets the matching source words or lyrics/).waitFor();
    await page.getByText(/Uses local Whisper small \(downloaded if needed\)/).waitFor();
    await page.getByRole('button', {name: /Audio context/}).click();
    assert.equal(await page.getByText('On', {exact: true}).isVisible(), true,
      'the collapsed panel shows when either transcript option is active');

    await page.evaluate(() => {
      const s = window.store.getState();
      window.store.setState({studioVideoWorkflow: 'references', params: {...s.params, image_mode: 0,
        minimax_h3_references: [{id: 'voice', type: 'audio', path: '/uploads/voice.wav', audio_intent: 'voice'}]}});
    });
    assert.equal(await page.getByRole('region', {name: 'Source audio transcript context'}).count(), 0,
      'voice references do not activate the panel');
    await page.evaluate(() => {
      const s = window.store.getState();
      window.store.setState({params: {...s.params, minimax_h3_references: [
        {id: 'style', type: 'audio', path: '/uploads/style.wav', audio_intent: 'style'},
      ]}});
    });
    assert.equal(await page.getByRole('region', {name: 'Source audio transcript context'}).count(), 0,
      'music and sound-style references do not activate the panel');
    await page.evaluate(() => {
      const s = window.store.getState();
      window.store.setState({params: {...s.params, minimax_h3_references: [
        {id: 'video-style', type: 'video', path: '/uploads/style.mp4', video_audio_role: 'style'},
      ]}});
    });
    assert.equal(await page.getByRole('region', {name: 'Source audio transcript context'}).count(), 0,
      'video style references do not activate the panel');
    await page.evaluate(() => {
      const s = window.store.getState();
      window.store.setState({params: {...s.params, minimax_h3_references: [
        {id: 'video-drive', type: 'video', path: '/uploads/drive.mp4', video_audio_role: 'drive'},
      ]}});
    });
    await page.getByRole('region', {name: 'Source audio transcript context'}).waitFor();

    await page.evaluate(() => {
      const s = window.store.getState();
      window.store.setState({h3WindowPlan: {plan_kind: 'sliding_window'}, params: {
        ...s.params, prompt: 'Window one.\nWindow two.', _ltx_original_prompt: 'The source idea.',
        _h3_original_prompt: 'The H3 source idea.',
        ltx_window_prompts: ['Window one.', 'Window two.'],
        _prompt_enhancement: {version: 1, state: 'complete', original_prompt: 'The source idea.',
          enhanced_prompt: 'Window one.\nWindow two.'},
      }});
      s.setParam('source_audio_transcript_for_enhancement', false);
    });
    assert.equal((await state()).h3WindowPlan, null, 'changing transcript context invalidates reviewed H3 windows');
    assert.equal((await state()).params.prompt, 'The source idea.', 'invalidating LTX windows restores the original source prompt');
    assert.equal((await state()).params._ltx_original_prompt, 'The source idea.', 'the LTX source prompt remains available');
    assert.equal((await state()).params._h3_original_prompt, 'The H3 source idea.', 'the H3 source prompt remains available');
    assert.equal((await state()).params.ltx_window_prompts, undefined, 'stale LTX window prompts are removed');

    await page.evaluate(() => {
      const s = window.store.getState();
      window.store.setState({studioVideoWorkflow: 'frames', h3WindowPlan: {plan_kind: 'sliding_window'}, params: {
        ...s.params, prompt: 'Old first window.\nOld second window.', _ltx_original_prompt: 'The source idea.',
        ltx_window_prompts: ['Old first window.', 'Old second window.'], audio_frame_offset: 240,
        source_audio_transcript_for_enhancement: true,
      }});
      window.store.getState().setParam('audio_guide', '/uploads/replacement.wav');
    });
    assert.equal((await state()).h3WindowPlan, null, 'a replacement source invalidates the old transcript plan');
    assert.equal((await state()).params.audio_frame_offset, 0, 'a replacement source starts at its own beginning');
    assert.equal((await state()).params.prompt, 'The source idea.', 'replacing source audio retains the original idea');
    assert.equal((await state()).params.ltx_window_prompts, undefined);

    const modelType = 'ltx2_22B_distilled_1_1';
    await page.evaluate(modelType => {
      const store = window.store, s = store.getState();
      const model = {model_type: modelType, name: 'LTX', family: 'ltx2', architecture: 'ltx2',
        is_i2v: true, is_t2v: true, supports_audio_input: true};
      const modelOptions = {model_type: modelType, family: 'ltx2', architecture: 'ltx2', fps: 25,
        multi_window_sequence_controls: false, sliding_window: false};
      store.setState({generationMode: 'video', studioVideoWorkflow: 'frames',
        models: [model], enabledModels: new Set([modelType]), selectedModelPerMode: {video: modelType},
        modelVisibilityOverrides: {'video:frames': {[modelType]: true}},
        modelOptions, studioVideoEffectiveCreateRoute: 'generate', activeWorkspace: 'test',
        durationSeconds: 4, slidingWindowSeconds: 4, slidingWindowOverlap: 0,
        startImage: null, endImage: null, imageRefs: [], h3WindowPlan: null,
        jobs: [], isGenerating: false, isEnhancing: false, promptEnhanceError: null,
        params: {...s.params, model_type: modelType, prompt: 'A singer performs.', resolution: '1280x720',
          image_mode: 0, video_length: 100, audio_guide: '/uploads/source.wav', audio_prompt_type: 'A',
          audio_frame_offset: 24,
          source_audio_transcript_in_prompt: true, source_audio_transcript_for_enhancement: true,
          ltx_multi_window: false, activated_loras: [], loras_multipliers: ''},
      });
    }, modelType);
    await page.evaluate(() => window.store.getState().enhancePrompt());
    assert.equal(enhancementRequests.length, 1, 'manual Enhance reaches the ordinary prompt endpoint');
    assert.equal(enhancementRequests[0].audio_guide, '/uploads/source.wav');
    assert.equal(enhancementRequests[0].audio_frame_offset, 24);
    assert.equal(enhancementRequests[0].audio_prompt_type, 'A');
    assert.equal(enhancementRequests[0].window_overlap_seconds, 0);
    assert.equal(enhancementRequests[0].window_discard_seconds, 0);
    assert.equal(enhancementRequests[0].source_audio_transcript_in_prompt, true);
    assert.equal(enhancementRequests[0].source_audio_transcript_for_enhancement, true);

    await page.evaluate(() => {
      window.store.setState({enhanceOnGeneration: true});
      return window.store.getState().startGeneration('queue');
    });
    assert.equal(generationRequests.length, 1, `deferred prompt enhancement still submits through generation: ${await page.evaluate(() => JSON.stringify({
      mode: window.store.getState().generationMode,
      model: window.store.getState().params.model_type,
      error: window.store.getState().promptEnhanceError,
      route: window.store.getState().studioVideoEffectiveCreateRoute,
      generating: window.store.getState().isGenerating,
    }))}`);
    assert.equal(generationRequests[0]._enhance_on_generation, true);
    assert.equal(generationRequests[0].source_audio_transcript_in_prompt, true);
    assert.equal(generationRequests[0].source_audio_transcript_for_enhancement, true);
    assert.equal(generationRequests[0].audio_guide, '/uploads/source.wav');

    const h3FramesType = 'minimax_h3_fused_turbo';
    const h3FramesOptions = {model_type: h3FramesType, architecture: 'minimax_h3', fps: 24,
      frames_minimum: 124, frames_maximum: 345, frames_steps: 17, sliding_window: true,
      sliding_window_auto_prompt_pacing: true,
      sliding_window_defaults: {window_min: 124, window_max: 345, window_step: 17,
        window_default: 124, overlap_min: 1, overlap_max: 103, overlap_step: 17,
        overlap_offset: 1, overlap_default: 18, discard_last_frames: 0}};
    await page.evaluate(({modelType, modelOptions}) => {
      const store = window.store, s = store.getState();
      const model = {model_type: modelType, name: 'H3 Frames', family: 'minimax_h3', architecture: 'minimax_h3',
        is_i2v: true, is_t2v: true, supports_audio_input: true};
      store.setState({generationMode: 'video', studioVideoWorkflow: 'frames', models: [model],
        enabledModels: new Set([modelType]), modelVisibilityOverrides: {'video:frames': {[modelType]: true}},
        selectedModelPerMode: {video: modelType}, modelOptions, studioVideoEffectiveCreateRoute: 'generate',
        durationSeconds: 20, slidingWindowSeconds: 124 / 24, slidingWindowOverlap: 18,
        startImage: null, endImage: null, imageRefs: [], h3WindowPlan: null,
        jobs: [], isGenerating: false, isEnhancing: false, promptEnhanceError: null,
        params: {...s.params, model_type: modelType, prompt: 'A singer performs through the song.',
          resolution: '864x480', image_mode: 0, video_length: 480, sliding_window_size: 124,
          sliding_window_overlap: 18, minimax_h3_multi_window: true, minimax_h3_window_storyboard: true,
          _duration_planning_mode: 'duration', audio_guide: '/uploads/source.wav', audio_prompt_type: 'A',
          audio_frame_offset: 48, source_audio_transcript_in_prompt: true,
          source_audio_transcript_for_enhancement: true, activated_loras: [], loras_multipliers: ''},
      });
    }, {modelType: h3FramesType, modelOptions: h3FramesOptions});
    await page.evaluate(() => window.store.getState().enhancePrompt());
    assert.equal(windowPlanRequests.length, 1, 'multi-window H3 Frames uses the structured planner');
    assert.equal(windowPlanRequests[0].audio_guide, '/uploads/source.wav');
    assert.equal(windowPlanRequests[0].audio_prompt_type, 'A');
    assert.equal(windowPlanRequests[0].audio_frame_offset, 48);
    assert.equal(windowPlanRequests[0].source_audio_transcript_in_prompt, true);
    assert.equal(windowPlanRequests[0].source_audio_transcript_for_enhancement, true);

    const h3ReferenceType = 'minimax_h3_ref2va_fused_turbo';
    const h3ReferenceOptions = {...h3FramesOptions, model_type: h3ReferenceType,
      architecture: 'minimax_h3_ref2va', omni_reference: true, sliding_window_auto_prompt_pacing: false};
    await page.evaluate(({modelType, modelOptions}) => {
      const store = window.store, s = store.getState();
      const model = {model_type: modelType, name: 'H3 References', family: 'minimax_h3',
        architecture: 'minimax_h3_ref2va', omni_reference: true, is_i2v: true, is_t2v: true};
      store.setState({generationMode: 'video', studioVideoWorkflow: 'references', models: [model],
        enabledModels: new Set([modelType]), modelVisibilityOverrides: {'video:references': {[modelType]: true}},
        selectedModelPerMode: {video: modelType}, modelOptions, studioVideoEffectiveCreateRoute: 'omni',
        durationSeconds: 20, slidingWindowSeconds: 124 / 24, slidingWindowOverlap: 18,
        startImage: null, endImage: null, imageRefs: [], h3WindowPlan: null,
        jobs: [], isGenerating: false, isEnhancing: false, promptEnhanceError: null,
        params: {...s.params, model_type: modelType, prompt: 'A performer follows the song.',
          resolution: '864x480', image_mode: 0, video_length: 480,
          minimax_h3_reference_sequence: true, minimax_h3_sequence_prompt_mode: 'adaptive',
          minimax_h3_references: [{id: 'driver', type: 'audio', path: '/uploads/driver.wav', audio_intent: 'drive'}],
          audio_frame_offset: 72, source_audio_transcript_in_prompt: true,
          source_audio_transcript_for_enhancement: true, activated_loras: [], loras_multipliers: ''},
      });
    }, {modelType: h3ReferenceType, modelOptions: h3ReferenceOptions});
    await page.evaluate(() => window.store.getState().enhancePrompt());
    assert.equal(sequencePlanRequests.length, 1, 'multi-window H3 References uses its sequence planner');
    assert.equal(sequencePlanRequests[0].references[0].path, '/uploads/driver.wav');
    assert.equal(sequencePlanRequests[0].audio_frame_offset, 72);
    assert.equal(sequencePlanRequests[0].source_audio_transcript_in_prompt, true);
    assert.equal(sequencePlanRequests[0].source_audio_transcript_for_enhancement, true);

    const imageType = 'flux2_klein_9b';
    await page.evaluate(imageType => {
      const store = window.store, s = store.getState();
      const model = {model_type: imageType, name: 'Flux', family: 'flux2', architecture: 'flux2', is_i2v: true, is_t2v: true};
      store.setState({generationMode: 'image', studioImageWorkflow: 'generate', models: [model],
        enabledModels: new Set([imageType]), modelVisibilityOverrides: {'image:generate': {[imageType]: true}},
        modelOptions: {model_type: imageType, family: 'flux2', architecture: 'flux2'},
        params: {...s.params, model_type: imageType, prompt: 'A garden.', image_mode: 1,
          source_audio_transcript_in_prompt: true, source_audio_transcript_for_enhancement: true},
      });
    }, imageType);
    await page.evaluate(() => window.store.getState().startGeneration('queue'));
    assert.equal(generationRequests.length, 2, 'unrelated image mode remains generatable');
    assert.equal(Object.hasOwn(generationRequests[1], 'source_audio_transcript_in_prompt'), false,
      'video transcript options never leak into image generation');
    assert.equal(Object.hasOwn(generationRequests[1], 'source_audio_transcript_for_enhancement'), false,
      'video enhancement options never leak into image generation');
    console.log('Source audio transcript context: driver gating, opt-ins, stale-plan invalidation, LLM/H3 request paths, and generation passed');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
