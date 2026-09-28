// Real useStore enhancement/submission regression with all network calls stubbed.
// Run: node tests/ui/h3_frames_timing.cjs
const assert = require('node:assert/strict');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const playwright = require(process.env.MAESTRO_PLAYWRIGHT ||
  'C:/Users/bliza/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

const modelType = 'minimax_h3_fused_turbo';
const modelOptions = {
  model_type: modelType, architecture: modelType, family: 'minimax_h3', fps: 24,
  frames_minimum: 124, frames_maximum: 345, frames_steps: 17,
  sliding_window: true, sliding_window_auto_prompt_pacing: true,
  sliding_window_defaults: {window_min: 124, window_max: 345, window_step: 17,
    window_default: 124, overlap_min: 1, overlap_max: 103, overlap_step: 17,
    overlap_offset: 1, overlap_default: 18, discard_last_frames: 0},
  resolution_preset_order: ['480p', '720p'], supports_auto_aspect: true,
  resolution_presets: {'480p': {values: {'16:9': '864x480'}}},
};

function plannedWindows(total, window, overlap, discard) {
  const stride = Math.max(1, window - overlap - discard);
  const count = total <= window ? 1 : 1 + Math.ceil((total - window + discard) / stride);
  const windows = [];
  let start = 0;
  let end = Math.min(total, window - (count > 1 ? discard : 0));
  windows.push([start, end]);
  while (end < total) {
    start = end;
    end = Math.min(total, start + stride);
    windows.push([start, end]);
  }
  return windows;
}

(async () => {
  const bundle = await esbuild.build({stdin: {contents: `
    import {useStore} from './src/stores/useStore';
    window.store = useStore;
  `, resolveDir: path.join(root, 'ui'), loader: 'tsx'}, bundle: true, write: false,
    jsx: 'automatic', tsconfigRaw: {compilerOptions: {jsx: 'react-jsx'}},
    define: {'process.env.NODE_ENV': '"development"'}, logLevel: 'silent'});
  const browser = await playwright.chromium.launch({headless: true,
    ...(process.platform === 'win32' ? {executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'} : {})});
  try {
    const page = await browser.newPage();
    const requests = [], plans = [];
    await page.route('**/*', async route => {
      const url = new URL(route.request().url());
      const json = body => route.fulfill({contentType: 'application/json', body: JSON.stringify(body)});
      if (url.pathname === '/api/v1/jobs') return json({jobs: []});
      if (url.pathname === `/api/v1/model-options/${modelType}`) return json(modelOptions);
      if (url.pathname === '/api/v1/llm/plan-h3-windows') {
        const body = route.request().postDataJSON();
        plans.push(body);
        const spans = plannedWindows(body.total_frames, body.window_frames, body.overlap_frames, body.discard_frames);
        const windows = spans.map(([start_frame, end_frame], index) => ({index: index + 1,
          title: `Window ${index + 1}`, start_frame, end_frame, start_seconds: start_frame / 24,
          end_seconds: end_frame / 24, prompt: `Planned scene ${index + 1}.`}));
        return json({...body, source_prompt: body.prompt, plan_kind: 'sliding_window',
          signature: `plan-${plans.length}`, planned_by: 'llm', effective_window_frames: body.window_frames,
          total_frames: body.total_frames, overlap_frames: body.overlap_frames,
          discard_frames: body.discard_frames, window_count: windows.length, windows,
          window_prompts: windows.map(window => window.prompt)});
      }
      if (url.pathname === '/api/v1/generate') {
        requests.push(route.request().postDataJSON());
        return json({job_id: String(requests.length), status: 'held'});
      }
      if (url.pathname === '/') return route.fulfill({contentType: 'text/html', body: '<div></div>'});
      if (url.pathname.startsWith('/api/')) return json({jobs: [], outputs: [], items: [], configured: true});
      return route.fulfill({status: 404, body: ''});
    });
    await page.goto('http://h3-timing.test');
    await page.addScriptTag({content: bundle.outputFiles[0].text});
    await page.evaluate(({modelType, modelOptions}) => {
      const store = window.store;
      const s = store.getState();
      store.setState({
        models: [{model_type: modelType, architecture: modelType, family: 'minimax_h3', name: 'H3 Turbo', is_t2v: true, is_i2v: true}],
        enabledModels: new Set([modelType]), selectedModelPerMode: {video: modelType},
        modelOptions, generationMode: 'video', sidebarMode: 'studio', sidebarOpen: true,
        studioVideoWorkflow: 'frames', studioImageWorkflow: 'generate',
        studioVideoEffectiveCreateRoute: 'generate', durationSeconds: 14,
        resolutionPreset: '480p', aspectRatio: '16:9',
        slidingWindowSeconds: 124 / 24, slidingWindowOverlap: 17, slidingWindowLocked: false,
        systemStats: {gpu: {vram_total_gb: 24}}, startImage: null, endImage: null,
        imageRefs: [], h3WindowPlan: null, jobs: [], isGenerating: false, isEnhancing: false,
        promptEnhanceError: null, activeWorkspace: 'default',
        params: {...s.params, model_type: modelType, resolution: '864x480', prompt: 'A short story told in three scenes.',
          image_mode: 0, image_start: undefined, image_end: undefined, video_length: 336,
          sliding_window_size: 124, sliding_window_overlap: 17, sliding_window_discard_last_frames: 0,
          minimax_h3_multi_window: true, minimax_h3_window_storyboard: true, custom_settings: {},
          _duration_planning_mode: 'duration'},
      });
    }, {modelType, modelOptions});

    await page.evaluate(() => window.store.getState().enhancePrompt());
    assert.deepEqual(plans.map(plan => ({total: plan.total_frames, window: plan.window_frames,
      overlap: plan.overlap_frames, discard: plan.discard_frames, multi: plan.minimax_h3_multi_window})),
    [{total: 336, window: 124, overlap: 18, discard: 0, multi: true}]);
    assert.deepEqual(await page.evaluate(() => {
      const s = window.store.getState();
      return {frames: s.params.video_length, duration: s.durationSeconds,
        overlap: s.params.sliding_window_overlap,
        spans: s.h3WindowPlan.windows.map(window => [window.start_frame, window.end_frame])};
    }), {frames: 336, duration: 14, overlap: 18, spans: [[0, 124], [124, 230], [230, 336]]});

    await page.evaluate(() => window.store.getState().updateH3WindowPrompt(0, 'The reviewed opening stays exact.'));
    await page.evaluate(model => window.store.getState().loadModelOptions(model), modelType);
    assert.deepEqual(await page.evaluate(() => {
      const s = window.store.getState();
      return {duration: s.durationSeconds, frames: s.params.video_length,
        overlap: s.slidingWindowOverlap, firstPrompt: s.h3WindowPlan.windows[0].prompt};
    }), {duration: 14, frames: 336, overlap: 18, firstPrompt: 'The reviewed opening stays exact.'});

    await page.evaluate(() => window.store.getState().startGeneration('queue'));
    assert.equal(requests.length, 1, await page.evaluate(() => JSON.stringify({
      error: window.store.getState().promptEnhanceError,
      model: window.store.getState().params.model_type,
      route: window.store.getState().studioVideoEffectiveCreateRoute,
    })));
    assert.deepEqual({total: requests[0].video_length, window: requests[0].sliding_window_size,
      overlap: requests[0].sliding_window_overlap, discard: requests[0].sliding_window_discard_last_frames,
      count: requests[0].h3_window_prompts.length, reviewed: requests[0]._h3_window_plan_reviewed,
      firstPrompt: requests[0].h3_window_prompts[0]},
    {total: 336, window: 124, overlap: 18, discard: 0, count: 3, reviewed: true,
      firstPrompt: 'The reviewed opening stays exact.'});

    // A serialized plan with a different boundary but unchanged count is stale.
    await page.evaluate(() => {
      const s = window.store.getState();
      window.store.setState({h3WindowPlan: {...s.h3WindowPlan, windows: s.h3WindowPlan.windows.map((window, index) => (
        index === 0 ? {...window, end_frame: window.end_frame + 1} : window
      ))}});
    });
    await page.evaluate(() => window.store.getState().startGeneration('queue'));
    assert.equal(requests.length, 1, 'Stale reviewed timing does not submit');
    assert.match(await page.evaluate(() => window.store.getState().promptEnhanceError), /timing.*Press Enhance/);

    // A true rolling multi-pass duration one frame above the native max must
    // bypass both the max+1 clamp and lattice repair.
    await page.evaluate(() => {
      const s = window.store.getState();
      window.store.setState({jobs: [], isGenerating: false, h3WindowPlan: null,
        durationSeconds: 346 / 24, slidingWindowSeconds: 124 / 24, slidingWindowOverlap: 18,
        params: {...s.params, prompt: 'Window one.\nWindow two.\nWindow three.\nWindow four.', video_length: 346,
          sliding_window_size: 124, sliding_window_overlap: 18, minimax_h3_window_storyboard: false}});
    });
    await page.evaluate(() => window.store.getState().startGeneration('queue'));
    assert.equal(requests.length, 2);
    assert.equal(requests[1].video_length, 346);

    // A true checkbox with only one selected pass follows existing native
    // rounding, so 336 becomes 345 when the chosen window is 345.
    await page.evaluate(() => {
      const s = window.store.getState();
      window.store.setState({jobs: [], isGenerating: false, h3WindowPlan: null,
        durationSeconds: 14, slidingWindowSeconds: 345 / 24, slidingWindowOverlap: 18,
        params: {...s.params, prompt: 'One pass.', video_length: 336, sliding_window_size: 345,
          sliding_window_overlap: 18, minimax_h3_multi_window: true, minimax_h3_window_storyboard: true}});
    });
    await page.evaluate(() => window.store.getState().startGeneration('queue'));
    assert.equal(requests.length, 3);
    assert.equal(requests[2].video_length, 345);
    console.log('H3 Frames store flow: enhancement, normalized timing, refresh, reviewed submit, stale boundaries, 346-frame rolling and one-pass rounding passed');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
