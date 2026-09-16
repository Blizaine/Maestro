// Real React + Zustand regression checks in an isolated browser (no app writes).
// Run: node tests/ui/studio_duration.cjs
const assert = require('node:assert/strict');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const playwright = require(process.env.MAESTRO_PLAYWRIGHT ||
  'C:/Users/bliza/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

(async () => {
  const bundle = await esbuild.build({
    stdin: { contents: `
      import React from 'react';
      import { createRoot } from 'react-dom/client';
      import { useStore } from './src/stores/useStore';
      import { DurationSlider, WindowSettings } from './src/components/Sidebar/DurationSlider';
      window.store = useStore;
      window.mount = () => {
        window.root = createRoot(document.getElementById('root'));
        window.root.render(<><DurationSlider/><WindowSettings/></>);
      };
    `, resolveDir: path.join(root, 'ui'), loader: 'tsx' },
    bundle: true, write: false, jsx: 'automatic', define: { 'process.env.NODE_ENV': '"development"' },
    logLevel: 'silent',
  });
  const browser = await playwright.chromium.launch({headless: true,
    ...(process.env.MAESTRO_CHROME || process.platform === 'win32'
      ? {executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'} : {})});
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.route('**/*', route => route.fulfill({contentType: 'text/html', body: '<div id="root"></div>'}));
    await page.goto('http://studio.test');
    await page.addScriptTag({content: bundle.outputFiles[0].text});
    await page.evaluate(() => {
      const options = {
        architecture: 'minimax_h3_ref2va', model_type: 'minimax_h3_ref2va', fps: 24,
        frames_minimum: 124, frames_maximum: 345, frames_steps: 17,
        sliding_window: true, omni_reference: true,
        sliding_window_defaults: { window_min: 124, window_max: 345, window_step: 17, window_default: 243, overlap_default: 18, discard_last_frames: 0 },
        omni_sequence_memory_policy: { resolution_bands: [{min_pixels: 0, vram_tiers: [{max_vram_gb: null, frames: 243}]}], reference_margin_steps: 1 },
      };
      window.store.setState({ modelOptions: options, generationMode: 'video', studioVideoWorkflow: 'extend',
        durationSeconds: 5, slidingWindowSeconds: 243 / 24, slidingWindowOverlap: 18,
        systemStats: {gpu: {vram_total_gb: 24}}, h3WindowOverrides: {},
        params: {...window.store.getState().params, model_type: 'minimax_h3_ref2va',
          resolution: '864x480', prompt: 'A calm 6-second shot.', image_mode: 3,
          _duration_planning_mode: 'auto', minimax_h3_reference_sequence: false},
      });
      window.updates = 0;
      window.durationHistory = [];
      window.store.subscribe(s => {
        window.durationHistory.push([s.durationSeconds, s.slidingWindowSeconds, s.params.minimax_h3_reference_sequence]);
        if (++window.updates > 200) throw new Error('Duration never converged');
      });
      window.mount();
    });
    await page.waitForTimeout(300);
    if (errors.length) console.error('Recent duration/window/sequence states:', await page.evaluate(() => window.durationHistory.slice(-20)));
    assert.deepEqual(errors, [], 'Restored Extend settings must render without a duration update loop');
    assert.ok(await page.locator('#root').innerText(), 'Studio controls remain mounted');
    const settled = await page.evaluate(() => window.updates);
    await page.waitForTimeout(100);
    assert.equal(await page.evaluate(() => window.updates), settled, 'Automatic duration reaches a stable state');
    console.log('Extend restored settings: React rendered and duration converged');
    await page.getByRole('button', {name: 'Time', exact: true}).click();
    await page.evaluate(() => {
      const s = window.store.getState();
      window.store.setState({studioVideoWorkflow: 'references', slidingWindowLocked: false,
        params: {...s.params, image_mode: 0, _duration_planning_mode: 'duration'}});
      window.store.getState().setDurationSeconds(124 / 24);
    });
    const slider = page.getByRole('slider', {name: 'Duration model duration'});
    await slider.focus();
    for (const frames of [141, 158, 175]) {
      await slider.press('ArrowRight');
      const s = await page.evaluate(() => {
        const s = window.store.getState();
        return {frames: s.params.video_length, window: s.params.sliding_window_size};
      });
      assert.deepEqual(s, {frames, window: frames}, 'Native duration grows the window one model step');
    }
    await slider.press('End');
    let state = await page.evaluate(() => {
      const s = window.store.getState();
      return {duration: s.durationSeconds, window: s.params.sliding_window_size, sequence: s.params.minimax_h3_reference_sequence};
    });
    assert.ok(state.duration <= 300 && state.duration > 299, 'Slider stops at the final native step within five minutes');
    assert.equal(state.window, 226, 'Automatic window retains GPU headroom instead of growing with a five-minute timeline');
    assert.equal(state.sequence, true);
    await page.getByRole('button', {name: '60m', exact: true}).click();
    assert.ok(await page.evaluate(() => window.store.getState().durationSeconds > 3590), 'Hour preset remains available');
    await page.evaluate(() => {
      const s = window.store.getState();
      s.setSlidingWindowLocked(true);
      s.setSlidingWindowSeconds(345 / 24);
      s.setDurationSeconds(345 / 24);
    });
    await page.waitForTimeout(100);
    assert.equal(await page.evaluate(() => window.store.getState().params.sliding_window_size), 345, 'User window override permits 14.4 seconds');
    assert.equal(await page.evaluate(() => window.store.getState().params.minimax_h3_reference_sequence), false);
    await page.evaluate(() => {
      const s = window.store.getState();
      s.setSlidingWindowLocked(false);
      s.setDurationSeconds(124 / 24);
    });
    await page.getByRole('button', {name: 'Window', exact: true}).click();
    await page.waitForTimeout(100);
    assert.equal(await page.evaluate(() => window.store.getState().params.video_length), 226, 'Entering Window mode uses stable GPU capacity');
    const beforeNoop = await page.evaluate(() => window.updates);
    await page.evaluate(() => {
      const s = window.store.getState();
      for (let i = 0; i < 20; i++) s.setDurationSeconds(s.durationSeconds);
    });
    assert.equal(await page.evaluate(() => window.updates), beforeNoop, 'Repeated canonical writes do not notify React');
    assert.deepEqual(errors, []);
    console.log('Time slider: native steps, GPU cap, manual 14.4s cap, 60m preset, Window mode and idempotence passed');
    await page.evaluate(() => {
      const s = window.store.getState();
      window.store.setState({studioVideoWorkflow: 'animate',
        modelOptions: {...s.modelOptions, model_type: 'viggle_animate', architecture: 'viggle_animate',
          omni_reference: false, frames_maximum: 124, sliding_window_memory_policy: null,
          sliding_window_defaults: {...s.modelOptions.sliding_window_defaults, window_max: 124}},
        params: {...s.params, model_type: 'viggle_animate', video_guide: '/uploads/control.mp4',
          _viggle_source_seconds: 9.584, _duration_planning_mode: 'auto'}});
    });
    await page.waitForTimeout(100);
    assert.equal(await page.evaluate(() => window.store.getState().params.video_length), 230);
    assert.ok((await page.locator('#root').innerText()).includes('2 windows'), 'Millisecond container rounding must not add a third window');
    assert.ok(!(await page.locator('#root').innerText()).includes('3 windows'));
    console.log('Viggle: source metadata rounds to two native windows, without a phantom extra pass');
    let submitted;
    const uploads = [];
    page.on('request', request => {if (request.url().includes('/upload')) uploads.push(request.url());});
    await page.route('**/api/v1/generate', route => {
      submitted = route.request().postDataJSON();
      return route.fulfill({status: 400, json: {detail: 'Submission captured by isolated test'}});
    });
    await page.evaluate(async () => {
      const s = window.store.getState();
      window.store.setState({startImage: new File(['stale'], 'old-start.png'),
        endImage: new File(['stale'], 'old-end.png'),
        imageRefs: [new File(['stale'], 'old-reference.png')],
        params: {...s.params, prompt: '', _viggle_edited_frame: '/uploads/edit.png',
          image_start: '/uploads/old-start.png', image_end: '/uploads/old-end.png',
          audio_prompt_type: 'K', num_inference_steps: 50, flow_shift: 12}});
      await window.store.getState().startGeneration();
    });
    assert.ok(submitted, 'Viggle can submit without a user text prompt');
    assert.equal(submitted.model_type, 'viggle_animate');
    assert.equal(submitted.video_length, 230);
    assert.equal(submitted.sliding_window_size, 124);
    assert.equal(submitted.sliding_window_overlap, 18);
    assert.equal(submitted.num_inference_steps, 3);
    assert.equal(submitted.flow_shift, 3);
    assert.equal(submitted.video_prompt_type, 'IVU');
    assert.deepEqual(submitted.image_refs, ['/uploads/edit.png']);
    assert.equal(submitted.image_start, undefined);
    assert.equal(submitted.image_end, undefined);
    assert.deepEqual(uploads, [], 'Hidden frame/reference inputs must not leak into Animate');
    console.log('Viggle submission: blank prompt accepted, fixed recipe, exact timeline, stale media excluded');
    await page.route('**/fresh-edit.png', route => route.fulfill({contentType:'image/png', body:Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jVZkAAAAASUVORK5CYII=', 'base64')}));
    await page.route('**/api/v1/upload', route => route.fulfill({json:{path:'/uploads/returned-edit.png', filename:'returned-edit.png'}}));
    const returned = await page.evaluate(async () => {
      window.root.unmount();
      const s = window.store.getState();
      window.store.setState({generationMode:'image', resolutionPreset:'auto',
        models:[{model_type:'viggle_animate', architecture:'viggle_animate', family:'minimax_h3', name:'Viggle'}],
        enabledModels:new Set(['viggle_animate']), selectedModelPerMode:{video:'viggle_animate'},
        savedParamsPerMode:{video:{_studio_video_workflow:'animate', _duration_planning_mode:'duration',
          video_guide:'/uploads/control.mp4', video_length:720, durationSeconds:30, resolution:'auto_480p'}},
        editReturnTarget:{anchor:'animate', savedResolutionPreset:'480p', previousImages:['old.png'],
          savedImageRefs:[], savedImageRefType:'', clipPath:'/uploads/control.mp4'},
        outputs:[{type:'image',name:'old.png',url:'http://studio.test/old.png'},
          {type:'image',name:'fresh-edit.png',url:'http://studio.test/fresh-edit.png'}],
        params:{...s.params, model_type:'flux2_klein_9b', image_mode:1}});
      await window.store.getState().applyOutputAsAnchor();
      const result = window.store.getState();
      return {mode:result.generationMode, model:result.params.model_type, source:result.params.video_guide,
        image:result.params._viggle_edited_frame, preset:result.resolutionPreset,
        duration:result.durationSeconds, planning:result.params._duration_planning_mode, target:result.editReturnTarget};
    });
    assert.deepEqual(returned, {mode:'video',model:'viggle_animate',source:'/uploads/control.mp4',
      image:'/uploads/returned-edit.png',preset:'480p',duration:30,planning:'duration',target:null});
    console.log('Viggle image return: fresh edit selected, source, resolution and manual duration restored');
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exitCode = 1; });
