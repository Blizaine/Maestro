// Real React + Zustand regression checks in an isolated browser (no app writes).
// Run: node tests/ui/studio_duration.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const playwright = require(process.env.MAESTRO_PLAYWRIGHT ||
  'playwright');

(async () => {
  const bundle = await esbuild.build({
    stdin: { contents: `
      import React from 'react';
      import { createRoot } from 'react-dom/client';
      import { useStore } from './src/stores/useStore';
      import { DurationSlider, WindowSettings } from './src/components/Sidebar/DurationSlider';
      window.store = useStore;
      window.mount = (compact = false) => {
        window.root = createRoot(document.getElementById('root'));
        window.root.render(compact ? <DurationSlider includeWindowSettings/> : <><DurationSlider/><WindowSettings/></>);
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

    const configureDurationScenario = async scenario => {
      await page.evaluate(scenario => {
        const current = window.store.getState();
        const h3 = scenario.kind === 'h3';
        const ltx = scenario.kind === 'ltx';
        const omni = h3 && scenario.omni !== false;
        const options = h3 ? {
          architecture: omni ? 'minimax_h3_ref2va' : 'minimax_h3',
          model_type: omni ? 'minimax_h3_ref2va' : 'minimax_h3', fps: 24,
          frames_minimum: 124, frames_maximum: 345, frames_steps: 17,
          sliding_window: true, omni_reference: omni,
          sliding_window_defaults: {window_min:124, window_max:345, window_step:17,
            overlap_min:1, overlap_max:97, overlap_step:17, overlap_offset:1, overlap_default:18, discard_last_frames:0},
          ...(omni ? {omni_sequence_memory_policy: {resolution_bands:[{min_pixels:0,
            vram_tiers:[{max_vram_gb:null,frames:243}]}], reference_margin_steps:1}} : {}),
        } : {
          architecture: ltx ? 'ltx-duration-test' : 'sliding-duration-test',
          model_type: ltx ? 'ltx-duration-test' : 'sliding-duration-test', fps:24,
          frames_minimum:24, frames_maximum:144, frames_steps:1, sliding_window:true,
          ...(ltx ? {multi_window_sequence_controls:true} : {}),
          sliding_window_defaults: {window_min:48, window_max:144, window_step:24,
            overlap_min:4, overlap_max:20, overlap_step:4, discard_last_frames:4},
        };
        const params = {
          ...current.params,
          model_type: options.model_type,
          image_mode: 0,
          prompt: 'A traveler crosses a quiet garden in warm morning light.',
          _duration_planning_mode: scenario.mode,
          video_length: scenario.durationFrames,
          sliding_window_size: scenario.windowFrames,
          sliding_window_overlap: scenario.overlapFrames,
          minimax_h3_extended_duration: false,
          minimax_h3_reference_sequence: h3 && omni && scenario.sequence,
          minimax_h3_multi_window: h3 && !omni && scenario.sequence,
          minimax_h3_sequence_continuity: scenario.continuity !== false,
          minimax_h3_sequence_clip_frames: scenario.windowFrames,
          ltx_multi_window: ltx && scenario.sequence,
          minimax_h3_references: scenario.mediaSeconds ? [{
            type:'video', path:'/uploads/timeline.mp4', duration_seconds:scenario.mediaSeconds,
            follow_timeline:true, video_intent:'environment',
          }] : [],
          video_guide: '', audio_guide: '',
        };
        window.store.setState({
          generationMode:'video',
          studioVideoWorkflow:scenario.workflow,
          durationSeconds:scenario.durationFrames / 24,
          slidingWindowSeconds:scenario.windowFrames / 24,
          slidingWindowOverlap:scenario.overlapFrames,
          slidingWindowLocked:scenario.locked,
          modelOptions:options,
          h3WindowPlan:null,
          promptEnhanceError:null,
          params,
        });
      }, scenario);
      await page.waitForTimeout(100);
      await page.evaluate(scenario => {
        const current = window.store.getState();
        const isH3 = String(current.modelOptions?.architecture || '').startsWith('minimax_h3');
        const isLtx = current.modelOptions?.multi_window_sequence_controls === true;
        const isOmni = current.modelOptions?.omni_reference === true;
        window.store.setState({
          durationSeconds:scenario.durationFrames / 24,
          slidingWindowSeconds:scenario.windowFrames / 24,
          slidingWindowOverlap:scenario.overlapFrames,
          slidingWindowLocked:scenario.locked,
          params:{
            ...current.params,
            _duration_planning_mode:scenario.mode,
            video_length:scenario.durationFrames,
            sliding_window_size:scenario.windowFrames,
            sliding_window_overlap:scenario.overlapFrames,
            minimax_h3_reference_sequence:isH3 && isOmni && scenario.sequence,
            minimax_h3_multi_window:isH3 && !isOmni && scenario.sequence,
            ltx_multi_window:isLtx && scenario.sequence,
          },
        });
      }, scenario);
      await page.waitForTimeout(100);
    };
    const resizeWindow = async frames => {
      await page.evaluate(frames => window.store.getState().setSlidingWindowSeconds(frames / 24), frames);
      await page.waitForTimeout(100);
      return page.evaluate(() => {
        const s = window.store.getState();
        return {durationFrames:s.params.video_length, windowFrames:s.params.sliding_window_size,
          duration:s.durationSeconds, sequence:s.params.minimax_h3_reference_sequence,
          h3MultiWindow:s.params.minimax_h3_multi_window, ltxSequence:s.params.ltx_multi_window,
          mode:s.params._duration_planning_mode,
          clipFrames:s.params.minimax_h3_sequence_clip_frames,
          h3Plan:s.h3WindowPlan, enhanceError:s.promptEnhanceError};
      });
    };

    await configureDurationScenario({kind:'h3', workflow:'references', mode:'duration',
      durationFrames:226, windowFrames:243, overlapFrames:18, sequence:false,
      continuity:true, locked:false});
    await page.getByRole('button', {name:'Window', exact:true}).click();
    const h3WindowSlider = page.getByRole('slider', {name:'Window length', exact:true});
    await h3WindowSlider.focus();
    await h3WindowSlider.press('ArrowRight');
    await page.waitForTimeout(100);
    let resized = await page.evaluate(() => {
      const s = window.store.getState();
      return {durationFrames:s.params.video_length, windowFrames:s.params.sliding_window_size,
        sequence:s.params.minimax_h3_reference_sequence, mode:s.params._duration_planning_mode};
    });
    assert.deepEqual(resized, {durationFrames:243, windowFrames:243, sequence:false, mode:'windows'},
      'One H3 window grows on the native frame lattice without enabling a sequence');
    await h3WindowSlider.press('ArrowLeft');
    await h3WindowSlider.press('ArrowLeft');
    await page.waitForTimeout(100);
    resized = await page.evaluate(() => {
      const s = window.store.getState();
      return {durationFrames:s.params.video_length, windowFrames:s.params.sliding_window_size,
        sequence:s.params.minimax_h3_reference_sequence};
    });
    assert.deepEqual(resized, {durationFrames:209, windowFrames:209, sequence:false},
      'One H3 window shrinks with its total and stays a single pass');

    await configureDurationScenario({kind:'h3', workflow:'references', mode:'windows',
      durationFrames:468, windowFrames:243, overlapFrames:18, sequence:true,
      continuity:true, locked:true});
    await h3WindowSlider.focus();
    await h3WindowSlider.press('ArrowRight');
    await page.waitForTimeout(100);
    resized = await page.evaluate(() => {
      const s = window.store.getState();
      return {durationFrames:s.params.video_length, windowFrames:s.params.sliding_window_size,
        sequence:s.params.minimax_h3_reference_sequence};
    });
    assert.deepEqual(resized, {durationFrames:502, windowFrames:260, sequence:true},
      'Two continuation windows retain their count when the native H3 length grows');
    await h3WindowSlider.press('ArrowLeft');
    await h3WindowSlider.press('ArrowLeft');
    await page.waitForTimeout(100);
    resized = await page.evaluate(() => {
      const s = window.store.getState();
      return {durationFrames:s.params.video_length, windowFrames:s.params.sliding_window_size,
        sequence:s.params.minimax_h3_reference_sequence};
    });
    assert.deepEqual(resized, {durationFrames:434, windowFrames:226, sequence:true},
      'Two continuation windows retain their count when the native H3 length shrinks');

    await configureDurationScenario({kind:'h3', workflow:'extend', mode:'windows',
      durationFrames:451, windowFrames:243, overlapFrames:18, sequence:true,
      continuity:true, locked:true});
    resized = await resizeWindow(260);
    assert.deepEqual({durationFrames:resized.durationFrames, windowFrames:resized.windowFrames,
      sequence:resized.sequence}, {durationFrames:485, windowFrames:260, sequence:true},
      'Extend preserves two windows using overlap minus one frame as first-pass context');
    resized = await resizeWindow(226);
    assert.deepEqual({durationFrames:resized.durationFrames, windowFrames:resized.windowFrames,
      sequence:resized.sequence}, {durationFrames:417, windowFrames:226, sequence:true},
      'Extend keeps its two-window count while shrinking the native pass');

    await configureDurationScenario({kind:'h3', omni:false, workflow:'frames', mode:'windows',
      durationFrames:468, windowFrames:243, overlapFrames:18, sequence:true,
      continuity:true, locked:true});
    resized = await resizeWindow(260);
    assert.deepEqual({durationFrames:resized.durationFrames, windowFrames:resized.windowFrames,
      sequence:resized.sequence, h3MultiWindow:resized.h3MultiWindow},
      {durationFrames:502, windowFrames:260, sequence:false, h3MultiWindow:true},
      'Non-Omni H3 Frames uses the First/Last multi-window sequence flag');

    await configureDurationScenario({kind:'h3', workflow:'references', mode:'windows',
      durationFrames:486, windowFrames:243, overlapFrames:18, sequence:true,
      continuity:false, locked:true});
    await page.evaluate(() => window.store.setState({
      h3WindowPlan:{plan_kind:'reference_sequence'}, promptEnhanceError:'Stale window plan',
    }));
    resized = await resizeWindow(260);
    assert.deepEqual({durationFrames:resized.durationFrames, windowFrames:resized.windowFrames,
      sequence:resized.sequence, clipFrames:resized.clipFrames},
      {durationFrames:520, windowFrames:260, sequence:true, clipFrames:260},
      'Hard-cut Omni windows keep two independent clips without overlap');
    assert.equal(resized.h3Plan, null, 'Changing window length invalidates a reviewed H3 window plan');
    assert.equal(resized.enhanceError, null, 'Changing window length clears stale enhancement errors');
    resized = await resizeWindow(226);
    assert.deepEqual({durationFrames:resized.durationFrames, windowFrames:resized.windowFrames,
      sequence:resized.sequence}, {durationFrames:452, windowFrames:226, sequence:true},
      'Hard-cut Omni clips keep their count while shrinking');

    await configureDurationScenario({kind:'ltx', workflow:'references', mode:'windows',
      durationFrames:176, windowFrames:96, overlapFrames:8, sequence:true,
      continuity:true, locked:true});
    resized = await resizeWindow(120);
    assert.deepEqual({durationFrames:resized.durationFrames, windowFrames:resized.windowFrames,
      ltxSequence:resized.ltxSequence}, {durationFrames:224, windowFrames:120, ltxSequence:true},
      'LTX preserves two windows using overlap and discarded tail frames');
    resized = await resizeWindow(72);
    assert.deepEqual({durationFrames:resized.durationFrames, windowFrames:resized.windowFrames,
      ltxSequence:resized.ltxSequence}, {durationFrames:128, windowFrames:72, ltxSequence:true},
      'LTX keeps the same count when window length decreases');

    await configureDurationScenario({kind:'sliding', workflow:'references', mode:'windows',
      durationFrames:176, windowFrames:96, overlapFrames:8, sequence:false,
      continuity:true, locked:true});
    resized = await resizeWindow(120);
    assert.deepEqual({durationFrames:resized.durationFrames, windowFrames:resized.windowFrames,
      sequence:resized.sequence, ltxSequence:resized.ltxSequence},
      {durationFrames:224, windowFrames:120, sequence:false, ltxSequence:false},
      'Ordinary sliding windows preserve count using overlap and discard geometry');
    const beforeWindowNoop = await page.evaluate(() => window.updates);
    await resizeWindow(120);
    assert.equal(await page.evaluate(() => window.updates), beforeWindowNoop,
      'Setting the current window length is idempotent');

    await configureDurationScenario({kind:'h3', workflow:'references', mode:'duration',
      durationFrames:243, windowFrames:243, overlapFrames:18, sequence:false,
      continuity:true, locked:true});
    resized = await resizeWindow(345);
    assert.equal(resized.durationFrames, 243, 'Time keeps its runtime while the window grows from 243 to 345 frames');
    assert.equal(resized.windowFrames, 345);
    assert.equal(resized.sequence, false, 'A 243-frame runtime still fits a 345-frame pass');
    resized = await resizeWindow(124);
    assert.equal(resized.durationFrames, 243, 'Time keeps its runtime while the window shrinks to 124 frames');
    assert.equal(resized.windowFrames, 124);
    assert.equal(resized.sequence, true, 'Mounted duration reconciliation enables continuation when the pass becomes shorter');

    await configureDurationScenario({kind:'h3', workflow:'references', mode:'auto',
      durationFrames:840, windowFrames:243, overlapFrames:18, sequence:true,
      continuity:true, locked:true, mediaSeconds:35});
    resized = await resizeWindow(260);
    assert.ok(Math.abs(resized.duration - 35) < 1e-8,
      'Auto continues to follow the full 35-second reference video when window length changes');
    assert.equal(resized.durationFrames, 840);

    await configureDurationScenario({kind:'h3', workflow:'references', mode:'windows',
      durationFrames:86400, windowFrames:243, overlapFrames:18, sequence:true,
      continuity:true, locked:true});
    resized = await resizeWindow(260);
    assert.ok(resized.duration <= 3600 && resized.durationFrames <= 86400,
      'Window mode respects the one-hour ceiling');
    const completeH3Windows = 1 + (resized.durationFrames - resized.windowFrames)
      / (resized.windowFrames - 18);
    assert.ok(Number.isInteger(completeH3Windows), 'The capped H3 timeline ends on a complete native window');
    assert.deepEqual(errors, [], 'Window length changes remain stable across Window, Time, and Auto modes');
    console.log('Window length: count-preserving H3, Extend, hard-cut Omni, LTX and sliding timelines passed');

    await page.evaluate(() => {window.updates = 0; window.durationHistory = [];});
    const extendedPhaseErrorCount = errors.length;
    const readExtendedState = () => page.evaluate(() => {
      const s = window.store.getState();
      return {
        mode:s.params._duration_planning_mode,
        extended:s.params.minimax_h3_extended_duration === true,
        durationFrames:s.params.video_length,
        windowFrames:s.params.sliding_window_size,
      };
    });
    const dragExtendedWindowTo = async frames => {
      const slider = page.getByRole('slider', {name:'Window length', exact:true});
      await slider.scrollIntoViewIfNeeded();
      const bounds = await slider.boundingBox();
      const limits = await slider.evaluate(input => ({min:Number(input.min), max:Number(input.max)}));
      const current = await page.evaluate(() => window.store.getState().params.sliding_window_size);
      const xAt = value => bounds.x + 8
        + ((value - limits.min) / (limits.max - limits.min)) * (bounds.width - 16);
      const y = bounds.y + bounds.height / 2;
      await page.mouse.move(xAt(current), y);
      await page.mouse.down();
      await page.mouse.move(xAt(frames), y, {steps:4});
      await page.mouse.up();
      await page.waitForTimeout(100);
      return readExtendedState();
    };
    const exactWindowCount = async () => Number(
      await page.getByRole('spinbutton', {name:'Window count', exact:true}).inputValue(),
    );
    for (const omni of [false, true]) {
      const workflow = omni ? 'references' : 'frames';
      const h3Fixture = {kind:'h3', omni, workflow, overlapFrames:18, continuity:true, locked:true};
      await configureDurationScenario({...h3Fixture, mode:'windows', durationFrames:243,
        windowFrames:243, sequence:false});
      await page.getByRole('checkbox', {name:/Allow 30s clips/}).check();
      assert.deepEqual(await readExtendedState(), {mode:'windows', extended:true,
        durationFrames:719, windowFrames:719}, `${workflow}: extending a one-window plan keeps Window mode and its count`);
      assert.equal(await exactWindowCount(), 1);
      let extended = await dragExtendedWindowTo(345);
      assert.deepEqual(extended, {mode:'windows', extended:true, durationFrames:345, windowFrames:345},
        `${workflow}: a one-window plan follows a shorter extended length`);
      extended = await dragExtendedWindowTo(719);
      assert.deepEqual(extended, {mode:'windows', extended:true, durationFrames:719, windowFrames:719},
        `${workflow}: a one-window plan follows a longer extended length`);
      assert.equal(await exactWindowCount(), 1);

      await configureDurationScenario({...h3Fixture, mode:'windows', durationFrames:468,
        windowFrames:243, sequence:true});
      await page.getByRole('checkbox', {name:/Allow 30s clips/}).check();
      assert.deepEqual(await readExtendedState(), {mode:'windows', extended:true,
        durationFrames:1420, windowFrames:719}, `${workflow}: extending keeps two windows`);
      assert.equal(await exactWindowCount(), 2);
      extended = await dragExtendedWindowTo(345);
      assert.deepEqual(extended, {mode:'windows', extended:true, durationFrames:672, windowFrames:345},
        `${workflow}: two windows follow a shorter extended length`);
      assert.equal(await exactWindowCount(), 2);
      extended = await dragExtendedWindowTo(719);
      assert.deepEqual(extended, {mode:'windows', extended:true, durationFrames:1420, windowFrames:719},
        `${workflow}: two windows follow a longer extended length`);
      await page.getByRole('checkbox', {name:/Allow 30s clips/}).uncheck();
      assert.deepEqual(await readExtendedState(), {mode:'windows', extended:false,
        durationFrames:672, windowFrames:345}, `${workflow}: disabling normalizes capacity and retains the two-window count`);
      assert.equal(await exactWindowCount(), 2);
      await page.getByRole('checkbox', {name:/Allow 30s clips/}).check();
      await page.getByRole('button', {name:/^Auto/}).click();
      await page.waitForTimeout(100);
      assert.equal((await readExtendedState()).mode, 'auto', `${workflow}: Auto remains selected`);
      assert.equal((await readExtendedState()).extended, false, `${workflow}: Auto exits the experiment`);
      const autoState = await page.evaluate(() => {
        const s = window.store.getState();
        return {locked:s.slidingWindowLocked, windowFrames:s.params.sliding_window_size};
      });
      assert.deepEqual(autoState, {locked:false, windowFrames:omni ? 226 : 345},
        `${workflow}: Auto restores its ordinary window recommendation`);
      assert.equal(await page.getByRole('slider', {name:'Window length', exact:true}).getAttribute('max'), '345',
        `${workflow}: Auto restores the native recommendation range`);

      await configureDurationScenario({...h3Fixture, mode:'duration', durationFrames:243,
        windowFrames:243, sequence:false});
      await page.getByRole('checkbox', {name:/Allow 30s clips/}).check();
      assert.deepEqual(await readExtendedState(), {mode:'duration', extended:true,
        durationFrames:243, windowFrames:719}, `${workflow}: Time retains its target while exposing the 719-frame window`);
      await page.getByRole('button', {name:'Window', exact:true}).click();
      assert.deepEqual(await readExtendedState(), {mode:'windows', extended:true,
        durationFrames:719, windowFrames:719}, `${workflow}: entering Window uses the selected extended capacity`);
      await page.getByRole('checkbox', {name:/Allow 30s clips/}).uncheck();
      assert.deepEqual(await readExtendedState(), {mode:'windows', extended:false,
        durationFrames:345, windowFrames:345}, `${workflow}: disabling in Window retains one window`);
      assert.equal(await exactWindowCount(), 1);
    }
    assert.equal(errors.length, extendedPhaseErrorCount,
      'Allow 30s Window interactions converge within the existing update guard');
    console.log('Allow 30s: Frames and References preserve Window counts, Time targets and Auto recommendations');

    // The opt-in must survive real UI reconciliation and the generation request.
    await page.evaluate(() => {window.updates = 0; window.durationHistory = [];});
    let extendedSubmission;
    await page.route('**/api/v1/generate', route => {
      extendedSubmission = route.request().postDataJSON();
      return route.fulfill({status:400, json:{detail:'Captured experimental request'}});
    });
    for (const omni of [true, false]) {
      extendedSubmission = undefined;
      await page.evaluate(omni => {
        const s = window.store.getState();
        const model = omni ? 'minimax_h3_ref2va' : 'minimax_h3';
        window.store.setState({studioVideoWorkflow:omni ? 'references' : 'frames',
          studioVideoEffectiveCreateRoute:omni ? 'omni' : 'generate',
          models:[{...s.modelOptions, model_type:model, architecture:model, omni_reference:omni}],
          modelOptions:{...s.modelOptions, model_type:model, architecture:model, omni_reference:omni},
          startImage:null, endImage:null, imageRefs:[], isGenerating:false, promptEnhanceError:null,
          params:{...s.params, model_type:model, image_mode:0, prompt:'A quiet garden at sunrise.',
            _duration_planning_mode:'duration', minimax_h3_extended_duration:false,
            minimax_h3_reference_sequence:false, minimax_h3_multi_window:false,
            minimax_h3_references:omni ? [{type:'image',path:'/ref.png',role:'Garden'}] : [],
            image_start:undefined, image_end:undefined}});
      }, omni);
      await page.getByRole('checkbox', {name:/Allow 30s clips/}).check();
      assert.equal(await page.getByRole('slider', {name:'Window length', exact:true}).getAttribute('max'), '719');
      await page.evaluate(() => window.store.getState().setDurationSeconds(30));
      await page.waitForTimeout(80);
      const selected = await page.evaluate(() => {
        const s = window.store.getState();
        return {frames:s.params.video_length, window:s.params.sliding_window_size,
          sequence:!!(s.params.minimax_h3_multi_window || s.params.minimax_h3_reference_sequence)};
      });
      assert.deepEqual(selected, {frames:719, window:719, sequence:false}, '30s rounds to one native pass');
      await page.evaluate(() => window.store.getState().startGeneration());
      assert.ok(extendedSubmission, `Experimental request reaches the API: ${await page.evaluate(() => window.store.getState().promptEnhanceError)}`);
      assert.equal(extendedSubmission.minimax_h3_extended_duration, true);
      assert.equal(extendedSubmission.video_length, 719);
      assert.equal(extendedSubmission.sliding_window_size, 719);
      assert.equal(extendedSubmission.sliding_window_memory_override, true);
      const modelOptions = await page.evaluate(() => window.store.getState().modelOptions);
      await page.route('**/api/v1/model-options/*', route => route.fulfill({json:modelOptions}));
      await page.evaluate(() => window.store.getState().loadModelOptions(window.store.getState().params.model_type));
      assert.equal(await page.evaluate(() => window.store.getState().params.sliding_window_size), 719,
        'Refreshing native model options must retain an experimental job limit');
      await page.getByRole('checkbox', {name:/Allow 30s clips/}).uncheck();
      assert.equal(await page.getByRole('slider', {name:'Window length', exact:true}).getAttribute('max'), '345');
    }
    await page.getByRole('checkbox', {name:/Allow 30s clips/}).check();
    await page.getByRole('button', {name:/^Auto/}).click();
    assert.equal(await page.evaluate(() => window.store.getState().params.minimax_h3_extended_duration), false,
      'Auto explicitly exits the experiment');
    // Exercise the compact popup controls used by the real Studio sidecar.
    await page.evaluate(() => {window.root.unmount(); window.mount(true);});
    await page.getByRole('checkbox', {name:/Allow 30s clips/}).check();
    await page.getByRole('button', {name:'Window', exact:true}).click();
    await page.getByRole('button', {name:'1', exact:true}).click();
    assert.equal(await page.evaluate(() => window.store.getState().params.video_length), 719);
    const assets = path.join(root, 'ui/dist/assets');
    await page.addStyleTag({content:fs.readFileSync(path.join(assets, fs.readdirSync(assets).find(f => f.endsWith('.css'))), 'utf8')});
    await page.setViewportSize({width:390, height:800});
    await page.evaluate(() => {document.body.style.padding='12px'; document.getElementById('root').style.width='100%';});
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= 390), 'Extended controls fit mobile width');
    const output = path.join(root, '.codex-tmp/sidebar-validation');
    fs.mkdirSync(output, {recursive:true});
    await page.screenshot({path:path.join(output, 'h3-extended-30s.png')});
    await page.getByRole('switch', {name:'Automatic duration'}).click();
    assert.equal(await page.evaluate(() => window.store.getState().params.minimax_h3_extended_duration), false);
    await page.evaluate(() => {window.root.unmount(); window.mount();});
    assert.deepEqual(errors, []);
    console.log('Experimental H3: both modes submit 719-frame single passes, toggle off and Auto restore native limits');
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
    // The return starts an asynchronous model-options request. Let the mocked
    // request finish before installing the independent LongCat fixture below.
    await page.waitForFunction(() => !window.store.getState().modelOptionsLoading);
    await page.evaluate(() => {
      const s = window.store.getState();
      window.store.setState({generationMode: 'video', studioVideoWorkflow: 'avatar',
        durationSeconds: 1275 / 16, slidingWindowSeconds: 93 / 16,
        slidingWindowOverlap: 13, slidingWindowLocked: false,
        modelOptions: {model_type: 'longcat_avatar', architecture: 'longcat_avatar',
          fps: 16, frames_minimum: 5, frames_steps: 4, sliding_window: true,
          sliding_window_defaults: {window_min: 17, window_max: 93, window_step: 4,
            window_default: 93, overlap_min: 1, overlap_max: 13, overlap_step: 4,
            overlap_default: 13, discard_last_frames: 0}},
        params: {...s.params, model_type: 'longcat_avatar', image_mode: 0,
          prompt: 'Man sings', audio_guide: '/voice.wav', video_guide: '',
          _duration_planning_mode: 'auto', video_length: 1275, sliding_window_size: 93,
          minimax_h3_extended_duration: false, minimax_h3_references: []}});
      window.mount();
    });
    await page.waitForTimeout(100);
    const longcatWindow = page.getByRole('slider', {name: 'Window size', exact: true});
    assert.equal(await longcatWindow.getAttribute('max'), String(93 / 16));
    assert.equal(await longcatWindow.getAttribute('step'), String(4 / 16));
    assert.equal(await page.evaluate(() => window.store.getState().params.video_length), 1275,
      'Avatar auto duration preserves the full audio timeline across short windows');
    assert.ok((await page.locator('#root').innerText()).includes('16 windows'));
    await longcatWindow.focus();
    await longcatWindow.press('ArrowLeft');
    assert.equal(await page.evaluate(() => window.store.getState().params.sliding_window_size), 89,
      'LongCat window controls use the four-frame model step');
    await longcatWindow.press('End');
    assert.equal(await page.evaluate(() => window.store.getState().params.sliding_window_size), 93,
      'The slider cannot submit an inherited forty-second window');
    assert.equal(await page.evaluate(() => window.store.getState().params.video_length), 1275);
    assert.deepEqual(errors, []);
    console.log('LongCat Avatar: 93-frame cap, native window steps, full audio duration and 16-pass continuation schedule passed');
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exitCode = 1; });
