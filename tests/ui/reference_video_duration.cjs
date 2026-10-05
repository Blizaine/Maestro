// Real React + Zustand duration-source checks in an isolated browser.
// Run: node tests/ui/reference_video_duration.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const playwright = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');

const fps = 24;
const h3Options = {
  architecture: 'minimax_h3_ref2va', model_type: 'minimax_h3_ref2va', fps,
  frames_minimum: 124, frames_maximum: 345, frames_steps: 17,
  sliding_window: true, omni_reference: true,
  sliding_window_defaults: {
    window_min: 124, window_max: 345, window_step: 17,
    window_default: 243, overlap_default: 18, discard_last_frames: 0,
  },
  omni_sequence_memory_policy: {
    resolution_bands: [{min_pixels: 0, vram_tiers: [{max_vram_gb: null, frames: 243}]}],
    reference_margin_steps: 1,
  },
};

const video = (id, fields = {}) => ({type: 'video', id, path: `/uploads/${id}.mp4`, ...fields});

(async () => {
  const bundle = await esbuild.build({
    stdin: { contents: `
      import React from 'react';
      import { createRoot } from 'react-dom/client';
      import { useStore } from './src/stores/useStore';
      import { OutputFormatControls } from './src/components/Sidebar/OutputFormatControls';
      window.store = useStore;
      window.mount = () => {
        window.root = createRoot(document.getElementById('root'));
        window.root.render(<OutputFormatControls/>);
      };
    `, resolveDir: path.join(root, 'ui'), loader: 'tsx' },
    bundle: true, write: false, jsx: 'automatic',
    define: {'process.env.NODE_ENV': '"development"'}, logLevel: 'silent',
    plugins: [{
      name: 'resolve-from-ui-with-node',
      setup(build) {
        const uiRoot = path.join(root, 'ui');
        const extensions = ['.tsx', '.ts', '.jsx', '.js', '.mjs', '.cjs', '.json', '.css'];
        build.onResolve({filter: /.*/}, args => {
          if (args.path.startsWith('.') || path.isAbsolute(args.path)) {
            const base = path.resolve(args.resolveDir || path.dirname(args.importer), args.path);
            const candidates = [base, ...extensions.map(extension => `${base}${extension}`),
              ...extensions.map(extension => path.join(base, `index${extension}`))];
            const resolved = candidates.find(candidate => {
              try { return fs.statSync(candidate).isFile(); } catch { return false; }
            });
            if (resolved) return {path: resolved};
          } else {
            try { return {path: require.resolve(args.path, {paths: [uiRoot]})}; } catch {}
          }
          throw new Error(`Unable to resolve UI bundle import: ${args.path}`);
        });
      },
    }],
  });
  const browser = await playwright.chromium.launch({headless: true,
    ...(process.env.MAESTRO_CHROME || process.platform === 'win32'
      ? {executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'} : {})});
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/*', route => route.fulfill({contentType: 'text/html', body: '<div id="root"></div>'}));
    await page.goto('http://studio.test');
    await page.addScriptTag({content: bundle.outputFiles[0].text});
    await page.evaluate(({options}) => {
      const s = window.store.getState();
      window.store.setState({
        modelOptions: options, generationMode: 'video', studioVideoWorkflow: 'references',
        durationSeconds: 5, slidingWindowSeconds: 243 / 24, slidingWindowOverlap: 18,
        slidingWindowLocked: false, systemStats: {gpu: {vram_total_gb: 24}}, h3WindowOverrides: {},
        params: {...s.params, model_type: 'minimax_h3_ref2va', image_mode: 0,
          prompt: 'A quiet garden at sunrise as a person enters the scene.', resolution: '864x480',
          _duration_planning_mode: 'auto', minimax_h3_extended_duration: false,
          minimax_h3_reference_sequence: false, minimax_h3_multi_window: false,
          minimax_h3_references: [], audio_guide: '', video_guide: ''},
      });
      window.durationNotifications = 0;
      window.durationStableFrames = 0;
      window.durationStableSnapshot = null;
      window.store.subscribe(() => {
        window.durationNotifications += 1;
        window.durationStableFrames = 0;
        window.durationStableSnapshot = null;
      });
      const observeStableDuration = () => {
        const current = window.store.getState();
        const snapshot = JSON.stringify({
          seconds: current.durationSeconds, frames: current.params.video_length,
          windowFrames: current.params.sliding_window_size,
          sequence: current.params.minimax_h3_reference_sequence,
          planningMode: current.params._duration_planning_mode,
          workflow: current.studioVideoWorkflow,
          references: current.params.minimax_h3_references,
        });
        if (window.durationStableSnapshot === snapshot) window.durationStableFrames += 1;
        else { window.durationStableSnapshot = snapshot; window.durationStableFrames = 0; }
        requestAnimationFrame(observeStableDuration);
      };
      requestAnimationFrame(observeStableDuration);
      window.mount();
    }, {options: h3Options});

    const read = () => page.evaluate(() => {
      const state = window.store.getState();
      const dialog = document.querySelector('[role="dialog"][id$="-duration"]');
      return {
        seconds: state.durationSeconds,
        frames: state.params.video_length,
        windowFrames: state.params.sliding_window_size,
        sequence: state.params.minimax_h3_reference_sequence,
        planningMode: state.params._duration_planning_mode,
        dialogHidden: !!dialog?.closest('[hidden]'),
        dialogText: dialog?.textContent || '',
        notifications: window.durationNotifications,
        stableFrames: window.durationStableFrames,
      };
    });
    const settle = async label => {
      await page.waitForFunction(() => window.durationStableFrames >= 6,
        null, {timeout: 5000});
      const current = await read();
      await page.waitForFunction(frames => window.durationStableFrames > frames,
        current.stableFrames, {timeout: 5000});
      const later = await read();
      assert.equal(later.notifications, current.notifications,
        `${label}: duration reaches a stable Zustand state`);
      assert.deepEqual(errors, [], `${label}: no React errors`);
      assert.equal(current.dialogHidden, true, `${label}: OutputFormatControls duration popup stays closed`);
      return current;
    };
    const setReferences = async (references, workflow = 'references', planningMode = 'auto') => {
      await page.evaluate(({references, workflow, planningMode}) => {
        const state = window.store.getState();
        window.store.setState({studioVideoWorkflow: workflow,
          params: {...state.params, minimax_h3_references: references,
            _duration_planning_mode: planningMode, audio_guide: '', video_guide: ''}});
      }, {references, workflow, planningMode});
      return settle(`references=${references.length}, workflow=${workflow}, plan=${planningMode}`);
    };
    const frameSeconds = seconds => Math.round(seconds * fps) / fps;

    const promptDuration = (await settle('initial prompt plan')).seconds;
    assert.ok(promptDuration >= 124 / fps && promptDuration <= 345 / fps,
      'Prompt auto sizing retains the normal native H3 duration');

    // The real DurationSlider remains mounted in OutputFormatControls while its
    // dialog is hidden; a non-lattice source longer than one pass drives the timeline.
    const nonLattice = 42.347;
    let state = await setReferences([video('long', {
      duration_seconds: nonLattice, source_duration_seconds: 40, follow_timeline: true,
      effective_duration_seconds: 9.5,
    })]);
    assert.equal(state.seconds, frameSeconds(nonLattice));
    assert.equal(state.frames, Math.round(nonLattice * fps));
    assert.ok(state.frames > state.windowFrames, 'A source longer than one window enables H3 reference sequencing');
    assert.equal(state.sequence, true);
    assert.match(state.dialogText, /complete reference video timeline/i);

    // Auto follows the longest timeline reference, not a sum of parallel videos.
    const longest = 73.128;
    state = await setReferences([
      video('short-parallel', {duration_seconds: 12.25}),
      video('long-parallel', {duration_seconds: longest, video_intent: 'scene'}),
      video('static-video', {duration_seconds: 114, follow_timeline: false}),
    ]);
    assert.equal(state.seconds, frameSeconds(longest));
    assert.equal(state.frames, Math.round(longest * fps));

    // Missing or invalid full-source metadata falls back to source duration;
    // per-window effective duration never substitutes for the full timeline.
    const fallback = 18.6;
    state = await setReferences([video('source-fallback', {
      duration_seconds: 0, source_duration_seconds: fallback, effective_duration_seconds: 98,
    })]);
    assert.equal(state.seconds, frameSeconds(fallback));
    state = await setReferences([video('unknown', {
      duration_seconds: 'unknown', source_duration_seconds: Number.NaN,
      effective_duration_seconds: 700,
    })]);
    assert.equal(state.seconds, promptDuration, 'Unknown duration returns to prompt-based auto sizing');

    // Images, static videos, default saved-character videos, RefMod identities,
    // and generated continuation media do not drive; an explicit saved-character
    // timeline opt-in remains eligible.
    state = await setReferences([
      {type: 'image', id: 'still', duration_seconds: 125, follow_timeline: true},
      video('static', {duration_seconds: 126, follow_timeline: false}),
      video('saved-default', {duration_seconds: 127, library_character_id: 'saved-1', video_intent: 'character'}),
      video('refmod', {duration_seconds: 128, library_character_id: 'saved-2', video_intent: 'character',
        follow_timeline: true, refmod_path: '/characters/refmod.safetensors'}),
      video('generated-continuation', {duration_seconds: 129, follow_timeline: true,
        _maestro_generated_continuity: true}),
    ]);
    assert.equal(state.seconds, promptDuration, 'Excluded media does not change prompt auto sizing');
    const explicitSavedTimeline = 23.125;
    state = await setReferences([
      video('saved-opt-in', {duration_seconds: explicitSavedTimeline,
        library_character_id: 'saved-3', video_intent: 'character', follow_timeline: true}),
    ]);
    assert.equal(state.seconds, frameSeconds(explicitSavedTimeline), 'Explicit true opts a saved character into timeline following');

    // Audio drive retains precedence over longer video references.
    const audioDrive = 21.875;
    state = await setReferences([
      {type: 'audio', id: 'drive', audio_intent: 'drive', duration_seconds: audioDrive},
      video('longer-video', {duration_seconds: 92}),
    ]);
    assert.equal(state.seconds, frameSeconds(audioDrive));
    assert.match(state.dialogText, /music \/ performance timeline/i);

    // Short source lengths still obey H3's native minimum. Automatic source
    // selection also retains the existing one-hour cap.
    state = await setReferences([video('short-native', {duration_seconds: 3.2})]);
    assert.equal(state.frames, 124);
    assert.equal(state.seconds, 124 / fps);
    state = await setReferences([video('over-hour', {duration_seconds: 3700})]);
    assert.equal(state.seconds, 3600);
    assert.equal(state.frames, 3600 * fps);

    // Replacing, removing, and toggling a timeline source updates Auto without
    // opening the popup. Removing the last source restores prompt sizing.
    state = await setReferences([video('inserted', {duration_seconds: 44})]);
    assert.equal(state.seconds, frameSeconds(44));
    state = await setReferences([video('replacement', {duration_seconds: 61})]);
    assert.equal(state.seconds, frameSeconds(61));
    state = await setReferences([video('toggle', {duration_seconds: 28.375, follow_timeline: false})]);
    assert.equal(state.seconds, promptDuration);
    state = await setReferences([video('toggle', {duration_seconds: 28.375, follow_timeline: true})]);
    assert.equal(state.seconds, frameSeconds(28.375));
    state = await setReferences([video('toggle', {duration_seconds: 28.375, follow_timeline: false})]);
    assert.equal(state.seconds, promptDuration);
    state = await setReferences([]);
    assert.equal(state.seconds, promptDuration);

    // User-selected duration and window-count planning remain stable as media changes.
    for (const [mode, manualSeconds] of [['duration', 31.125], ['windows', 32.25]]) {
      await page.evaluate(({mode, manualSeconds}) => {
        const store = window.store.getState();
        store.setParam('_duration_planning_mode', mode);
        store.setDurationSeconds(manualSeconds);
      }, {mode, manualSeconds});
      state = await settle(`${mode} manual plan selected`);
      assert.equal(state.planningMode, mode);
      const preservedSeconds = state.seconds;
      state = await setReferences([video(`manual-${mode}`, {duration_seconds: 107})], 'references', mode);
      assert.equal(state.seconds, preservedSeconds, `${mode} runtime survives a new video reference`);
      state = await setReferences([], 'references', 'auto');
      assert.equal(state.seconds, promptDuration, `${mode} returns to prompt auto sizing after its video is removed`);
    }

    // Video metadata only drives this plan in the References workflow.
    state = await setReferences([video('outside-references', {duration_seconds: 86})], 'frames');
    assert.equal(state.seconds, promptDuration);
    assert.ok(state.seconds < 86);
    assert.deepEqual(errors, []);
    console.log('Reference video duration: hidden popup updates, longest eligible source, exclusions, audio priority, manual modes, replacement/removal, and H3 caps passed');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
