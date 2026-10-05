// Real React + Zustand object-reference checks in an isolated browser. All
// network requests are intercepted; no live Studio state or generation is used.
// Run: node tests/ui/object_reference.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const playwright = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');

const modelType = 'minimax_h3_ref2va';
const fps = 24;
const options = {
  architecture: modelType,
  model_type: modelType,
  fps,
  frames_minimum: 124,
  frames_maximum: 345,
  frames_steps: 17,
  sliding_window: true,
  omni_reference: true,
  omni_reference_limits: {image: 9, video: 3, audio: 6, total: 18},
  sliding_window_defaults: {
    window_min: 124,
    window_max: 345,
    window_step: 17,
    window_default: 243,
    overlap_default: 18,
    discard_last_frames: 0,
  },
  omni_sequence_memory_policy: {
    resolution_bands: [{min_pixels: 0, vram_tiers: [{max_vram_gb: null, frames: 243}]}],
    reference_margin_steps: 1,
  },
};

const model = {
  model_type: modelType,
  name: 'H3 References',
  family: 'video',
  architecture: modelType,
  is_i2v: true,
  is_t2v: true,
  fps,
  omni_reference: true,
};

const references = [
  {
    id: 'sentinel-image', type: 'image', path: '/uploads/sentinel.png',
    filename: 'sentinel.png', role: 'Sentinel', character_name: 'Sentinel',
    library_character_id: 'sentinel', image_intent: 'identity',
  },
  {
    id: 'atat-object', type: 'image', path: '/uploads/atat.png',
    filename: 'atat.png', role: 'AT-AT', image_intent: 'identity', remove_background: false,
  },
  {
    id: 'portrait-image', type: 'image', path: '/uploads/portrait.png',
    filename: 'portrait.png', role: 'Knight portrait', image_intent: 'identity',
  },
  {
    id: 'scene-image', type: 'image', path: '/uploads/scene.png',
    filename: 'scene.png', role: 'Temple scene', image_intent: 'scene',
  },
  {
    id: 'sentinel-voice', type: 'audio', path: '/uploads/sentinel-voice.wav',
    filename: 'sentinel-voice.wav', role: 'Sentinel', character_name: 'Sentinel',
    library_character_id: 'sentinel', audio_intent: 'sound', duration_seconds: 7.1,
  },
  {
    id: 'laser', type: 'audio', path: '/uploads/laser.wav', filename: 'laser.wav',
    role: 'Laser blaster', audio_intent: 'voice', duration_seconds: 3.23, has_audio: true,
  },
  {
    id: 'style', type: 'audio', path: '/uploads/style.wav', filename: 'style.wav',
    role: 'Ambient synth style', audio_intent: 'style', duration_seconds: 46,
  },
  {
    id: 'video', type: 'video', path: '/uploads/scene.mp4', filename: 'scene.mp4',
    role: 'Street scene', has_audio: true, include_audio: true,
  },
  {
    id: 'score', type: 'audio', path: '/uploads/score.wav', filename: 'score.wav',
    role: 'Exact score', audio_intent: 'drive', duration_seconds: 23,
  },
];

(async () => {
  const bundle = await esbuild.build({
    stdin: { contents: `
      import React from 'react';
      import { createRoot } from 'react-dom/client';
      import { useStore } from './src/stores/useStore';
      import { OmniReferenceSection } from './src/components/Sidebar/OmniReferenceSection';
      window.store = useStore;
      window.mount = () => {
        window.root = createRoot(document.getElementById('root'));
        window.root.render(<OmniReferenceSection scope="studio"/>);
      };
    `, resolveDir: path.join(root, 'ui'), loader: 'tsx' },
    bundle: true,
    write: false,
    jsx: 'automatic',
    define: {'process.env.NODE_ENV': '"development"'},
    logLevel: 'silent',
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
    const page = await browser.newPage({viewport: {width: 390, height: 844}});
    const errors = [];
    let enhanceRequest = null;
    let generateRequest = null;
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/*', async route => {
      const request = route.request();
      const url = new URL(request.url());
      const json = value => route.fulfill({
        contentType: 'application/json', body: JSON.stringify(value),
      });
      if (url.hostname === 'studio.test' && url.pathname === '/') {
        return route.fulfill({contentType: 'text/html', body: '<div id="root"></div>'});
      }
      if (url.pathname === '/api/v1/characters') return json({characters: []});
      if (url.pathname === '/api/v1/jobs') return json({jobs: []});
      if (url.pathname === '/api/v1/llm/enhance-prompt') {
        enhanceRequest = JSON.parse(request.postData() || '{}');
        return json({original: enhanceRequest.prompt, enhanced: enhanceRequest.prompt, warnings: []});
      }
      if (url.pathname === '/api/v1/generate') {
        generateRequest = JSON.parse(request.postData() || '{}');
        return json({job_id: 'object-reference-job', status: 'held'});
      }
      if (url.pathname.startsWith('/api/v1/jobs/')) {
        return json({job_id: 'object-reference-job', status: 'held', progress: 0});
      }
      return json({});
    });
    await page.goto('http://studio.test');
    await page.addScriptTag({content: bundle.outputFiles[0].text});
    await page.evaluate(({options, model, references}) => {
      const store = window.store.getState();
      window.h3Options = options;
      window.store.setState({
        modelOptions: options,
        models: [model],
        enabledModels: new Set([model.model_type]),
        selectedModelPerMode: {...store.selectedModelPerMode, video: model.model_type},
        studioVideoModelPerCreateRoute: {generate: model.model_type, omni: model.model_type},
        studioVideoEffectiveCreateRoute: 'omni',
        generationMode: 'video',
        studioVideoWorkflow: 'references',
        durationSeconds: 12,
        slidingWindowSeconds: 243 / 24,
        slidingWindowOverlap: 18,
        slidingWindowLocked: false,
        systemStats: {gpu: {vram_total_gb: 24}},
        h3WindowOverrides: {},
        params: {...store.params,
          model_type: model.model_type,
          image_mode: 0,
          resolution: '864x480',
          prompt: 'The Sentinel fires the laser blaster when raising a hand.',
          video_length: 288,
          sliding_window_size: 243,
          sliding_window_overlap: 18,
          minimax_h3_reference_sequence: false,
          minimax_h3_references: references,
          audio_guide: '',
          video_guide: '',
        },
      });
      window.mount();
    }, {options, model, references});

    await page.waitForFunction(() => window.store.getState().params.minimax_h3_references
      .find(reference => reference.id === 'sentinel-voice')?.audio_intent === 'voice');
    const initialDuration = await page.evaluate(() => ({
      seconds: window.store.getState().durationSeconds,
      frames: window.store.getState().params.video_length,
    }));
    assert.deepEqual(initialDuration, {seconds: 12, frames: 288});

    await page.getByRole('button', {name: 'Edit Sentinel reference'}).click();
    const savedCharacterSettings = page.getByRole('group', {name: 'Sentinel reference settings'});
    assert.match(await savedCharacterSettings.innerText(), /Voice reference/);
    assert.equal(await savedCharacterSettings.getByRole('combobox').count(), 0,
      'A saved character voice stays locked to voice intent');
    await page.getByRole('button', {name: 'Close reference settings'}).click();

    await page.getByRole('button', {name: 'Edit Picture 3 reference'}).click();
    assert.ok(await page.getByRole('checkbox', {name: 'Isolate subject background'}).isVisible(),
      'Identity image references retain optional subject background isolation');
    await page.getByRole('button', {name: 'Close reference settings'}).click();

    await page.getByRole('button', {name: 'Edit Picture 2 reference'}).click();
    const imageType = page.getByRole('combobox', {name: 'Picture 2 type'});
    assert.ok(await imageType.locator('option[value="object"]').count(),
      'Object / prop is available in the image intent menu');
    await imageType.selectOption('object');
    const objectHelp = page.getByText(/Keeps the object's design, shape, proportions/i);
    assert.ok(await objectHelp.isVisible(), 'Object help is visible in the mobile reference editor');
    const objectHelpBox = await objectHelp.boundingBox();
    assert.ok(objectHelpBox && objectHelpBox.x >= 0 && objectHelpBox.x + objectHelpBox.width <= 390,
      'Object help fits within the mobile viewport');
    assert.match(await objectHelp.innerText(), /materials, colors and details/i);
    assert.match(await objectHelp.innerText(), /placement, scale, action and count/i);
    assert.match(await objectHelp.innerText(), /source scene, framing, background and pose are ignored/i);
    const objectBackground = page.getByRole('checkbox', {name: 'Isolate object background'});
    assert.ok(await objectBackground.isVisible(), 'Object background isolation is available');
    await objectBackground.check();

    const afterObjectSelection = await page.evaluate(() => {
      const reference = window.store.getState().params.minimax_h3_references
        .find(item => item.id === 'atat-object');
      return {
        seconds: window.store.getState().durationSeconds,
        frames: window.store.getState().params.video_length,
        imageIntent: reference?.image_intent,
        removeBackground: reference?.remove_background,
      };
    });
    assert.deepEqual(afterObjectSelection, {
      seconds: 12, frames: 288, imageIntent: 'object', removeBackground: true,
    }, 'Object intent and background isolation leave the selected duration unchanged');

    // Imported metadata can retain a legacy character id. The object must stay
    // independently editable and must not bind as a saved-character Subject.
    await page.evaluate(() => {
      const state = window.store.getState();
      window.store.setState({params: {...state.params,
        minimax_h3_references: state.params.minimax_h3_references.map(reference => reference.id === 'atat-object'
          ? {...reference, library_character_id: 'legacy-object-character', character_name: 'Legacy AT-AT'}
          : reference),
      }});
    });
    assert.ok(await imageType.isVisible(), 'An object with imported character metadata remains an editable Picture');
    assert.equal(await imageType.inputValue(), 'object');

    await page.getByRole('button', {name: 'Edit Audio 2 reference'}).click();
    const audioType = page.getByRole('combobox', {name: 'Audio 2 type'});
    assert.ok(await audioType.locator('option[value="sound"]').count(),
      'Sound effect reference is available in the audio intent menu');
    await audioType.selectOption('sound');
    const soundHelp = page.getByText(/Each window gets the same short sample/i);
    assert.ok(await soundHelp.isVisible(), 'Intent help is visible in the mobile reference editor');
    assert.match(await soundHelp.innerText(), /Describe the effect and when it happens in your prompt/i);
    assert.match(await soundHelp.innerText(), /does not play or loop the exact waveform or set total duration/i);

    const afterSoundSelection = await page.evaluate(() => {
      const refs = window.store.getState().params.minimax_h3_references;
      return {
        seconds: window.store.getState().durationSeconds,
        frames: window.store.getState().params.video_length,
        objectIntent: refs.find(reference => reference.id === 'atat-object')?.image_intent,
        objectBackground: refs.find(reference => reference.id === 'atat-object')?.remove_background,
        sound: refs.find(reference => reference.id === 'laser')?.audio_intent,
        savedVoice: refs.find(reference => reference.id === 'sentinel-voice')?.audio_intent,
        existingStyle: refs.find(reference => reference.id === 'style')?.audio_intent,
      };
    });
    assert.deepEqual(afterSoundSelection, {
      seconds: 12,
      frames: 288,
      objectIntent: 'object',
      objectBackground: true,
      sound: 'sound',
      savedVoice: 'voice',
      existingStyle: 'style',
    }, 'Sound intent is reusable and does not alter duration or migrate other selections');

    await page.evaluate(async () => window.store.getState().enhancePrompt());
    assert.ok(enhanceRequest,
      `Enhance Now sends the Omni reference inventory (${await page.evaluate(() => window.store.getState().promptEnhanceError || 'no request')})`);
    const inventory = enhanceRequest.reference_context || '';
    assert.deepEqual(enhanceRequest.image_paths, [
      '/uploads/sentinel.png', '/uploads/atat.png', '/uploads/portrait.png', '/uploads/scene.png',
    ], 'Object references remain Pictures and their image paths reach the enhancer');
    assert.match(inventory, /<Picture 1>: visual identity\/appearance reference for Sentinel;/,
      'The existing identity reference keeps Picture 1');
    const objectLine = inventory.split('\n').find(line => line.startsWith('<Picture 2>:')) || '';
    assert.match(objectLine, /intent=OBJECT REFERENCE; image_intent=object; retention=fully_preserved;/,
      'The object remains a fully preserved Picture design reference');
    assert.match(objectLine, /preserve the object's design, shape, proportions, materials, colors, and visible details from this Picture/i);
    assert.match(objectLine, /ignore its source scene, framing, background, and pose/i);
    assert.match(objectLine, /follow the prompt for placement, scale, action, and count/i);
    assert.doesNotMatch(objectLine, /Subject|speaker|character identity/i,
      'The object is not represented as a character or speaker');
    assert.doesNotMatch(inventory, /Saved character "Legacy AT-AT"/,
      'Imported object metadata cannot bind it to a saved character');
    assert.match(inventory, /<Picture 3>: visual identity\/appearance reference for Knight portrait;/,
      'An identity image retains its expected Picture label');
    assert.match(inventory, /<Picture 4>: visual identity\/appearance reference for Temple scene;/,
      'Picture numbering continues unchanged after an object reference');
    assert.match(inventory, /<Audio 2>: Laser blaster; intent=SOUND EFFECT REFERENCE;/,
      'The sound effect occupies the expected Audio tag');
    assert.match(inventory, /<Audio 3>: Ambient synth style; intent=AUDIO REFERENCE;/,
      'The existing style reference keeps its tag and intent');
    assert.match(inventory, /<Audio 4>: soundtrack paired with <Video 1>/,
      'A video soundtrack follows preceding numbered audio references');
    assert.match(inventory, /Exact target soundtrack: Exact score; intent=AUDIO REUSE \/ PERFORMANCE DRIVER/,
      'The exact soundtrack keeps its unnumbered driver semantics');
    const savedCharacterLine = inventory.split('\n').find(line => line.startsWith('Saved character "Sentinel"')) || '';
    assert.match(savedCharacterLine, /<Picture 1> \+ <Audio 1>/,
      'The saved character remains bound only to its appearance and voice');
    assert.doesNotMatch(savedCharacterLine, /<Audio 2>/,
      'The sound effect is not attached to the saved character voice');
    assert.doesNotMatch(savedCharacterLine, /<Picture 2>/,
      'The object is not attached to the saved character appearance');
    const soundLine = inventory.split('\n').find(line => line.includes('intent=SOUND EFFECT REFERENCE')) || '';
    assert.match(soundLine, /explicitly requested sound effect attached to its named action/i);
    assert.match(soundLine, /do not play, copy, or loop its waveform/i);
    assert.match(soundLine, /not treat it as voice identity, music, or a performance driver/i);
    assert.match(soundLine, /synchronize the effect to its requested action rather than the recording’s original timing/i);

    await page.evaluate(async () => window.store.getState().startGeneration('queue'));
    assert.ok(generateRequest, 'The queue submission reached the mocked generation endpoint');
    assert.equal(generateRequest._queue_mode, 'held');
    assert.equal(generateRequest.minimax_h3_references.find(reference => reference.id === 'laser').audio_intent,
      'sound', 'The sound intent survives generation submission');
    const queuedObject = generateRequest.minimax_h3_references.find(reference => reference.id === 'atat-object');
    assert.equal(queuedObject.image_intent, 'object', 'The object intent survives queue submission');
    assert.equal(queuedObject.remove_background, true, 'Object background isolation survives queue submission');
    assert.equal(queuedObject.path, '/uploads/atat.png', 'The object remains an image reference in the queue payload');

    await page.evaluate(async savedParams => {
      window.store.setState({
        selectedOutputMeta: {source: 'sidecar', params: savedParams},
        selectedOutput: 0,
        outputs: [{name: 'object-reference.mp4', type: 'video', url: '/output.mp4'}],
        loadLoras: async () => {},
        loadModelOptions: async () => window.store.setState({modelOptions: window.h3Options}),
      });
      await window.store.getState().loadSettingsFromOutput();
    }, generateRequest);
    const restoredIntents = await page.evaluate(() => {
      const refs = window.store.getState().params.minimax_h3_references;
      return {
        audio: refs.find(reference => reference.id === 'laser')?.audio_intent,
        image: refs.find(reference => reference.id === 'atat-object')?.image_intent,
        background: refs.find(reference => reference.id === 'atat-object')?.remove_background,
      };
    });
    assert.deepEqual(restoredIntents, {audio: 'sound', image: 'object', background: true},
      'Output settings restore preserves object intent, background isolation and existing sound intent');
    assert.deepEqual(errors, [], 'The reference editor and store actions render without React errors');
    console.log('Object reference: mobile help, background isolation, unchanged duration, Picture inventory and image paths, saved-character exclusion, queue submission and output restore passed');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
