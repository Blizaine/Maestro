// Exercise the real App startup path against a fully intercepted fake API.
// The fixtures contain no real model files, media, or user preferences.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const playwright = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');
const host = 'maestro-startup.test';
const origin = 'http://' + host;
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));

function deferred() {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return {promise, resolve};
}

async function withTimeout(promise, ms, label) {
  let timer;
  try {
    return await Promise.race([
      promise,
      new Promise((_, reject) => { timer = setTimeout(() => reject(new Error('Timed out waiting for ' + label)), ms); }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

const models = [
  {model_type: 'fixture-video', name: 'Fixture Video', family: 'wan', architecture: 'fixture_video', is_i2v: true, is_t2v: true, guidance_max_phases: 1, fps: 16, is_downloaded: true},
  {model_type: 'fixture-h3', name: 'Fixture H3 References', family: 'wan', architecture: 'fixture_h3', is_i2v: false, is_t2v: true, guidance_max_phases: 1, fps: 16, is_downloaded: true, omni_reference: true},
  {model_type: 'fixture-video-b', name: 'Fixture Video B', family: 'wan', architecture: 'fixture_video_b', is_i2v: true, is_t2v: true, guidance_max_phases: 1, fps: 16, is_downloaded: true},
  {model_type: 'fixture-image', name: 'Fixture Image', family: 'flux', architecture: 'fixture_image', is_i2v: false, is_t2v: false, guidance_max_phases: 1, fps: 16, is_downloaded: true},
  {model_type: 'fixture-audio', name: 'Fixture Audio', family: 'tts', architecture: 'fixture_audio', is_i2v: false, is_t2v: false, guidance_max_phases: 1, fps: 16, is_downloaded: true, supports_audio: true, generates_audio: true},
];

const catalogue = {
  families: [
    {id: 'wan', label: 'Fixture Video', order: 1},
    {id: 'flux', label: 'Fixture Image', order: 2},
    {id: 'tts', label: 'Fixture Audio', order: 3},
  ],
  models,
};
const refreshedCatalogue = {
  ...catalogue,
  models: models.map(model => model.model_type === 'fixture-video'
    ? {...model, name: 'Fresh Fixture Video Catalogue'}
    : model),
};

function modelOptions(modelType) {
  const model = models.find(candidate => candidate.model_type === modelType) || models[0];
  const audio = model.family === 'tts';
  const omniReference = model.model_type === 'fixture-h3';
  return {
    model_type: model.model_type,
    architecture: model.architecture,
    guidance_max_phases: 1,
    default_guidance_scale: 2.5,
    lock_guidance_phases: false,
    sliding_window: false,
    motion_amplitude: false,
    flow_shift: false,
    tea_cache: false,
    returns_audio: false,
    any_audio_prompt: false,
    audio_scale_name: '',
    lock_inference_steps: false,
    lock_guidance_scale: false,
    no_negative_prompt: false,
    i2v_class: !audio && model.family === 'wan',
    t2v_class: !audio && model.family === 'wan',
    image_outputs: model.family === 'flux',
    supports_end_frame: false,
    guide_preprocessing: null,
    guide_custom_choices: null,
    image_ref_choices: null,
    audio_prompt_type_sources: null,
    background_removal_label: null,
    sample_solvers: null,
    fps: 16,
    resolution_preset_order: ['720p'],
    resolution_presets: { '720p': {label: '720p', values: {'16:9': '1280x720'}} },
    supports_auto_aspect: true,
    audio_only: audio,
    omni_reference: omniReference,
  };
}

function studioPreferences(mode, configured = false) {
  return {
    configured,
    ...(configured ? {
      generation_mode: mode,
      selected_model_per_mode: {video: 'fixture-video', image: 'fixture-image', audio: 'fixture-audio'},
      selected_model_per_audio_sub_mode: {speech: 'fixture-audio'},
      audio_sub_mode: 'speech',
      studio_video_workflow: 'frames',
      studio_image_workflow: 'generate',
      inference_steps_per_model: {},
      h3_optimizations: {},
    } : {}),
  };
}

async function buildBundle() {
  const bundle = await esbuild.build({
    stdin: {contents: [
      "import React from 'react'; import {createRoot} from 'react-dom/client';",
      "import App from './src/App'; import {useStore} from './src/stores/useStore';",
      "window.store = useStore; const app = <App/>;",
      "createRoot(document.getElementById('root')).render(location.search.includes('strict') ? <React.StrictMode>{app}</React.StrictMode> : app);",
    ].join('\n'), resolveDir: path.join(root, 'ui'), loader: 'tsx'},
    bundle: true,
    write: false,
    jsx: 'automatic',
    define: {'process.env.NODE_ENV': '"development"'},
    logLevel: 'silent',
  });
  const assets = path.join(root, 'ui/dist/assets');
  const cssFile = fs.readdirSync(assets).find(name => name.endsWith('.css'));
  assert.ok(cssFile, 'the built UI stylesheet is present');
  return {
    js: bundle.outputFiles[0].text,
    css: fs.readFileSync(path.join(assets, cssFile), 'utf8'),
  };
}

async function openFixture(browser, bundle, scenario, viewport = {width: 1360, height: 900}) {
  const context = await browser.newContext({viewport});
  const page = await context.newPage();
  const errors = [];
  const consoleErrors = [];
  const externalRequests = [];
  const unexpectedApiPaths = [];
  const mutations = [];
  const requests = [];
  const active = new Map();
  const maxInFlight = new Map();
  const counts = new Map();
  const llmModelsStarted = deferred();
  const preferencesRetryStarted = deferred();
  const optionsRetryStarted = deferred();
  const defaultsRetryStarted = deferred();
  const preferencesRetryGate = deferred();
  const optionsRetryGate = deferred();
  const defaultsRetryGate = deferred();
  let llmModelsSettled = false;
  let mainFrameNavigations = 0;

  page.on('pageerror', error => errors.push(error.stack || error.message));
  page.on('console', message => { if (message.type() === 'error') consoleErrors.push(message.text()); });
  page.on('framenavigated', frame => { if (frame === page.mainFrame()) mainFrameNavigations += 1; });
  await page.addInitScript(({mode, deferAutoRetry, initialInnerHeight}) => {
    if (deferAutoRetry) {
      const nativeSetTimeout = window.setTimeout.bind(window);
      window.setTimeout = (callback, delay, ...args) => nativeSetTimeout(
        callback,
        delay === 1000 ? 30000 : delay,
        ...args,
      );
    }
    localStorage.setItem('maestro_welcome_seen_v1', '1');
    localStorage.setItem('maestro_mode_settings', JSON.stringify({
      generationMode: mode,
      selectedModelPerMode: {video: 'fixture-video', image: 'fixture-image', audio: 'fixture-audio'},
      savedParamsPerMode: {}, savedLoraPerMode: {}, savedPromptPerMode: {},
    }));
    if (Number.isFinite(initialInnerHeight)) {
      Object.defineProperty(window, 'innerHeight', {configurable: true, value: initialInnerHeight});
    }
  }, {
    mode: scenario.savedMode || 'video',
    deferAutoRetry: scenario.deferAutoRetry === true,
    initialInnerHeight: scenario.initialInnerHeight,
  });

  const json = (route, body, status = 200) => route.fulfill({
    status,
    contentType: 'application/json',
    body: JSON.stringify(body),
  });

  await page.route('**/*', async route => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.origin !== origin) {
      externalRequests.push(request.url());
      await route.abort();
      return;
    }
    if (url.pathname === '/') {
      return route.fulfill({contentType: 'text/html', body:
        '<meta name="viewport" content="width=device-width,initial-scale=1">' +
        '<style>html,body,#root{height:100%;margin:0}</style>' +
        '<link rel="stylesheet" href="/app.css"><div id="root"></div>' +
        '<script src="/bundle.js"></script>'});
    }
    if (url.pathname === '/bundle.js') return route.fulfill({contentType: 'application/javascript', body: bundle.js});
    if (url.pathname === '/app.css') return route.fulfill({contentType: 'text/css', body: bundle.css});
    if (!url.pathname.startsWith('/api/')) {
      return route.fulfill({status: 200, contentType: 'application/octet-stream', body: ''});
    }

    const method = request.method();
    const pathname = url.pathname;
    const key = method + ' ' + pathname;
    const entry = {method, pathname, search: url.search, body: request.postData() || null, startedAt: Date.now()};
    requests.push(entry);
    counts.set(key, (counts.get(key) || 0) + 1);
    const inFlight = (active.get(key) || 0) + 1;
    active.set(key, inFlight);
    maxInFlight.set(key, Math.max(maxInFlight.get(key) || 0, inFlight));
    if (method !== 'GET' && method !== 'HEAD' && method !== 'OPTIONS') mutations.push(entry);

    try {
      if (method === 'GET' && pathname === '/api/v1/models') {
        const attempt = counts.get(key);
        if (scenario.failModelsFirst && attempt === 1) {
          entry.failed = true;
          entry.completedAt = Date.now();
          return json(route, {detail: 'fixture unavailable'}, 503);
        }
        if (scenario.failModelsAfterReady && attempt === 2) {
          entry.failed = true;
          entry.completedAt = Date.now();
          return json(route, {detail: 'fixture unavailable'}, 503);
        }
        entry.failed = false;
        entry.completedAt = Date.now();
        if (scenario.freshCatalogueAfterFailure && attempt >= 3) return json(route, refreshedCatalogue);
        return json(route, catalogue);
      }
      if (method === 'GET' && pathname === '/api/v1/characters') return json(route, {characters: []});
      if (method === 'GET' && pathname === '/api/v1/services-config') {
        return json(route, {configured: true, nsfw_mode: false, auto_performance: false});
      }
      if (method === 'GET' && pathname === '/api/v1/model-visibility') {
        return json(route, {configured: true, enabled_models: models.map(model => model.model_type), mode_overrides: {}, initialized_mature_models: [], defaults_version: 23});
      }
      if (method === 'GET' && pathname === '/api/v1/presets') return json(route, {presets: []});
      if (method === 'GET' && pathname === '/api/v1/h3-window-overrides') return json(route, {overrides: {}});
      if (pathname === '/api/v1/studio-preferences' && method === 'GET') {
        const attempt = counts.get(key);
        if (scenario.failPreferencesUntilRecovered && !scenario.preferencesRecovered) {
          entry.failed = true;
          entry.completedAt = Date.now();
          return json(route, {detail: 'fixture unavailable'}, 503);
        }
        if (scenario.failPreferencesFirst && attempt === 1) return json(route, {detail: 'fixture unavailable'}, 503);
        if (scenario.gatePreferencesRetry && attempt === 2) {
          preferencesRetryStarted.resolve();
          await preferencesRetryGate.promise;
        }
        entry.failed = false;
        entry.completedAt = Date.now();
        return json(route, studioPreferences(scenario.preferenceMode || 'video', scenario.preferencesConfigured === true));
      }
      if (pathname === '/api/v1/studio-preferences' && method === 'PUT') {
        return json(route, {configured: true});
      }
      if (method === 'GET' && pathname.startsWith('/api/v1/model-options/')) {
        const attempt = counts.get(key);
        if (scenario.failOptionsFirst && attempt === 1) return json(route, {detail: 'fixture unavailable'}, 503);
        if (scenario.gateOptionsRetry && attempt === 2) {
          optionsRetryStarted.resolve();
          await optionsRetryGate.promise;
        }
        return json(route, modelOptions(decodeURIComponent(pathname.slice('/api/v1/model-options/'.length))));
      }
      if (method === 'GET' && pathname.startsWith('/api/v1/defaults/')) {
        const modelType = decodeURIComponent(pathname.slice('/api/v1/defaults/'.length));
        const attempt = counts.get(key);
        if ((scenario.failDefaultsFirst || scenario.failDefaultsModelsFirst?.includes(modelType)) && attempt === 1) {
          entry.failed = true;
          entry.completedAt = Date.now();
          return json(route, {detail: 'fixture unavailable'}, 503);
        }
        if ((scenario.gateDefaultsRetry || scenario.gateDefaultsModelRetry === modelType) && attempt === 2) {
          defaultsRetryStarted.resolve();
          await defaultsRetryGate.promise;
        }
        entry.failed = false;
        entry.completedAt = Date.now();
        return json(route, {num_inference_steps: 9, guidance_scale: 2.5});
      }
      if (method === 'GET' && pathname === '/api/v1/llm/models') {
        llmModelsStarted.resolve();
        if (scenario.slowLlmModels) await pause(scenario.slowLlmModels);
        llmModelsSettled = true;
        return json(route, {models: []});
      }
      if (method === 'GET' && pathname === '/api/v1/llm/status') return json(route, {loaded: false, status: 'unloaded'});
      if (method === 'GET' && pathname === '/api/v1/workspaces') return json(route, {active: 'default', workspaces: [{name: 'default', path: 'default'}]});
      if (method === 'GET' && pathname === '/api/v1/outputs') return json(route, {outputs: [], total: 0});
      if (method === 'GET' && pathname === '/api/v1/system-config') return json(route, {});
      if (method === 'GET' && pathname === '/api/v1/system-detect') return json(route, {hardware: {gpu_vram_gb: 0}, checks: [], recommendation: {}});
      if (method === 'GET' && pathname === '/api/v1/system/preflight') return json(route, {ok: true, checks: []});
      if (method === 'GET' && pathname === '/api/v1/system-stats') return json(route, {
        gpu: {available: false, percent: 0, vram_used_gb: 0, vram_total_gb: 0, vram_percent: 0},
        ram: {percent: 0, used_gb: 0, total_gb: 0}, cpu: {percent: 0},
        model: {name: null, model_type: null, loaded: false},
      });
      if (method === 'GET' && pathname === '/api/v1/downloads/active') return json(route, {downloads: [], version: 'fixture'});
      if (method === 'GET' && pathname === '/api/v1/models/downloads/status') return json(route, {downloads: {}});
      if (method === 'GET' && pathname === '/api/v1/media-flow/capabilities') return json(route, {
        neural_rendering: {available: false, reason: 'Isolated browser test'},
        frame_generation: {available: false, factors: [], reason: 'Isolated browser test'},
      });
      if (method === 'GET' && pathname === '/api/v1/jobs') return json(route, {jobs: []});
      if (method === 'GET' && pathname === '/api/v1/director/queue') return json(route, {entries: [], running: false, paused: false, current: null});
      if (method === 'GET' && pathname === '/api/v1/director/pipelines') return json(route, {pipelines: []});
      if (method === 'GET' && pathname === '/api/v1/loras/installed') return json(route, {loras: []});
      if (method === 'GET' && pathname.startsWith('/api/v1/loras/') && pathname.endsWith('/details')) {
        return json(route, {loras: [], guidance_max_phases: 1, manifest_last_check_at: null});
      }
      if (method === 'GET' && pathname.startsWith('/api/v1/loras/')) return json(route, {loras: [], guidance_max_phases: 1});
      if (method === 'POST' && pathname === '/api/v1/loras/check-updates') {
        return json(route, {checked: 0, updates_available: 0, errors: [], skipped: true, reason: 'fresh'});
      }
      if (method === 'GET' && pathname === '/api/v1/resolutions') return json(route, {resolutions: []});

      // This catches any incidental App reads so the browser can never fall
      // through to a real API. New startup dependencies should add a focused
      // fixture above; the collected path is included in an assertion below.
      unexpectedApiPaths.push(key + (url.search || ''));
      return json(route, {});
    } catch (error) {
      unexpectedApiPaths.push(key + ' route error: ' + error.message);
      return json(route, {});
    } finally {
      active.set(key, Math.max(0, (active.get(key) || 1) - 1));
    }
  });

  await page.goto(origin + '/' + (scenario.strict ? '?strict=1' : ''), {waitUntil: 'domcontentloaded'});
  return {
    context, page, errors, consoleErrors, externalRequests, unexpectedApiPaths, mutations,
    requests, maxInFlight, counts, llmModelsStarted, preferencesRetryStarted,
    optionsRetryStarted, defaultsRetryStarted, preferencesRetryGate, optionsRetryGate,
    defaultsRetryGate,
    get llmModelsSettled() { return llmModelsSettled; },
    get mainFrameNavigations() { return mainFrameNavigations; },
  };
}

async function waitForReady(page, timeout = 6000, guidanceScale = 2.5) {
  await page.waitForFunction(expectedGuidanceScale => {
    const state = window.store && window.store.getState();
    return state && state.modelsLoaded && !state.modelsLoading && !state.modelsLoadError
      && !state.modelOptionsLoading && state.modelOptions?.model_type === state.params.model_type
      && state.params.num_inference_steps === 9 && state.params.guidance_scale === expectedGuidanceScale;
  }, guidanceScale, {timeout});
}

function assertIsolated(fixture) {
  assert.deepEqual(fixture.externalRequests, [], 'all browser requests stay on the intercepted fake origin');
  const expectedBootstrapMutations = new Set([
    'PUT /api/v1/studio-preferences',
    'POST /api/v1/loras/check-updates',
  ]);
  const unexpectedMutations = fixture.mutations.filter(entry => !expectedBootstrapMutations.has(entry.method + ' ' + entry.pathname));
  assert.deepEqual(unexpectedMutations, [], 'startup did not issue a generation, settings, or other backend mutation');
  assert.deepEqual(fixture.errors, [], 'the App produced no uncaught browser errors');
  assert.deepEqual(fixture.unexpectedApiPaths, [], 'all API reads used by this fixture have explicit synthetic responses');
}

async function testControlsBeforeBackground(bundle, browser) {
  const fixture = await openFixture(browser, bundle, {slowLlmModels: 1500});
  const {page} = fixture;
  try {
    await withTimeout(fixture.llmModelsStarted.promise, 4500, 'the deliberately slow background LLM models read');
    await waitForReady(page);
    assert.equal(fixture.llmModelsSettled, false, 'model controls hydrate while the background LLM models read is pending');
    const sidebar = page.locator('.maestro-sidebar');
    for (const label of ['Video', 'Image', 'Audio']) {
      const button = sidebar.getByRole('button', {name: label, exact: true});
      assert.equal(await button.isVisible(), true, label + ' mode control is visible on desktop');
      assert.equal(await button.isDisabled(), false, label + ' mode control is enabled after controls hydrate');
    }
    assert.deepEqual(await page.evaluate(() => {
      const state = window.store.getState();
      return [state.params.model_type, state.modelOptions.model_type, state.params.num_inference_steps, state.params.guidance_scale];
    }), ['fixture-video', 'fixture-video', 9, 2.5], 'the selected model, options, and tuned defaults are ready independently of LLM discovery');
    assert.equal(fixture.counts.get('GET /api/v1/system-detect') || 0, 0, 'hidden Settings has not loaded hardware detection');

    await page.getByRole('button', {name: 'Settings', exact: true}).click();
    await page.waitForFunction(() => window.store.getState().systemDetect !== null, null, {timeout: 3000});
    assert.equal(fixture.counts.get('GET /api/v1/system-detect'), 1, 'hardware detection starts when Settings is first opened');
    await page.evaluate(() => window.store.getState().setSettingsOpen(false));
    await page.setViewportSize({width: 390, height: 844});
    await page.getByRole('button', {name: 'Open sidecar', exact: true}).click();
    for (const label of ['Video', 'Image', 'Audio']) {
      assert.equal(await page.locator('.maestro-sidebar').getByRole('button', {name: label, exact: true}).isVisible(), true,
        label + ' mode control is visible on mobile after opening the sidecar');
    }
    await withTimeout((async () => {
      while (!fixture.llmModelsSettled) await pause(20);
    })(), 2500, 'the isolated slow LLM response to finish');
    assertIsolated(fixture);
    console.log('PASS App startup: desktop/mobile controls and model defaults are ready before slow background reads');
  } catch (error) {
    console.error('startup fixture diagnostics', JSON.stringify({
      requests: fixture.requests.map(entry => entry.method + ' ' + entry.pathname + entry.search),
      apiFallbacks: fixture.unexpectedApiPaths,
      consoleErrors: fixture.consoleErrors,
      browserErrors: fixture.errors,
      state: await page.evaluate(() => {
        const state = window.store?.getState();
        return state && {modelsLoaded: state.modelsLoaded, modelsLoading: state.modelsLoading,
          modelsLoadError: state.modelsLoadError, params: state.params, options: state.modelOptions};
      }).catch(() => null),
    }));
    throw error;
  } finally {
    await fixture.context.close();
  }
}

async function setVisualViewport(page, {height, offsetTop}) {
  await page.evaluate(({height, offsetTop}) => {
    const viewport = window.visualViewport;
    Object.defineProperties(viewport, {
      height: {configurable: true, value: height},
      offsetTop: {configurable: true, value: offsetTop},
    });
    viewport.dispatchEvent(new Event('resize'));
    viewport.dispatchEvent(new Event('scroll'));
  }, {height, offsetTop});
}

async function prepareMobileReferenceWorkflow(page) {
  await waitForReady(page);
  await page.evaluate(() => window.store.getState().selectModel('fixture-h3'));
  await waitForReady(page);
  await page.evaluate(() => window.store.getState().setStudioVideoWorkflow('references'));
  await page.waitForFunction(() => window.store.getState().studioVideoWorkflow === 'references');
  await page.getByRole('button', {name: 'Open sidecar', exact: true}).click();
  await page.locator('.maestro-sidebar').waitFor({state: 'visible'});
}

async function assertMobileReferenceControlsVisible(page, message) {
  const sidebar = page.locator('.maestro-sidebar');
  const visible = {
    videoMode: await sidebar.getByRole('button', {name: 'Video', exact: true}).isVisible(),
    imageMode: await sidebar.getByRole('button', {name: 'Image', exact: true}).isVisible(),
    audioMode: await sidebar.getByRole('button', {name: 'Audio', exact: true}).isVisible(),
    workflow: await sidebar.getByRole('button', {name: 'Workflow: References', exact: true}).isVisible(),
    referenceInput: await sidebar.locator('.studio-media-add button[aria-label="Add reference"]').isVisible(),
    hardware: await sidebar.locator('.studio-hardware').isVisible(),
  };
  assert.deepEqual(visible, {
    videoMode: true,
    imageMode: true,
    audioMode: true,
    workflow: true,
    referenceInput: true,
    hardware: true,
  }, message);
}

async function testMobileViewportWithoutEditableKeepsControls(bundle, browser) {
  const fixture = await openFixture(browser, bundle, {initialInnerHeight: 1100}, {width: 390, height: 844});
  const {page} = fixture;
  try {
    await prepareMobileReferenceWorkflow(page);
    assert.equal(await page.evaluate(() => {
      const active = document.activeElement;
      return !active || !active.matches('textarea, input:not([type="range"]):not([type="checkbox"]):not([type="file"]), [contenteditable="true"]');
    }), true,
      'the mobile sidecar opens without an editable field focused');
    await assertMobileReferenceControlsVisible(page,
      'a stale larger innerHeight on cold load must not hide modes, workflow, references, or hardware');

    await setVisualViewport(page, {height: 700, offsetTop: 42});
    await page.waitForTimeout(50);
    await assertMobileReferenceControlsVisible(page,
      'address-bar viewport resize and offset changes without editable focus keep sidecar controls visible');

    await page.evaluate(() => {
      window.dispatchEvent(new Event('pageshow'));
      document.dispatchEvent(new Event('visibilitychange'));
    });
    await page.waitForTimeout(50);
    await assertMobileReferenceControlsVisible(page,
      'returning to a visible page without editable focus keeps sidecar controls visible');
    assert.equal(fixture.mainFrameNavigations, 1, 'viewport recovery does not require a page refresh');
    assertIsolated(fixture);
    console.log('PASS App startup: mobile sidecar controls remain visible through no-focus viewport changes');
  } finally {
    await fixture.context.close();
  }
}

async function testMobileFocusedKeyboardCompactsAndRestores(bundle, browser) {
  const fixture = await openFixture(browser, bundle, {}, {width: 390, height: 844});
  const {page} = fixture;
  try {
    await prepareMobileReferenceWorkflow(page);
    await assertMobileReferenceControlsVisible(page, 'mobile controls start visible before the keyboard opens');

    await page.evaluate(() => window.store.getState().setParam('minimax_h3_references', [{
      id: 'viewport-fixture-picture', type: 'image', filename: 'fixture.png', path: '/fixture.png',
      url: 'data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs=', role: 'Fixture person', image_intent: 'identity',
    }]));
    await page.getByRole('button', {name: 'Edit Picture 1 reference', exact: true}).click();
    const referenceName = page.getByRole('textbox', {name: 'Picture 1 name'});
    await referenceName.waitFor({state: 'visible'});

    const prompt = page.locator('textarea[placeholder="Describe your video..."]');
    await referenceName.focus();
    await setVisualViewport(page, {height: 520, offsetTop: 0});
    await page.waitForFunction(() => document.querySelector('.maestro-sidebar')?.getAttribute('data-keyboard-open') === 'true');
    assert.equal(await referenceName.evaluate(element => document.activeElement === element), true,
      'the reference name remains focused while the simulated mobile keyboard is open');
    assert.equal(await page.locator('.maestro-sidebar .studio-workflow-header').isVisible(), false,
      'a focused keyboard compacts the workflow header');
    assert.equal(await page.locator('.maestro-sidebar .studio-hardware').isVisible(), false,
      'a focused keyboard compacts the hardware footer');

    const focusTransitions = await page.evaluate(async () => {
      const values = [];
      const sidebar = document.querySelector('.maestro-sidebar');
      const observer = new MutationObserver(() => values.push(sidebar?.getAttribute('data-keyboard-open')));
      observer.observe(sidebar, {attributes: true, attributeFilter: ['data-keyboard-open']});
      const prompt = document.querySelector('textarea[placeholder="Describe your video..."]');
      const referenceName = document.querySelector('input[aria-label="Picture 1 name"]');
      referenceName?.focus();
      prompt?.focus();
      await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      observer.disconnect();
      return values;
    });
    assert.equal(await prompt.evaluate(element => document.activeElement === element), true,
      'the focus handoff lands on the prompt inside the sidecar');
    assert.equal(focusTransitions.includes('false'), false,
      'moving focus between text fields does not briefly clear keyboard compaction');
    assert.equal(await page.locator('.maestro-sidebar').getAttribute('data-keyboard-open'), 'true',
      'keyboard compaction remains active after the focus handoff');

    await prompt.evaluate(element => element.blur());
    await page.waitForFunction(() => document.querySelector('.maestro-sidebar')?.getAttribute('data-keyboard-open') === 'false');
    await assertMobileReferenceControlsVisible(page,
      'blurring the active field restores all sidecar controls before the viewport changes');
    await setVisualViewport(page, {height: 844, offsetTop: 0});
    await page.waitForFunction(() => document.querySelector('.maestro-sidebar')?.getAttribute('data-keyboard-open') === 'false');
    await assertMobileReferenceControlsVisible(page,
      'closing the keyboard keeps all sidecar controls restored without refresh');
    assert.equal(fixture.mainFrameNavigations, 1, 'keyboard close recovery does not navigate or refresh the page');
    assertIsolated(fixture);
    console.log('PASS App startup: focused mobile keyboard compacts and restores the sidecar controls');
  } finally {
    await fixture.context.close();
  }
}

async function testStrictModeModelsRecovery(bundle, browser) {
  const fixture = await openFixture(browser, bundle, {strict: true, failModelsFirst: true});
  try {
    await waitForReady(fixture.page, 7000);
    assert.equal(fixture.counts.get('GET /api/v1/models'), 2, 'one failed catalog read is followed by one fresh recovery read');
    assert.equal(fixture.maxInFlight.get('GET /api/v1/models'), 1, 'StrictMode does not create concurrent identical catalog reads');
    assert.equal(fixture.mainFrameNavigations, 1, 'startup recovers without refreshing the page');
    assertIsolated(fixture);
    console.log('PASS App startup: StrictMode catalog failure recovers without refresh or duplicate concurrent reads');
  } finally {
    await fixture.context.close();
  }
}

async function testPreferenceHydrationGatesWrites(bundle, browser) {
  const fixture = await openFixture(browser, bundle, {
    savedMode: 'image',
    failPreferencesFirst: true,
    gatePreferencesRetry: true,
    preferenceMode: 'image',
    preferencesConfigured: true,
  });
  try {
    await fixture.page.getByRole('alert').waitFor({state: 'visible', timeout: 5000});
    await withTimeout(fixture.preferencesRetryStarted.promise, 5000, 'the retrying Studio preferences read');
    const prefWritesWhileReadPending = fixture.mutations.filter(entry => entry.method === 'PUT' && entry.pathname === '/api/v1/studio-preferences');
    assert.equal(prefWritesWhileReadPending.length, 0, 'failed preferences are not replaced by defaults while the successful retry is pending');
    const savedDuringRetry = await fixture.page.evaluate(() => JSON.parse(localStorage.getItem('maestro_mode_settings')));
    assert.equal(savedDuringRetry.generationMode, 'image', 'the saved mode remains intact while preferences are unavailable');
    assert.equal(savedDuringRetry.selectedModelPerMode.image, 'fixture-image', 'the saved model remains intact while preferences are unavailable');

    fixture.preferencesRetryGate.resolve();
    await waitForReady(fixture.page);
    const hydrated = await fixture.page.evaluate(() => ({
      mode: window.store.getState().generationMode,
      model: window.store.getState().params.model_type,
      selected: window.store.getState().selectedModelPerMode.image,
    }));
    assert.deepEqual(hydrated, {mode: 'image', model: 'fixture-image', selected: 'fixture-image'}, 'successful preferences restore the saved model and mode');
    await fixture.page.waitForFunction(() => JSON.parse(localStorage.getItem('maestro_mode_settings') || '{}').generationMode === 'image', null, {timeout: 2000});
    await pause(80);
    const writes = fixture.mutations.filter(entry => entry.method === 'PUT' && entry.pathname === '/api/v1/studio-preferences');
    assert.ok(writes.length > 0, 'the successful hydration can persist the validated preference record');
    const writeIndex = fixture.requests.indexOf(writes[0]);
    const successfulReadIndex = fixture.requests.findIndex(entry => entry.method === 'GET'
      && entry.pathname === '/api/v1/studio-preferences' && entry.completedAt);
    assert.ok(successfulReadIndex >= 0 && writeIndex > successfulReadIndex, 'the preference write happens after successful preference hydration');
    const writeBody = JSON.parse(writes[0].body);
    assert.equal(writeBody.generation_mode, 'image');
    assert.equal(writeBody.selected_model_per_mode.image, 'fixture-image');
    assertIsolated(fixture);
    console.log('PASS App startup: preference failures cannot persist defaults before hydration');
  } finally {
    fixture.preferencesRetryGate.resolve();
    await fixture.context.close();
  }
}

async function testCachedPreferencesRecoverOnOnline(bundle, browser) {
  const scenario = {
    savedMode: 'image',
    failPreferencesUntilRecovered: true,
    preferenceMode: 'video',
    preferencesConfigured: true,
  };
  const fixture = await openFixture(browser, bundle, scenario);
  const {page} = fixture;
  try {
    await page.waitForFunction(() => {
      const state = window.store && window.store.getState();
      return state && !state.modelsLoading && !state.modelsLoaded
        && state.modelsLoadError?.includes('Using cached preferences')
        && state.generationMode === 'image'
        && state.params.model_type === 'fixture-image'
        && state.modelOptions?.model_type === 'fixture-image';
    }, null, {timeout: 5000});

    const cachedState = await page.evaluate(() => {
      const state = window.store.getState();
      return {
        loaded: state.modelsLoaded,
        mode: state.generationMode,
        model: state.params.model_type,
        selectedModel: state.selectedModelPerMode.image,
        optionsModel: state.modelOptions.model_type,
      };
    });
    assert.deepEqual(cachedState, {
      loaded: false,
      mode: 'image',
      model: 'fixture-image',
      selectedModel: 'fixture-image',
      optionsModel: 'fixture-image',
    }, 'cached mode, model, and options remain visible while preferences are unavailable');
    assert.equal(await page.getByRole('button', {name: 'Retry', exact: true}).isVisible(), true,
      'the cached fallback exposes a Retry action');
    assert.equal(fixture.counts.get('GET /api/v1/studio-preferences'), 1,
      'the first preference read is still failing before the browser recovery signal');
    assert.equal(fixture.requests.find(entry => entry.method === 'GET'
      && entry.pathname === '/api/v1/studio-preferences').failed, true,
    'the cached view is backed by a failed preferences response');
    assert.equal(fixture.mutations.filter(entry => entry.method === 'PUT'
      && entry.pathname === '/api/v1/studio-preferences').length, 0,
    'cached preferences are never written back as server defaults during the outage');

    const cachedPrompt = page.locator('textarea[placeholder="Describe your image..."]');
    await cachedPrompt.fill('keep this cached-mode prompt through reconnect');
    assert.equal(await cachedPrompt.inputValue(), 'keep this cached-mode prompt through reconnect');
    assert.equal(fixture.mutations.filter(entry => entry.method === 'PUT'
      && entry.pathname === '/api/v1/studio-preferences').length, 0,
    'typing while cached settings are active does not persist incomplete preferences');

    scenario.preferencesRecovered = true;
    await page.evaluate(() => window.dispatchEvent(new Event('online')));
    await waitForReady(page);
    const recoveredState = await page.evaluate(() => {
      const state = window.store.getState();
      return {mode: state.generationMode, model: state.params.model_type, loaded: state.modelsLoaded};
    });
    assert.deepEqual(recoveredState, {mode: 'video', model: 'fixture-video', loaded: true},
      'the hydrated server preference record supersedes the stale browser cache');
    assert.equal(await page.locator('textarea[placeholder="Describe your video..."]').inputValue(),
      'keep this cached-mode prompt through reconnect', 'the prompt typed during the outage survives preference hydration');

    const preferenceReads = fixture.requests.filter(entry => entry.method === 'GET'
      && entry.pathname === '/api/v1/studio-preferences');
    assert.ok(preferenceReads.length >= 2, 'the online signal causes a fresh preferences read');
    const successfulReadIndex = fixture.requests.findIndex(entry => entry.method === 'GET'
      && entry.pathname === '/api/v1/studio-preferences' && entry.failed === false);
    assert.ok(successfulReadIndex >= 0, 'the recovered server preference response is recorded');
    const preferenceWrites = fixture.mutations.filter(entry => entry.method === 'PUT'
      && entry.pathname === '/api/v1/studio-preferences');
    if (preferenceWrites.length) {
      assert.ok(fixture.requests.indexOf(preferenceWrites[0]) > successfulReadIndex,
        'preference persistence only starts after the recovered server read');
      const writeBody = JSON.parse(preferenceWrites[0].body);
      assert.equal(writeBody.generation_mode, 'video');
      assert.equal(writeBody.selected_model_per_mode.video, 'fixture-video');
    }
    assert.equal(fixture.maxInFlight.get('GET /api/v1/studio-preferences'), 1,
      'the reconnect does not issue concurrent identical preference reads');
    assertIsolated(fixture);
    console.log('PASS App startup: cached preferences remain usable through outage and online recovery wins');
  } catch (error) {
    console.error('cached preferences recovery diagnostics', JSON.stringify({
      requests: fixture.requests.map(entry => entry.method + ' ' + entry.pathname + entry.search + (entry.failed ? ' (503)' : '')),
      apiFallbacks: fixture.unexpectedApiPaths,
      consoleErrors: fixture.consoleErrors,
      browserErrors: fixture.errors,
      state: await page.evaluate(() => {
        const state = window.store?.getState();
        return state && {modelsLoaded: state.modelsLoaded, modelsLoading: state.modelsLoading,
          modelsLoadError: state.modelsLoadError, generationMode: state.generationMode,
          params: state.params, options: state.modelOptions};
      }).catch(() => null),
    }));
    throw error;
  } finally {
    await fixture.context.close();
  }
}

async function testOptionsRecoveryKeepsPrompt(bundle, browser) {
  const fixture = await openFixture(browser, bundle, {failOptionsFirst: true, gateOptionsRetry: true});
  const {page} = fixture;
  try {
    await page.getByRole('alert').waitFor({state: 'visible', timeout: 5000});
    const prompt = page.locator('textarea[placeholder="Describe your video..."]');
    await prompt.fill('keep this draft while controls recover');
    assert.equal(await prompt.inputValue(), 'keep this draft while controls recover', 'the prompt accepts text while model options are unavailable');
    await page.evaluate(() => window.store.getState().setParam('guidance_scale', 6.4));
    assert.equal(await page.evaluate(() => window.store.getState().params.guidance_scale), 6.4,
      'a guidance edit made before Retry is retained while the error is visible');
    await page.getByRole('button', {name: 'Retry', exact: true}).click();
    await withTimeout(fixture.optionsRetryStarted.promise, 3000, 'the retrying active model options read');
    assert.equal(await page.evaluate(() => window.store.getState().params.guidance_scale), 6.4,
      'a previously successful defaults read is not reapplied over the guidance edit at retry start');
    await page.evaluate(() => window.store.getState().setParam('guidance_scale', 7.2));
    assert.equal(await prompt.inputValue(), 'keep this draft while controls recover', 'retrying model options leaves the typed prompt untouched');
    fixture.optionsRetryGate.resolve();
    await waitForReady(page, 6000, 7.2);
    assert.equal(await prompt.inputValue(), 'keep this draft while controls recover', 'the prompt remains after the active model options recover');
    assert.equal(await page.evaluate(() => window.store.getState().params.guidance_scale), 7.2,
      'the guidance edit made during the pending options retry survives recovery');
    assert.equal(fixture.counts.get('GET /api/v1/model-options/fixture-video'), 2, 'one failed active options read is followed by one fresh read');
    assertIsolated(fixture);
    console.log('PASS App startup: active model options recover while preserving the prompt and guidance edits');
  } finally {
    fixture.optionsRetryGate.resolve();
    await fixture.context.close();
  }
}

async function testDefaultsRecoveryKeepsGuidance(bundle, browser) {
  const fixture = await openFixture(browser, bundle, {failDefaultsFirst: true, gateDefaultsRetry: true});
  const {page} = fixture;
  try {
    await page.getByRole('alert').waitFor({state: 'visible', timeout: 5000});
    assert.equal(fixture.requests.find(entry => entry.method === 'GET'
      && entry.pathname === '/api/v1/defaults/fixture-video').failed, true,
    'the startup error is caused by a failed active model defaults read');
    await page.evaluate(() => window.store.getState().setParam('guidance_scale', 6.4));
    assert.equal(await page.evaluate(() => window.store.getState().params.guidance_scale), 6.4,
      'a guidance edit made before Retry is present while defaults are unavailable');

    await page.getByRole('button', {name: 'Retry', exact: true}).click();
    await withTimeout(fixture.defaultsRetryStarted.promise, 3000, 'the retrying active model defaults read');
    assert.equal(await page.evaluate(() => window.store.getState().params.guidance_scale), 6.4,
      'the pre-retry guidance value remains visible while the defaults response is pending');
    await page.evaluate(() => window.store.getState().setParam('guidance_scale', 7.2));
    fixture.defaultsRetryGate.resolve();
    await waitForReady(page, 6000, 7.2);
    assert.equal(await page.evaluate(() => window.store.getState().params.guidance_scale), 7.2,
      'a guidance edit made during the pending defaults retry wins over the late defaults response');
    assert.equal(fixture.counts.get('GET /api/v1/defaults/fixture-video'), 2,
      'one failed defaults read is followed by a fresh recovery read');
    assert.equal(fixture.requests.filter(entry => entry.method === 'GET'
      && entry.pathname === '/api/v1/defaults/fixture-video')[1].failed, false,
    'the second defaults read succeeds');
    assertIsolated(fixture);
    console.log('PASS App startup: active defaults recovery preserves guidance edits made before and during Retry');
  } catch (error) {
    console.error('defaults recovery diagnostics', JSON.stringify({
      requests: fixture.requests.map(entry => entry.method + ' ' + entry.pathname + entry.search + (entry.failed ? ' (503)' : '')),
      apiFallbacks: fixture.unexpectedApiPaths,
      consoleErrors: fixture.consoleErrors,
      browserErrors: fixture.errors,
      state: await page.evaluate(() => {
        const state = window.store?.getState();
        return state && {modelsLoaded: state.modelsLoaded, modelsLoading: state.modelsLoading,
          modelsLoadError: state.modelsLoadError, modelOptions: state.modelOptions, params: state.params};
      }).catch(() => null),
    }));
    throw error;
  } finally {
    fixture.defaultsRetryGate.resolve();
    await fixture.context.close();
  }
}

async function testModelSwitchDuringStartupErrorRetriesActiveDefaults(bundle, browser) {
  const fixture = await openFixture(browser, bundle, {
    failDefaultsModelsFirst: ['fixture-video', 'fixture-video-b'],
    gateDefaultsModelRetry: 'fixture-video-b',
    deferAutoRetry: true,
  });
  const {page} = fixture;
  try {
    await page.getByRole('alert').waitFor({state: 'visible', timeout: 5000});
    assert.equal(fixture.requests.find(entry => entry.method === 'GET'
      && entry.pathname === '/api/v1/defaults/fixture-video').failed, true,
    'the startup alert begins with model A defaults unavailable');
    assert.equal(await page.evaluate(() => window.store.getState().params.model_type), 'fixture-video',
      'model A is active when the first defaults request fails');

    const sidebar = page.locator('.maestro-sidebar');
    await sidebar.getByRole('button', {name: 'Choose model', exact: true}).click();
    const modelB = page.locator('button[title="Fixture Video B"]');
    await modelB.waitFor({state: 'visible', timeout: 2000});
    await modelB.click();
    await page.waitForFunction(() => {
      const state = window.store?.getState();
      return state && !state.modelsLoading && !state.modelOptionsLoading
        && state.params.model_type === 'fixture-video-b'
        && state.modelOptions?.model_type === 'fixture-video-b';
    }, null, {timeout: 4000});
    await withTimeout((async () => {
      while (!fixture.requests.some(entry => entry.method === 'GET'
        && entry.pathname === '/api/v1/defaults/fixture-video-b' && entry.completedAt)) await pause(10);
    })(), 3000, 'model B’s first defaults read to fail');
    assert.equal(fixture.requests.find(entry => entry.method === 'GET'
      && entry.pathname === '/api/v1/defaults/fixture-video-b').failed, true,
    'model B’s first defaults read also fails after its options have loaded');
    assert.ok(await page.evaluate(() => window.store.getState().modelsLoadError),
      'successful model B options do not clear the startup error while model B defaults are unavailable');
    assert.equal(await page.getByRole('alert').isVisible(), true,
      'the Retry alert remains visible for the active model with unavailable defaults');

    await page.getByRole('button', {name: 'Retry', exact: true}).click();
    await withTimeout(fixture.defaultsRetryStarted.promise, 3000, 'model B defaults recovery request');
    assert.equal(fixture.counts.get('GET /api/v1/defaults/fixture-video-b'), 2,
      'Retry fetches defaults for the newly active model B again');
    assert.equal(fixture.maxInFlight.get('GET /api/v1/defaults/fixture-video-b'), 1,
      'model B defaults recovery does not overlap its failed request');
    fixture.defaultsRetryGate.resolve();
    await waitForReady(page);
    assert.equal(await page.evaluate(() => {
      const state = window.store.getState();
      return state.params.model_type === 'fixture-video-b'
        && state.selectedModelPerMode.video === 'fixture-video-b'
        && state.modelOptions?.model_type === 'fixture-video-b';
    }), true, 'controls become ready for model B only after its defaults recovery succeeds');
    assert.equal(fixture.counts.get('GET /api/v1/defaults/fixture-video'), 1,
      'Retry does not waste the recovery on the no-longer-active model A');
    assert.equal(fixture.requests.filter(entry => entry.method === 'GET'
      && entry.pathname === '/api/v1/defaults/fixture-video-b')[1].failed, false,
    'model B’s second defaults response succeeds');
    assertIsolated(fixture);
    console.log('PASS App startup: switching models during an error retries defaults for the active model');
  } catch (error) {
    console.error('model switch/defaults recovery diagnostics', JSON.stringify({
      requests: fixture.requests.map(entry => entry.method + ' ' + entry.pathname + entry.search + (entry.failed ? ' (503)' : '')),
      apiFallbacks: fixture.unexpectedApiPaths,
      consoleErrors: fixture.consoleErrors,
      browserErrors: fixture.errors,
      state: await page.evaluate(() => {
        const state = window.store?.getState();
        return state && {modelsLoaded: state.modelsLoaded, modelsLoading: state.modelsLoading,
          modelsLoadError: state.modelsLoadError, params: state.params, modelOptions: state.modelOptions};
      }).catch(() => null),
    }));
    throw error;
  } finally {
    fixture.defaultsRetryGate.resolve();
    await fixture.context.close();
  }
}

async function testCatalogueRefreshRetryPreservesSession(bundle, browser) {
  const fixture = await openFixture(browser, bundle, {
    failModelsAfterReady: true,
    freshCatalogueAfterFailure: true,
  });
  const {page} = fixture;
  try {
    await waitForReady(page);
    const prompt = page.locator('textarea[placeholder="Describe your video..."]');
    await prompt.fill('keep this prompt and mode through catalogue recovery');
    await page.evaluate(async () => { await window.store.getState().loadModels(); });
    await page.waitForFunction(() => {
      const state = window.store?.getState();
      return state && !state.modelsLoading && !!state.modelsLoadError;
    }, null, {timeout: 4000});
    assert.equal(fixture.counts.get('GET /api/v1/models'), 2,
      'the explicit full catalog refresh issues a second models read');
    assert.equal(fixture.requests.filter(entry => entry.method === 'GET'
      && entry.pathname === '/api/v1/models')[1].failed, true,
    'the post-startup catalogue refresh receives the fixture 503');
    assert.deepEqual(await page.evaluate(() => {
      const state = window.store.getState();
      return {loaded: state.modelsLoaded, mode: state.generationMode,
        model: state.params.model_type, prompt: state.params.prompt};
    }), {
      loaded: true,
      mode: 'video',
      model: 'fixture-video',
      prompt: 'keep this prompt and mode through catalogue recovery',
    }, 'a failed full catalogue refresh keeps the current navigation and prompt intact');

    await page.getByRole('button', {name: 'Retry', exact: true}).click();
    await page.waitForFunction(() => {
      const state = window.store?.getState();
      return state && state.modelsLoaded && !state.modelsLoading && !state.modelsLoadError
        && state.models.some(model => model.model_type === 'fixture-video'
          && model.name === 'Fresh Fixture Video Catalogue');
    }, null, {timeout: 7000});
    assert.equal(fixture.counts.get('GET /api/v1/models'), 3,
      'Retry fetches the catalogue again instead of clearing the error after a metadata-only recovery');
    assert.equal(fixture.requests.filter(entry => entry.method === 'GET'
      && entry.pathname === '/api/v1/models')[2].failed, false,
    'the retry receives a successful refreshed catalogue');
    assert.equal(await prompt.inputValue(), 'keep this prompt and mode through catalogue recovery',
      'the composer prompt survives the successful full catalogue reload');
    assert.deepEqual(await page.evaluate(() => {
      const state = window.store.getState();
      return {mode: state.generationMode, model: state.params.model_type,
        selected: state.selectedModelPerMode.video, prompt: state.params.prompt};
    }), {
      mode: 'video',
      model: 'fixture-video',
      selected: 'fixture-video',
      prompt: 'keep this prompt and mode through catalogue recovery',
    }, 'the active mode, selected model, and prompt survive catalogue recovery');
    assert.equal(fixture.mainFrameNavigations, 1, 'catalogue recovery does not refresh the page');
    const sidebar = page.locator('.maestro-sidebar');
    assert.equal((await sidebar.getByRole('button', {name: 'Choose model', exact: true}).getAttribute('title'))
      .startsWith('Fresh Fixture Video Catalogue'), true,
    'the model picker displays the refreshed catalogue entry');
    assertIsolated(fixture);
    console.log('PASS App startup: failed full catalogue refresh retries from the server and preserves the session');
  } catch (error) {
    console.error('catalogue recovery diagnostics', JSON.stringify({
      requests: fixture.requests.map(entry => entry.method + ' ' + entry.pathname + entry.search + (entry.failed ? ' (503)' : '')),
      apiFallbacks: fixture.unexpectedApiPaths,
      consoleErrors: fixture.consoleErrors,
      browserErrors: fixture.errors,
      state: await page.evaluate(() => {
        const state = window.store?.getState();
        return state && {modelsLoaded: state.modelsLoaded, modelsLoading: state.modelsLoading,
          modelsLoadError: state.modelsLoadError, generationMode: state.generationMode,
          params: state.params, models: state.models?.map(model => ({model_type: model.model_type, name: model.name}))};
      }).catch(() => null),
    }));
    throw error;
  } finally {
    await fixture.context.close();
  }
}

(async () => {
  const bundle = await buildBundle();
  const browser = await playwright.chromium.launch({
    headless: true,
    ...(process.platform === 'win32' ? {executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'} : {}),
  });
  try {
    await testMobileFocusedKeyboardCompactsAndRestores(bundle, browser);
    await testMobileViewportWithoutEditableKeepsControls(bundle, browser);
    await testControlsBeforeBackground(bundle, browser);
    await testStrictModeModelsRecovery(bundle, browser);
    await testPreferenceHydrationGatesWrites(bundle, browser);
    await testCachedPreferencesRecoverOnOnline(bundle, browser);
    await testOptionsRecoveryKeepsPrompt(bundle, browser);
    await testDefaultsRecoveryKeepsGuidance(bundle, browser);
    await testModelSwitchDuringStartupErrorRetriesActiveDefaults(bundle, browser);
    await testCatalogueRefreshRetryPreservesSession(bundle, browser);
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
