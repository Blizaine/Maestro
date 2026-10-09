// Run with: node tests/ui/h3_vae_upsampling.cjs
// Covers capability scoping and the real Studio submit payload after switching
// away from a cached H3 model.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const {chromium} = require(process.env.MAESTRO_PLAYWRIGHT ||
  'C:/Users/bliza/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

function loadTypeScript(filePath, dependencies = {}) {
  const transformed = esbuild.transformSync(fs.readFileSync(filePath, 'utf8'), {
    loader: 'ts',
    format: 'cjs',
    target: 'node22',
  });
  const loaded = {exports: {}};
  const localRequire = dependency => Object.hasOwn(dependencies, dependency)
    ? dependencies[dependency]
    : require(dependency);
  new Function('module', 'exports', 'require', transformed.code)(
    loaded, loaded.exports, localRequire,
  );
  return loaded.exports;
}

(async () => {
  const {
    H3_VAE_UPSAMPLER,
    h3VaeUpsamplingHelp,
    modelOptionsForSelectedModel,
    sanitizeH3VaeUpsampling,
    supportsH3VaeUpsampling,
  } = loadTypeScript(path.join(root, 'ui/src/lib/h3VaeUpsampling.ts'));

  const h3VideoOptions = {
    model_type: 'minimax_h3_video_fixture',
    architecture: 'minimax_h3',
    vae_upsamplers: {h3_vae: [0]},
  };
  const h3ImageAndVideoOptions = {
    ...h3VideoOptions,
    vae_upsamplers: {h3_vae: [0, 1]},
  };
  const fluxOptions = {
    model_type: 'flux2_klein_9b',
    architecture: 'flux2_klein_9b',
    vae_upsamplers: {},
  };
  assert.equal(supportsH3VaeUpsampling(h3VideoOptions, 0), true, 'advertised video mode is available');
  assert.equal(supportsH3VaeUpsampling(h3VideoOptions, 1), false, 'H3 image mode stays hidden until advertised');
  assert.equal(supportsH3VaeUpsampling(h3VideoOptions, -1), false, 'audio never inherits the video capability');
  assert.equal(supportsH3VaeUpsampling(h3ImageAndVideoOptions, 1), true, 'an explicitly advertised image mode is available');
  assert.equal(modelOptionsForSelectedModel(h3VideoOptions, fluxOptions.model_type), null,
    'a cached H3 response is not active after switching to another model');
  assert.equal(modelOptionsForSelectedModel(h3VideoOptions, h3VideoOptions.model_type), h3VideoOptions,
    'Director keeps options only when they belong to its selected video model');
  assert.equal(sanitizeH3VaeUpsampling(H3_VAE_UPSAMPLER, fluxOptions, 0), '',
    'a non-H3 request drops a cached H3 method');
  assert.equal(sanitizeH3VaeUpsampling(H3_VAE_UPSAMPLER, h3VideoOptions, 1), '',
    'a supported H3 model still drops the method in an unadvertised mode');
  assert.equal(sanitizeH3VaeUpsampling('lanczos2', fluxOptions, 0), 'lanczos2',
    'ordinary upsampling string values pass through unchanged');
  assert.match(h3VaeUpsamplingHelp('960x544'), /output is 1920×1088/,
    'the help reports the current doubled canvas when dimensions are known');
  assert.match(h3VaeUpsamplingHelp('auto_540p'), /540p at 16:9 \(960×544\) becomes 1920×1088/,
    'auto dimensions use a concrete 540p example without claiming it is the live canvas');
  assert.match(h3VaeUpsamplingHelp('960x544'), /Keeps denoising at the selected resolution\./);

  const targetModel = 'flux2_klein_9b';
  const targetOptions = {
    model_type: targetModel,
    architecture: targetModel,
    fps: 24,
    frames_minimum: 1,
    frames_steps: 1,
    default_num_inference_steps: 4,
    default_guidance_scale: 1,
    hide_resolution_presets: false,
    resolution_preset_order: ['auto'],
    resolution_presets: {auto: {values: {auto: 'auto'}}},
    vae_upsampler_modes: [],
    vae_upsamplers: {},
    audio_only: false,
    sliding_window: false,
    sliding_window_defaults: null,
  };
  const modelDefinition = {
    model_type: targetModel,
    name: 'Flux fixture',
    family: 'flux2',
    architecture: targetModel,
    is_i2v: false,
    is_t2v: false,
    guidance_max_phases: 1,
    fps: 24,
    is_downloaded: true,
  };
  const bundle = await esbuild.build({
    stdin: {
      contents: "import {useStore} from './src/stores/useStore'; window.store = useStore;",
      resolveDir: path.join(root, 'ui'),
      loader: 'ts',
    },
    bundle: true,
    write: false,
    define: {'process.env.NODE_ENV': '"development"'},
    logLevel: 'silent',
  });
  const browser = await chromium.launch({
    headless: true,
    ...(process.platform === 'win32' ? {
      executablePath: process.env.MAESTRO_CHROME ||
        'C:/Program Files/Google/Chrome/Application/chrome.exe',
    } : {}),
  });

  try {
    const page = await browser.newPage();
    const requests = [];
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/*', async route => {
      const requestUrl = new URL(route.request().url());
      const json = body => route.fulfill({contentType: 'application/json', body: JSON.stringify(body)});
      if (requestUrl.pathname === '/') {
        return route.fulfill({contentType: 'text/html', body: '<div></div>'});
      }
      if (requestUrl.pathname.startsWith('/api/v1/model-options/')) return json(targetOptions);
      if (requestUrl.pathname.startsWith('/api/v1/defaults/')) {
        return json({num_inference_steps: 4, guidance_scale: 1});
      }
      if (requestUrl.pathname === '/api/v1/generate') {
        requests.push(requestUrl && route.request().postDataJSON());
        return json({job_id: 'h3-vae-switch', status: 'held'});
      }
      return json({});
    });

    await page.goto('http://h3-vae-switch.test');
    await page.addScriptTag({content: bundle.outputFiles[0].text});
    await page.evaluate(({targetModel, modelDefinition, h3VideoOptions, h3Method}) => {
      const current = window.store.getState();
      window.store.setState({
        models: [modelDefinition],
        families: [{id: 'flux2', label: 'Flux 2', order: 1}],
        enabledModels: new Set([targetModel]),
        selectedModelPerMode: {image: targetModel},
        generationMode: 'image',
        sidebarMode: 'studio',
        modelOptions: h3VideoOptions,
        modelOptionsLoading: false,
        spatialUpsampling: h3Method,
        studioImageWorkflow: 'generate',
        startImage: null,
        endImage: null,
        imageRefs: [],
        h3WindowPlan: null,
        jobs: [],
        isGenerating: false,
        isEnhancing: false,
        params: {
          ...current.params,
          model_type: targetModel,
          prompt: 'A red kite above a green hill.',
          resolution: '1024x1024',
          image_mode: 1,
          video_length: 1,
          num_inference_steps: 4,
          guidance_scale: 1,
          seed: 42,
          spatial_upsampling: h3Method,
          custom_settings: {},
        },
      });
    }, {
      targetModel,
      modelDefinition,
      h3VideoOptions,
      h3Method: H3_VAE_UPSAMPLER,
    });

    await page.evaluate(() => window.store.getState().startGeneration('queue'));
    assert.equal(requests.length, 1, await page.evaluate(() =>
      'generation request count; UI error: ' + window.store.getState().promptEnhanceError));
    assert.equal(requests[0].model_type, targetModel);
    assert.equal(requests[0].image_mode, 1);
    assert.equal(requests[0].spatial_upsampling, '',
      'the real submit payload clears the cached H3 selection after resolving the selected non-H3 model');
    assert.deepEqual(errors, [], 'store request completes without browser errors');
    console.log('H3 VAE capability switching, request sanitization, Director model scoping, and output dimensions passed');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
