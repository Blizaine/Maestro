// Model-resolved two-expert settings survive refresh and reset on model switch.
// Run: node tests/ui/wan22_guidance_defaults.cjs
const assert = require('node:assert/strict');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const {chromium} = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');

const options = {
  'wan22-a': {
    model_type: 'wan22-a', architecture: 'wan', fps: 16,
    guidance_max_phases: 3, default_guidance_phases: 2,
    default_guidance2_scale: 1, default_switch_threshold: 900,
  },
  'wan22-b': {
    model_type: 'wan22-b', architecture: 'wan', fps: 16,
    guidance_max_phases: 3, default_guidance_phases: 3,
    default_guidance2_scale: 4.5, default_switch_threshold: 640,
  },
  'wan22-no-expert-defaults': {
    model_type: 'wan22-no-expert-defaults', architecture: 'wan', fps: 16,
    guidance_max_phases: 3,
  },
};

(async () => {
  const bundle = await esbuild.build({
    stdin: {contents: "import {useStore} from './src/stores/useStore'; window.store = useStore;",
      resolveDir: path.join(root, 'ui'), loader: 'ts'},
    bundle: true, write: false,
    define: {'process.env.NODE_ENV': '"development"'}, logLevel: 'silent',
  });
  const browser = await chromium.launch({headless: true, ...(process.platform === 'win32' ? {
    executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  } : {})});
  try {
    const page = await browser.newPage();
    await page.route('**/*', route => {
      const endpoint = new URL(route.request().url()).pathname;
      if (endpoint === '/') return route.fulfill({contentType: 'text/html', body: '<div>fixture</div>'});
      const modelType = decodeURIComponent(endpoint.split('/').pop());
      return route.fulfill({json: options[modelType] || {model_type: modelType, fps: 16}});
    });
    await page.goto('http://studio.test');
    await page.addScriptTag({content: bundle.outputFiles[0].text});
    await page.evaluate(() => {
      const state = window.store.getState();
      window.store.setState({
        ...state,
        modelOptions: {model_type: 'wan-old', guidance_max_phases: 3},
        params: {...state.params, model_type: 'wan22-a', guidance_phases: 3,
          guidance2_scale: 7, switch_threshold: 250},
      });
    });
    const load = model => page.evaluate(model => window.store.getState().loadModelOptions(model), model);
    const read = () => page.evaluate(() => ({
      model: window.store.getState().modelOptions?.model_type,
      phases: window.store.getState().params.guidance_phases,
      guidance2: window.store.getState().params.guidance2_scale,
      threshold: window.store.getState().params.switch_threshold,
    }));

    await load('wan22-a');
    assert.deepEqual(await read(), {model: 'wan22-a', phases: 2, guidance2: 1, threshold: 900},
      'A model uses its resolved defaults rather than the three-phase capability maximum or stale global values');

    await page.evaluate(() => window.store.setState(state => ({params: {
      ...state.params, guidance_phases: 1, guidance2_scale: 6.5, switch_threshold: 420,
    }})));
    await load('wan22-a');
    assert.deepEqual(await read(), {model: 'wan22-a', phases: 1, guidance2: 6.5, threshold: 420},
      'Refreshing the active model preserves explicit user overrides');

    await page.evaluate(() => window.store.setState(state => ({
      modelOptions: {...state.modelOptions},
      params: {...state.params, model_type: 'wan22-b'},
    })));
    await load('wan22-b');
    assert.deepEqual(await read(), {model: 'wan22-b', phases: 3, guidance2: 4.5, threshold: 640},
      'Switching models applies the newly resolved defaults');

    await page.evaluate(() => window.store.setState(state => ({
      modelOptions: {...state.modelOptions},
      params: {...state.params, model_type: 'wan22-no-expert-defaults',
        guidance_phases: 3, guidance2_scale: 6.5, switch_threshold: 420},
    })));
    await load('wan22-no-expert-defaults');
    const missingDefaults = await read();
    assert.equal(missingDefaults.model, 'wan22-no-expert-defaults');
    assert.equal(missingDefaults.phases, 1,
      'A missing selected phase does not fall back to capability max');
    assert.equal(missingDefaults.guidance2, undefined,
      'A missing second-expert scale does not inherit stale model values');
    assert.equal(missingDefaults.threshold, undefined,
      'A missing switch threshold does not inherit stale model values');

    console.log('Wan 2.2 model-resolved guidance defaults: refresh overrides, model switching, and missing-default fallback passed');
  } finally {
    await browser.close();
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
