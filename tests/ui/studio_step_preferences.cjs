// Real Zustand/client lifecycle; every request is served by an isolated fixture.
// Run: node tests/ui/studio_step_preferences.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
// Use an installed Playwright package, or point to a shared installation.
const {chromium} = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');
const frames = 'minimax_h3_fused_turbo', references = 'minimax_h3_ref2va_fused_turbo';
const locked = 'viggle_animate', regular = 'minimax_h3';

(async () => {
  const defaults = Object.fromEntries([frames, references, locked, regular].map(id =>
    [id, JSON.parse(fs.readFileSync(path.join(root, 'app/defaults', id + '.json'), 'utf8'))]));
  const options = Object.fromEntries(Object.entries(defaults).map(([id, preset]) => [id, {
    ...preset.model, model_type:id, fps:24, default_num_inference_steps:preset.num_inference_steps,
    sliding_window:true, frames_minimum:124, frames_maximum:345, frames_steps:17,
    sliding_window_defaults:{window_min:124, window_max:345, window_step:17, window_default:243, overlap_default:18},
    omni_reference:id === references,
  }]));
  options[locked] = {...options[locked], lock_inference_steps:true, default_num_inference_steps:3};
  options[regular] = {...options[regular], default_num_inference_steps:20, minimax_h3_turbo:{
    preset_id:'pdd', filename:'acc.safetensors', steps:8, presets:[{id:'pdd', filename:'acc.safetensors', steps:8}],
  }};
  const models = Object.keys(defaults).map(id => ({...options[id], name:id, family:'minimax_h3', is_downloaded:true}));
  const bundle = await esbuild.build({stdin:{contents:
    "import {useStore} from './src/stores/useStore'; window.store = useStore;",
    resolveDir:path.join(root,'ui'), loader:'ts'}, bundle:true, write:false,
    define:{'process.env.NODE_ENV':'"development"'}, logLevel:'silent'});
  const browser = await chromium.launch({headless:true, ...(process.platform === 'win32' ? {
    executablePath:process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  } : {})});
  try {
    for (const defaultsFirst of [true, false]) {
      let durable = {configured:true, generation_mode:'video', studio_video_workflow:'frames',
        selected_model_per_mode:{video:frames}, inference_steps_per_model:{}};
      let offlinePreferences = false;
      const context = await browser.newContext();
      const errors = [];
      await context.addInitScript(() => {
        const fetch = window.fetch.bind(window);
        window.pendingRequests = 0;
        window.lastRequestActivity = Date.now();
        window.fetch = (...args) => {
          window.pendingRequests++;
          window.lastRequestActivity = Date.now();
          return fetch(...args).finally(() => {
            window.pendingRequests--;
            window.lastRequestActivity = Date.now();
          });
        };
      });
      // Load-state networkidle can already be satisfied before a store action
      // starts its requests. Wait for these fetches and their queued saves.
      const settle = page => page.waitForFunction(() =>
        window.pendingRequests === 0 && Date.now() - window.lastRequestActivity >= 150);
      const open = async origin => {
        const page = await context.newPage();
        page.on('pageerror', error => errors.push(error.message));
        await page.route('**/*', async route => {
          const endpoint = new URL(route.request().url()).pathname;
          const json = body => route.fulfill({json:body});
          if (endpoint === '/') return route.fulfill({contentType:'text/html', body:'<div>Studio preferences fixture</div>'});
          if (endpoint === '/api/v1/models') return json({models, families:[{id:'minimax_h3', label:'H3', order:1}]});
          if (endpoint === '/api/v1/model-visibility') return json({configured:true, enabled_models:Object.keys(defaults), initialized_mature_models:[], defaults_version:11});
          if (endpoint === '/api/v1/studio-preferences') {
            if (offlinePreferences) return route.fulfill({status:503, body:'Offline'});
            if (route.request().method() === 'PUT') durable = {...durable, ...route.request().postDataJSON()};
            return json(durable);
          }
          if (endpoint.includes('/model-options/') || endpoint.includes('/defaults/')) {
            const isDefaults = endpoint.includes('/defaults/');
            await new Promise(resolve => setTimeout(resolve, isDefaults === defaultsFirst ? 5 : 80));
            const id = decodeURIComponent(endpoint.split('/').pop());
            return json(isDefaults ? defaults[id] || {} : options[id] || {});
          }
          return json({loras:[], presets:[], overrides:{}, items:[], jobs:[], outputs:[], total:0});
        });
        await page.goto(origin);
        await page.addScriptTag({content:bundle.outputFiles[0].text});
        await page.evaluate(() => window.store.getState().loadModels());
        await settle(page);
        return page;
      };
      const steps = page => page.evaluate(() => window.store.getState().params.num_inference_steps);
      const choose = async (page, model) => {
        await page.evaluate(model => window.store.getState().selectModel(model), model);
        await settle(page);
      };
      let page = await open('http://studio.test:42015');
      assert.equal(await steps(page),4,'New models start at four steps');
      await page.evaluate(() => window.store.getState().setParam('num_inference_steps',12));
      await settle(page);
      assert.equal(durable.inference_steps_per_model[frames],12);
      await choose(page,references);
      assert.equal(await steps(page),4,'A different model has its own default');
      await page.evaluate(() => window.store.getState().setParam('num_inference_steps',9));
      await settle(page);
      assert.equal(durable.inference_steps_per_model[references],9);
      await choose(page,frames);
      assert.equal(await steps(page),12,'Model switches restore remembered steps');
      // A user changes steps while both default fetches are still in flight.
      await page.evaluate(model => {
        window.store.getState().selectModel(model);
        window.store.getState().setParam('num_inference_steps',11);
      }, frames);
      await settle(page);
      assert.equal(await steps(page),11,'Late options/defaults cannot overwrite a new user choice');
      await page.evaluate(() => {
        window.store.getState().setParam('num_inference_steps',12);
        window.store.getState().setParam('prompt','Temporary scene');
        window.store.getState().setParam('seed',123);
      });
      await settle(page);
      const persisted = await page.evaluate(() => JSON.parse(localStorage.getItem('maestro_mode_settings')));
      assert.equal(persisted.inferenceStepsPerModel[frames],12);
      assert.equal(durable.prompt,undefined);
      await page.close();
      page = await open('http://studio.test:42015');
      assert.equal(await steps(page),12,'Full reload restores steps');
      assert.equal(await page.evaluate(() => window.store.getState().params.prompt),'','The working prompt still starts fresh');
      assert.notEqual(await page.evaluate(() => window.store.getState().params.seed),123);
      await page.close();
      page = await open('http://studio.test:42199');
      assert.equal(await steps(page),12,'A new Pinokio port restores from server preferences');
      await choose(page,references);
      assert.equal(await steps(page),9);
      await choose(page,frames);
      await page.close();
      offlinePreferences = true;
      page = await open('http://studio.test:42199');
      assert.equal(await steps(page),12,'Browser cache works when server preferences are unavailable');
      await page.close();
      offlinePreferences = false;
      durable.inference_steps_per_model[frames] = 99;
      page = await open('http://studio.test:42200');
      assert.equal(await steps(page),12,'Restored values respect model limits');
      await page.evaluate(({locked,regular}) => window.store.setState({inferenceStepsPerModel:{[locked]:12,[regular]:30}}),{locked,regular});
      await choose(page,locked);
      assert.equal(await steps(page),3,'Locked model recipes retain their required step count');
      await page.evaluate(model => {
        window.store.getState().selectModel(model);
        window.store.getState().setParam('minimax_h3_turbo_mode',true);
      },regular);
      await settle(page);
      assert.equal(await steps(page),8,'Managed Turbo recipes take precedence over remembered steps');
      assert.deepEqual(errors,[]);
      await context.close();
      console.log(`Step persistence passed with ${defaultsFirst ? 'defaults' : 'options'} resolving first: model switches, reload, changed port, cache fallback, late responses, clamping and fixed recipes.`);
    }
  } finally {await browser.close();}
})().catch(error => {console.error(error); process.exitCode=1;});
