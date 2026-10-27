// PTD detail round-trip through the real store, Studio settings, and reference editor.
// API traffic is mocked; no live generation or model runtime is used.
// Run: node tests/ui/h3_ptd_reference_detail.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const playwright = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');

const PTD = 'minimax_h3_ref2va_ptd_refasymopd';
const ORDINARY = 'minimax_h3_ref2va';
const ptdOptions = {
  model_type: PTD,
  architecture: 'minimax_h3_ref2va',
  omni_reference: true,
  omni_reference_limits: {image: 9, video: 3, audio: 3, total: 12},
  omni_reference_detail_choices: [['PTD Recipe (768px short edge)', 'ptd768']],
  omni_reference_detail_default: 'ptd768',
  fps: 24,
  frames_minimum: 124,
  frames_maximum: 345,
  frames_steps: 17,
  default_num_inference_steps: 8,
  lock_inference_steps: true,
  loras_disabled: true,
};
const ordinaryOptions = {
  ...ptdOptions,
  model_type: ORDINARY,
  omni_reference_detail_choices: [
    ['Match output (faster)', 'match'],
    ['High detail (official PDD recipe)', 'max'],
  ],
  omni_reference_detail_default: 'match',
};

async function buildBrowserBundle() {
  const result = await esbuild.build({
    stdin: {contents: `
      import React from 'react';
      import {createRoot} from 'react-dom/client';
      import {useStore} from './src/stores/useStore';
      import {AdvancedSettings} from './src/components/Sidebar/AdvancedSettings';
      import {OmniReferenceSection} from './src/components/Sidebar/OmniReferenceSection';
      window.store = useStore;
      window.optionsByModel = ${JSON.stringify({[PTD]: ptdOptions, [ORDINARY]: ordinaryOptions})};
      window.configStudio = (modelType, options, referenceDetail) => {
        const current = useStore.getState();
        useStore.setState({
          params: {
            ...current.params,
            model_type: modelType,
            prompt: 'A person turns toward the garden.',
            resolution: '640x480',
            image_mode: 0,
            seed: 27,
            num_inference_steps: 8,
            guidance_scale: 1,
            activated_loras: [],
            loras_multipliers: '',
            minimax_h3_references: [{id: 'ref-1', type: 'image', path: '/uploads/ref.png', filename: 'ref.png'}],
            ...(referenceDetail === undefined ? {} : {minimax_h3_reference_detail: referenceDetail}),
          },
          models: [{...options, family: 'minimax_h3', is_downloaded: true, is_t2v: true, is_i2v: true}],
          modelOptions: options,
          enabledModels: new Set([modelType]),
          generationMode: 'video',
          studioVideoWorkflow: 'references',
          selectedModelPerMode: {...current.selectedModelPerMode, video: modelType},
          activeWorkspace: 'PTD UI regression',
          selectedOutputMeta: null,
          selectedOutput: -1,
          jobs: [],
          presets: [],
          servicesConfig: {},
          systemStats: {gpu: {vram_total_gb: 24}, ram: {total_gb: 64}},
          availableLoras: [],
        });
      };
      window.configDirector = (modelType, referenceDetail) => {
        const current = useStore.getState();
        useStore.setState({
          sidebarMode: 'director',
          generationMode: 'video',
          selectedModelPerMode: {...current.selectedModelPerMode, video: modelType},
          directorH3References: [{id: 'director-ref', type: 'image', path: '/uploads/ref.png', filename: 'ref.png'}],
          directorH3ReferenceDetail: referenceDetail,
          activeWorkspace: 'PTD UI regression',
        });
      };
      window.mountStudio = () => {
        if (!window.reactRoot) window.reactRoot = createRoot(document.getElementById('root'));
        window.reactRoot.render(<><AdvancedSettings/><OmniReferenceSection scope="studio"/></>);
      };
      window.mountDirector = () => {
        if (!window.reactRoot) window.reactRoot = createRoot(document.getElementById('root'));
        window.reactRoot.render(<OmniReferenceSection scope="director"/>);
      };
      window.submitQueued = () => useStore.getState().startGeneration('queue');
    `, resolveDir: path.join(root, 'ui'), loader: 'tsx'},
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
  return result.outputFiles[0].text;
}

(async () => {
  const bundle = await buildBrowserBundle();
  const browser = await playwright.chromium.launch({
    headless: true,
    ...(process.env.MAESTRO_CHROME || process.platform === 'win32'
      ? {executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'}
      : {}),
  });
  const submitted = [];
  const pageErrors = [];
  try {
    const page = await browser.newPage({viewport: {width: 1280, height: 900}});
    page.on('pageerror', error => pageErrors.push(error.message));
    await page.route('**/*', async route => {
      const request = route.request();
      const url = new URL(request.url());
      const json = value => route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(value),
      });
      if (url.hostname === 'studio.test' && url.pathname === '/') {
        return route.fulfill({contentType: 'text/html', body: '<div id="root"></div>'});
      }
      if (url.pathname === '/api/v1/characters') return json({characters: []});
      if (url.pathname === '/api/v1/jobs') return json({jobs: []});
      if (url.pathname === '/api/v1/studio-preferences') return json({});
      if (url.pathname.startsWith('/api/v1/media-flow/capabilities')) {
        return json({
          neural_rendering: {available: false, reason: 'browser fixture', scales: []},
          frame_generation: {available: false, reason: 'browser fixture', factors: []},
        });
      }
      if (url.pathname.startsWith('/api/v1/model-options/')) {
        const modelType = decodeURIComponent(url.pathname.split('/').at(-1));
        return json(modelType === PTD ? ptdOptions : ordinaryOptions);
      }
      if (url.pathname === '/api/v1/generate') {
        submitted.push(JSON.parse(request.postData() || '{}'));
        return json({job_id: `ptd-detail-${submitted.length}`, status: 'held'});
      }
      if (url.pathname.endsWith('/presets') || url.pathname.includes('/presets/')) {
        return json({presets: []});
      }
      return json({});
    });
    await page.goto('http://studio.test/');
    await page.addScriptTag({content: bundle});
    const errors = [];

    // Studio: a stale max value is reconciled to the only PTD model-option,
    // and the real startGeneration serializer posts that PTD value.
    await page.evaluate(() => {
      window.configStudio(window.optionsByModel.minimax_h3_ref2va_ptd_refasymopd.model_type,
        window.optionsByModel.minimax_h3_ref2va_ptd_refasymopd, 'max');
      window.mountStudio();
    });
    await page.waitForFunction(() => window.store.getState().params.minimax_h3_reference_detail === 'ptd768');
    await page.getByRole('button', {name: /^Advanced settings/}).click();
    await page.getByText('Performance', {exact: true}).click();
    const studioDetail = page.getByLabel('Reference detail', {exact: true});
    await studioDetail.waitFor();
    assert.equal(await studioDetail.inputValue(), 'ptd768');
    assert.deepEqual(await studioDetail.locator('option').evaluateAll(nodes => nodes.map(node => node.value)), ['ptd768']);
    await page.evaluate(() => window.submitQueued());
    await page.waitForFunction(() => window.store.getState().jobs.some(job => job.id === 'ptd-detail-1')
      || window.store.getState().promptEnhanceError);
    assert.equal(submitted[0]?.minimax_h3_reference_detail, 'ptd768', JSON.stringify(submitted[0]));
    console.log('Studio PTD stale max -> ptd768 UI and actual queued request serializer passed');

    // Studio with an absent detail also materializes the model default before
    // submission; switching back to an ordinary model clears a stale PTD value.
    await page.evaluate(() => {
      window.configStudio(window.optionsByModel.minimax_h3_ref2va_ptd_refasymopd.model_type,
        window.optionsByModel.minimax_h3_ref2va_ptd_refasymopd, undefined);
      window.mountStudio();
    });
    await page.waitForFunction(() => window.store.getState().params.minimax_h3_reference_detail === 'ptd768');
    await page.evaluate(() => window.submitQueued());
    await page.waitForFunction(() => window.store.getState().jobs.some(job => job.id === 'ptd-detail-2')
      || window.store.getState().promptEnhanceError);
    assert.equal(submitted[1]?.minimax_h3_reference_detail, 'ptd768', JSON.stringify(submitted[1]));

    await page.evaluate(() => {
      window.configStudio(window.optionsByModel.minimax_h3_ref2va.model_type,
        window.optionsByModel.minimax_h3_ref2va, 'ptd768');
      window.mountStudio();
    });
    await page.waitForFunction(() => window.store.getState().params.minimax_h3_reference_detail === 'match');
    await studioDetail.waitFor();
    assert.deepEqual(await studioDetail.locator('option').evaluateAll(nodes => nodes.map(node => node.value)), ['match', 'max']);
    assert.equal(await studioDetail.inputValue(), 'match');
    await studioDetail.selectOption('max');
    assert.equal(await page.evaluate(() => window.store.getState().params.minimax_h3_reference_detail), 'max');
    await page.evaluate(() => window.submitQueued());
    await page.waitForFunction(() => window.store.getState().jobs.some(job => job.id === 'ptd-detail-3')
      || window.store.getState().promptEnhanceError);
    assert.equal(submitted[2]?.minimax_h3_reference_detail, 'max', JSON.stringify(submitted[2]));
    assert.notEqual(submitted[2]?.minimax_h3_reference_detail, 'ptd768');
    console.log('Studio ordinary options retain match/max and serializer posts max, never ptd768');

    // Director uses the same PTD model-options contract and reconciles stale
    // values in its independent preference state before presenting the control.
    await page.evaluate(() => {
      window.configDirector('minimax_h3_ref2va_ptd_refasymopd', 'max');
      window.mountDirector();
    });
    await page.waitForFunction(() => window.store.getState().directorH3ReferenceDetail === 'ptd768');
    const directorDetail = page.locator('select[title^="PTD RefAsymOPD"]');
    await directorDetail.waitFor();
    assert.equal(await directorDetail.inputValue(), 'ptd768');
    assert.deepEqual(await directorDetail.locator('option').evaluateAll(nodes => nodes.map(node => node.value)), ['ptd768']);
    await page.evaluate(() => {
      window.configDirector('minimax_h3_ref2va', 'ptd768');
      window.mountDirector();
    });
    await page.waitForFunction(() => window.store.getState().directorH3ReferenceDetail === 'match');
    const directorOrdinary = page.locator('select').last();
    await directorOrdinary.waitFor();
    assert.deepEqual(await directorOrdinary.locator('option').evaluateAll(nodes => nodes.map(node => node.value)), ['match', 'max']);
    assert.equal(await directorOrdinary.inputValue(), 'match');
    await directorOrdinary.selectOption('max');
    assert.equal(await page.evaluate(() => window.store.getState().directorH3ReferenceDetail), 'max');
    console.log('Director PTD and ordinary model-options reconcile the model-specific detail state');

    if (pageErrors.length) errors.push(...pageErrors);
    assert.deepEqual(errors, [], 'components render without browser errors');
    console.log('PTD reference-detail browser/store regression passed');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
