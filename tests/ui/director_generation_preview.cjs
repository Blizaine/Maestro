// Exercise Director discovery and gallery tiles with the real store/components.
// All HTTP requests are intercepted. No app, GPU, user settings or render is touched.
// Run: MAESTRO_PLAYWRIGHT=<package path> node tests/ui/director_generation_preview.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const playwright = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
const media = (id, revision = 1) => ({
  url: '/preview/' + id + '?revision=' + revision, kind: 'video',
  revision, mode: 'tiny_vae_video', window: 1, total_windows: 1,
  clip: 1, total_clips: 1,
});
const job = (id, owner = null, extra = {}) => ({
  job_id: id, status: 'running', progress: 50, step: 3, total_steps: 6,
  phase: 'Denoising', message: 'Generating...', output_files: [], error: null,
  show_in_gallery: true, director_pipeline_id: owner, director_detached_operation: false,
  ...extra,
});
const pipeline = clip => ({
  id: 'parent', status: 'running', phase: 'generating_video',
  clip_plans: [], planned_clips: [], clip_images: [], output_files: [],
  progress: {current: clip, total: 23, current_clip: clip, total_clips: 23,
    message: 'Clip ' + clip + '/23: Denoising', step: 3, total_steps: 6,
    clip_eta_seconds: 60, project_eta_seconds: 5400},
});

(async () => {
  const bundle = await esbuild.build({stdin: {contents: [
    "import React from 'react'; import {createRoot} from 'react-dom/client';",
    "import {MainContent} from './src/components/MainContent/MainContent';",
    "import {useStore} from './src/stores/useStore';",
    "window.store = useStore; const idle = async () => {};",
    "useStore.setState({jobs: [], outputs: [], outputsTotal: 0, outputsLoading: false, activeWorkspace: 'Test',",
    "workspaces: [{name:'Test',path:'Test'}], pipelineId: 'parent', pipelinePolling: true,",
    "refreshOutputs: idle, loadOutputs: idle, directorStep: 'review_video'});",
    "createRoot(document.getElementById('root')).render(<MainContent/>);",
    "useStore.getState().pollPipelineStatus();",
  ].join('\n'), resolveDir: path.join(root, 'ui'), loader: 'tsx'},
    bundle: true, write: false, jsx: 'automatic',
    define: {'process.env.NODE_ENV': '"development"'}, logLevel: 'silent'});
  const assets = path.join(root, 'ui/dist/assets');
  const css = fs.readFileSync(path.join(assets, fs.readdirSync(assets).find(name => name.endsWith('.css'))), 'utf8');
  const browser = await playwright.chromium.launch({headless: true,
    ...(process.platform === 'win32' ? {executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'} : {})});
  try {
    for (const viewport of [{width: 1360, height: 900}, {width: 390, height: 844}]) {
      const page = await browser.newPage({viewport});
      let current = pipeline(1);
      let activeJobs = [];
      const statuses = new Map();
      const errors = [], mutations = [];
      let listRequests = 0, simultaneousLists = 0, maxSimultaneousLists = 0;
      let listDelay = 100;
      await page.addInitScript(() => {
        const nativeSetAttribute = Element.prototype.setAttribute;
        Element.prototype.setAttribute = function(name, value) {
          if (this instanceof HTMLVideoElement && name.toLowerCase() === 'src') {
            this.dataset.previewSrc = String(value);
            return;
          }
          nativeSetAttribute.call(this, name, value);
        };
        HTMLMediaElement.prototype.play = function() {
          this.dataset.playback = 'playing';
          return Promise.resolve();
        };
        HTMLMediaElement.prototype.pause = function() { this.dataset.playback = 'paused'; };
      });
      page.on('pageerror', error => errors.push(error.message));
      await page.route('**/*', async route => {
        const request = route.request();
        const endpoint = new URL(request.url()).pathname;
        const json = body => route.fulfill({json: body});
        if (endpoint === '/') return route.fulfill({contentType: 'text/html', body:
          '<meta name="viewport" content="width=device-width,initial-scale=1"><style>html,body,#root{height:100%;margin:0}</style><link rel="stylesheet" href="/app.css"><div id="root"></div><script src="/bundle.js"></script>'});
        if (endpoint === '/bundle.js') return route.fulfill({contentType: 'application/javascript', body: bundle.outputFiles[0].text});
        if (endpoint === '/app.css') return route.fulfill({contentType: 'text/css', body: css});
        if (request.method() !== 'GET') {
          mutations.push(endpoint);
          return json({});
        }
        if (endpoint === '/preview/scene.svg') return route.fulfill({contentType: 'image/svg+xml',
          body: '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360"><rect width="640" height="360" fill="#334155"/></svg>'});
        if (endpoint === '/api/v1/director/pipeline/parent') return json(current);
        if (endpoint === '/api/v1/jobs') {
          listRequests++;
          simultaneousLists++;
          maxSimultaneousLists = Math.max(maxSimultaneousLists, simultaneousLists);
          // Exercise overlapping Director and startup/reconnect discovery.
          await delay(listDelay);
          simultaneousLists--;
          return json({jobs: activeJobs});
        }
        if (endpoint.startsWith('/api/v1/status/')) {
          const id = endpoint.slice('/api/v1/status/'.length);
          return json(statuses.get(id) || job(id, null, {status: 'completed'}));
        }
        return json({outputs: [], total: 0, entries: [], jobs: [], pipelines: [], queue: [], capabilities: {}});
      });
      await page.goto('http://director-preview.test/');
      await page.waitForFunction(() => window.store?.getState().pipelineStatus?.id === 'parent');
      const card = page.locator('[data-director-pipeline-id="parent"]');
      await card.waitFor();
      assert.equal(await page.locator('[data-generation-preview]').count(), 0, 'pre-preview phase keeps the Director progress card');
      await page.waitForFunction(() => window.store?.getState().pipelineStatus?.phase === 'generating_video');
      activeJobs = [job('child-1', 'parent')];
      statuses.set('child-1', activeJobs[0]);
      await page.waitForFunction(() => window.store.getState().jobs.some(job => job.id === 'child-1'), null, {timeout: 6000});
      assert.equal(await page.locator('[data-generation-job-id="child-1"]').count(), 0, 'the discovered child does not create a duplicate progress card');
      statuses.set('child-1', job('child-1', 'parent', {
        preview: media('child-1'), h3_window_plan: {signature: 'first-plan', window_count: 1,
          windows: [{index: 1, title: 'First clip', start_frame: 0, start_seconds: 0,
            end_seconds: 10.125, prompt: 'Exact first clip prompt.'}]},
      }));
      await card.locator('video').waitFor({timeout: 5000});
      assert.equal(await page.locator('[data-generation-preview]').count(), 1, 'live preview appears without refreshing');
      // The pipeline can enter final processing before the child's next status poll.
      // Ownership still prevents a duplicate child tile during that short transition.
      current = {...pipeline(1), phase: 'post_processing'};
      await page.waitForFunction(() => window.store.getState().pipelineStatus?.phase === 'post_processing', null, {timeout: 5000});
      assert.equal(await page.locator('[data-generation-job-id="child-1"]').count(), 0, 'finishing phases retain one parent tile');
      assert.equal(await card.locator('[data-generation-preview]').count(), 1, 'last preview stays on the Director tile during final processing');
      current = pipeline(1);
      await card.getByText('Full Director render ~1h 30m', {exact: false}).waitFor();
      assert.equal(await card.getByText('Full Studio render', {exact: false}).count(), 0, 'parent retains Director estimates');
      await card.getByText('Clip 1/23 · Window 1/1').waitFor();
      await card.getByText('Exact H3 prompt · Window 1/1').waitFor();
      await card.getByRole('button', {name: 'View all', exact: true}).click();
      await card.locator('pre').getByText('Exact first clip prompt.', {exact: true}).waitFor();
      await card.getByRole('button', {name: 'Hide generation information'}).click();
      await card.getByRole('button', {name: 'Show generation information'}).waitFor();
      assert.equal(await card.getByText('Full Director render', {exact: false}).isVisible(), false, 'tap hides overlay information');
      assert.equal(await card.locator('video').getAttribute('data-playback'), 'playing', 'tap leaves video playing');
      await card.getByRole('button', {name: 'Pause live preview'}).click();
      await card.getByRole('button', {name: 'Play live preview'}).waitFor();
      assert.equal(await card.getByRole('button', {name: 'Show generation information'}).count(), 1, 'pause does not toggle information');

      current = pipeline(2);
      statuses.set('child-1', job('child-1', 'parent', {status: 'completed'}));
      activeJobs = [job('child-2', 'parent')];
      statuses.set('child-2', activeJobs[0]);
      await page.waitForFunction(() => window.store.getState().jobs.some(job => job.id === 'child-2'), null, {timeout: 9000});
      await card.locator('video').waitFor({state: 'detached'});
      statuses.set('child-2', job('child-2', 'parent', {preview: media('child-2')}));
      await card.locator('video').waitFor({timeout: 5000});
      await card.getByRole('button', {name: 'Play live preview'}).waitFor();
      assert.equal(await card.locator('video').getAttribute('data-playback'), 'paused', 'pause preference survives a new clip job');
      assert.equal(await card.getByRole('button', {name: 'Show generation information'}).count(), 1, 'overlay preference survives clip transition');
      await card.getByRole('button', {name: 'Show generation information'}).click();
      await card.getByText('Clip 2/23 · Window 1/1').waitFor();

      activeJobs = [
        statuses.get('child-2'),
        job('studio', null),
        job('other-parent', 'different-parent'),
        job('detached-repair', 'parent', {director_detached_operation: true}),
      ];
      activeJobs.forEach(item => statuses.set(item.job_id, item));
      await page.evaluate(() => Promise.all([
        window.store.getState().reconnectJobs(),
        window.store.getState().reconnectJobs(),
      ]));
      assert.equal(maxSimultaneousLists, 1, 'concurrent discovery shares one request');
      assert.equal(await page.evaluate(() => new Set(window.store.getState().jobs.map(job => job.id)).size), 4, 'concurrent discovery keeps one entry per job');
      assert.equal(await page.locator('[data-generation-job-id="child-2"]').count(), 0, 'associated child remains merged');
      for (const id of ['studio', 'other-parent', 'detached-repair']) {
        await page.locator('[data-generation-job-id="' + id + '"]').waitFor();
      }

      // A confirmed retry must be adopted immediately, even while history is slow.
      listDelay = 800;
      statuses.set('accepted-retry', job('accepted-retry', null, {status: 'queued'}));
      const retryLatency = await page.evaluate(async () => {
        const state = window.store.getState();
        const restoring = state.reconnectJobs();
        await new Promise(resolve => setTimeout(resolve, 30));
        const source = state.jobs.find(job => job.id === 'studio');
        const started = performance.now();
        await state.reconnectJobs({...source, id: 'accepted-retry', status: 'queued', showInGallery: true});
        const latency = performance.now() - started;
        if (!window.store.getState().jobs.some(job => job.id === 'accepted-retry')) throw Error('accepted retry missing');
        await restoring;
        return latency;
      });
      assert.ok(retryLatency < 400, 'accepted retry bypasses slow recovery (' + retryLatency + 'ms)');
      listDelay = 100;

      // Simulate a fresh page's startup discovery racing its Director polling.
      await page.reload();
      await page.waitForFunction(() => !!window.store);
      await page.evaluate(() => window.store.getState().reconnectJobs());
      await card.locator('video').waitFor({timeout: 6000});
      assert.equal(await page.locator('[data-director-pipeline-id]').count(), 1);
      assert.equal(await page.locator('[data-generation-job-id="child-2"]').count(), 0, 'refresh still renders one Director card');
      await card.getByRole('button', {name: 'Stop', exact: true}).click();
      await page.waitForFunction(() => window.store.getState().pipelineId === null);
      assert.deepEqual(mutations, ['/api/v1/director/pipeline/parent/stop'], 'Stop targets the parent pipeline');
      // A new project gets fresh overlay/playback preferences.
      statuses.set('other-parent', job('other-parent', 'different-parent', {preview: media('other-parent')}));
      await page.evaluate(status => window.store.setState({
        pipelineId: 'different-parent', pipelineStatus: status, pipelinePolling: false,
      }), {...current, id: 'different-parent'});
      const freshCard = page.locator('[data-director-pipeline-id="different-parent"]');
      await freshCard.locator('video').waitFor({timeout: 5000});
      await freshCard.getByRole('button', {name: 'Hide generation information'}).waitFor();
      await freshCard.getByRole('button', {name: 'Pause live preview'}).waitFor();
      assert.equal(await freshCard.locator('video').getAttribute('data-playback'), 'playing', 'new Director project starts with fresh playback preferences');
      current = {...pipeline(1), phase: 'generating_images'};
      const imageJob = job('scene-image', 'parent', {preview: {
        ...media('scene-image'), kind: 'image', url: '/preview/scene.svg',
      }});
      activeJobs = [imageJob];
      statuses.set('scene-image', imageJob);
      await page.evaluate(() => {
        window.store.setState({pipelineId: 'parent', pipelineStatus: null,
          pipelinePolling: true, jobs: []});
        window.store.getState().pollPipelineStatus();
      });
      await card.locator('img').waitFor({timeout: 6000});
      assert.equal(await page.locator('[data-generation-job-id="scene-image"]').count(), 0, 'supported scene-image previews also share their Director tile');
      assert.deepEqual(errors, [], 'gallery and polling raise no browser errors');
      console.log('PASS Director live preview ' + viewport.width + 'px: discovery, transitions, dedup, refresh, controls (' + listRequests + ' discoveries)');
      await page.close();
    }
  } finally {
    await browser.close();
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
