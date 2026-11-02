// Real job cards/store, with all requests intercepted. No GPU or live app calls.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const playwright = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');
const origin = 'http://failed-job-details.test';

const details = { workspace: 'Turbo tests', params: {
  model_type: 'custom_h3_frames', prompt: 'A boat moves.', resolution: '704x1280',
  seed: 42, num_inference_steps: 8, guidance_scale: 1,
  activated_loras: ['turbo-one.safetensors', 'style-two.safetensors'], loras_multipliers: '0.80 0',
}, performance_plan: {profile: 2, transformer_budget_mb: 4096} };
const failed = { job_id: 'failed1', status: 'failed', kind: 'generation', progress: 0,
  step: 0, total_steps: 8, phase: '', message: 'Adapter load failed', error: 'Incompatible adapter',
  output_files: [], created_at: 1, generation_details: details };

(async () => {
  const bundle = await esbuild.build({stdin: {resolveDir: path.join(root, 'ui'), loader: 'tsx', contents: `
    import React from 'react'; import {createRoot} from 'react-dom/client';
    import {useStore} from './src/stores/useStore';
    import {JobPlaceholder} from './src/components/MainContent/MainContent';
    import {captureGenerationDetails} from './src/lib/generationDetails';
    window.store = useStore; window.captureDetails = captureGenerationDetails;
    useStore.setState({models: [{model_type:'custom_h3_frames',name:'Custom H3 Frames'}],
      params: {prompt:'NEW prompt', seed:999, num_inference_steps:50, activated_loras:['NEW.safetensors']},
      loadOutputs: async () => {}, refreshOutputs: async () => {}});
    function Cards() { const jobs=useStore(s=>s.jobs); return <div>{jobs.map(job=><JobPlaceholder
      key={job.id} job={job} onDismiss={()=>useStore.getState().dismissJob(job.id)}/>)}</div> }
    createRoot(document.getElementById('root')).render(<Cards/>);
  `}, bundle: true, write: false, jsx: 'automatic', define: {'process.env.NODE_ENV': '"development"'}, logLevel: 'silent'});
  const cssName = fs.readdirSync(path.join(root, 'ui/dist/assets')).find(name => name.endsWith('.css'));
  const css = fs.readFileSync(path.join(root, 'ui/dist/assets', cssName), 'utf8');
  const browser = await playwright.chromium.launch({headless: true,
    executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'});
  try {
    for (const viewport of [{width: 390, height: 844}, {width: 1280, height: 900}]) {
      const page = await browser.newPage({viewport});
      const errors = [], requests = [];
      page.on('pageerror', error => errors.push(error.stack || error.message));
      let dismissed = false, live = false, polls = 0;
      await page.route('**/*', route => {
        const request = route.request(), url = new URL(request.url());
        assert.equal(url.origin, origin, 'all traffic stays on the test origin');
        const json = body => route.fulfill({json: body});
        if (url.pathname === '/') return route.fulfill({contentType:'text/html',body:'<meta name="viewport" content="width=device-width,initial-scale=1"><div id="root"></div>'});
        requests.push(`${request.method()} ${url.pathname}`);
        if (url.pathname === '/api/v1/jobs' && request.method() === 'GET') return json({jobs: dismissed ? [] : [{...failed, ...(live ? {status:'running',error:null,message:'Loading'} : {})}]});
        if (url.pathname === '/api/v1/status/failed1') { polls++; return json(failed); }
        if (url.pathname === '/api/v1/jobs/failed1' && request.method() === 'DELETE') { dismissed = true; return json({dismissed: true}); }
        throw new Error(`Unexpected request: ${request.method()} ${url.pathname}`);
      });
      await page.goto(origin);
      await page.addStyleTag({content: css});
      await page.addScriptTag({content: bundle.outputFiles[0].text});
      await page.evaluate(() => window.store.getState().reconnectJobs());
      const card = page.locator('[data-generation-job-id="failed1"]');
      await card.getByText('Generation Failed', {exact:true}).waitFor();
      await card.getByText('Settings & LoRAs', {exact:true}).click();
      await card.getByText('Custom H3 Frames', {exact:true}).waitFor();
      assert.ok((await card.innerText()).includes('turbo-one.safetensors'));
      assert.ok((await card.innerText()).includes('0.80×'));
      assert.ok((await card.innerText()).includes('0×'), 'zero weight is retained');
      assert.ok((await card.innerText()).includes('A boat moves.'));
      assert.ok(!(await card.innerText()).includes('NEW.safetensors'), 'saved attempt does not read current form');
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false, 'mobile details fit the viewport');
      const rowsAligned = await card.locator('dl').first().evaluate(list => {
        const labels = [...list.querySelectorAll('dt')], values = [...list.querySelectorAll('dd')];
        return labels.every((label, index) => {
          const a = label.getBoundingClientRect(), b = values[index].getBoundingClientRect();
          return Math.abs(a.top - b.top) < 1 && b.left > a.left;
        });
      });
      assert.ok(rowsAligned, 'each setting label stays beside its value');
      const screenshotDir = path.join(root, '.codex-tmp/eros-lora-review-20261023');
      fs.mkdirSync(screenshotDir, {recursive: true});
      await page.screenshot({path:path.join(screenshotDir, `failed-details-${viewport.width}.png`),fullPage:true});

      await page.evaluate(() => window.store.getState().dismissJob('failed1'));
      await card.waitFor({state:'detached'});
      await page.waitForFunction(() => window.store.getState().jobs.length === 0);
      await page.evaluate(() => window.store.getState().reconnectJobs());
      assert.equal(await page.locator('[data-generation-job-id]').count(), 0, 'dismissed history does not reappear');
      assert.ok(requests.includes('DELETE /api/v1/jobs/failed1'));

      // A refreshed browser adopting a running job must keep its card and
      // settings when the first poll reports failure, rather than dropping it.
      dismissed = false; live = true;
      await page.evaluate(() => window.store.getState().reconnectJobs());
      await card.getByText('Generation Failed', {exact:true}).waitFor();
      assert.ok(polls > 0);
      assert.equal(await page.evaluate(() => window.store.getState().jobs[0].generationDetails.params.seed), 42);

      // Pre-queue rejection keeps the original submission, including a cloned
      // LoRA array, and strips tokens from URL-selected adapter names.
      const snapshot = await page.evaluate(() => {
        const original = {seed:0, activated_loras:['https://example.invalid/turbo%20adapter.safetensors?token=secret'],
          loras_multipliers:'0', api_key:'secret', image_start:'C:/private/input.png'};
        const captured = window.captureDetails(original, 'Test');
        original.activated_loras[0]='changed.safetensors';
        return captured;
      });
      assert.deepEqual(snapshot.params.activated_loras, ['turbo adapter.safetensors']);
      assert.equal(snapshot.params.seed, 0);
      assert.ok(!JSON.stringify(snapshot).includes('secret'));
      await page.evaluate(() => window.store.setState({jobs:[{id:'legacy',status:'failed',progress:0,step:0,totalSteps:0,
        phase:'',message:'Old error',error:'Old error',outputFiles:[]}]}));
      await page.getByText('Settings & LoRAs', {exact:true}).click();
      await page.getByText('Settings were not captured for this older job.').waitFor();
      assert.deepEqual(errors, []);
      await page.close();
    }
  } finally { await browser.close(); }
  console.log('Failed job details: mobile/desktop snapshots, reconnect/polling, dismissal and legacy fallback passed.');
})().catch(error => {console.error(error);process.exitCode=1;});
