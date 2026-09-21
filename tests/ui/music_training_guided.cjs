// Real training UI with isolated APIs: no recordings or GPU jobs are changed.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const {chromium} = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');

(async () => {
  const bundle = await esbuild.build({stdin: {contents: `import React from 'react'; import {createRoot} from 'react-dom/client'; import {MyMusicDialog} from './src/components/Sidebar/MyMusicDialog'; const root = createRoot(document.getElementById('root')); window.mount = () => root.render(<MyMusicDialog onClose={() => root.render(null)} onSelect={style => window.selected = style}/>); window.mount();`, resolveDir: path.join(root, 'ui'), loader: 'tsx'}, bundle: true, write: false, jsx: 'automatic'});
  const assets = path.join(root, 'ui/dist/assets');
  const css = fs.readFileSync(path.join(assets, fs.readdirSync(assets).find(name => name.endsWith('.css'))), 'utf8');
  const browser = await chromium.launch({headless: true, ...(process.platform === 'win32' ? {executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'} : {})});
  try {
    for (const width of [1100, 390]) {
      const page = await browser.newPage({viewport: {width, height: 900}});
      page.setDefaultTimeout(12000);
      const errors = [], calls = [];
      page.on('pageerror', error => errors.push(error.message));
      const parent = {id: 'voice-project', name: 'My recordings', status: 'draft', progress: 0, message: 'Ready to train', tokenizer_pair: 'v9', tracks: [], checkpoints: []};
      let child, failTrain = false;
      const legacy = {id: 'legacy', name: 'Earlier style project', status: 'cancelled', progress: 80, message: 'Stopped at step 650', tokenizer_pair: 'v4', tracks: [], checkpoints: [{step: 650, file: 'step-650.safetensors', scores: {}}], completed_steps: 650, prepared: {}, resume_available: true, training_options: {steps: 800, rank: 32, seed: 777, learning_rate: .00002, lyric_alignment: true}, alignment: {ready: true, tracks: []}};
      await page.route('**/*', async route => {
        const request = route.request(), url = new URL(request.url());
        let result = {};
        if (url.pathname === '/') return route.fulfill({contentType: 'text/html', body: `<style>${css}</style><div id="root"></div><script>${bundle.outputFiles[0].text}</script>`});
        if (url.pathname === '/api/v1/music-styles') result = {styles: []};
        else if (url.pathname === '/api/v1/music-training/projects') result = {projects: [parent, ...(child ? [child] : []), legacy]};
        else if (request.method() === 'POST') {
          const op = url.pathname.split('/').pop(), body = request.postDataJSON();
          calls.push({op, body});
          if (op === 'prepare-pair') parent.pair_prepared = {version: 1};
          else if (op === 'adapt-pair') Object.assign(parent, {pair_completed_steps: body.steps, pair_resume_available: true, pair_training_options: body, status: 'completed', pair_checkpoints: [{step: body.steps, file: `head-${body.steps}.safetensors`, scores: {}, audio_updates: 1}]});
          else if (op === 'select-pair') {
            child = {id: 'style-project', name: 'My recordings · adapted 100', status: 'draft', message: 'Ready for song style', tracks: [], progress: 0, tokenizer_pair: 'v9', checkpoints: [], adapted_pair: {step: 100, source_project: parent.id}};
            result = child;
          } else if (op === 'prepare') child.prepared = {version: 1};
          else if (op === 'train') {
            if (failTrain) return route.fulfill({status: 400, contentType: 'application/json', body: JSON.stringify({detail: 'Please review the recordings'})});
            if (url.pathname.includes('/style-project/')) Object.assign(child, {training_options: body, completed_steps: body.steps, resume_available: true, status: 'completed', checkpoints: [{step: body.steps, file: `step-${body.steps}.safetensors`, scores: {}}]});
          } else if (op === 'audition-style') result = {id: 'trained-lora', name: 'My music', trigger: 'My voice'};
          else throw new Error(`Unexpected operation: ${op}`);
        }
        return route.fulfill({contentType: 'application/json', body: JSON.stringify(result)});
      });
      await page.goto('http://guided-music.test/');
      await page.getByRole('button', {name: 'Train a style', exact: true}).click();
      assert.equal(await page.getByLabel('Audio tokenizer and decoder').isVisible(), false, 'New projects hide implementation choices');
      await page.getByLabel('Training project', {exact: true}).selectOption(parent.id);
      assert.equal(await page.getByLabel('Training method', {exact: true}).isVisible(), false);
      assert.equal(await page.getByLabel('Sound adaptation steps').isVisible(), false);
      await page.getByRole('button', {name: 'Prepare voice training', exact: true}).click();
      await page.getByRole('button', {name: 'Train voice & sound', exact: true}).click();
      assert.deepEqual(calls.at(-1), {op: 'adapt-pair', body: {steps: 100, resume: false, seed: 22005}});
      await page.getByRole('button', {name: 'Continue to song style', exact: true}).click();
      assert.deepEqual(calls.at(-1), {op: 'select-pair', body: {checkpoint: 'head-100.safetensors'}});
      await page.getByText('Voice training is already included', {exact: false}).waitFor();
      const methods = await page.getByLabel('Training method', {exact: true}).locator('option').evaluateAll(options => options.map(option => option.value));
      assert.ok(!methods.includes('author'), 'Style stage cannot accidentally restart voice training');
      await page.getByRole('button', {name: 'Prepare song-style training'}).click();
      assert.equal(await page.getByLabel('Adapter rank').isVisible(), false);
      assert.equal(await page.getByLabel('Add extra word-timing guidance').isVisible(), false);
      await page.getByText('Your supplied lyrics are always used', {exact: false}).waitFor();
      failTrain = true;
      await page.getByRole('button', {name: 'Train song style', exact: true}).click();
      await page.getByRole('alert').waitFor();
      assert.equal(calls.at(-1).body.steps, 200);
      assert.equal(calls.at(-1).body.rank, 64);
      assert.equal(calls.at(-1).body.lyric_alignment, false);
      assert.equal(child.completed_steps, undefined, 'Failure does not advance the workflow');
      failTrain = false;
      await page.getByRole('button', {name: 'Train song style', exact: true}).click();
      await page.getByRole('button', {name: 'Use LoRA & try a song', exact: true}).waitFor();
      await page.getByText('Expert settings', {exact: true}).click();
      await page.getByRole('button', {name: 'Open original sound training', exact: true}).click();
      const before = calls.length;
      await page.getByRole('button', {name: 'Continue to song style', exact: true}).click();
      await page.getByRole('button', {name: 'Use LoRA & try a song', exact: true}).waitFor();
      assert.equal(calls.length, before, 'Returning to a learned voice reopens its existing style project');
      await page.getByText('Expert settings', {exact: true}).click();
      assert.equal(await page.getByLabel('Adapter rank').isDisabled(), true);
      assert.equal(await page.getByLabel('Total training steps').inputValue(), '400');
      await page.getByRole('button', {name: 'Continue song-style training', exact: true}).click();
      assert.equal(calls.at(-1).body.resume, true);
      assert.equal(calls.at(-1).body.steps, 400);
      await page.getByText('Song style saved at 400 steps.', {exact: false}).waitFor();
      await page.getByText('Expert settings', {exact: true}).click();
      const dialog = page.getByRole('dialog', {name: 'My music'});
      assert.ok(await dialog.evaluate(el => el.scrollWidth <= el.clientWidth + 1));
      const screenshots = path.join(root, '.codex-tmp/music-guided-ui'); fs.mkdirSync(screenshots, {recursive: true});
      await page.getByLabel('Training project', {exact: true}).scrollIntoViewIfNeeded();
      await page.screenshot({path: path.join(screenshots, `guided-${width}.png`)});
      await page.getByLabel('Training project', {exact: true}).selectOption('legacy');
      await page.getByText('Expert settings', {exact: true}).click();
      assert.equal(await page.getByLabel('Adapter rank').inputValue(), '32', 'Project switches cannot inherit another adapter rank');
      assert.equal(await page.getByLabel('Total training steps').inputValue(), '800');
      await page.getByText('Extra word-timing guidance · experimental', {exact: true}).click();
      assert.equal(await page.getByLabel('Add extra word-timing guidance').isChecked(), true);
      assert.equal(await page.getByLabel('Add extra word-timing guidance').isDisabled(), true);
      await page.getByRole('button', {name: 'Continue song-style training', exact: true}).click();
      assert.deepEqual(calls.at(-1).body, {steps: 800, rank: 32, seed: 777, learning_rate: .00002, resume: true, lyric_alignment: true, audition: {enabled: false}});
      await page.getByLabel('Training project', {exact: true}).selectOption(child.id);
      child.status = 'training'; child.job_id = 'job-active';
      await page.getByRole('button', {name: 'Stop after current step'}).waitFor();
      await page.getByText('Expert settings', {exact: true}).click();
      assert.equal(await page.getByLabel('Training method').isDisabled(), true);
      assert.equal(await page.getByLabel('Total training steps').isDisabled(), true);
      assert.equal(await page.getByRole('button', {name: 'Continue song-style training', exact: true}).isDisabled(), true);
      await page.getByRole('button', {name: 'Use LoRA & try a song', exact: true}).click();
      await page.waitForFunction(() => window.selected?.id === 'trained-lora');
      assert.equal(calls.at(-1).op, 'audition-style', 'Try a song selects saved weights, without launching a GPU job');
      assert.deepEqual(errors, []);
      await page.close();
    }
    console.log('PASS guided training: desktop/mobile, 100→200 default path, expert isolation, error recovery, matched voice handoff, existing-project reuse, saved resume settings and active-job controls.');
  } finally {await browser.close();}
})().catch(error => {console.error(error); process.exitCode = 1;});
