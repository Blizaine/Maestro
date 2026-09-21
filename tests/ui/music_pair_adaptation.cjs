// Isolated author-adaptation controls; no real training requests.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const {chromium} = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');
(async () => {
  const bundle = await esbuild.build({stdin: {contents: `import React from 'react'; import {createRoot} from 'react-dom/client'; import {AuthorMusicTraining} from './src/components/Sidebar/AuthorMusicTraining'; const root = createRoot(document.getElementById('root')); window.show = (project, sourceProject) => root.render(<AuthorMusicTraining key={project.id} project={project} sourceProject={sourceProject} busy={false} run={async action => {await action()}} onOpenProject={id => window.selected = id}/>);`, resolveDir: path.join(root, 'ui'), loader: 'tsx'}, bundle: true, write: false, jsx: 'automatic'});
  const cssFile = fs.readdirSync(path.join(root, 'ui/dist/assets')).find(n => n.endsWith('.css'));
  const css = fs.readFileSync(path.join(root, 'ui/dist/assets', cssFile), 'utf8');
  const browser = await chromium.launch({headless: true, ...(process.platform === 'win32' ? {executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'} : {})});
  try {
    for (const width of [1000, 390]) {
      const page = await browser.newPage({viewport: {width, height: 1000}});
      const calls = [], errors = [];
      page.on('pageerror', e => errors.push(e.message));
      await page.route('**/*', async route => {
        const url = new URL(route.request().url());
        if (url.pathname === '/') return route.fulfill({contentType: 'text/html', body: `<style>${css}body{background:#151515;color:#eee}#root{max-width:640px;margin:12px auto;padding:12px}</style><div id="root"></div><script>${bundle.outputFiles[0].text}</script>`});
        calls.push({operation: url.pathname.split('/').pop(), body: route.request().postDataJSON()});
        await route.fulfill({contentType: 'application/json', body: JSON.stringify(url.pathname.endsWith('/select-pair') ? {id: 'fresh-project'} : {job_id: 'job-one'})});
      });
      await page.goto('http://pair-fixture/');
      const project = {id: 'test', name: 'Artist adaptation', status: 'draft', tracks: [], checkpoints: []};
      await page.evaluate(p => window.show(p), project);
      await page.getByRole('button', {name: 'Prepare voice training'}).click();
      assert.equal(calls.at(-1).operation, 'prepare-pair');
      project.pair_prepared = {version: 1};
      await page.evaluate(p => window.show(p), project);
      await page.getByRole('button', {name: 'Train voice & sound', exact: true}).click();
      assert.deepEqual(calls.at(-1).body, {steps: 100, resume: false, seed: 22005});
      project.status = 'completed'; project.pair_completed_steps = 25; project.pair_resume_available = true;
      project.pair_training_options = {seed: 7, steps: 100};
      project.pair_checkpoints = [{step: 25, file: 'head-25.safetensors', scores: {heldout_spectral: 1.2}, audio_updates: 8}];
      await page.evaluate(p => window.show(p), project);
      await page.getByText('Expert settings', {exact: true}).click();
      await page.getByRole('button', {name: 'Resume voice & sound adaptation'}).click();
      assert.equal(calls.at(-1).body.resume, true); assert.equal(calls.at(-1).body.seed, 7);
      assert.equal(calls.at(-1).body.steps, 100, 'Resume keeps the unfinished target');
      await page.getByText('Saved sound versions & comparisons', {exact: true}).click();
      await page.getByRole('button', {name: 'Compare original / before / after'}).click();
      assert.equal(calls.at(-1).operation, 'reconstruct-pair');
      assert.equal(calls.at(-1).body.checkpoint, 'head-25.safetensors');
      await page.getByRole('status').waitFor();
      assert.equal(await page.locator('#root').evaluate(e => e.scrollWidth <= e.clientWidth + 1), true);
      fs.mkdirSync(path.join(root, '.codex-tmp/music-pair-ui'), {recursive: true});
      await page.screenshot({path: path.join(root, `.codex-tmp/music-pair-ui/author-${width}.png`)});
      await page.getByRole('button', {name: 'Continue to song style'}).click();
      await page.waitForFunction(() => window.selected === 'fresh-project');
      project.status = 'training'; await page.evaluate(p => window.show(p), project);
      assert.ok(await page.getByRole('button', {name: 'Resume voice & sound adaptation'}).isDisabled());
      project.status = 'completed'; project.pair_completed_steps = 100;
      const child = {id: 'child', name: 'Artist adaptation · adapted 100', status: 'prepared', tracks: [], checkpoints: [],
        pair_prepared: {version: 1}, adapted_pair: {step: 100, source_project: project.id}};
      const beforeNavigation = calls.length;
      await page.evaluate(({child, source}) => window.show(child, source), {child, source: project});
      assert.equal(await page.getByRole('button', {name: 'Train voice & sound', exact: true}).count(), 0);
      await page.getByText('100 sound steps completed', {exact: false}).waitFor();
      await page.getByRole('button', {name: 'Open original sound training'}).click();
      await page.waitForFunction(() => window.selected === 'test');
      assert.equal(calls.length, beforeNavigation, 'Opening the source must not start training');
      await page.screenshot({path: path.join(root, `.codex-tmp/music-pair-ui/source-${width}.png`)});
      await page.evaluate(p => window.show(p), project);
      await page.getByText('Expert settings', {exact: true}).click();
      await page.getByRole('spinbutton', {name: 'Sound adaptation steps'}).fill('400');
      await page.getByRole('button', {name: 'Resume voice & sound adaptation'}).click();
      assert.deepEqual(calls.at(-1).body, {steps: 400, resume: true, seed: 7});
      assert.deepEqual(errors, []); await page.close();
    }
    console.log('Author adaptation desktop/mobile: prepare, start/resume, compare, child/source navigation, total-step resume and busy controls pass.');
  } finally {await browser.close();}
})().catch(error => {console.error(error); process.exitCode = 1;});
