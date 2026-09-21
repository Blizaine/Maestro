// Exercise the real dialog with an isolated backend; no user's GPU jobs change.
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
      let project, voice, styleProject, upload = 0;
      let styles = [];
      const errors = [], posts = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.route('**/*', async route => {
        const request = route.request(), url = new URL(request.url());
        let result = {};
        if (url.pathname === '/') return route.fulfill({contentType: 'text/html', body: `<style>${css}</style><div id="root"></div><script>${bundle.outputFiles[0].text}</script>`});
        if (url.pathname === '/api/v1/music-styles') result = {styles};
        else if (url.pathname === '/api/v1/music-training/projects') result = {projects: [project, voice, styleProject].filter(Boolean)};
        else if (url.pathname === '/api/v1/upload-audio') result = {path: `song-${++upload}.wav`};
        else if (request.method() === 'POST') {
          const op = url.pathname.startsWith('/api/v1/cancel/') ? 'cancel' : url.pathname.split('/').pop(); const body = request.postDataJSON(); posts.push({op, body});
          if (op === 'drafts') {
            project = {...body, id: 'auto-project', tracks: body.tracks.map((track, index) => ({...track, id: `track-${index}`})),
              preparation_draft: true, preparation: {version: 1, revision: 0, songs: {}}, status: 'draft', message: 'Ready', progress: 0, checkpoints: []};
            result = project;
          } else if (op === 'auto-train') {
            project.auto_training = {...project.auto_training, version: 1, ...body, status: 'queued', stage: project.auto_training?.stage || 'analyze-songs'};
            Object.assign(project, {status: 'queued', auto_training_root: project.id, job_id: 'auto-job', message: 'Waiting for the GPU'});
            result = {job_id: project.job_id};
          } else if (op === 'cancel') {result = {status: 'cancelled'};}
          else throw Error(`Unexpected frontend orchestration: ${op}`);
        }
        return route.fulfill({contentType: 'application/json', body: JSON.stringify(result)});
      });
      await page.goto('http://auto-training.test/');
      await page.getByRole('button', {name: 'Train a style', exact: true}).click();
      assert.equal(await page.getByRole('button', {name: 'Guided', exact: true}).getAttribute('aria-pressed'), 'true');
      await page.getByRole('button', {name: 'Auto', exact: true}).click();
      assert.equal(await page.getByLabel('Voice & sound steps', {exact: true}).inputValue(), '100');
      assert.equal(await page.getByLabel('Song-style steps', {exact: true}).inputValue(), '200');
      await page.getByLabel('Project name', {exact: true}).fill('Auto fixture');
      await page.getByLabel('Style trigger', {exact: true}).fill('Original fixture voice');
      await page.getByLabel('Recordings', {exact: true}).setInputFiles([
        {name: 'practice.wav', mimeType: 'audio/wav', buffer: Buffer.from('practice')},
        {name: 'check.wav', mimeType: 'audio/wav', buffer: Buffer.from('check')},
      ]);
      await page.getByLabel('Check only — do not train on this song (held out)').nth(1).waitFor();
      assert.equal(await page.getByLabel('Check only — do not train on this song (held out)').nth(1).isChecked(), true);
      await page.getByLabel('Voice & sound steps', {exact: true}).fill('0');
      assert.equal(await page.getByRole('button', {name: 'Start Auto training', exact: true}).isDisabled(), true);
      await page.getByLabel('Voice & sound steps', {exact: true}).fill('125');
      await page.getByLabel('Song-style steps', {exact: true}).fill('300');
      await page.getByRole('button', {name: 'Start Auto training', exact: true}).click();
      await page.getByText('Waiting for the GPU', {exact: true}).waitFor();
      assert.deepEqual(posts.map(row => row.op), ['drafts', 'auto-train']);
      assert.deepEqual(posts[1].body, {voice_steps: 125, style_steps: 300});
      assert.equal(posts[0].body.tokenizer_pair, 'v9');
      assert.equal(await page.getByRole('button', {name: 'Open prepared clips', exact: true}).isDisabled(), true);
      await page.getByRole('button', {name: 'Close My music', exact: true}).click();
      project.auto_training.status = 'running'; project.auto_training.stage = 'adapt-pair'; project.status = 'training'; project.message = 'Learning voice & sound'; project.progress = 40;
      await page.evaluate(() => window.mount());
      await page.getByRole('button', {name: 'Train a style', exact: true}).click();
      await page.getByLabel('Training project', {exact: true}).selectOption(project.id);
      await page.getByText('Learning voice & sound', {exact: true}).waitFor();
      assert.equal(posts.length, 2, 'Reopening only observes the backend-owned run');
      await page.getByRole('button', {name: 'Stop Auto training', exact: true}).click();
      await page.getByText('Stopping after the current step is saved…', {exact: true}).waitFor();
      assert.equal(await page.getByRole('button', {name: 'Open prepared clips', exact: true}).isDisabled(), true);
      project.status = 'cancelled'; project.auto_training.status = 'cancelled'; project.message = 'Stopped with checkpoint saved';
      await page.getByRole('button', {name: 'Resume Auto training', exact: true}).click();
      await page.getByText('Waiting for the GPU', {exact: true}).waitFor();
      assert.deepEqual(posts.at(-1), {op: 'auto-train', body: {voice_steps: 125, style_steps: 300}});
      project.status = 'failed'; project.auto_training.status = 'failed'; project.message = 'Could not finish the current stage; saved work can resume';
      await page.getByText(project.message, {exact: true}).waitFor();
      await page.getByRole('button', {name: 'Resume Auto training', exact: true}).click();
      await page.getByText('Waiting for the GPU', {exact: true}).waitFor();
      const before = posts.length;
      voice = {id: 'voice-project', name: 'Auto fixture · voice', trigger: project.trigger, auto_training_root: project.id,
        status: 'completed', message: 'Saved', progress: 100, tracks: [], checkpoints: [], pair_prepared: {}, pair_completed_steps: 125,
        pair_resume_available: true, pair_checkpoints: [{step: 125, file: 'head-125.safetensors', scores: {}}]};
      styleProject = {id: 'style-project', name: 'Auto fixture · song style', trigger: project.trigger, auto_training_root: project.id,
        status: 'completed', message: 'Saved', progress: 100, tracks: [], prepared: {}, completed_steps: 300, resume_available: true,
        checkpoints: [{step: 300, file: 'step-300.safetensors', scores: {}}], adapted_pair: {step: 125, source_project: voice.id}};
      Object.assign(project.auto_training, {status: 'completed', stage: 'publish', sound_project_id: voice.id, style_project_id: styleProject.id, style_id: 'auto-lora',
        selection_summary: {training_clips: 7, check_clips: 2, unreviewed_clips: 9, warnings: ['Automatic transcript used']}});
      Object.assign(project, {status: 'completed', progress: 100, message: 'Auto training complete'});
      styles = [{id: 'auto-lora', name: 'Auto fixture · voice-125 · style-300', trigger: project.trigger, license: 'CC BY-NC 4.0'}];
      await page.getByRole('button', {name: 'Use LoRA & try a song', exact: true}).waitFor();
      await page.getByRole('button', {name: 'Open voice training', exact: true}).click();
      await page.getByRole('button', {name: 'Continue to song style', exact: true}).waitFor();
      await page.getByText('Expert settings', {exact: true}).click();
      await page.getByRole('button', {name: 'Resume voice & sound adaptation', exact: true}).waitFor();
      await page.getByRole('button', {name: 'Back to Auto overview', exact: true}).click();
      await page.getByRole('button', {name: 'Open song-style training', exact: true}).click();
      await page.getByText('Expert settings', {exact: true}).click();
      await page.getByRole('button', {name: 'Continue song-style training', exact: true}).waitFor();
      await page.getByRole('button', {name: 'Back to Auto overview', exact: true}).click();
      assert.equal(posts.length, before, 'Opening a completed stage does not start it');
      const dialog = page.getByRole('dialog', {name: 'My music'});
      assert.ok(await dialog.evaluate(el => el.scrollWidth <= el.clientWidth + 1));
      const screenshots = path.join(root, '.codex-tmp/music-auto-ui'); fs.mkdirSync(screenshots, {recursive: true});
      await page.getByLabel('Training project', {exact: true}).scrollIntoViewIfNeeded();
      await page.screenshot({path: path.join(screenshots, `complete-${width}.png`)});
      await page.getByRole('button', {name: 'Use LoRA & try a song', exact: true}).click();
      await page.waitForFunction(() => window.selected?.id === 'auto-lora');
      assert.equal(posts.length, before, 'Trying the saved LoRA only selects it');
      assert.deepEqual(errors, []);
      await page.close();
    }
    console.log('PASS Auto training: opt-in, configurable 100/200 defaults, backend-only progression, reopen, stop/resume, error recovery, completed-stage extensions and desktop/mobile layout.');
  } finally {await browser.close();}
})().catch(error => {console.error(error); process.exitCode = 1;});
