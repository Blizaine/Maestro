// Isolated browser fixture: uploads, training and preferences never reach Maestro.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const {chromium} = require(process.env.MAESTRO_PLAYWRIGHT || 'C:/Users/bliza/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

// A real PCM recording lets the browser reject bad URLs or non-audio responses.
const recording = Buffer.alloc(44 + 16000);
recording.write('RIFF'); recording.writeUInt32LE(recording.length - 8, 4);
recording.write('WAVEfmt ', 8); recording.writeUInt32LE(16, 16);
recording.writeUInt16LE(1, 20); recording.writeUInt16LE(1, 22);
recording.writeUInt32LE(8000, 24); recording.writeUInt32LE(16000, 28);
recording.writeUInt16LE(2, 32); recording.writeUInt16LE(16, 34);
recording.write('data', 36); recording.writeUInt32LE(16000, 40);
for (let i = 0; i < 8000; i++) recording.writeInt16LE(Math.round(1000 * Math.sin(i * 2 * Math.PI * 220 / 8000)), 44 + i * 2);

async function verifyPlayback(player) {
  const playback = await player.evaluate(async audio => {
    audio.muted = true;
    await audio.play();
    await new Promise(resolve => audio.addEventListener('timeupdate', resolve, {once: true}));
    const result = {duration: audio.duration, time: audio.currentTime, error: audio.error?.message};
    audio.pause();
    return result;
  });
  assert.equal(playback.error, undefined);
  assert.equal(playback.duration, 1, 'Recording metadata contains the real duration');
  assert.ok(playback.time > 0, 'Recording playback advances');
}

(async () => {
  const bundle = await esbuild.build({stdin: {contents: `import React from 'react'; import {createRoot} from 'react-dom/client'; import {MyMusicDialog} from './src/components/Sidebar/MyMusicDialog'; window.mount = () => {const root = createRoot(document.getElementById('root')); root.render(<MyMusicDialog onClose={() => root.unmount()} onSelect={style => window.selected = style}/>);};`, resolveDir: path.join(root, 'ui'), loader: 'tsx'}, bundle: true, write: false, jsx: 'automatic'});
  const cssFile = fs.readdirSync(path.join(root, 'ui/dist/assets')).find(name => name.endsWith('.css'));
  const css = fs.readFileSync(path.join(root, 'ui/dist/assets', cssFile), 'utf8');
  const browser = await chromium.launch({headless: true,
    ...(process.platform === 'win32' ? {executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'} : {})});
  try {
    for (const width of [1100, 390]) {
      const page = await browser.newPage({viewport: {width, height: 820}});
      const errors = []; page.on('pageerror', error => errors.push(error.message));
      let project, upload = 0, submitted, audioSubmitted, cancelled = false, rendered;
      const style = {id: 'style-one', name: 'Original acoustic', trigger: 'My acoustic songs', license: 'CC BY-NC 4.0'};
      await page.route('**/*', async route => {
        const url = new URL(route.request().url()), body = route.request().postDataJSON;
        let result = {};
        if (url.pathname === '/') return route.fulfill({contentType: 'text/html', body: `<style>${css}</style><div id="root"></div><script>${bundle.outputFiles[0].text}</script><script>window.mount()</script>`});
        if (url.pathname.startsWith('/api/v1/uploads/') || url.pathname.includes('/recordings/')) {
          const valid = /^\/api\/v1\/uploads\/audio\/song-[12]\.wav$/.test(url.pathname)
            || /^\/api\/v1\/music-training\/projects\/project-one\/recordings\/track-[01]$/.test(url.pathname);
          if (!valid) return route.fulfill({status: 404, body: 'Recording not found'});
          const range = /^bytes=(\d+)-(\d*)$/.exec(route.request().headers().range || '');
          const start = range ? Number(range[1]) : 0, end = range?.[2] ? Math.min(Number(range[2]), recording.length - 1) : recording.length - 1;
          return route.fulfill({status: range ? 206 : 200, contentType: 'audio/wav',
            headers: {'Accept-Ranges': 'bytes', ...(range ? {'Content-Range': `bytes ${start}-${end}/${recording.length}`} : {})},
            body: recording.subarray(start, end + 1)});
        }
        if (url.pathname === '/api/v1/music-styles') result = {styles: [style]};
        else if (url.pathname === '/api/v1/upload-audio') result = {path: `uploads/audio/song-${++upload}.wav`, filename: `song-${upload}.wav`, url: `/api/v1/uploads/audio/song-${upload}.wav`};
        else if (url.pathname === '/api/v1/music-training/projects') {
          if (route.request().method() === 'POST') {
            submitted = body.call(route.request());
            project = {...submitted, tracks: submitted.tracks.map((track, index) => ({...track, id: `track-${index}`})), id: 'project-one', status: 'draft', progress: 0, message: 'Ready to prepare', checkpoints: [], reviewed_track_ids: []};
            result = project;
          } else result = {projects: project ? [project] : []};
        } else if (url.pathname.endsWith('/prepare')) {
          project = {...project, status: 'queued', job_id: 'training-job', message: 'Waiting to prepare music'};
          result = {job_id: 'training-job'};
        } else if (url.pathname.endsWith('/align-lyrics')) {
          project = {...project, alignment: {ready: true, tracks: []}};
        } else if (url.pathname.endsWith('/prepare-audio')) {
          project = {...project, audio_prepared: {version: 1}};
        } else if (url.pathname.endsWith('/adapt-audio')) {
          audioSubmitted = body.call(route.request());
          project = {...project, status: 'queued', job_id: 'training-job', message: 'Adapting source sound queued'};
        } else if (url.pathname.endsWith('/render-audition')) {
          rendered = body.call(route.request());
        } else if (url.pathname.endsWith('/review-data')) {
          project = {...project, review_drafts: {'track-0': {text: 'New draft words', note: 'Local draft, review required', language: 'en', seconds: 30, segments: [{text: 'New draft words', start: 0, end: 2, uncertain: true}]}}};
        } else if (url.pathname === '/api/v1/cancel/training-job') {
          cancelled = true; project = {...project, status: 'cancelled', message: 'Cancelled before starting', prepared: {version: 1}};
        }
        await route.fulfill({contentType: 'application/json', body: JSON.stringify(result)});
      });
      await page.goto('http://maestro-music.test/');
      const dialog = page.getByRole('dialog', {name: 'My music'});
      await dialog.waitFor();
      const bounds = await dialog.boundingBox();
      assert.ok(bounds.x >= 0 && bounds.x + bounds.width <= width, 'Music dialog stays within viewport');
      await page.getByRole('button', {name: 'Train a style', exact: true}).click();
      await page.getByLabel('Project name', {exact: true}).fill('My original songs');
      await page.getByLabel('Style trigger', {exact: true}).fill('My acoustic songs');
      await page.getByLabel('Recordings', {exact: true}).setInputFiles([
        {name: 'one.wav', mimeType: 'audio/wav', buffer: Buffer.from('one')},
        {name: 'two.wav', mimeType: 'audio/wav', buffer: Buffer.from('two')},
      ]);
      await page.getByLabel('Style caption', {exact: true}).nth(1).waitFor();
      await verifyPlayback(page.getByLabel('Recording one.wav', {exact: true}));
      await verifyPlayback(page.getByLabel('Recording two.wav', {exact: true}));
      for (let index = 0; index < 2; index++) {
        await page.getByLabel('Style caption', {exact: true}).nth(index).fill('Acoustic pop');
        await page.getByLabel('Full lyrics and sections', {exact: true}).nth(index).fill('[Verse]\nOriginal song');
      }
      assert.equal(await page.getByLabel('Held-out evaluation song').nth(1).isChecked(), true);
      await page.getByLabel('I listened and checked this caption and these lyrics', {exact: true}).first().check();
      await page.getByLabel('Style caption', {exact: true}).first().fill('Gentle acoustic pop');
      assert.equal(await page.getByLabel('I listened and checked this caption and these lyrics', {exact: true}).first().isChecked(), false, 'Editing a caption invalidates its review');
      await page.getByRole('button', {name: 'Create project', exact: true}).click();
      await page.getByRole('button', {name: 'Prepare audio tokens'}).click();
      await page.getByText('Waiting to prepare music', {exact: true}).waitFor();
      assert.equal(submitted.tracks.length, 2); assert.equal(submitted.tracks[1].holdout, true);
      await page.getByRole('button', {name: 'Stop after current step'}).click();
      await page.getByText('Cancelled before starting', {exact: true}).waitFor();
      assert.ok(cancelled);
      assert.equal(await page.getByLabel('Use lyric timing when training').isEnabled(), false);
      await page.getByRole('button', {name: 'Align lyrics', exact: true}).click();
      await page.getByLabel('Use lyric timing when training').check();
      await page.getByText('Adapt source sound · experimental', {exact: true}).click();
      await page.getByRole('button', {name: 'Prepare source sound', exact: true}).click();
      await page.getByLabel('Total audio training steps').fill('100');
      await page.getByLabel('Automatically audition checkpoints', {exact: true}).check();
      await page.getByLabel('Audition style caption', {exact: true}).fill('Solo voice and piano');
      await page.getByLabel('Audition lyrics', {exact: true}).fill('[Verse]\nNew test words for comparison');
      await page.getByLabel('Fixed seed', {exact: true}).fill('777');
      await page.getByLabel('Maximum seconds', {exact: true}).fill('30');
      assert.equal(await page.getByLabel('Music conditioning').inputValue(), '');
      await page.getByRole('button', {name: 'Adapt source sound', exact: true}).click();
      await page.getByText('Adapting source sound queued', {exact: true}).waitFor();
      assert.equal(audioSubmitted.steps, 100);
      assert.equal(audioSubmitted.conditioning_checkpoint, '');
      assert.equal(audioSubmitted.resume, false);
      assert.equal(audioSubmitted.audition.seed, 777);
      assert.equal(audioSubmitted.audition.enabled, true);
      assert.equal(await page.getByLabel('Fixed seed', {exact: true}).isDisabled(), true);
      await page.getByRole('button', {name: 'Stop after current step'}).click();
      await page.getByText('Cancelled before starting', {exact: true}).waitFor();
      project = {...project, checkpoints: [{step: 200, file: 'step-200.safetensors', scores: {heldout: 2.5}}], auditions: [{id: 'sample1', branch: 'style', step: 200, checkpoint: 'step-200.safetensors', request_id: 'same-test', status: 'completed', settings: audioSubmitted.audition, seconds: 28.2, engine: 'legacy'}]};
      const sample = page.getByLabel('style step 200 audition');
      await sample.waitFor();
      await sample.evaluate(audio => {audio.dataset.playbackIdentity = 'keep';});
      await page.waitForResponse(response => response.url().endsWith('/api/v1/music-training/projects'));
      assert.equal(await sample.getAttribute('data-playback-identity'), 'keep', 'Polling preserves audio playback element');
      assert.equal(await page.getByLabel('Fixed seed', {exact: true}).inputValue(), '777', 'Polling does not reset edited sample settings');
      await page.getByLabel('Automatically audition checkpoints', {exact: true}).scrollIntoViewIfNeeded();
      const reviewOutput = path.join(root, '.codex-tmp/sidebar-validation'); fs.mkdirSync(reviewOutput, {recursive: true});
      assert.equal(await dialog.evaluate(element => element.scrollWidth <= element.clientWidth + 1), true, 'Training controls do not overflow horizontally');
      await page.screenshot({path: path.join(reviewOutput, `my-music-auditions-${width}.png`)});
      await page.getByLabel('Render saved checkpoint').selectOption('0');
      assert.equal(rendered.checkpoint, 'step-200.safetensors');
      assert.equal(rendered.audition.seed, 777);
      await page.getByText('Recordings, captions & lyrics · 0/2 reviewed', {exact: true}).click();
      await verifyPlayback(page.getByLabel('Recording one.wav', {exact: true}));
      await verifyPlayback(page.getByLabel('Recording two.wav', {exact: true}));
      const savedLyrics = page.getByLabel(`Saved lyrics for ${project.tracks[0].name}`, {exact: true});
      assert.equal(await savedLyrics.inputValue(), '[Verse]\nOriginal song');
      assert.equal(await savedLyrics.getAttribute('readonly'), '');
      await page.getByRole('button', {name: 'Draft lyrics from recordings', exact: true}).click();
      await page.getByText('Local lyric draft · en · needs review', {exact: true}).click();
      await page.getByText('0.0–2.0s: New draft words (uncertain)', {exact: true}).waitFor();
      await page.screenshot({path: path.join(reviewOutput, `my-music-review-${width}.png`)});
      assert.equal(project.tracks[0].lyrics, '[Verse]\nOriginal song', 'Transcription does not replace source labels');
      await page.getByRole('button', {name: 'Edit captions and lyrics in a new project', exact: true}).click();
      assert.equal(await page.getByLabel('Project name', {exact: true}).inputValue(), 'My original songs · reviewed');
      await verifyPlayback(page.getByLabel('Recording one.wav', {exact: true}));
      assert.equal(await page.getByLabel('Held-out evaluation song').nth(1).isChecked(), true);
      await page.getByLabel('Full lyrics and sections', {exact: true}).nth(0).fill('[Verse]\nCorrected words');
      assert.equal(project.tracks[0].lyrics, '[Verse]\nOriginal song', 'Editing a draft preserves the original project');
      await page.getByRole('button', {name: 'Saved styles', exact: true}).click();
      await page.getByRole('button', {name: 'Use style', exact: true}).click();
      assert.equal(await page.evaluate(() => window.selected.id), style.id);
      await page.evaluate(() => window.mount());
      await dialog.waitFor();
      const output = path.join(root, '.codex-tmp/sidebar-validation'); fs.mkdirSync(output, {recursive: true});
      await page.screenshot({path: path.join(output, `my-music-${width}.png`)});
      await page.keyboard.press('Escape');
      await dialog.waitFor({state: 'detached'});
      assert.deepEqual(errors, []);
      await page.close();
    }
    console.log('PASS My music: desktop/mobile fit, draft/saved/copied recording playback, dataset upload, held-out labels, alignment review/copy, source-sound controls, shared-queue cancellation and style selection');
  } finally {await browser.close();}
})().catch(error => {console.error(error); process.exitCode = 1;});
