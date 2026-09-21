// Isolated desktop/mobile workflow: no requests reach the user's app or training jobs.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const {chromium} = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');
const recording = Buffer.alloc(44 + 32000);
recording.write('RIFF'); recording.writeUInt32LE(recording.length - 8, 4);
recording.write('WAVEfmt ', 8); recording.writeUInt32LE(16, 16);
recording.writeUInt16LE(1, 20); recording.writeUInt16LE(1, 22);
recording.writeUInt32LE(8000, 24); recording.writeUInt32LE(16000, 28);
recording.writeUInt16LE(2, 32); recording.writeUInt16LE(16, 34);
recording.write('data', 36); recording.writeUInt32LE(32000, 40);

(async () => {
  const bundle = await esbuild.build({stdin: {contents: `import React from 'react'; import {createRoot} from 'react-dom/client'; import {MyMusicDialog} from './src/components/Sidebar/MyMusicDialog'; createRoot(document.getElementById('root')).render(<MyMusicDialog onClose={() => {}} onSelect={() => {}}/>);`, resolveDir: path.join(root, 'ui'), loader: 'tsx'}, bundle: true, write: false, jsx: 'automatic'});
  const cssFile = fs.readdirSync(path.join(root, 'ui/dist/assets')).find(n => n.endsWith('.css'));
  const css = fs.readFileSync(path.join(root, 'ui/dist/assets', cssFile), 'utf8');
  const browser = await chromium.launch({headless: true, ...(process.platform === 'win32' ? {executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'} : {})});
  try {
    for (const width of [1100, 390]) {
      const page = await browser.newPage({viewport: {width, height: 880}});
      const errors = []; page.on('pageerror', e => errors.push(e.message));
      let draft, dataset, submitted, upload = 0, voiceRequest, saveRequest;
      const clip = {id: 'clip-1', start: .2, end: 1.4, lyrics: 'These are drafted words', style: 'Rhythmic music', delivery: 'unspecified', included: true, reviewed: false, warnings: ['Check the words.']};
      const song = held => ({duration: 30, language: 'en', holdout: held, speakers: [
        {id: 'lead', label: 'Voice 1', word_count: 100, confident_words: 92, seconds: 20, start: .2, end: 1.4},
        {id: 'guest', label: 'Voice 2', word_count: 10, confident_words: 8, seconds: 4, start: .4, end: 1.4}],
        selected_speakers: ['lead'], clips: [{...clip}], warnings: ['Voice labels are local to this song.']});
      await page.route('**/*', async route => {
        const url = new URL(route.request().url()), request = route.request();
        if (url.pathname === '/') return route.fulfill({contentType: 'text/html', body: `<style>${css}</style><div id="root"></div><script>${bundle.outputFiles[0].text}</script>`});
        if (url.pathname.includes('/audio') && request.method() === 'GET' || url.pathname.includes('/uploads/') || /\/recordings\/song-[01]$/.test(url.pathname)) {
          const range = /^bytes=(\d+)-(\d*)$/.exec(request.headers().range || '');
          const start = range ? +range[1] : 0, end = range?.[2] ? Math.min(+range[2], recording.length - 1) : recording.length - 1;
          return route.fulfill({status: range ? 206 : 200, contentType: 'audio/wav', headers: {'Accept-Ranges': 'bytes', ...(range ? {'Content-Range': `bytes ${start}-${end}/${recording.length}`} : {})}, body: recording.subarray(start, end + 1)});
        }
        let result;
        if (url.pathname.endsWith('/music-styles')) result = {styles: []};
        else if (url.pathname.endsWith('/music-training/projects')) result = {projects: [draft, dataset].filter(Boolean)};
        else if (url.pathname.endsWith('/upload-audio')) result = {path: `uploads/song-${++upload}.wav`, url: `/uploads/song-${upload}.wav`};
        else if (url.pathname.endsWith('/music-training/drafts')) {
          submitted = request.postDataJSON();
          draft = {...submitted, tracks: submitted.tracks.map((t, i) => ({...t, id: `song-${i}`})), id: 'draft-one', status: 'draft', progress: 0, message: 'Ready', checkpoints: [], preparation_draft: true, preparation: {version: 1, revision: 0, songs: {}}};
          result = draft;
        } else if (url.pathname.endsWith('/analyze-songs')) {
          draft = {...draft, status: 'prepared', message: 'Voices and clips ready', preparation: {...draft.preparation, revision: 1, songs: {'song-0': song(false), 'song-1': song(true)}}}; result = {job_id: 'analysis'};
        } else if (url.pathname.endsWith('/select-voices')) {
          voiceRequest = request.postDataJSON();
          const s = draft.preparation.songs[voiceRequest.track_id]; s.selected_speakers = voiceRequest.selected_speakers; s.clips = [{...clip}];
          draft.preparation.revision++; result = draft;
        } else if (url.pathname.endsWith('/save-preparation')) {
          saveRequest = request.postDataJSON();
          assert.equal(saveRequest.revision, draft.preparation.revision);
          Object.assign(draft.preparation.songs[saveRequest.track_id], {clips: saveRequest.clips, holdout: saveRequest.holdout}); draft.preparation.revision++; result = draft;
        } else if (url.pathname.endsWith('/build-dataset')) {
          assert.ok(Object.values(draft.preparation.songs).every(s => s.clips.every(c => c.reviewed)));
          dataset = {id: 'built-project', name: 'Reviewed music', status: 'draft', message: 'Ready to prepare audio tokens', progress: 0, tracks: submitted.tracks.map((t, i) => ({...t, id: `song-${i}`})), tokenizer_pair: 'v9', checkpoints: []};
          draft.preparation.dataset_project_id = dataset.id; result = {job_id: 'build'};
        } else throw new Error(`Unexpected fixture request: ${url.pathname}`);
        await route.fulfill({contentType: 'application/json', body: JSON.stringify(result)});
      });
      await page.goto('http://music-fixture/');
      await page.getByRole('button', {name: 'Train a style', exact: true}).click();
      assert.equal(await page.getByLabel('Audio tokenizer and decoder', {exact: true}).inputValue(), 'v9');
      await page.getByLabel('Project name', {exact: true}).fill('Song preparation test');
      await page.getByLabel('Style trigger', {exact: true}).fill('My test songs');
      await page.getByLabel('Recordings', {exact: true}).setInputFiles([{name: 'Train song.wav', mimeType: 'audio/wav', buffer: recording}, {name: 'Holdout song.wav', mimeType: 'audio/wav', buffer: recording}]);
      await page.getByText('Holdout song.wav', {exact: true}).waitFor();
      assert.equal(await page.getByLabel('Lyrics', {exact: true}).count(), 0);
      await page.getByRole('button', {name: 'Analyze songs', exact: true}).click();
      await page.getByText('Voices and clips ready', {exact: true}).waitFor();
      assert.equal(submitted.tokenizer_pair, 'v9'); assert.equal(submitted.tracks[0].lyrics, '');
      assert.ok(await page.getByRole('checkbox', {name: /Voice 1/}).isChecked());
      assert.ok(!await page.getByRole('checkbox', {name: /Voice 2/}).isChecked());
      assert.ok(await page.getByRole('button', {name: 'Create training dataset', exact: true}).isDisabled());
      await page.getByRole('button', {name: /Listen to voice 1/}).click();
      const player = page.getByLabel('Listen to voice 1', {exact: true});
      await player.evaluate(async audio => {
        if (audio.readyState < 1) await new Promise(resolve => audio.addEventListener('loadedmetadata', resolve, {once: true}));
        await new Promise(resolve => audio.addEventListener('timeupdate', resolve, {once: true}));
      });
      assert.equal(await player.evaluate(audio => audio.duration), 2);
      assert.ok(await player.evaluate(audio => audio.currentTime >= .2));
      await player.evaluate(audio => audio.pause());
      await page.getByRole('checkbox', {name: /Voice 2/}).check();
      await page.getByRole('button', {name: 'Suggest clips for selected voices', exact: true}).click();
      await page.waitForFunction(() => !document.body.innerText.includes('Working…'));
      assert.deepEqual(voiceRequest.selected_speakers, ['lead', 'guest']);
      await page.locator('summary').filter({hasText: 'Clip 1 ·'}).click();
      await page.getByLabel('Lyrics', {exact: true}).fill('These are corrected words');
      await page.getByLabel('Vocal delivery', {exact: true}).selectOption('rap');
      await page.getByRole('checkbox', {name: /I checked this clip/}).check();
      assert.ok(await page.getByLabel('Song to review', {exact: true}).isDisabled());
      await page.waitForTimeout(2700); // Polling must not erase unsaved labels.
      assert.equal(await page.getByLabel('Lyrics', {exact: true}).inputValue(), 'These are corrected words');
      await page.getByRole('button', {name: 'Save song review', exact: true}).click();
      await page.waitForFunction(() => !document.body.innerText.includes('Working…'));
      assert.equal(saveRequest.clips[0].delivery, 'rap'); assert.equal(saveRequest.holdout, false);
      await page.getByLabel('Song to review', {exact: true}).selectOption('song-1');
      await page.getByRole('checkbox', {name: /I have listened to and checked all included/}).check();
      await page.getByRole('button', {name: 'Save song review', exact: true}).click();
      await page.waitForFunction(() => !document.body.innerText.includes('Working…'));
      assert.equal(saveRequest.holdout, true);
      await page.getByRole('button', {name: 'Create training dataset', exact: true}).click();
      await page.getByRole('button', {name: 'Open training dataset', exact: true}).waitFor();
      const overflow = await page.locator('dialog').evaluate(el => el.scrollWidth > el.clientWidth + 1);
      assert.equal(overflow, false, `No horizontal scroll at ${width}px`);
      fs.mkdirSync(path.join(root, '.codex-tmp/music-dataset-ui'), {recursive: true});
      await page.screenshot({path: path.join(root, `.codex-tmp/music-dataset-ui/review-${width}.png`)});
      await page.getByRole('button', {name: 'Open training dataset', exact: true}).click();
      await page.getByRole('button', {name: 'Prepare voice training', exact: true}).waitFor();
      assert.equal(await page.getByLabel('Training project', {exact: true}).inputValue(), 'built-project');
      assert.deepEqual(errors, []);
      await page.close();
    }
    console.log('Automatic music preparation passes desktop/mobile, playback, draft review, polling and dataset handoff.');
  } finally {await browser.close();}
})().catch(error => {console.error(error); process.exitCode = 1;});
